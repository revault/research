"""
TODO:
Update sequences to handle transaction broadcast & finalize. 
"""

import logging
import random

from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame
from statemachine import StateMachine
from transactions import ConsolidateFanoutTx, CancelTx
from utils import (
    cf_tx_size,
    P2WPKH_INPUT_SIZE,
    P2WPKH_OUTPUT_SIZE,
    BLOCKS_PER_DAY,
)


class AllocationError(Exception):
    pass


class NoVaultToSpend(RuntimeError):
    pass


class Simulation(object):
    """Simulator for fee-reserve management of a Revault Watchtower."""

    def __init__(
        self,
        n_stk,
        n_man,
        hist_feerate_csv,
        reserve_strat,
        estimate_strat,
        o_version,
        i_version,
        allocate_version,
        exp_active_vaults,
        refill_excess,
        refill_period,
        spend_rate,
        invalid_spend_rate,
        catastrophe_rate,
        with_balance=False,
        with_divergence=False,
        with_op_cost=False,
        with_cum_op_cost=False,
        with_overpayments=False,
        with_coin_pool=False,
        with_coin_pool_age=False,
        with_risk_status=False,
        with_risk_time=False,
        with_fb_coins_dist=False,
    ):
        # Stakeholder parameters
        self.expected_active_vaults = exp_active_vaults
        # In general 2 with reserve_strat = CUMMAX95Q90 and 10 to 15 with reserve_strat = 95Q90
        self.refill_excess = refill_excess
        self.refill_period = refill_period
        self.spend_rate = spend_rate

        # Manager parameters
        self.invalid_spend_rate = invalid_spend_rate
        self.catastrophe_rate = catastrophe_rate

        # WT state machine
        self.wt = StateMachine(
            n_stk,
            n_man,
            hist_feerate_csv,
            reserve_strat,
            estimate_strat,
            o_version,
            i_version,
            allocate_version,
        )
        self.vault_count = 0
        self.vault_id = 0

        # Simulation configuration
        self.with_balance = with_balance
        self.balances = []
        self.with_divergence = with_divergence
        self.divergence = []
        self.with_op_cost = with_op_cost
        self.with_cum_op_cost = with_cum_op_cost
        self.costs = []
        self.with_overpayments = with_overpayments
        self.overpayments = []
        self.with_coin_pool = with_coin_pool
        self.pool_after_refill = []
        self.pool_after_cf = []
        self.pool_after_spend = []
        self.pool_after_cancel = []
        self.pool_after_catastrophe = []
        self.with_risk_status = with_risk_status
        self.risk_status = []
        self.with_coin_pool_age = with_coin_pool_age
        self.coin_pool_age = []
        self.with_risk_time = with_risk_time
        self.wt_risk_time = []
        self.with_fb_coins_dist = with_fb_coins_dist
        self.fb_coins_dist = []
        self.vm_values = []

        # Simulation report
        self.report_init = f"""\
        Watchtower config:\n\
        O_0_factor: {self.wt.O_0_factor}\n\
        O_1_factor: {self.wt.O_1_factor}\n\
        Refill excess: {self.refill_excess}\n\
        Expected active vaults: {self.expected_active_vaults}\n\
        Refill period: {self.refill_period}\n\
        Spend rate: {self.spend_rate}\n\
        Invalid spend rate: {self.invalid_spend_rate}\n\
        Catastrophe rate: {self.catastrophe_rate}\n\
        """

    def new_vault_id(self):
        self.vault_id += 1
        return self.vault_id

    def required_reserve(self, block_height):
        """The amount the WT should have in reserve based on the number of active vaults"""
        required_reserve_per_vault = self.wt.fee_reserve_per_vault(block_height)
        num_vaults = len(self.wt.list_vaults())
        return num_vaults * required_reserve_per_vault

    def amount_needed(self, block_height, expected_new_vaults):
        """Returns amount to refill to ensure WT has sufficient operating balance.
        Used by stakeholder wallet software.
        R(t) in the paper.

        Note: stakeholder knows WT's balance and num_vaults (or expected_active_vaults).
              Stakeholder doesn't know which coins are allocated or not.
        """
        bal = self.wt.balance()
        frpv = self.wt.fee_reserve_per_vault(block_height)
        reserve_total = frpv * (
            expected_new_vaults + len(self.wt.list_vaults()) + self.refill_excess
        )
        R = reserve_total - bal
        if R <= 0:
            return 0

        new_reserves = R // (frpv)

        # Expected CF Tx fee
        try:
            feerate = self.wt._estimate_smart_feerate(block_height)
        except (ValueError, KeyError):
            feerate = self.wt._feerate(block_height)
        expected_num_outputs = len(self.wt.fb_coins_dist(block_height)) * new_reserves
        # just incase all coins are slected, plus the new refill output
        # FIXME: way too much
        expected_num_inputs = self.wt.coin_pool.n_coins() + 1
        expected_cf_fee = (
            cf_tx_size(expected_num_inputs, expected_num_outputs) * feerate
        )

        R += expected_cf_fee
        return int(R)

    def _reserve_divergence(self, block_height):
        if self.wt.list_vaults() != []:
            divergence = []
            frpv = self.wt.fee_reserve_per_vault(block_height)
            for vault in self.wt.list_vaults():
                div = vault.reserve_balance() - frpv
                divergence.append(div)
            # block, mean div, min div, max div
            self.divergence.append(
                [
                    block_height,
                    sum(divergence) / len(self.wt.list_vaults()),
                    min(divergence),
                    max(divergence),
                ]
            )

    def initialize_sequence(self, block_height):
        logging.debug(f"Initialize sequence at block {block_height}")
        # Refill transition
        refill_amount = self.amount_needed(block_height, self.expected_active_vaults)
        if refill_amount <= 0:
            logging.debug(f"  Refill not required, WT has enough bitcoin")
        else:
            self.wt.refill(refill_amount)
            logging.debug(
                f"  Refill transition at block {block_height} by {refill_amount}"
            )

            # Track operational costs
            # TODO: 2in 2out
            try:
                self.refill_fee = 109.5 * self.wt._estimate_smart_feerate(block_height)
            # FIXME: why key error?
            except (ValueError, KeyError):
                self.refill_fee = 109.5 * self.wt._feerate(block_height)

            # snapshot coin pool after refill Tx
            if self.with_coin_pool:
                amounts = [coin.amount for coin in self.wt.list_coins()]
                self.pool_after_refill.append([block_height, amounts])

            # Consolidate-fanout transition
            self.cf_fee = self.wt.broadcast_consolidate_fanout(block_height)
            logging.debug(
                f"  Consolidate-fanout transition at block {block_height} with fee:"
                f" {self.cf_fee}"
            )

            # snapshot coin pool after CF Tx
            if self.with_coin_pool:
                amounts = [coin.amount for coin in self.wt.list_coins()]
                self.pool_after_cf.append([block_height, amounts])

    def refill_sequence(self, block_height):
        refill_amount = self.amount_needed(block_height, 0)
        if refill_amount > 0:
            logging.debug(f"Refill sequence at block {block_height}")
            # Refill transition
            logging.debug(
                f"  Refill transition at block {block_height} by {refill_amount}"
            )
            self.wt.refill(refill_amount)

            # TODO: 2-in 2-out
            try:
                self.refill_fee = 109.5 * self.wt._estimate_smart_feerate(block_height)
            # FIXME: why key error?
            except (ValueError, KeyError):
                self.refill_fee = 109.5 * self.wt._feerate(block_height)

            # snapshot coin pool after refill Tx
            if self.with_coin_pool:
                amounts = [coin.amount for coin in self.wt.list_coins()]
                self.pool_after_refill.append([block_height, amounts])

            # Consolidate-fanout transition
            # Wait for confirmation of refill, then CF Tx
            self.cf_fee = self.wt.broadcast_consolidate_fanout(block_height)
            logging.debug(
                f"  Consolidate-fanout transition at block {block_height} with fee:"
                f" {self.cf_fee}"
            )

            # snapshot coin pool after CF Tx confirmation
            if self.with_coin_pool:
                amounts = [coin.amount for coin in self.wt.list_coins()]
                self.pool_after_cf.append([block_height, amounts])

    def _spend_init(self, block_height):
        # Top up sequence
        # Top up delegations before processing a delegation, because time has passed, and we mustn't accept
        # delegation if the available coin pool is insufficient.
        self.top_up_sequence(block_height)

        # Allocation transition
        # If WT fails to acknowledge new delegation, raise AllocationError
        try:
            # Delegate a vault
            amount = int(10e10)  # 100 BTC
            vault_id = self.new_vault_id()
            logging.debug(
                f"  Allocation transition at block {block_height} to vault {vault_id}"
            )
            self.wt.allocate(vault_id, amount, block_height)
            self.vault_count += 1
        except (RuntimeError):
            logging.debug(f"  Allocation transition FAILED for vault {vault_id}")
            raise (AllocationError())

        # choose a random vault to spend
        return random.choice(self.wt.list_vaults()).id

    def top_up_sequence(self, block_height):
        # loop over copy since allocate may remove an element, changing list index
        for vault in list(self.wt.list_vaults()):
            try:
                # Allocation transition
                logging.debug(
                    f"  Allocation transition at block {block_height} to vault"
                    f" {vault.id}"
                )
                assert isinstance(vault.amount, int)
                self.wt.allocate(vault.id, vault.amount, block_height)
            except (RuntimeError):
                logging.debug(f"  Allocation transition FAILED for vault {vault.id}")
                raise (AllocationError())

    def spend_sequence(self, block_height):
        logging.debug(f"Spend sequence at block {block_height}")
        if len(self.wt.list_vaults()) == 0:
            raise NoVaultToSpend

        vaultID = self._spend_init(block_height)
        # Spend transition
        logging.debug(f"  Spend transition at block {block_height}")
        self.wt.spend(vaultID, block_height)

        # snapshot coin pool after spend attempt
        if self.with_coin_pool:
            amounts = [coin.amount for coin in self.wt.list_coins()]
            self.pool_after_spend.append([block_height, amounts])

    def cancel_sequence(self, block_height):
        logging.debug(f"Cancel sequence at block {block_height}")
        if len(self.wt.list_vaults()) == 0:
            raise NoVaultToSpend

        vault_id = self._spend_init(block_height)
        # Cancel transition
        cancel_inputs = self.wt.broadcast_cancel(vault_id, block_height)
        self.wt.finalize_cancel(vault_id)
        self.cancel_fee = sum(coin.amount for coin in cancel_inputs)
        logging.debug(
            f"  Cancel transition with vault {vault_id} for fee: {self.cancel_fee}"
        )

        # snapshot coin pool after cancel
        if self.with_coin_pool:
            amounts = [coin.amount for coin in self.wt.list_coins()]
            self.pool_after_cancel.append([block_height, amounts])

        # Compute overpayments
        if self.with_overpayments:
            try:
                feerate = self.wt._estimate_smart_feerate(block_height)
            except (ValueError, KeyError):
                feerate = self.wt._feerate(block_height)
            self.overpayments.append([block_height, self.cancel_fee - feerate])

    def catastrophe_sequence(self, block_height):
        if len(self.wt.list_vaults()) == 0:
            raise NoVaultToSpend

        # Topup sequence
        self.top_up_sequence(block_height)

        logging.debug(f"Catastrophe sequence at block {block_height}")
        for vault in self.wt.list_vaults():
            # Cancel transition
            cancel_inputs = self.wt.broadcast_cancel(vault.id, block_height)
            self.wt.finalize_cancel(vault.id)
            # If a cancel fee has already been paid this block, sum those fees
            # so that when plotting costs this will appear as one total operation
            # rather than several separate cancel operations
            try:
                cancel_fee = sum(coin.amount for coin in cancel_inputs)
                self.cancel_fee += cancel_fee
            except (TypeError):
                cancel_fee = sum(coin.amount for coin in cancel_inputs)
                self.cancel_fee = cancel_fee
            logging.debug(
                f"  Cancel transition with vault {vault.id} for fee: {cancel_fee}"
            )

        # snapshot coin pool after all spend attempts are cancelled
        if self.with_coin_pool:
            amounts = [coin.amount for coin in self.wt.list_coins()]
            self.pool_after_catastrophe.append([block_height, amounts])

    def confirm_sequence(self, height):
        """State transition which considers each tx in WT's mempool and checks if the offered
        fee-rate is sufficient.
        If so, applies the transaction to the state.
        If not, handles rejection for cancel transaction type or does nothing for others.
        """
        for tx in self.wt.unconfirmed_transactions():
            if isinstance(tx, ConsolidateFanoutTx):
                self.wt.finalize_consolidate_fanout(tx, height)
                self.top_up_sequence(height)
            elif isinstance(tx, CancelTx):
                self.wt.finalize_cancel(tx, height)
            else:
                raise

    def run(self, start_block, end_block):
        """Iterate from {start_block} to {end_block}, executing transitions
        according to configuration.
        """
        self.start_block, self.end_block = start_block, end_block
        self.refill_fee, self.cf_fee, self.cancel_fee = None, None, None
        switch = "good"

        self.initialize_sequence(start_block)

        # For each block in the range, simulate an action affecting the watchtower
        # (formally described as a sequence of transitions) based on the configured
        # probabilities and the historical data of the current block.
        # Then, populate some data at this block for later analysis (see the plot()
        # method).
        for block in range(start_block, end_block):
            # First of all, was any transaction confirmed in this block?
            self.confirm_sequence(block)

            try:
                # Refill once per refill period
                if block % self.refill_period == 0:
                    self.refill_sequence(block)

                # The spend rate is a rate per day
                if random.random() < self.spend_rate / BLOCKS_PER_DAY:
                    # generate invalid spend, requires cancel
                    if random.random() < self.invalid_spend_rate:
                        try:
                            self.cancel_sequence(block)
                        except NoVaultToSpend:
                            logging.debug("Failed to Cancel, no vault to spend")
                    # generate valid spend, requires processing
                    else:
                        try:
                            self.spend_sequence(block)
                        except NoVaultToSpend:
                            logging.debug("Failed to Spend, no vault to spend")

                # The catastrophe rate is a rate per day
                if random.random() < self.catastrophe_rate / BLOCKS_PER_DAY:
                    try:
                        self.catastrophe_sequence(block)
                    except NoVaultToSpend:
                        logging.debug(
                            "Failed to Cancel (catastrophe), no vault to spend"
                        )
                    # Reboot operation after catastrophe
                    self.initialize_sequence(block)
            except (AllocationError):
                self.end_block = block
                logging.error(f"Allocation error at block {block}")
                break

            if self.with_balance:
                self.balances.append(
                    [
                        block,
                        self.wt.balance(),
                        self.required_reserve(block),
                        self.wt.unallocated_balance(),
                    ]
                )

            if self.with_risk_status:
                status = self.wt.risk_status(block)
                if (status["vaults_at_risk"] != 0) or (
                    status["delegation_requires"] != 0
                ):
                    self.risk_status.append(status)
            if self.with_op_cost or self.with_cum_op_cost:
                self.costs.append(
                    [block, self.refill_fee, self.cf_fee, self.cancel_fee]
                )
                self.refill_fee, self.cf_fee, self.cancel_fee = None, None, None

            if self.with_coin_pool_age:
                try:
                    processed = [
                        coin for coin in self.wt.list_coins() if coin.is_processed()
                    ]
                    ages = [block - coin.fan_block for coin in processed]
                    age = sum(ages)
                    self.coin_pool_age.append([block, age])
                except:
                    pass  # If processed is empty, error raised

            if self.with_cum_op_cost:
                # Check if wt becomes risky
                if switch == "good":
                    for vault in self.wt.list_vaults():
                        if self.wt.under_requirement(vault, block) != 0:
                            switch = "bad"
                            break
                    if switch == "bad":
                        risk_on = block

                # Check if wt no longer risky
                if switch == "bad":
                    any_risk = []
                    for vault in self.wt.list_vaults():
                        if self.wt.under_requirement(vault, block) != 0:
                            any_risk.append(True)
                            break
                    if True not in any_risk:
                        switch = "good"
                        risk_off = block
                        self.wt_risk_time.append((risk_on, risk_off))

            if self.with_divergence:
                self._reserve_divergence(block)

            if self.with_fb_coins_dist:
                if block % 1000 == 0:
                    self.fb_coins_dist.append([block, self.wt.fb_coins_dist(block)])
                self.vm_values.append([block, self.wt.Vm(block)])

    def plot(self, output=None, show=False):
        """Plot info about the simulation stored according to configuration.
        If {output} is set, will write the plot image to this file.

        Returns a string containing a "report" about the simulation.
        """
        report = self.report_init
        plt.style.use(["plot_style.txt"])

        subplots_len = sum(
            int(a)
            for a in [
                self.with_balance,
                self.with_divergence,
                self.with_op_cost,
                self.with_cum_op_cost,
                self.with_overpayments,
                self.with_coin_pool,
                self.with_risk_status,
                self.with_coin_pool_age,
                self.with_risk_time,
                self.with_fb_coins_dist,
            ]
        )
        figure, axes = plt.subplots(
            subplots_len, 1, sharex=True, figsize=(5.4, subplots_len * 3.9)
        )
        plot_num = 0

        # Plot WT balance vs total required reserve
        if self.with_balance and self.balances != []:
            bal_df = DataFrame(
                self.balances,
                columns=["block", "balance", "required reserve", "unallocated balance"],
            )
            bal_df.set_index(["block"], inplace=True)
            bal_df.plot(ax=axes[plot_num], title="WT Balance", legend=True)
            axes[plot_num].set_xlabel("Block", labelpad=15)
            axes[plot_num].set_ylabel("Satoshis", labelpad=15)
            plot_num += 1

        costs_df = None
        if self.costs != []:
            costs_df = DataFrame(
                self.costs, columns=["block", "Refill Fee", "CF Fee", "Cancel Fee"]
            )
            report += f"Refill operations: {costs_df['Refill Fee'].count()}\n"

        # Plot refill amount vs block, operating expense vs block
        if self.with_op_cost and costs_df is not None:
            costs_df.plot.scatter(
                x="block",
                y="Refill Fee",
                s=6,
                color="r",
                ax=axes[plot_num],
                label="Refill Fee",
            )
            costs_df.plot.scatter(
                x="block", y="CF Fee", s=6, color="g", ax=axes[plot_num], label="CF Fee"
            )
            costs_df.plot.scatter(
                x="block",
                y="Cancel Fee",
                s=6,
                color="b",
                ax=axes[plot_num],
                label="Cancel Fee",
            )
            axes[plot_num].legend(loc="upper left")
            axes[plot_num].set_title("Operating Costs Breakdown")
            axes[plot_num].set_ylabel("Satoshis", labelpad=15)
            axes[plot_num].set_xlabel("Block", labelpad=15)

            plot_num += 1

        # Plot cumulative operating costs (CF, Cancel, Spend)
        if self.with_cum_op_cost and costs_df is not None:
            cumulative_costs_df = costs_df
            cumulative_costs_df.set_index(["block"], inplace=True)
            cumulative_costs_df = cumulative_costs_df.fillna(0).cumsum()
            cumulative_costs_df.plot.line(
                ax=axes[plot_num],
                color={"Refill Fee": "r", "CF Fee": "g", "Cancel Fee": "b"},
            )
            axes[plot_num].legend(loc="upper left")
            axes[plot_num].set_title("Cumulative Operating Costs")
            axes[plot_num].set_ylabel("Satoshis", labelpad=15)
            axes[plot_num].set_xlabel("Block", labelpad=15)
            report += (
                "Total cumulative cancel fee cost:"
                f" {cumulative_costs_df['Cancel Fee'].iloc[-1]}\n"
            )
            report += (
                "Total cumulative consolidate-fanout fee cost:"
                f" {cumulative_costs_df['CF Fee'].iloc[-1]}\n"
            )
            report += (
                "Total cumulative refill fee cost:"
                f" {cumulative_costs_df['Refill Fee'].iloc[-1]}\n"
            )
            report += (
                "Total cumulative cost:"
                f" {cumulative_costs_df['Cancel Fee'].iloc[-1]+cumulative_costs_df['CF Fee'].iloc[-1]+cumulative_costs_df['Refill Fee'].iloc[-1]}\n"
            )

            # Highlight the plot with areas that show when the WT is at risk due to at least one
            # insufficient vault fee-reserve
            for (risk_on, risk_off) in self.wt_risk_time:
                axes[plot_num].axvspan(risk_off, risk_on, color="red", alpha=0.25)

            report += f"Analysis time span: {self.start_block} to {self.end_block}\n"
            risk_time = 0
            for (risk_on, risk_off) in self.wt_risk_time:
                risk_time += risk_off - risk_on
            report += f"Total time at risk: {risk_time} blocks\n"

            # What about avg recovery time?
            recovery_times = []
            for (risk_on, risk_off) in self.wt_risk_time:
                recovery_times.append(risk_off - risk_on)
            if recovery_times != []:
                report += f"Mean recovery time: {np.mean(recovery_times)} blocks\n"
                report += f"Median recovery time: {np.median(recovery_times)} blocks\n"
                report += f"Max recovery time: {max(recovery_times)} blocks\n"

            plot_num += 1

        # Plot coin pool amounts vs block
        if (
            self.with_coin_pool
            and self.pool_after_refill != []
            and self.pool_after_cf != []
            and self.pool_after_spend != []
            and self.pool_after_cancel != []
            and self.pool_after_catastrophe != []
        ):
            for frame in self.pool_after_refill:
                tuples = list(zip([frame[0] for i in frame[1]], frame[1]))
                pool_df = DataFrame(tuples, columns=["block", "amount"])
                pool_df.plot.scatter(
                    x="block",
                    y="amount",
                    color="r",
                    alpha=0.1,
                    s=5,
                    ax=axes[plot_num],
                    # label="After Refill",
                )
            for frame in self.pool_after_cf:
                tuples = list(zip([frame[0] for i in frame[1]], frame[1]))
                pool_df = DataFrame(tuples, columns=["block", "amount"])
                pool_df.plot.scatter(
                    x="block",
                    y="amount",
                    color="g",
                    alpha=0.1,
                    s=5,
                    ax=axes[plot_num],
                    # label="After CF",
                )
            for frame in self.pool_after_spend:
                tuples = list(zip([frame[0] for i in frame[1]], frame[1]))
                pool_df = DataFrame(tuples, columns=["block", "amount"])
                pool_df.plot.scatter(
                    x="block",
                    y="amount",
                    color="b",
                    alpha=0.1,
                    s=5,
                    ax=axes[plot_num],
                    # label="After Spend",
                )
            for frame in self.pool_after_cancel:
                tuples = list(zip([frame[0] for i in frame[1]], frame[1]))
                pool_df = DataFrame(tuples, columns=["block", "amount"])
                pool_df.plot.scatter(
                    x="block",
                    y="amount",
                    color="k",
                    alpha=0.1,
                    s=5,
                    ax=axes[plot_num],
                    # label="After Cancel",
                )
            for frame in self.pool_after_catastrophe:
                tuples = list(zip([frame[0] for i in frame[1]], frame[1]))
                pool_df = DataFrame(tuples, columns=["block", "amount"])
                pool_df.plot.scatter(
                    x="block",
                    y="amount",
                    color="orange",
                    alpha=0.1,
                    s=5,
                    ax=axes[plot_num],
                    # label="After Catastrophe",
                )
            # handles, labels = axes[plot_num].get_legend_handles_labels()
            # FIXME
            # try:
            #     i = subplots.index("operations")
            #     handles, _labels = axes[i].get_legend_handles_labels()
            #     labels = set(labels)
            #     axes[plot_num].legend(handles, labels, loc="upper right")
            # except (ValueError):
            #     pass
            axes[plot_num].set_title("Feebump Coin Pool")
            axes[plot_num].set_ylabel("Coin Amount (Satoshis)", labelpad=15)
            axes[plot_num].set_xlabel("Block", labelpad=15)
            plot_num += 1

        # Plot vault reserves divergence
        if self.with_divergence and self.divergence != []:
            div_df = DataFrame(
                self.divergence,
                columns=["Block", "MeanDivergence", "MinDivergence", "MaxDivergence"],
            )
            div_df.set_index("Block", inplace=True)
            div_df["MeanDivergence"].plot(
                ax=axes[plot_num], label="mean divergence", legend=True
            )
            div_df["MinDivergence"].plot(
                ax=axes[plot_num], label="minimum divergence", legend=True
            )
            div_df["MaxDivergence"].plot(
                ax=axes[plot_num], label="max divergence", legend=True
            )
            axes[plot_num].set_xlabel("Block", labelpad=15)
            axes[plot_num].set_ylabel("Satoshis", labelpad=15)
            axes[plot_num].set_title("Vault Divergence \nfrom Requirement")
            plot_num += 1

        # Plot WT risk status
        if self.with_risk_status and self.risk_status != []:
            risk_status_df = DataFrame(self.risk_status)
            risk_status_df.set_index(["block"], inplace=True)
            risk_status_df["num_vaults"].plot(
                ax=axes[plot_num], label="number of vaults", color="r", legend=True
            )
            risk_status_df["vaults_at_risk"].plot(
                ax=axes[plot_num], label="vaults at risk", color="b", legend=True
            )
            ax2 = axes[plot_num].twinx()
            risk_status_df["delegation_requires"].plot(
                ax=ax2, label="new delegation requires", color="g", legend=True
            )
            risk_status_df["severity"].plot(
                ax=ax2, label="total severity of risk", color="k", legend=True
            )
            axes[plot_num].set_ylabel("Vaults", labelpad=15)
            axes[plot_num].set_xlabel("Block", labelpad=15)
            ax2.set_ylabel("Satoshis", labelpad=15)
            plot_num += 1

        # Plot overpayments
        if self.with_overpayments and self.overpayments != []:
            df = DataFrame(self.overpayments, columns=["block", "overpayments"])
            df["cumulative"] = df["overpayments"].cumsum()
            df.plot(
                ax=axes[plot_num],
                x="block",
                y="overpayments",
                color="k",
                alpha=0.5,
                kind="scatter",
                label="Single",
                legend=True,
            ).legend(loc="center left")
            df.set_index(["block"], inplace=True)
            ax2 = axes[plot_num].twinx()
            df["cumulative"].plot(
                ax=ax2, label="Cumulative", color="b", legend=True
            ).legend(loc="center right")
            axes[plot_num].set_xlabel("Block", labelpad=15)
            axes[plot_num].set_ylabel("Satoshis", labelpad=15)
            axes[plot_num].set_title("Cancel Fee Overpayments")
            plot_num += 1

        # Plot coin pool age
        if self.with_coin_pool_age and self.coin_pool_age != []:
            age_df = DataFrame(self.coin_pool_age, columns=["block", "age"])
            age_df.plot.scatter(
                x="block",
                y="age",
                s=6,
                color="orange",
                ax=axes[plot_num],
                label="Total coin pool age",
            )
            plot_num += 1

        # Plot fb_coins_dist
        if self.with_fb_coins_dist and self.fb_coins_dist != []:
            for frame in self.fb_coins_dist:
                tuples = list(zip([frame[0] for i in frame[1]], frame[1]))
                df = DataFrame(tuples, columns=["Block", "amount"])
                df.plot.scatter(
                    x="Block",
                    y="amount",
                    style="-",
                    alpha=1,
                    s=10,
                    ax=axes[plot_num],
                    legend=False,
                )
                axes[plot_num].set_title("Fee-bump Coins Distribution")
                axes[plot_num].set_ylabel("Satoshis", labelpad=15)
                axes[plot_num].set_xlabel("Block", labelpad=15)
            if self.vm_values != []:
                df = DataFrame(self.vm_values, columns=["Block", "Vm"])
                df.set_index("Block", inplace=True)
                df.plot(ax=axes[plot_num], legend=True)
                axes[plot_num].legend(["$V_m$"])

            plot_num += 1

        if output is not None:
            plt.savefig(f"{output}.png")

        if show:
            plt.show()

        return report

    def plot_fee_history(self, start_block, end_block, output=None, show=False):

        plt.style.use(["plot_style.txt"])
        fig = self.wt.hist_df["mean_feerate"][start_block:end_block].plot()
        plt.title("Feerate History")
        plt.xlabel("Block", labelpad=15)
        plt.ylabel("Feerate (sats/vByte)", labelpad=15)
        fig.size = 3, 7

        if output is not None:
            plt.savefig(f"{output}.png")

        if show:
            plt.show()

    def plot_frpv(self, start_block, end_block, output=None, show=False):
        plt.style.use(["plot_style.txt"])
        frpv = []
        for block in range(start_block, end_block):
            frpv.append((block, self.wt.fee_reserve_per_vault(block)))
        df = DataFrame(frpv, columns=["block", "frpv"])
        df.set_index("block", inplace=True)
        fig = df.plot()
        plt.title("Fee Reserve per Vault")

        plt.xlabel("Block", labelpad=15)
        plt.ylabel("Feerate (sats/vByte)", labelpad=15)
        fig.size = 3, 7

        if output is not None:
            plt.savefig(f"{output}.png")

        if show:
            plt.show()

# FIXME: eventually have some small pytests
if __name__ == "__main__":

    # logging.basicConfig(level=logging.DEBUG)
    sim = Simulation(
        n_stk=5,
        n_man=5,
        hist_feerate_csv="historical_fees.csv",
        reserve_strat="CUMMAX95Q90",
        estimate_strat="ME30",
        o_version=1,
        i_version=2,
        allocate_version=1,
        exp_active_vaults=5,
        refill_excess=4 * 5,
        refill_period=1008,
        spend_rate=1,
        invalid_spend_rate=0.1,
        catastrophe_rate=0.05,
        with_balance=False,
        with_divergence=False,
        with_op_cost=False,
        with_cum_op_cost=False,
        with_overpayments=False,
        with_coin_pool=False,
        with_coin_pool_age=False,
        with_risk_status=False,
        with_risk_time=False,
        with_fb_coins_dist=False,
    )

    start_block = 200000
    end_block = 680000

    sim.run(start_block, end_block)

    # sim.plot_frpv(start_block, end_block, show=True)
    # sim.plot_fee_history(start_block, end_block, show=True)
