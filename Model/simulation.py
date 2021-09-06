import logging
import random

from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame
from utils import TX_OVERHEAD_SIZE, P2WPKH_INPUT_SIZE, P2WPKH_OUTPUT_SIZE
from statemachine import StateMachine


class AllocationError(Exception):
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
        exp_active_vaults,
        refill_excess,
        refill_period,
        delegation_period,
        invalid_spend_rate,
        catastrophe_rate,
        with_balance=False,
        with_vault_excess=False,
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
        self.delegation_period = delegation_period

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
        )
        self.vault_count = 0
        self.vault_id = 0

        # Simulation configuration
        self.with_balance = with_balance
        self.balances = []
        self.with_vault_excess = with_vault_excess
        self.vault_excess_before_cf = []
        self.vault_excess_after_cf = []
        self.vault_excess_after_delegation = []
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
        Delegation period: {self.delegation_period}\n\
        Invalid spend rate: {self.invalid_spend_rate}\n\
        Catastrophe rate: {self.catastrophe_rate}\n\
        """

    def new_vault_id(self):
        self.vault_id += 1
        return self.vault_id

    def required_reserve(self, block_height):
        """The amount the WT should have in reserve based on the number of active vaults"""
        required_reserve_per_vault = self.wt.fee_reserve_per_vault(
            block_height)
        num_vaults = len(self.wt.vaults)
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
            expected_new_vaults + len(self.wt.vaults) + self.REFILL_EXCESS
        
        R = reserve_total - bal
        if R <= 0:
            return 0

        new_reserves = R // (frpv)

        # Expected CF Tx fee
        try:
            feerate = self.wt._estimate_smart_feerate(block_height)
        except (ValueError, KeyError):
            feerate = self.wt._feerate(block_height)
        expected_num_outputs = len(
            self.wt.fb_coins_dist(block_height)) * new_reserves
        # just incase all coins are slected, plus the new refill output
        expected_num_inputs = len(self.wt.fbcoins) + 1 
        expected_cf_fee = (
            TX_OVERHEAD_SIZE
            + expected_num_outputs * P2WPKH_OUTPUT_SIZE
            + expected_num_inputs * P2WPKH_INPUT_SIZE
        ) * feerate

        R += expected_cf_fee
        return R

    def initialize_sequence(self, block_height):
        logging.debug(f"Initialize sequence at block {block_height}")
        # Refill transition
        refill_amount = self.amount_needed(
            block_height, self.EXPECTED_ACTIVE_VAULTS)
        if refill_amount <= 0:
            logging.debug(f"  Refill not required, WT has enough bitcoin")
        else:
            self.wt.refill(refill_amount)
            logging.debug(
                f"  Refill transition at block {block_height} by {refill_amount}"
            )

            # Track operational costs
            try:
                self.refill_fee = 109.5 * \
                    self.wt._estimate_smart_feerate(block_height)
            except (ValueError, KeyError):
                self.refill_fee = 109.5 * self.wt._feerate(block_height)

            # snapshot coin pool after refill Tx
            if self.with_coin_pool:
                amounts = [coin["amount"] for coin in self.wt.fbcoins]
                self.pool_after_refill.append([block_height, amounts])

            # snapshot vault excesses before CF Tx
            if self.with_vault_excess:
                vault_requirement = self.wt.fee_reserve_per_vault(block_height)
                excesses = []
                for vault in self.wt.vaults:
                    excess = (
                        sum([coin["amount"] for coin in vault["fee_reserve"]])
                        - vault_requirement
                    )
                    if excess > 0:
                        excesses.append(excess)
                self.vault_excess_before_cf.append([block_height, excesses])

            # Consolidate-fanout transition
            self.cf_fee = self.wt.consolidate_fanout(block_height)
            logging.debug(
                f"  Consolidate-fanout transition at block {block_height} with fee: {self.cf_fee}"
            )

            # snapshot coin pool after CF Tx
            if self.with_coin_pool:
                amounts = [coin["amount"] for coin in self.wt.fbcoins]
                self.pool_after_cf.append([block_height, amounts])

        # Allocation transitions
        for i in range(0, self.expected_active_vaults):
            amount = 10e10  # 100 BTC
            self.wt.allocate(self.new_vault_id(), amount, block_height)
            self.vault_count += 1

        # snapshot vault excesses after delegations
        if self.with_vault_excess:
            vault_requirement = self.wt.fee_reserve_per_vault(block_height)
            excesses = []
            for vault in self.wt.vaults:
                excess = (
                    sum([coin["amount"] for coin in vault["fee_reserve"]])
                    - vault_requirement
                )
                if excess > 0:
                    excesses.append(excess)
            self.vault_excess_after_delegation.append([block_height, excesses])

    def refill_sequence(self, block_height):
        refill_amount = self.amount_needed(block_height, 0)
        if refill_amount > 0:
            logging.debug(f"Refill sequence at block {block_height}")
            # Refill transition
            logging.debug(
                f"  Refill transition at block {block_height} by {refill_amount}"
            )
            self.wt.refill(refill_amount)

            try:
                self.refill_fee = 109.5 * \
                    self.wt._estimate_smart_feerate(block_height)
            except (ValueError, KeyError):
                self.refill_fee = 109.5 * self.wt._feerate(block_height)

            # snapshot coin pool after refill Tx
            if self.with_coin_pool:
                amounts = [coin["amount"] for coin in self.wt.fbcoins]
                self.pool_after_refill.append([block_height, amounts])

            # snapshot vault excesses before CF Tx
            if self.with_vault_excess:
                vault_requirement = self.wt.fee_reserve_per_vault(block_height)
                excesses = []
                for vault in self.wt.vaults:
                    excess = (
                        sum([coin["amount"] for coin in vault["fee_reserve"]])
                        - vault_requirement
                    )
                    if excess > 0:
                        excesses.append(excess)
                self.vault_excess_before_cf.append([block_height, excesses])

            # Consolidate-fanout transition
            # Wait for confirmation of refill, then CF Tx
            self.cf_fee = self.wt.consolidate_fanout(block_height + 1)
            logging.debug(
                f"  Consolidate-fanout transition at block {block_height+1} with fee: {self.cf_fee}"
            )

            # snapshot coin pool after CF Tx confirmation
            if self.with_coin_pool:
                amounts = [coin["amount"] for coin in self.wt.fbcoins]
                self.pool_after_cf.append([block_height + 7, amounts])

            # snapshot vault excesses after CF Tx
            if self.with_vault_excess:
                excesses = []
                for vault in self.wt.vaults:
                    excess = (
                        sum([coin["amount"] for coin in vault["fee_reserve"]])
                        - vault_requirement
                    )
                    if excess > 0:
                        excesses.append(excess)
                self.vault_excess_after_cf.append([block_height + 7, excesses])

            # Top up sequence
            # Top up delegations after confirmation of CF Tx, because consolidating coins
            # can diminish the fee_reserve of a vault
            self.top_up_sequence(block_height + 7)

    def _spend_init(self, block_height):
        # Top up sequence
        # Top up delegations before processing a delegation, because time has passed, and we mustn't accept
        # delegation if the available coin pool is insufficient.
        self.top_up_sequence(block_height)

        # Allocation transition
        # If WT fails to acknowledge new delegation, raise AllocationError
        try:
            # Delegate a vault
            amount = 10e10  # 100 BTC
            vault_id = self.new_vault_id()
            logging.debug(f"  Allocation transition at block {block_height} to vault {vault_id}")
            self.wt.allocate(vault_id, amount, block_height)
            self.vault_count += 1
        except (RuntimeError):
            logging.debug(f"  Allocation transition FAILED for vault {vault_id}")
            raise (AllocationError())

        # snapshot vault excesses after delegation
        if self.with_vault_excess:
            vault_requirement = self.wt.fee_reserve_per_vault(block_height)
            excesses = []
            for vault in self.wt.vaults:
                excess = (
                    sum([coin["amount"] for coin in vault["fee_reserve"]])
                    - vault_requirement
                )
                if excess > 0:
                    excesses.append(excess)
            self.vault_excess_after_delegation.append([block_height, excesses])

        # choose a random vault to spend
        vaultID = random.choice(self.wt.vaults)["id"]

        return vaultID

    def top_up_sequence(self, block_height):
        # loop over copy since allocate may remove an element, changing list index
        for vault in list(self.wt.vaults):
            try:
                # Allocation transition
                logging.debug(f"  Allocation transition at block {block_height} to vault {vault['id']}")
                self.wt.allocate(vault["id"], vault["amount"], block_height)
            except (RuntimeError):
                logging.debug(f"  Allocation transition FAILED for vault {vault['id']}")
                raise (AllocationError())

    def spend_sequence(self, block_height):
        logging.debug(f"Spend sequence at block {block_height}")
        vaultID = self._spend_init(block_height)
        # Spend transition
        logging.debug(f"  Spend transition at block {block_height}")
        self.wt.process_spend(vaultID)

        # snapshot coin pool after spend attempt
        if self.with_coin_pool:
            amounts = [coin["amount"] for coin in self.wt.fbcoins]
            self.pool_after_spend.append([block_height, amounts])

    def cancel_sequence(self, block_height):
        logging.debug(f"Cancel sequence at block {block_height}")
        vaultID = self._spend_init(block_height)
        # Cancel transition
        cancel_inputs = self.wt.process_cancel(vaultID, block_height)
        self.wt.finalize_cancel(vaultID)
        self.cancel_fee = sum(coin["amount"] for coin in cancel_inputs)
        logging.debug(
            f"  Cancel transition with vault {vaultID} for fee: {self.cancel_fee}"
        )

        # snapshot coin pool after cancel
        if self.with_coin_pool:
            amounts = [coin["amount"] for coin in self.wt.fbcoins]
            self.pool_after_cancel.append([block_height, amounts])

        # Compute overpayments
        if self.with_overpayments:
            try:
                feerate = self.wt._estimate_smart_feerate(block_height)
            except (ValueError, KeyError):
                feerate = self.wt._feerate(block_height)
            self.overpayments.append([block_height, self.cancel_fee - feerate])

    def catastrophe_sequence(self, block_height):
        # Topup sequence
        self.top_up_sequence(block_height)

        logging.debug(f"Catastrophe sequence at block {block_height}")
        for vault in self.wt.vaults:
            # Cancel transition
            cancel_inputs = self.wt.process_cancel(vault["id"], block_height)
            self.wt.finalize_cancel(vault["id"])
            # If a cancel fee has already been paid this block, sum those fees
            # so that when plotting costs this will appear as one total operation
            # rather than several separate cancel operations
            try:
                cancel_fee = sum(coin["amount"] for coin in cancel_inputs)
                self.cancel_fee += cancel_fee
            except (TypeError):
                cancel_fee = sum(coin["amount"] for coin in cancel_inputs)
                self.cancel_fee = cancel_fee
            logging.debug(
                f"  Cancel transition with vault {vault['id']} for fee: {cancel_fee}"
            )

        # snapshot coin pool after all spend attempts are cancelled
        if self.with_coin_pool:
            amounts = [coin["amount"] for coin in self.wt.fbcoins]
            self.pool_after_catastrophe.append([block_height, amounts])

    def run(self, start_block, end_block):
        """Iterate from {start_block} to {end_block}, executing transitions
        according to configuration.
        """
        self.start_block, self.end_block = start_block, end_block
        self.refill_fee, self.cf_fee, self.cancel_fee = None, None, None
        switch = "good"

        self.initialize_sequence(start_block)

        for block in range(start_block, end_block):
            try:
                # Refill sequence spans 8 blocks, musn't begin another sequence
                # with period shorter than that.
                if block % self.refill_period == 0:  # once per refill period
                    self.refill_sequence(block)

                # Fixme: assumes self.delegation_period > 20
                if (
                    block % self.delegation_period == 20
                ):  # once per delegation period on the 20th block
                    # generate invalid spend, requires cancel
                    if random.random() < self.invalid_spend_rate:
                        self.cancel_sequence(block)

                    # generate valid spend, requires processing
                    else:
                        self.spend_sequence(block)

                if block % 144 == 70:  # once per day on the 70th block
                    if random.random() < self.catastrophe_rate:
                        self.catastrophe_sequence(block)

                        # Reboot operation after catastrophe
                        self.initialize_sequence(block + 10)
            # Stop simulation, exit loop and report results
            except (AllocationError):
                self.end_block = block
                logging.error(f"Allocation error at block {block}")
                break

            if self.with_balance:
                self.balances.append(
                    [block, self.wt.balance(), self.required_reserve(block)]
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
                        coin for coin in self.wt.fbcoins if coin["processed"] != None
                    ]
                    ages = [block - coin["processed"] for coin in processed]
                    age = sum(ages)
                    self.coin_pool_age.append([block, age])
                except:
                    pass  # If processed is empty, error raised

            if self.with_cum_op_cost:
                # Check if wt becomes risky
                if switch == "good":
                    for vault in self.wt.vaults:
                        if self.wt.under_requirement(vault["fee_reserve"], block) != 0:
                            switch = "bad"
                            break
                    if switch == "bad":
                        risk_on = block

                # Check if wt no longer risky
                if switch == "bad":
                    any_risk = []
                    for vault in self.wt.vaults:
                        if self.wt.under_requirement(vault["fee_reserve"], block) != 0:
                            any_risk.append(True)
                            break
                    if True not in any_risk:
                        switch = "good"
                        risk_off = block
                        self.wt_risk_time.append((risk_on, risk_off))

            if self.with_fb_coins_dist:
                if block % 1000 == 0:
                    self.fb_coins_dist.append(
                        [block, self.wt.fb_coins_dist(block)])
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
                self.with_vault_excess,
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
        figure, axes = plt.subplots(subplots_len, 1, sharex=True)
        plot_num = 0

        # Plot WT balance vs total required reserve
        if self.with_balance and self.balances != []:
            bal_df = DataFrame(
                self.balances, columns=["block", "balance", "required reserve"]
            )
            bal_df.set_index(["block"], inplace=True)
            bal_df.plot(ax=axes[plot_num], title="WT Balance", legend=True)
            axes[plot_num].set_xlabel("Block", labelpad=15)
            axes[plot_num].set_ylabel("Satoshis", labelpad=15)
            plot_num += 1

        costs_df = None
        if self.costs != []:
            costs_df = DataFrame(
                self.costs, columns=[
                    "block", "Refill Fee", "CF Fee", "Cancel Fee"]
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
            report += f"Total cumulative cancel fee cost: {cumulative_costs_df['Cancel Fee'].iloc[-1]}\n"
            report += f"Total cumulative consolidate-fanout fee cost: {cumulative_costs_df['CF Fee'].iloc[-1]}\n"
            report += f"Total cumulative refill fee cost: {cumulative_costs_df['Refill Fee'].iloc[-1]}\n"

            # Highlight the plot with areas that show when the WT is at risk due to at least one
            # insufficient vault fee-reserve
            for (risk_on, risk_off) in self.wt_risk_time:
                axes[plot_num].axvspan(
                    risk_off, risk_on, color="red", alpha=0.25)

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
                    label="After Refill",
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
                    label="After CF",
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
                    label="After Spend",
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
                    label="After Cancel",
                )
            for frame in self.pool_after_catastrophe:
                tuples = list(zip([frame[0] for i in frame[1]], frame[1]))
                pool_df = DataFrame(tuples, columns=["block", "amount"])
                pool_df.plot.scatter(
                    x="block",
                    y="amount",
                    color="o",
                    alpha=0.1,
                    s=5,
                    ax=axes[plot_num],
                    label="After Catastrophe",
                )
            handles, labels = axes[plot_num].get_legend_handles_labels()
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

        if (
            self.with_vault_excess
            and self.vault_excess_before_cf != []
            and self.vault_excess_after_cf != []
            and self.vault_excess_after_delegation != []
        ):
            # AS SCATTER
            # for frame in self.vault_excess_after_cf:
            #     tuples = list(zip([frame[0] for i in frame[1]], frame[1]))
            #     excesses_df = DataFrame(tuples, columns=['block', 'amount'])
            #     # , label="After CF")
            #     excesses_df.plot.scatter(
            #         x='block', y='amount', color='r', ax=axes[plot_num])
            # for frame in self.vault_excess_after_delegation:
            #     tuples = list(zip([frame[0] for i in frame[1]], frame[1]))
            #     excesses_df = DataFrame(tuples, columns=['block', 'amount'])
            #     # , label="After Delegation")
            #     excesses_df.plot.scatter(
            #         x='block', y='amount', color='g', ax=axes[plot_num])
            # for frame in self.vault_excess_before_cf:
            #     tuples = list(zip([frame[0] for i in frame[1]], frame[1]))
            #     excesses_df = DataFrame(tuples, columns=['block', 'amount'])
            #     excesses_df.plot.scatter(
            #         x='block', y='amount', color='b', ax=axes[plot_num])
            # axes[plot_num].set_ylabel("Vault Excess (Satoshis)", labelpad=15)
            # axes[plot_num].set_xlabel("Block", labelpad=15)
            # plot_num += 1

            # Normalised sum of vault excesses
            # excesses_df = DataFrame(columns=['block', 'amount'])
            vault_excess_after_cf = []
            for frame in self.vault_excess_after_cf:
                vault_excess_after_cf.append(
                    (frame[0], sum(frame[1]) / self.expected_active_vaults)
                )
            excesses_df = DataFrame(
                vault_excess_after_cf, columns=["block", "After CF"]
            )
            excesses_df.plot.scatter(
                x="block", y="After CF", ax=axes[plot_num], color="r", label="After CF"
            )
            vault_excess_after_delegation = []
            for frame in self.vault_excess_after_delegation:
                vault_excess_after_delegation.append(
                    (frame[0], sum(frame[1]) / self.expected_active_vaults)
                )
            excesses_df = DataFrame(
                vault_excess_after_delegation, columns=[
                    "block", "After delegation"]
            )
            excesses_df.plot.scatter(
                x="block",
                y="After delegation",
                ax=axes[plot_num],
                color="g",
                label="After delegation",
            )
            vault_excess_before_cf = []
            for frame in self.vault_excess_before_cf:
                vault_excess_before_cf.append(
                    (frame[0], sum(frame[1]) / self.expected_active_vaults)
                )
            excesses_df = DataFrame(
                vault_excess_before_cf, columns=["block", "Before CF"]
            )
            excesses_df.plot.scatter(
                x="block",
                y="Before CF",
                ax=axes[plot_num],
                color="b",
                label="Before CF",
            )
            axes[plot_num].set_title("Mean Excess per Vault")
            axes[plot_num].set_ylabel("Satoshis", labelpad=15)
            axes[plot_num].set_xlabel("Block", labelpad=15)
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
            df = DataFrame(self.overpayments, columns=[
                           "block", "overpayments"])
            df["cumulative"] = df["overpayments"].cumsum()
            df.plot(ax=axes[plot_num], 
                    x="block",
                    y="overpayments",
                    color="k",
                    alpha=0.5,
                    kind="scatter",
                    label="Single",
                    legend=True).legend(loc="center left")
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

        figure.size = 5.4, plot_num*3.9

        if output is not None:
            plt.savefig(f"{output}.png")

        if show:
            plt.show()

        return report

    def plot_fee_history(start_block, end_block, output=None, show=False):

        plt.style.use(["plot_style.txt"])
        # Plot fee history
        self.wt.feerate_df["MeanFeerate"][start_block:end_block].plot(
            ax=axes[1])
        axes[1].set_ylabel("Feerate (sats/vByte)")
        axes[1].set_title("Historic Feerates")

        if output is not None:
            plt.savefig(f"{output}")

        if show:
            plt.show()
