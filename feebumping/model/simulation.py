import logging
import math
import numpy as np
import random

from matplotlib import pyplot as plt
from pandas import DataFrame
from statemachine import StateMachine, AllocationError, ProcessingState
from transactions import ConsolidateFanoutTx, CancelTx
from utils import (
    cf_tx_size,
    P2WPKH_INPUT_SIZE,
    P2WPKH_OUTPUT_SIZE,
    TX_OVERHEAD_SIZE,
    BLOCKS_PER_DAY,
    REFILL_TX_SIZE,
    MAX_TX_SIZE,
    VAULT_AMOUNT,
)


class NoVaultToSpend(RuntimeError):
    pass


class Simulation(object):
    """Simulator for fee-reserve management of a Revault Watchtower."""

    def __init__(
        self,
        n_stk,
        n_man,
        locktime,
        hist_feerate_csv,
        reserve_strat,
        fallback_est_strat,
        cf_coin_selec,
        cancel_coin_selec,
        num_vaults,
        refill_excess,
        refill_period,
        unvault_rate,
        invalid_spend_rate,
        catastrophe_rate,
        delegate_rate,
        with_balance=False,
        with_divergence=False,
        with_op_cost=False,
        with_cum_op_cost=False,
        with_overpayments=False,
        with_risk_status=False,
        with_risk_time=False,
        with_fb_coins_dist=False,
    ):
        # Simulation parameters
        self.num_vaults = num_vaults
        self.refill_excess = refill_excess
        self.refill_period = refill_period
        self.unvault_rate = unvault_rate
        self.delegate_rate = delegate_rate

        # Manager parameters
        self.invalid_spend_rate = invalid_spend_rate
        self.catastrophe_rate = catastrophe_rate

        # WT state machine
        self.wt = StateMachine(
            n_stk,
            n_man,
            locktime,
            hist_feerate_csv,
            reserve_strat,
            fallback_est_strat,
            cf_coin_selec,
            cancel_coin_selec,
        )
        self.vault_id = 0

        # Plots configuration
        self.with_balance = with_balance
        self.balances = []
        self.with_divergence = with_divergence
        self.divergence = []
        self.with_op_cost = with_op_cost
        self.with_cum_op_cost = with_cum_op_cost
        self.costs = []
        self.wt_risk_time = []
        self.with_overpayments = with_overpayments
        self.overpayments = []
        self.with_risk_status = with_risk_status
        self.risk_status = []
        self.with_fb_coins_dist = with_fb_coins_dist
        self.fb_coins_dist = []
        self.scale_fixed = delegate_rate is None

        # Simulation report
        self.delegation_failures = 0
        self.delegation_successes = 0
        self.report_init = f"""\
        Watchtower config:\n\
            n_stk: {n_stk}\n\
            n_man: {n_man}\n\
            locktime: {locktime}\n\
            hist_feerate_csv: {hist_feerate_csv}\n\
            reserve_strat: {reserve_strat}\n\
            fallback_est_strat: {fallback_est_strat}\n\
            cf_coin_selec: {cf_coin_selec}\n\
            cancel_coin_selec: {cancel_coin_selec}\n\
        Simulation config:\n\
            Number of vaults: {self.num_vaults}\n\
            Refill excess: {self.refill_excess}\n\
            Expected active vaults: {self.num_vaults}\n\
            Refill period: {self.refill_period}\n\
            Unvault rate: {self.unvault_rate}\n\
            Invalid spend rate: {self.invalid_spend_rate}\n\
            Catastrophe rate: {self.catastrophe_rate}\n\
            Delegate rate: {self.delegate_rate}\n\
        """
        self.report_df = DataFrame(
            columns=[
                "mean_balance",
                "cum_ops_cost",
                "cum_cancel_fee",
                "cum_cf_fee",
                "cum_refill_fee",
                "time_at_risk",
                "mean_recovery_time",
                "median_recovery_time",
                "max_recovery_time",
                "delegation_failure_count",
                "delegation_failure_rate",
                "max_cancel_conf_time",
                "max_cf_conf_time",
                "max_risk_coef",
            ],
            index=[0],
        )

    def new_vault_id(self):
        self.vault_id += 1
        return self.vault_id

    def required_reserve_per_vault(self, block_height):
        """The absolute amount of sats the WT should have in reserve per vault.

        Note how the required reserve differs from the reserve feerate times the
        cancel transaction size and the number of vaults: the absolute amount of
        BTC also accounts for the cost of including a coin in the tx vin.
        """
        return sum(self.wt.fb_coins_dist(block_height))

    def required_reserve(self, block_height):
        """The total absolute amount of sats the WT should have in reserve."""
        return self.wt.vaults_count() * self.required_reserve_per_vault(block_height)

    def refill_amount(self, block_height, expected_new_vaults):
        """Returns amount to refill to ensure WT has sufficient operating balance.
        Used by stakeholder wallet software. R(t, E) in the paper.

        Note: stakeholder knows WT's balance, num_vaults, fb_coins_dist.
              Stakeholder doesn't know which coins are allocated or not.
        """
        # First of all get the overall needed reserve
        balance = self.wt.balance()
        target_dist = self.wt.fb_coins_dist(block_height)
        amount_needed_per_vault = sum(target_dist)
        reserve_needed = amount_needed_per_vault * (
            expected_new_vaults + self.wt.vaults_count() + self.refill_excess
        )

        # Refill if the balance is very low. Always refill an amount sufficient to create
        # at least one dist.
        if balance > reserve_needed * 1.01:
            return 0
        refill_amount = max(reserve_needed - balance, amount_needed_per_vault)

        # In addition to the absolute amount the WT wallet will need to pay the fee
        # for the CF transaction(s). Since it'll do it immediately we can estimate
        # the worst case feerate and add some slack to it.
        # FIXME: we assume there is only a single one..
        cf_feerate = self.wt.next_block_feerate(block_height)
        # We can't know how many inputs there will be to the CF tx since we don't know
        # how many coins will be consolidated. We arbitrarily assume 2% of thecoins will (FIXME)
        consolidated_coins_count = int(self.wt.coin_pool.n_coins() // 50) + 1
        cf_num_inputs = 1 + consolidated_coins_count
        # Same for the number of outputs. We assume the WT will create as many distributions
        # as it can from the new refill (rounded up), plus the number of coins it consolidated
        # (hence overestimating)
        new_dists = math.ceil(refill_amount / amount_needed_per_vault)
        cf_num_outputs = new_dists * len(target_dist)
        cf_fee = cf_tx_size(cf_num_inputs, cf_num_outputs) * cf_feerate

        refill_amount += cf_fee
        return int(refill_amount)

    def compute_reserve_divergence(self, block_height):
        """Compute how far the vault's reserves have divereged from the current fee reserve per vault.
        Compute the risk status; the total amount (satoshis) below the required reserve among available vaults."""
        vaults = self.wt.list_available_vaults()
        if len(vaults) == 0 or not (self.with_divergence or self.with_risk_status):
            return

        divergence = []
        for vault in vaults:
            div = vault.reserve_balance() - self.required_reserve_per_vault(
                block_height
            )
            divergence.append(div)
        if self.with_divergence:
            self.divergence.append(
                [
                    block_height,
                    sum(divergence) / len(vaults),
                    min(divergence),
                    max(divergence),
                ]
            )

        if self.with_risk_status:
            risk_by_vault = [abs(div) for div in divergence if div < 0]
            nominal_risk = sum(risk_by_vault)
            risk_coefficient = len(risk_by_vault) / len(vaults) * nominal_risk
            self.risk_status.append((block_height, risk_coefficient))

    def broadcast_cf_tx(self, block_height):
        cf_fee = self.wt.broadcast_consolidate_fanout(block_height)
        logging.info(
            f"  Consolidate-fanout transition at block {block_height} with fee:"
            f" {cf_fee}"
        )
        if self.cf_fee is None:
            self.cf_fee = 0
        self.cf_fee += cf_fee

    def refill_sequence(self, block_height, expected_new_vaults):
        refill_amount = self.refill_amount(block_height, expected_new_vaults)
        if refill_amount == 0:
            logging.info("  Refill not required, WT has enough bitcoin")
            return

        logging.info(f"Refill sequence at block {block_height}")
        logging.info(f"  Refill transition at block {block_height} by {refill_amount}")
        self.wt.refill(refill_amount)

        feerate = self.wt.next_block_feerate(block_height)
        self.refill_fee = REFILL_TX_SIZE * feerate
        self.broadcast_cf_tx(block_height)

    def delegate_sequence(self, block_height):
        # Top up allocations before processing a delegation, because time has passed, and
        # we mustn't accept a delegation if the available coin pool is insufficient.
        self.top_up_sequence(block_height)

        # Try to allocate fb coins from the pool to the new vault.
        vault_id = self.new_vault_id()
        logging.info(
            f"  Allocation transition at block {block_height} to vault {vault_id}"
        )
        try:
            self.wt.allocate(vault_id, VAULT_AMOUNT, block_height)
            self.delegation_successes += 1
        except AllocationError as e:
            logging.error(
                f"  Allocation transition FAILED for vault {vault_id}: {str(e)}"
            )
            self.delegation_failures += 1

    def top_up_sequence(self, block_height):
        # FIXME: that's ugly and confusing. What happens here is that allocate() will
        # return early if the vault doesn't need to be allocated, hence not raising. We
        # should check this here instead.

        # loop over copy since allocate may remove an element, changing list index
        for vault in list(self.wt.list_available_vaults()):
            try:
                # Allocation transition
                logging.info(
                    f"  Allocation transition at block {block_height} to vault"
                    f" {vault.id}"
                )
                assert isinstance(vault.amount, int)
                self.wt.allocate(vault.id, vault.amount, block_height)
            except AllocationError as e:
                logging.error(
                    f"  Allocation transition FAILED for vault {vault.id}: {str(e)}"
                )

    def spend(self, block_height):
        if len(self.wt.list_available_vaults()) == 0:
            raise NoVaultToSpend

        vault_id = random.choice(self.wt.list_available_vaults()).id
        logging.info(
            f"  Spend transition with vault {vault_id} at block {block_height}"
        )
        self.wt.spend(vault_id, block_height)

    def cancel(self, block_height):
        if len(self.wt.list_available_vaults()) == 0:
            raise NoVaultToSpend

        vault_id = random.choice(self.wt.list_available_vaults()).id
        # Cancel transition
        cancel_inputs = self.wt.broadcast_cancel(vault_id, block_height)
        self.cancel_fee = sum(coin.amount for coin in cancel_inputs)
        logging.info(
            f"  Cancel transition with vault {vault_id} for fee: {self.cancel_fee}"
        )

        # Compute overpayments
        if self.with_overpayments:
            feerate = self.wt.next_block_feerate(block_height)
            needed_fee = self.wt.cancel_tx_fee(feerate, len(cancel_inputs))
            self.overpayments.append([block_height, self.cancel_fee - needed_fee])

    def catastrophe_sequence(self, block_height):
        if len(self.wt.list_available_vaults()) == 0:
            raise NoVaultToSpend

        self.top_up_sequence(block_height)
        logging.info(f"Catastrophe sequence at block {block_height}")
        for vault in self.wt.list_available_vaults():
            # Cancel transition
            cancel_inputs = self.wt.broadcast_cancel(vault.id, block_height)
            # If a cancel fee has already been paid this block, sum those fees
            # so that when plotting costs this will appear as one total operation
            # rather than several separate cancel operations
            cancel_fee = sum(coin.amount for coin in cancel_inputs)
            if self.cancel_fee is not None:
                self.cancel_fee += cancel_fee
            else:
                self.cancel_fee = cancel_fee
            logging.info(
                f"  Cancel transition with vault {vault.id} for fee: {cancel_fee}"
            )

    def confirm_sequence(self, height):
        """State transition which considers each tx in WT's mempool and checks if the offered
        fee-rate is sufficient.
        If so, applies the transaction to the state.
        If not, handles rejection for cancel transaction type or does nothing for others.
        """
        for tx in self.wt.unconfirmed_transactions():
            if isinstance(tx, ConsolidateFanoutTx):
                assert (
                    len(tx.txins) * P2WPKH_INPUT_SIZE
                    + len(tx.txouts) * P2WPKH_OUTPUT_SIZE
                    + TX_OVERHEAD_SIZE
                    <= MAX_TX_SIZE
                )
                logging.debug(
                    f"  Consolidate-fanout confirm transition at block {height}"
                )
                self.wt.finalize_consolidate_fanout(tx, height)
                # Some vaults may have had (some of) their coins consolidated
                # during the cf tx, need to top up those vaults when the new
                # fanout coins become available again.
                self.top_up_sequence(height)
                # If there was a change output, proceed with the next CF tx.
                if tx.txouts[-1].processing_state == ProcessingState.UNPROCESSED:
                    self.broadcast_cf_tx(height)
            elif isinstance(tx, CancelTx):
                logging.debug(f"  Cancel confirm transition at block {height}")
                confirmed = self.wt.finalize_cancel(tx, height)

                # Compute overpayments
                if self.with_overpayments:
                    if confirmed:
                        feerate = self.wt.next_block_feerate(height)
                        needed_fee = self.wt.cancel_tx_fee(feerate, len(tx.fbcoins))
                        # Note: negative overpayments (underpayments) possible if minfee for block was 0
                        self.overpayments.append([height, tx.fee - needed_fee])

            else:
                raise

    def run(self, start_block, end_block):
        """Iterate from {start_block} to {end_block}, executing transitions
        according to configuration.
        """
        self.start_block, self.end_block = start_block, end_block
        self.refill_fee, self.cf_fee, self.cancel_fee = None, None, None
        # A switch we use to determine whether we are under requirements
        is_risky = False

        # At startup allocate as many reserves as we expect to have vaults
        logging.info(
            f"Initializing at block {start_block} with {self.num_vaults} new vaults"
        )
        self.refill_sequence(start_block, self.num_vaults)

        # For each block in the range, simulate an action affecting the watchtower
        # (formally described as a sequence of transitions) based on the configured
        # probabilities and the historical data of the current block.
        # Then, populate some data at this block for later analysis (see the plot()
        # method).
        for block in range(start_block, end_block):
            # First of all, was any transaction confirmed in this block?
            self.confirm_sequence(block)

            # Refill once per refill period
            if block % self.refill_period == 0:
                self.refill_sequence(block, 0)

            if self.scale_fixed:
                # We always try to keep the number of expected vaults under watch. We might
                # not be able to allocate if a CF tx is pending but not yet confirmed.
                for _ in range(self.wt.vaults_count(), self.num_vaults):
                    try:
                        self.wt.allocate(self.new_vault_id(), VAULT_AMOUNT, block)
                    except AllocationError as e:
                        logging.error(
                            "Not enough funds to allocate all the expected vaults at"
                            f" block {block}: {str(e)}"
                        )
                        break
            else:
                # The delegate rate is per day
                if random.random() < self.delegate_rate / BLOCKS_PER_DAY:
                    self.delegate_sequence(block)

            # The spend rate is per day
            if random.random() < self.unvault_rate / BLOCKS_PER_DAY:
                if self.scale_fixed:
                    self.delegate_sequence(block)

                # generate invalid spend, requires cancel
                if random.random() < self.invalid_spend_rate:
                    try:
                        self.cancel(block)
                    except NoVaultToSpend:
                        logging.info("Failed to Cancel, no vault to spend")
                # generate valid spend, requires processing
                else:
                    try:
                        self.spend(block)
                    except NoVaultToSpend:
                        logging.info("Failed to Spend, no vault to spend")

            # The catastrophe rate is a rate per day
            if random.random() < self.catastrophe_rate / BLOCKS_PER_DAY:
                try:
                    self.catastrophe_sequence(block)
                except NoVaultToSpend:
                    logging.info("Failed to Cancel (catastrophe), no vault to spend")
                # Reboot operation after catastrophe
                self.refill_sequence(block, self.num_vaults)

            if self.with_balance:
                self.balances.append(
                    [
                        block,
                        self.wt.balance(),
                        self.required_reserve(block),
                        self.wt.unallocated_balance(),
                    ]
                )

            if self.with_op_cost or self.with_cum_op_cost:
                self.costs.append(
                    [block, self.refill_fee, self.cf_fee, self.cancel_fee]
                )
                self.refill_fee, self.cf_fee, self.cancel_fee = None, None, None

            if self.with_cum_op_cost:
                was_risky = is_risky
                is_risky = any(
                    self.wt.under_requirement(v, block)
                    for v in self.wt.list_available_vaults()
                )
                # If its state changed, record the block
                if not was_risky and is_risky:
                    risk_on = block
                elif was_risky and not is_risky:
                    risk_off = block
                    self.wt_risk_time.append((risk_on, risk_off))

            if self.with_divergence or self.with_risk_status:
                self.compute_reserve_divergence(block)

            if self.with_fb_coins_dist:
                if block % 10_000 == 0:
                    self.fb_coins_dist.append([block, self.wt.fb_coins_dist(block)])

            if self.wt.mempool != []:
                for tx in self.wt.mempool:
                    if isinstance(tx, CancelTx):
                        self.report_df["max_cancel_conf_time"].loc[0] = max(
                            self.report_df["max_cancel_conf_time"].loc[0],
                            block - tx.broadcast_height,
                        )
                        if block - tx.broadcast_height >= self.wt.locktime:
                            logging.info(
                                f"Transaction {tx} was not confirmed before the"
                                " expiration of the locktime!"
                            )
                            raise (
                                RuntimeError(
                                    "Watchtower failed to confirm cancel"
                                    " transaction in time. All your base are belong"
                                    " to us."
                                )
                            )
                    if isinstance(tx, ConsolidateFanoutTx):
                        self.report_df["max_cf_conf_time"].loc[0] = max(
                            self.report_df["max_cf_conf_time"].loc[0],
                            block - tx.broadcast_height,
                        )

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
                self.with_risk_status,
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
                columns=["block", "Balance", "Required Reserve", "Unallocated Balance"],
            )
            bal_df.set_index(["block"], inplace=True)
            bal_df.plot(ax=axes[plot_num], title="WT Balance", legend=True)
            axes[plot_num].set_xlabel("Block", labelpad=15)
            axes[plot_num].set_ylabel("Satoshis", labelpad=15)
            self.report_df["mean_balance"] = bal_df["Balance"].mean()
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
                s=10,
                color="r",
                ax=axes[plot_num],
                label="Refill Fee",
            )
            costs_df.plot.scatter(
                x="block",
                y="CF Fee",
                s=10,
                color="g",
                ax=axes[plot_num],
                label="CF Fee",
            )
            costs_df.plot.scatter(
                x="block",
                y="Cancel Fee",
                s=10,
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
            self.report_df["cum_cancel_fee"].loc[0] = cumulative_costs_df[
                "Cancel Fee"
            ].iloc[-1]
            self.report_df["cum_cf_fee"].loc[0] = cumulative_costs_df["CF Fee"].iloc[-1]
            self.report_df["cum_refill_fee"].loc[0] = cumulative_costs_df[
                "Refill Fee"
            ].iloc[-1]
            self.report_df["cum_ops_cost"].loc[0] = (
                self.report_df["cum_refill_fee"].loc[0]
                + self.report_df["cum_cf_fee"].loc[0]
                + self.report_df["cum_cancel_fee"].loc[0]
            )
            report += (
                "Total cumulative cancel fee cost:"
                f" {self.report_df['cum_cancel_fee'].loc[0]}\n"
            )
            report += (
                "Total cumulative consolidate-fanout fee cost:"
                f" {self.report_df['cum_cf_fee'].loc[0]}\n"
            )
            report += (
                "Total cumulative refill fee cost:"
                f" {self.report_df['cum_refill_fee'].loc[0]}\n"
            )
            report += (
                f"Total cumulative cost: {self.report_df['cum_ops_cost'].loc[0]}\n"
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
            self.report_df["time_at_risk"].loc[0] = risk_time

            # What about avg recovery time?
            recovery_times = []
            for (risk_on, risk_off) in self.wt_risk_time:
                recovery_times.append(risk_off - risk_on)
            if recovery_times != []:
                self.report_df["mean_recovery_time"].loc[0] = np.mean(recovery_times)
                report += (
                    f"Mean recovery time: {self.report_df['mean_recovery_time'].loc[0]}"
                    " blocks\n"
                )
                self.report_df["median_recovery_time"].loc[0] = np.median(
                    recovery_times
                )
                report += (
                    "Median recovery time:"
                    f" {self.report_df['median_recovery_time'].loc[0]} blocks\n"
                )
                self.report_df["max_recovery_time"].loc[0] = max(recovery_times)
                report += (
                    f"Max recovery time: {self.report_df['max_recovery_time'].loc[0]}"
                    " blocks\n"
                )

            plot_num += 1

        # Plot vault reserves divergence
        if self.with_divergence and self.divergence != []:
            div_df = DataFrame(
                self.divergence,
                columns=["Block", "MeanDivergence", "MinDivergence", "MaxDivergence"],
            )
            div_df.set_index("Block", inplace=True)
            div_df["MeanDivergence"].plot(
                ax=axes[plot_num], label="Mean Divergence", legend=True
            )
            div_df["MinDivergence"].plot(
                ax=axes[plot_num], label="Minimum Divergence", legend=True
            )
            div_df["MaxDivergence"].plot(
                ax=axes[plot_num], label="Max Divergence", legend=True
            )
            axes[plot_num].set_xlabel("Block", labelpad=15)
            axes[plot_num].set_ylabel("Satoshis", labelpad=15)
            axes[plot_num].set_title("Vault Reserve \n Divergence from Requirement")
            plot_num += 1

        # Plot WT risk status
        if self.with_risk_status and self.risk_status != []:
            risk_status_df = DataFrame(
                self.risk_status, columns=["block", "risk coefficient"]
            )
            self.report_df["max_risk_coef"] = risk_status_df["risk coefficient"].max()
            risk_status_df.set_index(["block"], inplace=True)
            risk_status_df.plot(ax=axes[plot_num])
            axes[plot_num].set_title("Risk Coefficient, $\Omega$")
            axes[plot_num].set_ylabel("Severity", labelpad=15)
            axes[plot_num].set_xlabel("Block", labelpad=15)
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

            plot_num += 1

        # Report confirmation tracking
        report += (
            "Max confirmation time for a Cancel Tx:"
            f" {self.report_df['max_cancel_conf_time'].loc[0]}\n"
        )
        report += (
            "Max confirmation time for a Consolidate-fanout Tx:"
            f" {self.report_df['max_cf_conf_time'].loc[0]}\n"
        )

        if output is not None:
            plt.savefig(f"{output}.png")

        if show:
            plt.show()

        if self.delegation_failures > 0 or self.delegation_successes > 0:
            self.report_df["delegation_failure_count"].loc[0] = self.delegation_failures
            self.report_df["delegation_failure_rate"].loc[
                0
            ] = self.delegation_failures / (
                self.delegation_successes + self.delegation_failures
            )

            self.report_df["delegation_failure_rate"].loc[0] = None
            report += (
                f"Delegation failures: {self.delegation_failures} /"
                f" { (self.delegation_successes + self.delegation_failures)}"
                f" ({(self.delegation_failures /  (self.delegation_successes + self.delegation_failures) )* 100}%)\n"
            )

        return (report, self.report_df)

    def plot_fee_history(self, start_block, end_block, output=None, show=False):

        plt.style.use(["plot_style.txt"])
        fig, axes = plt.subplots(1, 1, figsize=(5.4, 3.9))
        self.wt.hist_df["mean_feerate"][start_block:end_block].plot(color="black")
        axes.set_ylabel("Satoshis per Weight Unit", labelpad=15)
        axes.set_xlabel("Block", labelpad=15)

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

    def plot_fee_estimate(
        self, comp_strat, start_block, end_block, output=None, show=False
    ):
        plt.style.use(["plot_style.txt"])
        estimates = []
        for block in range(start_block, end_block):
            est1 = self.wt.next_block_feerate(block)
            self.wt.fallback_est_strat = comp_strat
            est2 = self.wt._feerate(block)
            estimates.append([block, est1, est2])

        est_df = DataFrame(
            estimates, columns=["block", "estimateSmartFee-1", comp_strat]
        )
        est_df.set_index("block", inplace=True)
        fig = est_df.plot()

        plt.title("Feerate Estimates")
        plt.xlabel("Block", labelpad=15)
        plt.ylabel("Feerate (sats/vByte)", labelpad=15)

        if output is not None:
            plt.savefig(f"{output}.png")

        if show:
            plt.show()
