from WTSM import *
import random

# Size of the non-input and non-output parts of a Segwit tx, in virtual bytes
TX_OVERHEAD_SIZE = 10.5
# Size of a P2WPKH input in virtual bytes
P2WPKH_INPUT_SIZE = 68
# Size of a P2WPKH output in virtual bytes
P2WPKH_OUTPUT_SIZE = 31

class AllocationError(Exception):
    pass


class WTSim(object):
    """Simulator for fee-reserve management of a Revault Watchtower."""

    def __init__(self, config, fname):
        # Stakeholder parameters
        self.EXPECTED_ACTIVE_VAULTS = 5  # Units: fee reserve per vault
        # In general 2 with reserve_strat = CUMMAX95Q90 and 10 to 15 with reserve_strat = 95Q90
        self.REFILL_EXCESS = 3 * self.EXPECTED_ACTIVE_VAULTS
        self.REFILL_PERIOD = 144 * 7
        self.DELEGATION_PERIOD = 144

        # Manager parameters
        self.INVALID_SPEND_RATE = 0.1
        self.CATASTROPHE_RATE = 0.005

        # WT state machine
        self.wt = WTSM(config)
        self.vaults_df = read_csv("vaultIDs.csv")

        # Simulation report
        self.fname = fname
        self.report_init = f"""\
        Watchtower config:\n\
        {config}O_0_factor: {self.wt.O_0_factor}\n\
        O_1_factor: {self.wt.O_1_factor}\n\
        Refill excess: {self.REFILL_EXCESS}\n\
        Expected active vaults: {self.EXPECTED_ACTIVE_VAULTS}\n\
        Refill period: {self.REFILL_PERIOD}\n\
        Delegation period: {self.DELEGATION_PERIOD}\n\
        Invalid spend rate: {self.INVALID_SPEND_RATE}\n\
        Catastrophe rate: {self.CATASTROPHE_RATE}\n\
        """

    def required_reserve(self, block_height):
        """The amount the WT should have in reserve based on the number of active vaults"""
        required_reserve_per_vault = self.wt.fee_reserve_per_vault(block_height)
        num_vaults = len(self.wt.vaults)
        return num_vaults * required_reserve_per_vault

    def R(self, block_height):
        """Returns amount to refill to ensure WT has sufficient operating balance.
        Used by stakeholder wallet software.

        Note: stakeholder knows WT's balance and num_vaults (or EXPECTED_ACTIVE_VAULTS).
              Stakeholder doesn't know which coins are allocated or not.
        """
        bal = self.wt.balance()
        frpv = self.wt.fee_reserve_per_vault(block_height)
        reserve_total = frpv * (
            self.EXPECTED_ACTIVE_VAULTS + self.REFILL_EXCESS
        )  # Should really be len(self.wt.vaults) not EXPECTED_ACTIVE_VAULTS but then initialise sequence wouldn't work
        R = reserve_total - bal
        if R <= 0:
            return 0

        new_reserves = R // (frpv)

        # Expected CF Tx fee
        try:
            feerate = self.wt._estimate_smart_feerate(block_height)
        except (ValueError, KeyError):
            feerate = self.wt._feerate(block_height)
        expected_num_outputs = len(self.wt.O(block_height)) * new_reserves
        expected_num_inputs = len(self.wt.O(block_height)) * len(self.wt.vaults)
        expected_cf_fee = (
            TX_OVERHEAD_SIZE
            + expected_num_outputs * P2WPKH_OUTPUT_SIZE
            + expected_num_inputs * P2WPKH_INPUT_SIZE
        ) * feerate

        R += expected_cf_fee
        return R

    def initialize_sequence(self, block_height):
        print(f"Initialize sequence at block {block_height}")
        ## Refill transition
        refill_amount = self.R(block_height)
        if refill_amount <= 0:
            print(f"  Refill not required, WT has enough bitcoin")
        else:
            self.wt.refill(refill_amount)
            print(f"  Refill transition at block {block_height} by {refill_amount}")

            # Track operational costs
            try:
                self.refill_fee = 109.5 * self.wt._estimate_smart_feerate(block_height)
            except (ValueError, KeyError):
                self.refill_fee = 109.5 * self.wt._feerate(block_height)

            # snapshot coin pool after refill Tx
            if "coin_pool" in self.subplots:
                amounts = [coin["amount"] for coin in self.wt.fbcoins]
                self.pool_after_refill.append([block_height, amounts])

            # snapshot vault excesses before CF Tx
            if "vault_excesses" in self.subplots:
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

            ## Consolidate-fanout transition
            self.cf_fee = self.wt.consolidate_fanout(block_height)
            print(
                f"  Consolidate-fanout transition at block {block_height} with fee: {self.cf_fee}"
            )

            # snapshot coin pool after CF Tx
            if "coin_pool" in self.subplots:
                amounts = [coin["amount"] for coin in self.wt.fbcoins]
                self.pool_after_cf.append([block_height, amounts])

        ## Allocation transitions
        for i in range(0, self.EXPECTED_ACTIVE_VAULTS):
            vaultID = self.vaults_df["vaultID"][self.vault_count]
            self.vault_count += 1
            amount = 10e10  # 100 BTC
            self.wt.allocate(vaultID, amount, block_height)

        # snapshot vault excesses after delegations
        if "vault_excesses" in self.subplots:
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
        refill_amount = self.R(block_height)
        if refill_amount > 0:
            print(f"Refill sequence at block {block_height}")
            ## Refill transition
            print(f"  Refill transition at block {block_height} by {refill_amount}")
            self.wt.refill(refill_amount)

            try:
                self.refill_fee = 109.5 * self.wt._estimate_smart_feerate(block_height)
            except (ValueError, KeyError):
                self.refill_fee = 109.5 * self.wt._feerate(block_height)

            # snapshot coin pool after refill Tx
            if "coin_pool" in self.subplots:
                amounts = [coin["amount"] for coin in self.wt.fbcoins]
                self.pool_after_refill.append([block_height, amounts])

            # snapshot vault excesses before CF Tx
            if "vault_excesses" in self.subplots:
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

            ## Consolidate-fanout transition
            # Wait for confirmation of refill, then CF Tx
            self.cf_fee = self.wt.consolidate_fanout(block_height + 1)
            print(
                f"  Consolidate-fanout transition at block {block_height+1} with fee: {self.cf_fee}"
            )

            # snapshot coin pool after CF Tx confirmation
            if "coin_pool" in self.subplots:
                amounts = [coin["amount"] for coin in self.wt.fbcoins]
                self.pool_after_cf.append([block_height + 7, amounts])

            # snapshot vault excesses after CF Tx
            if "vault_excesses" in self.subplots:
                excesses = []
                for vault in self.wt.vaults:
                    excess = (
                        sum([coin["amount"] for coin in vault["fee_reserve"]])
                        - vault_requirement
                    )
                    if excess > 0:
                        excesses.append(excess)
                self.vault_excess_after_cf.append([block_height + 7, excesses])

            ## Top up sequence
            # Top up delegations after confirmation of CF Tx, because consolidating coins
            # can diminish the fee_reserve of a vault
            self.top_up_sequence(block_height + 7)

    def _spend_init(self, block_height):
        ## Top up sequence
        # Top up delegations before processing a delegation, because time has passed, and we mustn't accept
        # delegation if the available coin pool is insufficient.
        self.top_up_sequence(block_height)

        # Delegate a vault
        vaultID = self.vaults_df["vaultID"][self.vault_count]
        self.vault_count += 1
        amount = 10e10  # 100 BTC

        ## Allocation transition
        # If WT fails to acknowledge new delegation, raise AllocationError
        try:
            self.wt.allocate(vaultID, amount, block_height)
        except (RuntimeError):
            raise (AllocationError())

        # snapshot vault excesses after delegation
        if "vault_excesses" in self.subplots:
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
                ## Allocation transition
                self.wt.allocate(vault["id"], vault["amount"], block_height)
            except (RuntimeError):
                print(f"  Allocation transition FAILED for vault {vault['id']}")
                raise (AllocationError())

    def spend_sequence(self, block_height):
        print(f"Spend sequence at block {block_height}")
        vaultID = self._spend_init(block_height)
        ## Spend transition
        print(f"  Spend transition at block {block_height}")
        self.wt.process_spend(vaultID)

        # snapshot coin pool after spend attempt
        if "coin_pool" in self.subplots:
            amounts = [coin["amount"] for coin in self.wt.fbcoins]
            self.pool_after_spend.append([block_height, amounts])

    def cancel_sequence(self, block_height):
        print(f"Cancel sequence at block {block_height}")
        vaultID = self._spend_init(block_height)
        ## Cancel transition
        cancel_inputs = self.wt.process_cancel(vaultID, block_height)
        self.wt.finalize_cancel(vaultID)
        self.cancel_fee = sum(coin["amount"] for coin in cancel_inputs)
        print(f"  Cancel transition with vault {vaultID} for fee: {self.cancel_fee}")

        # snapshot coin pool after cancel
        if "coin_pool" in self.subplots:
            amounts = [coin["amount"] for coin in self.wt.fbcoins]
            self.pool_after_cancel.append([block_height, amounts])

        # Compute overpayments
        if "overpayments" in subplots:
            try:
                feerate = self.wt._estimate_smart_feerate(block_height)
            except (ValueError, KeyError):
                feerate = self.wt._feerate(block_height)
            self.overpayments.append([block_height, self.cancel_fee - feerate])

    def catastrophe_sequence(self, block_height):
        print(f"Catastrophe sequence at block {block_height}")
        for vault in self.wt.vaults:
            ## Cancel transition
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
            print(f"  Cancel transition with vault {vault['id']} for fee: {cancel_fee}")

        # snapshot coin pool after all spend attempts are cancelled
        if "coin_pool" in self.subplots:
            amounts = [coin["amount"] for coin in self.wt.fbcoins]
            self.pool_after_catastrophe.append([block_height, amounts])

    def plot_simulation(self, start_block, end_block, subplots):

        plt.style.use(["plot_style.txt"])

        self.refill_fee, self.cf_fee, self.cancel_fee = None, None, None
        self.vault_count = 1

        self.subplots = subplots
        self.pool_after_refill = []
        self.pool_after_cf = []
        self.pool_after_spend = []
        self.pool_after_cancel = []
        self.pool_after_catastrophe = []
        self.vault_excess_before_cf = []
        self.vault_excess_after_cf = []
        self.vault_excess_after_delegation = []
        self.overpayments = []
        balances = []
        risk_status = []
        costs = []
        coin_pool_age = []
        wt_risk_time = []

        switch = "good"
        report = self.report_init

        self.initialize_sequence(start_block)

        for block in range(start_block, end_block):
            try:
                # Refill sequence spans 8 blocks, musn't begin another sequence
                # with period shorter than that.
                if block % self.REFILL_PERIOD == 0:  # once per refill period
                    self.refill_sequence(block)

                # Fixme: assumes self.DELEGATION_PERIOD > 20
                if (
                    block % self.DELEGATION_PERIOD == 20
                ):  # once per delegation period on the 20th block
                    # generate invalid spend, requires cancel
                    if random.random() < self.INVALID_SPEND_RATE:
                        self.cancel_sequence(block)

                    # generate valid spend, requires processing
                    else:
                        self.spend_sequence(block)

                if block % 144 == 70:  # once per day on the 70th block
                    if random.random() < self.CATASTROPHE_RATE:
                        self.catastrophe_sequence(block)

                        # Reboot operation after catastrophe
                        self.initialize_sequence(block + 10)
            # Stop simulation, exit loop and report results
            except (AllocationError):
                print(f"Allocation error at block {block}")
                break

            if "balance" in subplots:
                balances.append(
                    [block, self.wt.balance(), self.required_reserve(block)]
                )
            if "risk_status" in subplots:
                status = self.wt.risk_status(block)
                if (status["vaults_at_risk"] != 0) or (
                    status["delegation_requires"] != 0
                ):
                    risk_status.append(status)
            if "operations" in subplots or "cumulative_ops" in subplots:
                costs.append([block, self.refill_fee, self.cf_fee, self.cancel_fee])
                self.refill_fee, self.cf_fee, self.cancel_fee = None, None, None

            if "coin_pool_age" in subplots:
                try:
                    processed = [
                        coin for coin in self.wt.fbcoins if coin["processed"] != None
                    ]
                    ages = [block - coin["processed"] for coin in processed]
                    age = sum(ages)
                    coin_pool_age.append([block, age])
                except:
                    pass  # If processed is empty, error raised

            if "cumulative_ops" in subplots:
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
                        wt_risk_time.append((risk_on, risk_off))

        figure, axes = plt.subplots(len(subplots), 1, sharex=True)
        plot_num = 0

        # Plot WT balance vs total required reserve
        if "balance" in subplots:
            bal_df = DataFrame(
                balances, columns=["block", "balance", "required reserve"]
            )
            bal_df.set_index(["block"], inplace=True)
            bal_df.plot(ax=axes[plot_num], title="WT Balance", legend=True)
            axes[plot_num].set_xlabel("Block", labelpad=15)
            axes[plot_num].set_ylabel("Satoshis", labelpad=15)
            plot_num += 1

        costs_df = DataFrame(
            costs, columns=["block", "Refill Fee", "CF Fee", "Cancel Fee"]
        )
        report += f"Refill operations: {costs_df['Refill Fee'].count()}\n"

        # Plot refill amount vs block, operating expense vs block
        if "operations" in subplots:
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
        if "cumulative_ops" in subplots:
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
            for (risk_on, risk_off) in wt_risk_time:
                axes[plot_num].axvspan(risk_off, risk_on, color="red", alpha=0.25)

            report += f"Analysis time span: {start_block} to {end_block}\n"
            risk_time = 0
            for (risk_on, risk_off) in wt_risk_time:
                risk_time += risk_off - risk_on
            report += f"Total time at risk: {risk_time} blocks\n"

            # What about avg recovery time?
            recovery_times = []
            for (risk_on, risk_off) in wt_risk_time:
                recovery_times.append(risk_off - risk_on)
            if recovery_times != []:
                report += f"Mean recovery time: {np.mean(recovery_times)} blocks\n"
                report += f"Median recovery time: {np.median(recovery_times)} blocks\n"
                report += f"Max recovery time: {max(recovery_times)} blocks\n"

            plot_num += 1

        # Plot coin pool amounts vs block
        if "coin_pool" in subplots:
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
            try:
                i = subplots.index("operations")
                handles, _labels = axes[i].get_legend_handles_labels()
                labels = set(labels)
                axes[plot_num].legend(handles, labels, loc="upper right")
            except (ValueError):
                pass
            axes[plot_num].set_title("Feebump Coin Pool")
            axes[plot_num].set_ylabel("Coin Amount (Satoshis)", labelpad=15)
            axes[plot_num].set_xlabel("Block", labelpad=15)
            plot_num += 1

        if "vault_excesses" in subplots:
            ## AS SCATTER
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

            ## Normalised sum of vault excesses
            # excesses_df = DataFrame(columns=['block', 'amount'])
            vault_excess_after_cf = []
            for frame in self.vault_excess_after_cf:
                vault_excess_after_cf.append(
                    (frame[0], sum(frame[1]) / self.EXPECTED_ACTIVE_VAULTS)
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
                    (frame[0], sum(frame[1]) / self.EXPECTED_ACTIVE_VAULTS)
                )
            excesses_df = DataFrame(
                vault_excess_after_delegation, columns=["block", "After delegation"]
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
                    (frame[0], sum(frame[1]) / self.EXPECTED_ACTIVE_VAULTS)
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
            axes[plot_num].set_ylabel("Mean Excess per Vault (Satoshis)", labelpad=15)
            axes[plot_num].set_xlabel("Block", labelpad=15)
            plot_num += 1

        # Plot WT risk status
        if "risk_status" in subplots:
            if risk_status != []:
                risk_status_df = DataFrame(risk_status)
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
        if "overpayments" in subplots:
            df = DataFrame(self.overpayments, columns=["block", "overpayments"])
            df["cumulative"] = df["overpayments"].cumsum()
            df.set_index(["block"], inplace=True)
            df["overpayments"].plot(ax=axes[plot_num], label="Singular", legend=True)
            ax2 = axes[plot_num].twinx()
            df["cumulative"].plot(
                ax=ax2, label="Cumulative", color="orange", legend=True
            )
            axes[plot_num].set_xlabel("Block", labelpad=15)
            axes[plot_num].set_ylabel("Satoshis", labelpad=15)
            axes[plot_num].set_title("Cancel Fee Overpayments")
            plot_num += 1

        # Plot coin pool age
        if "coin_pool_age" in subplots:
            age_df = DataFrame(coin_pool_age, columns=["block", "age"])
            age_df.plot.scatter(
                x="block",
                y="age",
                s=6,
                color="orange",
                ax=axes[plot_num],
                label="Total coin pool age",
            )
            plot_num += 1

        with open(f"Results/{self.fname}.txt", "w+", encoding="utf-8") as f:
            f.write(report)
        plt.savefig(f"Results/{self.fname}.png")
        # plt.show()

    def plot_strategic_values(
        self, start_block, end_block, estimate_strat, reserve_strat, O_version
    ):
        plt.style.use(["plot_style.txt"])

        figure, axes = plt.subplots(3, 1, sharex=True)
        self.wt.estimate_strat = estimate_strat
        self.wt.reserve_strat = reserve_strat
        self.wt.O_version = O_version

        # Plot strategic values & estimateSmartFee history
        rows = []
        fees_paid_rows = []
        Os = []
        for block in range(start_block, end_block):
            if block % 1000 == 0:
                print(f"Processing block {block}")
                Os.append([block, self.wt.O(block)])
            rows.append(
                [block, self.wt.Vm(block), self.wt.fee_reserve_per_vault(block)]
            )

        strategy_df = DataFrame(rows, columns=["Block", "Vm", "Required Reserve"])
        strategy_df.set_index("Block", inplace=True)
        strategy_df["Vm"].plot(ax=axes[0], title="Vm", legend=True)
        strategy_df["Required Reserve"].plot(
            ax=axes[0], title="Required Reserve", legend=True
        )
        axes[0].set_ylabel("Sats")
        axes[0].set_title(
            f"Strategic Values\nestimate_strat = {estimate_strat}\nreserve_strat = {reserve_strat})",
            pad=15,
        )

        # Plot fee history
        self.wt.feerate_df["MeanFeerate"][start_block:end_block].plot(ax=axes[1])
        axes[1].set_ylabel("Feerate (sats/vByte)")
        axes[1].set_title("Historic Feerates")

        # Plot amounts of O(t)
        for frame in Os:
            tuples = list(zip([frame[0] for i in frame[1]], frame[1]))
            Os_df = DataFrame(tuples, columns=["block", "amount"])
            Os_df.plot.scatter(
                x="block",
                y="amount",
                style="-",
                alpha=1,
                s=10,
                ax=axes[2],
                legend=False,
            )
            axes[2].set_title("O")
            axes[2].set_ylabel("Coin Amount (Satoshis)", labelpad=15)
            axes[2].set_xlabel("Block", labelpad=15)

        plt.savefig("Results/StrategicValues.png")
        # plt.show()


if __name__ == "__main__":
    random.seed(21000000)

    # note: fee_estimates_fine.csv starts on block 415909 at 2016-05-18 02:00:00
    config = {
        "n_stk": 7,
        "n_man": 3,
        "reserve_strat": "CUMMAX95Q90",
        "estimate_strat": "ME30",
        "O_version": 1,
        "I_version": 2,
        "feerate_src": "../block_fees/historical_fees.csv",
        "estimate_smart_feerate_src": "fee_estimates_fine.csv",
        "weights_src": "tx_weights.csv",
        "block_datetime_src": "block_height_datetime.csv",
    }
    fname = "TestReport"

    sim = WTSim(config, fname)

    start_block = 200000
    end_block = 681000

    # "operations", "coin_pool_age", "coin_pool", "risk_status"
    subplots = ["balance", "vault_excesses", "cumulative_ops", "overpayments"]
    sim.plot_simulation(start_block, end_block, subplots)

    sim.plot_strategic_values(
        start_block, end_block, "ME30", "CUMMAX95Q90", O_version=1
    )
