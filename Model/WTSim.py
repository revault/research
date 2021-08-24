from WTSM import *

class DelegationError(Exception):
    pass
class SpendAttemptError(Exception):
    pass

class WTSim(object):
    """Simulator for fee-reserve management of a Revault Watchtower.
    """

    def __init__(self, config, fname):
        # Stakeholder parameters
        self.excess_delegations = 3
        self.expected_active_vaults = 5

        # Manager parameters
        self.INVALID_SPEND_RATE = 0.1
        self.CATASTROPHE_RATE = 0.005 

        # WT state machine
        self.wt = WTSM(config)
        self.vaults_df = read_csv("vaultIDs.csv")

        # Simulation report
        self.fname = fname
        self.report_init = f"Watchtower config:\n{config}\nExcess delegations: {self.excess_delegations}\nExpected active vaults: {self.expected_active_vaults}\nInvalid spend rate: {self.INVALID_SPEND_RATE}\nCatastrophe rate: {self.CATASTROPHE_RATE}\n"

    def required_reserve(self, block_height):
        required_reserve_per_vault = self.wt.fee_reserve_per_vault(
            block_height)
        num_vaults = len(self.wt.vaults)
        return (num_vaults + self.excess_delegations)*required_reserve_per_vault

    def R(self, block_height):
        """Returns amount to refill to ensure WT has sufficient balance.
           Used by stakeholder wallet software. 

           Note: stakeholder knows WT's balance and num_vaults. Stakeholder
                 doesn't know which coins are allocated or not. 
        """
        bal = self.wt.balance()
        reserve_total = sum(self.wt.O(block_height))*(len(self.wt.vaults)+self.excess_delegations)
        R = reserve_total - bal
        if R <= 0:
            return 0

        new_reserves = R//(sum(self.wt.O(block_height)))

        # Expected CF Tx fee
        try: 
            feerate = self.wt._estimate_smart_feerate(block_height)
        except(ValueError, KeyError):
            feerate = self.wt._feerate(block_height)
        P2WPKH_INPUT_vBytes = 67.75
        P2WPKH_OUTPUT_vBytes = 31
        expected_num_outputs = len(self.wt.O(block_height))*new_reserves
        expected_num_inputs = len(self.wt.O(block_height))*len(self.wt.vaults)  # guess...
        expected_cf_fee = (10.75 + expected_num_outputs*P2WPKH_OUTPUT_vBytes +
                            expected_num_inputs*P2WPKH_INPUT_vBytes)*feerate

        print(f"    Expected CF Tx fee: {expected_cf_fee}")
        print(f"    Expected num outputs: {expected_num_outputs}")

        R += expected_cf_fee
        return R

    def initialize_sequence(self, block_height):
        ## Refill transition
        # WT provided with enough to allocate up to the expected number of active vaults
        initial_reserve_buffer_factor = 2
        reserve_total =  sum(self.wt.O(block_height))*self.expected_active_vaults*initial_reserve_buffer_factor

        # Expected CF Tx fee
        try: 
            feerate = self.wt._estimate_smart_feerate(block_height)
        except(ValueError, KeyError):
            feerate = self.wt._feerate(block_height)
        P2WPKH_INPUT_vBytes = 67.75
        P2WPKH_OUTPUT_vBytes = 31
        expected_num_outputs = len(self.wt.O(block_height))*self.expected_active_vaults
        expected_num_inputs = 1
        expected_cf_fee = (10.75 + expected_num_outputs*P2WPKH_OUTPUT_vBytes +
                            expected_num_inputs*P2WPKH_INPUT_vBytes)*feerate

        refill_amount = reserve_total + expected_cf_fee - self.wt.balance()
        self.wt.refill(refill_amount)
        print(f"inatialising WT with refill at block {block_height} by {refill_amount}")

        # Track operational costs
        try:
            self.refill_fee = 109.5 * self.wt._estimate_smart_feerate(block_height)
        except(ValueError, KeyError):
            self.refill_fee = 109.5 * self.wt._feerate(block_height)

        # snapshot coin pool after refill Tx
        if "coin_pool" in self.subplots:
            amounts = [coin['amount'] for coin in self.wt.fbcoins]
            self.pool_after_refill.append([block_height, amounts])

        # snapshot vault excesses before CF Tx
        if "vault_excesses" in self.subplots:
            vault_requirement = self.wt.fee_reserve_per_vault(block_height)
            excesses = []
            for vault in self.wt.vaults:
                excess = sum(
                    [coin['amount'] for coin in vault['fee_reserve']]) - vault_requirement
                if excess > 0:
                    excesses.append(excess)
            self.vault_excess_before_cf.append([block_height, excesses])

        ## Consolidate-fanout transition
        self.cf_fee = self.wt.consolidate_fanout(block_height)
        print(f"consolidate-fanout Tx at block {block_height} with fee: {self.cf_fee}")

        # snapshot coin pool after CF Tx
        if "coin_pool" in self.subplots:
            amounts = [coin['amount'] for coin in self.wt.fbcoins]
            self.pool_after_cf.append([block_height, amounts])

        ## Allocation transitions
        for i in range(0,self.expected_active_vaults):
            print(f"processing delegation at block {block_height}")
            vaultID = self.vaults_df['vaultID'][self.vault_count]
            self.vault_count += 1
            amount = 10e10 # 100 BTC
            self.wt.process_delegation(vaultID, amount, block_height)

        # snapshot vault excesses after delegations
        if "vault_excesses" in self.subplots:
            vault_requirement = self.wt.fee_reserve_per_vault(block_height)
            excesses = []
            for vault in self.wt.vaults:
                excess = sum(
                    [coin['amount'] for coin in vault['fee_reserve']]) - vault_requirement
                if excess > 0:
                    excesses.append(excess)
            self.vault_excess_after_delegation.append([block_height, excesses])


    def refill_sequence(self, block_height):
        refill_amount = self.R(block_height)
        if refill_amount > 0:
            ## Refill transition
            print(f"refilling at block {block_height} by {refill_amount} to fill reserve")
            self.wt.refill(refill_amount)
            try:
                self.refill_fee = 109.5 * self.wt._estimate_smart_feerate(block_height)
            except(ValueError, KeyError):
                self.refill_fee = 109.5 * self.wt._feerate(block_height)

            # snapshot coin pool after refill Tx
            if "coin_pool" in self.subplots:
                amounts = [coin['amount'] for coin in self.wt.fbcoins]
                self.pool_after_refill.append([block_height, amounts])

            # snapshot vault excesses before CF Tx
            if "vault_excesses" in self.subplots:
                vault_requirement = self.wt.fee_reserve_per_vault(block_height)
                excesses = []
                for vault in self.wt.vaults:
                    excess = sum(
                        [coin['amount'] for coin in vault['fee_reserve']]) - vault_requirement
                    if excess > 0:
                        excesses.append(excess)
                self.vault_excess_before_cf.append([block_height, excesses])

            ## Consolidate-fanout transition
            # Wait for confirmation of refill, then CF Tx
            self.cf_fee = self.wt.consolidate_fanout(block_height+1)
            print(f"consolidate-fanout Tx at block {block_height+1} with fee: {self.cf_fee}")

            # snapshot coin pool after CF Tx confirmation
            if "coin_pool" in self.subplots:
                amounts = [coin['amount'] for coin in self.wt.fbcoins]
                self.pool_after_cf.append([block_height+7, amounts])

            # snapshot vault excesses after CF Tx
            if "vault_excesses" in self.subplots:
                excesses = []
                for vault in self.wt.vaults:
                    excess = sum(
                        [coin['amount'] for coin in vault['fee_reserve']]) - vault_requirement
                    if excess > 0:
                        excesses.append(excess)
                self.vault_excess_after_cf.append([block_height, excesses])

            ## Top up transition
            # Top up delegations after confirmation of CF Tx, because consolidating coins
            # reduces the fee_reserve of a vault
            try:
                self.wt.top_up_delegations(block_height+7)
            except(RuntimeError):
                pass

    def _spend_init(self,block_height):
        ## Top up transition
        # Top up delegations before processing a delegation, because time has passed, and we mustn't accept
        # delegation if the available coin pool is insufficient.
        try:
            self.wt.top_up_delegations(block_height)
        except(RuntimeError):
            pass

        # Delegate a vault
        print(f"processing delegation at block {block_height}")
        vaultID = self.vaults_df['vaultID'][self.vault_count]
        self.vault_count += 1
        amount = 10e10  # 100 BTC

        ## Allocation transition
        # If WT fails to acknowledge delegation, exit the simulation and plot the outcome so-far
        try:
            self.wt.process_delegation(vaultID, amount, block_height)
        except(RuntimeError):
            raise(DelegationError())

        # snapshot vault excesses after delegation
        if "vault_excesses" in self.subplots:
            vault_requirement = self.wt.fee_reserve_per_vault(block_height)
            excesses = []
            for vault in self.wt.vaults:
                excess = sum(
                    [coin['amount'] for coin in vault['fee_reserve']]) - vault_requirement
                if excess > 0:
                    excesses.append(excess)
            self.vault_excess_after_delegation.append([block_height, excesses])


        # choose a random vault to spend
        try:
            vaultID = choice(self.wt.vaults)['id']
        except IndexError:
            raise SpendAttemptError(
                "Attempted spend without existing delegations")

        return vaultID

    def spend_sequence(self, block_height):
        vaultID = self._spend_init(block_height)
        ## Spend transition
        print(f"processing spend at block {block_height}")
        self.wt.process_spend(vaultID)

        # snapshot coin pool after spend attempt
        if "coin_pool" in self.subplots:
            amounts = [coin['amount'] for coin in self.wt.fbcoins]
            self.pool_after_spend.append([block_height, amounts])

    def cancel_sequence(self, block_height):
        vaultID = self._spend_init(block_height)
        ## Cancel transition
        print(f"processing cancel at block {block_height}")
        cancel_inputs = self.wt.process_cancel(vaultID, block_height)
        self.wt.finalize_cancel(vaultID)
        self.cancel_fee = sum(coin['amount'] for coin in cancel_inputs)
        print(f"Cancel! fee = {self.cancel_fee}")

        # snapshot coin pool after cancel
        if "coin_pool" in self.subplots:
            amounts = [coin['amount'] for coin in self.wt.fbcoins]
            self.pool_after_cancel.append([block_height, amounts])

    def catastrophe_sequence(self, block_height):
        print(f"Catastrophic breach detected at block {block_height}")
        for vault in self.wt.vaults:
            ## Cancel transition
            cancel_inputs = self.wt.process_cancel(vault['id'], block_height)
            self.wt.finalize_cancel(vault['id'])
            # If a cancel fee has already been paid this block, sum those fees
            # so that when plotting costs this will appear as one total operation
            # rather than several separate cancel operations
            try:
                cancel_fee = sum(coin['amount'] for coin in cancel_inputs)
                self.cancel_fee += cancel_fee
            except(TypeError):
                cancel_fee = sum(coin['amount'] for coin in cancel_inputs)
                self.cancel_fee = cancel_fee
            print(f"Canceled spend from vault {vault['id']}! fee = {cancel_fee}")

        # snapshot coin pool after all spend attempts are cancelled
        if "coin_pool" in self.subplots:
            amounts = [coin['amount'] for coin in self.wt.fbcoins]
            self.pool_after_catastrophe.append([block_height, amounts])

    def plot_simulation(self, start_block, end_block, subplots):

        plt.style.use(['plot_style.txt'])              

        self.refill_fee = None
        self.cf_fee = None
        self.cancel_fee = None
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
        balances = []
        risk_status = []
        costs = []
        coin_pool_age = []
        wt_risk_time = []
        
        switch = "good"
        report = self.report_init

        self.initialize_sequence(start_block)

        for block in range(start_block, end_block):
            # if block % 144 == 0: # once per day
            if block % 1 == 0: # each block
                self.refill_sequence(block)

            if block % 144 == 10: # once per day on the 10th block
                # generate invalid spend, requires cancel
                if random() < self.INVALID_SPEND_RATE:
                    self.cancel_sequence(block)
                # generate valid spend, requires processing
                else:
                    try:
                        self.spend_sequence(block)
                    # Stop simulation, exit loop and report results
                    except(DelegationError):
                        print("Process Delegation FAILED.")
                        break
                    except(SpendAttemptError):
                        print("Process Spend FAILED.")
                        break

            if block % 144 == 50: #once per day on the 50th block
                if random() < self.CATASTROPHE_RATE:
                    self.catastrophe_sequence(block)

                    # Reboot operation after catastrophe
                    self.initialize_sequence(block+10)


            if "balance" in subplots:
                balances.append([block, self.wt.balance(),
                             self.required_reserve(block)])
            if "risk_status" in subplots:
                status = self.wt.risk_status(block)
                if (status['vaults_at_risk'] != 0) or (status['delegation_requires'] != 0):
                    risk_status.append(status)
            if "operations" in subplots or "cumulative_ops"in subplots:
                costs.append([block, self.refill_fee, self.cf_fee, self.cancel_fee])
                self.refill_fee, self.cf_fee, self.cancel_fee = None, None, None

            if "coin_pool_age" in subplots:
                try:
                    processed = [
                        coin for coin in self.wt.fbcoins if coin['processed'] != None]
                    ages = [block - coin['processed'] for coin in processed]
                    age = sum(ages)
                    coin_pool_age.append([block, age])
                except:
                    pass  # If processed is empty, error raised

            if "operations" in subplots:
                # Check if wt becomes risky
                if switch == "good":
                    for vault in self.wt.vaults:
                        if self.wt.under_requirement(vault['fee_reserve'], block) != 0:
                            switch = "bad"
                            break
                    if switch == "bad":
                        risk_on = block

                # Check if wt no longer risky
                if switch == "bad":
                    any_risk = []
                    for vault in self.wt.vaults:
                        if self.wt.under_requirement(vault['fee_reserve'], block) != 0:
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
            bal_df = DataFrame(balances, columns=[
                "block", "balance", "required reserve"])
            bal_df.set_index(['block'], inplace=True)
            bal_df.plot(ax=axes[plot_num], title="WT Balance", legend=True)
            axes[plot_num].set_xlabel("Block", labelpad=15)
            axes[plot_num].set_ylabel("Satoshis", labelpad=15)
            plot_num += 1

        costs_df = DataFrame(
            costs, columns=['block', 'Refill Fee', 'CF Fee', 'Cancel Fee'])
        # Plot refill amount vs block, operating expense vs block
        if "operations" in subplots:
            costs_df.plot.scatter(x='block', y='Refill Fee', s=6,
                                  color='r', ax=axes[plot_num], label="Refill Fee")
            costs_df.plot.scatter(x='block', y='CF Fee', s=6,
                                  color='g', ax=axes[plot_num], label="CF Fee")
            costs_df.plot.scatter(x='block', y='Cancel Fee', s=6,
                                  color='b', ax=axes[plot_num], label="Cancel Fee")
            axes[plot_num].legend(loc='upper left')
            axes[plot_num].set_title("Operating Costs Breakdown")
            axes[plot_num].set_ylabel("Satoshis", labelpad=15)
            axes[plot_num].set_xlabel("Block", labelpad=15)

            # Highlight the plot with areas that show when the WT is at risk due to at least one
            # insufficient vault fee-reserve
            for (risk_on, risk_off) in wt_risk_time:
                axes[plot_num].axvspan(
                    risk_off, risk_on, color='red', alpha=0.25)

            report += f"Analysis time span: {start_block} to {end_block}\n"
            risk_time = 0
            for (risk_on, risk_off) in wt_risk_time:
                risk_time += (risk_off - risk_on)
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

        # Plot cumulative operating costs (CF, Cancel, Spend)
        if "cumulative_ops" in subplots:
            cumulative_costs_df = costs_df
            cumulative_costs_df.set_index(['block'], inplace=True)
            cumulative_costs_df = cumulative_costs_df.fillna(0).cumsum()
            cumulative_costs_df.plot.line(ax=axes[plot_num], color={
                                          'Refill Fee': 'r', 'CF Fee': 'g', 'Cancel Fee': 'b'})
            axes[plot_num].legend(loc='upper left')
            axes[plot_num].set_title("Cumulative Operating Costs")
            axes[plot_num].set_ylabel("Satoshis", labelpad=15)
            axes[plot_num].set_xlabel("Block", labelpad=15)
            plot_num += 1

        # Plot coin pool amounts vs block
        if "coin_pool" in subplots:
            for frame in self.pool_after_refill:
                tuples = list(zip([frame[0] for i in frame[1]], frame[1]))
                pool_df = DataFrame(tuples, columns=['block', 'amount'])
                pool_df.plot.scatter(x='block', y='amount', color='r',
                                     alpha=0.1, s=5, ax=axes[plot_num], label="After Refill")
            for frame in self.pool_after_cf:
                tuples = list(zip([frame[0] for i in frame[1]], frame[1]))
                pool_df = DataFrame(tuples, columns=['block', 'amount'])
                pool_df.plot.scatter(x='block', y='amount', color='g',
                                     alpha=0.1, s=5, ax=axes[plot_num], label="After CF")
            for frame in self.pool_after_spend:
                tuples = list(zip([frame[0] for i in frame[1]], frame[1]))
                pool_df = DataFrame(tuples, columns=['block', 'amount'])
                pool_df.plot.scatter(x='block', y='amount', color='b', alpha=0.1,
                                     s=5, ax=axes[plot_num], label="After Spend")
            for frame in self.pool_after_cancel:
                tuples = list(zip([frame[0] for i in frame[1]], frame[1]))
                pool_df = DataFrame(tuples, columns=['block', 'amount'])
                pool_df.plot.scatter(x='block', y='amount', color='k', alpha=0.1,
                                     s=5, ax=axes[plot_num], label="After Cancel")
            for frame in self.pool_after_catastrophe:
                tuples = list(zip([frame[0] for i in frame[1]], frame[1]))
                pool_df = DataFrame(tuples, columns=['block', 'amount'])
                pool_df.plot.scatter(x='block', y='amount', color='o', alpha=0.1,
                                     s=5, ax=axes[plot_num], label="After Catastrophe")
            handles, labels = axes[plot_num].get_legend_handles_labels()
            try:
                i = subplots.index("operations")
                handles, _labels = axes[i].get_legend_handles_labels()
                labels = set(labels)
                axes[plot_num].legend(handles, labels, loc='upper right')
            except(ValueError):
                pass
            axes[plot_num].set_title("Feebump Coin Pool")
            axes[plot_num].set_ylabel("Coin Amount (Satoshis)", labelpad=15)
            axes[plot_num].set_xlabel("Block", labelpad=15)
            plot_num += 1

        if "vault_excesses" in subplots:
            for frame in self.vault_excess_after_cf:
                tuples = list(zip([frame[0] for i in frame[1]], frame[1]))
                excesses_df = DataFrame(tuples, columns=['block', 'amount'])
                # , label="After CF")
                excesses_df.plot.scatter(
                    x='block', y='amount', color='r', ax=axes[plot_num])
            for frame in self.vault_excess_after_delegation:
                tuples = list(zip([frame[0] for i in frame[1]], frame[1]))
                excesses_df = DataFrame(tuples, columns=['block', 'amount'])
                # , label="After Delegation")
                excesses_df.plot.scatter(
                    x='block', y='amount', color='g', ax=axes[plot_num])
            for frame in self.vault_excess_before_cf:
                tuples = list(zip([frame[0] for i in frame[1]], frame[1]))
                excesses_df = DataFrame(tuples, columns=['block', 'amount'])
                excesses_df.plot.scatter(
                    x='block', y='amount', color='b', ax=axes[plot_num])
            axes[plot_num].set_ylabel("Vault Excess (Satoshis)", labelpad=15)
            axes[plot_num].set_xlabel("Block", labelpad=15)
            plot_num += 1

        # Plot WT number of active vaults
        if "risk_status" in subplots:
            if risk_status != []:
                risk_status_df = DataFrame(risk_status)
                risk_status_df.set_index(['block'], inplace=True)
                risk_status_df['num_vaults'].plot(ax=axes[plot_num], label="number of vaults", color='r', legend=True)
                risk_status_df['vaults_at_risk'].plot(ax=axes[plot_num], label="vaults at risk", color='b', legend=True)
                ax2 = axes[plot_num].twinx()
                risk_status_df['delegation_requires'].plot(ax=ax2,  label="new delegation requires", color='g', legend=True)
                risk_status_df['severity'].plot(ax=ax2,  label="total severity of risk", color='k', legend=True)
                axes[plot_num].set_ylabel("Vaults", labelpad=15)
                axes[plot_num].set_xlabel("Block", labelpad=15)
                ax2.set_ylabel("Satoshis", labelpad=15)
                plot_num += 1

        # Plot coin pool age
        if "coin_pool_age" in subplots:
            age_df = DataFrame(coin_pool_age, columns=['block', 'age'])
            age_df.plot.scatter(x='block', y='age', s=6, color='orange',
                                ax=axes[plot_num], label="Total coin pool age")
            plot_num += 1

        with open(f"Results/{self.fname}.txt",'w+',encoding = 'utf-8') as f:
            f.write(report)
        plt.savefig(f"Results/{self.fname}.png")
        # plt.show()

    def plot_strategic_values(self, start_block, end_block, estimate_strat, reserve_strat):
        plt.style.use(['plot_style.txt'])

        figure, axes = plt.subplots(3, 1)
        self.wt.estimate_strat = estimate_strat
        self.wt.reserve_strat = reserve_strat

        # Plot strategic values & estimateSmartFee history
        rows = []
        fees_paid_rows = []
        for block in range(start_block, end_block):
            if block % 1000 == 0:
                print(f"Processing block {block}")
            datetime = self.wt._get_block_datetime(block)
            rows.append([datetime, self.wt.Vm(block), self.wt.Vb(
                block), self.wt.fee_reserve_per_vault(block)])
            fees_paid_rows.append([datetime, self.wt._feerate(block)])

        strategy_df = DataFrame(
            rows, columns=['DateTime', 'Vm', 'Vb', 'Required Reserve'])
        strategy_df.set_index('DateTime', inplace=True)
        strategy_df['Vm'].plot(ax=axes[0], title='Vm', legend=True)
        strategy_df['Vb'].plot(ax=axes[0], title='Vb', legend=True)
        strategy_df['Required Reserve'].plot(
            ax=axes[0], title='Required Reserve', legend=True)
        axes[0].set_ylabel("Sats")
        axes[0].set_title(f"Strategic Values (estimate_strat = {estimate_strat}, reserve_strat = {reserve_strat})")

        # Plot fee history
        start = self.wt.feerate_df.index.get_loc(
            key=self.wt._get_block_datetime(start_block), method="ffill")
        end = self.wt.feerate_df.index.get_loc(
            key=self.wt._get_block_datetime(end_block), method="ffill")
        self.wt.feerate_df['FeeRate'][start:end].plot(ax=axes[1])
        axes[1].set_ylabel("Feerate (sats/vByte)")
        axes[1].set_title("Historic Feerates")

        # Plot estimateSmartFee history against block data based A
        fee_df = DataFrame(fees_paid_rows, columns=['DateTime', 'Fee'])
        fee_df.set_index('DateTime', inplace=True)
        fee_df['Fee'].plot(ax=axes[2], title="Fees Paid", legend=False)

        axes[2].set_ylabel("FeeRate estimate (sats/vByte)")
        axes[2].set_title(f"{self.wt.estimate_strat} of actual fees paid")

        plt.show()


if __name__ == '__main__':
    # note: fee_estimates_fine.csv starts on block 415909 at 2016-05-18 02:00:00
    config = {
        "n_stk": 7, "n_man": 3, "reserve_strat": "CUMMAX95Q90", "estimate_strat": "ME30",
        "O_version": 0, "I_version": 1, "feerate_src": "../block_fees/historical_fees.csv", 
        "estimate_smart_feerate_src": "fee_estimates_fine.csv", "weights_src": "tx_weights.csv",
        "block_datetime_src": "block_height_datetime.csv"
    }
    fname = "TestReport"

    sim = WTSim(config, fname)

    start_block = 100000 
    end_block = 681000

    # "vault_excesses", "coin_pool_age", "coin_pool", "risk_status"]
    subplots = ["balance", "operations", "cumulative_ops"]
    sim.plot_simulation(start_block, end_block, subplots)

    # sim.plot_strategic_values(start_block, end_block, "ME30", "CUMMAX95Q90")

    # sim.plot_cf_cost_per_refill_amount(start_block, end_block, 1, 5)
