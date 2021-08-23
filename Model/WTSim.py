from WTSM import *

# Timestamp from my node 2016-05-18 02:01:08, data starts at 2016-05-18 02:00:00
FIRST_BLOCK_IN_DATA = 415909
MAX_TX_SIZE = 100000  # vBytes

class WTSim(object):
    """Simulator for fee-reserve management of a Revault Watchtower.
    """

    def __init__(self, config, fname):
        # Stakeholder parameters
        self.excess_delegations = 7
        self.expected_active_vaults = 3

        # Manager parameters
        self.INVALID_SPEND_RATE = 0.05
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
        required_reserve = self.required_reserve(block_height)
        R = required_reserve - bal
        if R <= 0:
            return 0

        new_reserves = R//(sum(self.wt.O(block_height)))

        # Expected CF Tx fee
        feerate = self.wt._feerate(block_height)  # FIXME estimateSmartFee
        P2WPKH_INPUT_vBytes = 67.75
        P2WPKH_OUTPUT_vBytes = 31
        expected_num_outputs = len(self.wt.O(block_height))*new_reserves
        expected_num_inputs = len(self.wt.O(block_height))*len(self.wt.vaults)  # guess...
        expected_cf__fee = (10.75 + expected_num_outputs*P2WPKH_OUTPUT_vBytes +
                            expected_num_inputs*P2WPKH_INPUT_vBytes)*feerate

        print(f"    Expected CF Tx fee: {expected_cf__fee}")
        print(f"    Expected num outputs: {expected_num_outputs}")

        R += expected_cf__fee
        return R

    def plot_simulation(self, start_block, end_block, subplots):

        plt.style.use(['plot_style.txt'])

        refill_fees = []
        refill_amounts = []
        cf_fees = []
        cancel_fees = []
        balances = []
        vault_count = 1
        risk_status = []
        costs = []
        coin_pool_age = []
        pool_after_refill = []
        pool_after_cf = []
        pool_after_spend = []
        wt_risk_time = []
        vault_excess_before_cf = []
        vault_excess_after_cf = []
        vault_excess_after_delegation = []
        switch = "good"
        report = self.report_init


        #Delegate several vaults initially
        refill_amount = self.R(start_block)
        refill_amount = refill_amount*self.expected_active_vaults
        self.wt.refill(refill_amount)
        self.wt.consolidate_fanout(start_block)

        for i in range(0,self.expected_active_vaults):
            print(f"processing delegation at block {start_block}")
            vaultID = self.vaults_df['vaultID'][vault_count]
            vault_count += 1
            amount = 10e10 # 100 BTC
            self.wt.process_delegation(vaultID, amount, start_block)


        for block in range(start_block, end_block):
            refill_amount = 0
            refill_fee = None
            cf_fee = None
            cancel_fee = None

            # Refill if the balance is low
            # if block % 144 == 0: # once per day
            if block % 1 == 0: # each block
                refill_amount = self.R(block)
                if refill_amount > 0:
                    print(f"refilling at block {block} by {refill_amount} to fill reserve to {self.required_reserve(block)}")
                    self.wt.refill(refill_amount)
                    refill_amounts.append([block, refill_amount])
                    # use estimate smart fee
                    refill_fee = 109.5 * self.wt._feerate(block)
                    refill_fees.append([block, refill_fee])

                    # snapshot coin pool after refill Tx
                    if "coin_pool" in subplots:
                        amounts = [coin['amount'] for coin in self.wt.fbcoins]
                        pool_after_refill.append([block, amounts])

                    # snapshot vault excesses before CF Tx
                    if "vault_excesses" in subplots:
                        vault_requirement = self.wt.fee_reserve_per_vault(block)
                        excesses = []
                        for vault in self.wt.vaults:
                            excess = sum(
                                [coin['amount'] for coin in vault['fee_reserve']]) - vault_requirement
                            if excess > 0:
                                excesses.append(excess)
                        vault_excess_before_cf.append([block, excesses])

                    # Wait for confirmation of refill, then CF Tx
                    cf_fee = self.wt.consolidate_fanout(block+1)
                    print(f"consolidate-fanout Tx at block {block+1} with fee: {cf_fee}")
                    cf_fees.append([block, cf_fee])

                    # snapshot coin pool after CF Tx confirmation
                    if "coin_pool" in subplots:
                        amounts = [coin['amount'] for coin in self.wt.fbcoins]
                        pool_after_cf.append([block+7, amounts])

                    # snapshot vault excesses after CF Tx
                    if "vault_excesses" in subplots:
                        excesses = []
                        for vault in self.wt.vaults:
                            excess = sum(
                                [coin['amount'] for coin in vault['fee_reserve']]) - vault_requirement
                            if excess > 0:
                                excesses.append(excess)
                        vault_excess_after_cf.append([block, excesses])

                    # Top up delegations after confirmation of CF Tx, because consolidating coins
                    # reduces the fee_reserve of a vault
                    try:
                        self.wt.top_up_delegations(block+7)
                    except(RuntimeError):
                        pass

            # Once per day, after balance maintenance, process delegation for a spend
            if block % 144 == 0:
                # Top up delegations before processing a delegation, because time has passed, and we mustn't accept
                # delegation if the available coin pool is insufficient.
                try:
                    self.wt.top_up_delegations(block)
                except(RuntimeError):
                    pass

                # Delegate a vault
                print(f"processing delegation at block {block}")
                vaultID = self.vaults_df['vaultID'][vault_count]
                vault_count += 1
                amount = 10e10  # 100 BTC

                # If WT fails to acknowledge delegation, exit the simulation and plot the outcome so-far
                try:
                    self.wt.process_delegation(vaultID, amount, block)
                except(RuntimeError):
                    print("Process Delegation FAILED.")
                    break

                # snapshot vault excesses after delegation
                if "vault_excesses" in subplots:
                    vault_requirement = self.wt.fee_reserve_per_vault(block)
                    excesses = []
                    for vault in self.wt.vaults:
                        excess = sum(
                            [coin['amount'] for coin in vault['fee_reserve']]) - vault_requirement
                        if excess > 0:
                            excesses.append(excess)
                    vault_excess_after_delegation.append([block, excesses])

                # Process spend attempt
                print(f"processing spend at block {block}")
                # choose a random vault to spend
                try:
                    vaultID = choice(self.wt.vaults)['id']
                except IndexError:
                    raise RuntimeError(
                        "Attempted spend without existing delegations")

                # generate invalid spend, requires cancel
                if random() < self.INVALID_SPEND_RATE:
                    cancel_inputs = self.wt.process_cancel(vaultID, block)
                    self.wt.finalize_cancel(vaultID)
                    cancel_fee = sum(coin['amount'] for coin in cancel_inputs)
                    cancel_fees.append([block, cancel_fees])
                    print(f"Cancel! fee = {cancel_fee}")
                # generate valid spend, requires processing
                else:
                    self.wt.process_spend(vaultID)

                # snapshot coin pool after spend attempt
                if "coin_pool" in subplots:
                    amounts = [coin['amount'] for coin in self.wt.fbcoins]
                    pool_after_spend.append([block, amounts])

            # Catastrophe
            if block % 144 == 1:
                if random() < self.CATASTROPHE_RATE:
                    print(f"Catastrophic breach detected at block {block}")
                    for vault in self.wt.vaults:
                        cancel_inputs = self.wt.process_cancel(vault['id'], block)
                        self.wt.finalize_cancel(vault['id'])
                        cancel_fee = sum(coin['amount'] for coin in cancel_inputs)
                        cancel_fees.append([block, cancel_fees])
                        print(f"Canceled spend from vault {vault['id']}! fee = {cancel_fee}")

                    # snapshot coin pool after all spend attempts are cancelled
                    if "coin_pool" in subplots:
                        amounts = [coin['amount'] for coin in self.wt.fbcoins]
                        pool_after_spend.append([block, amounts])


                    # Reboot operational flow with several vaults
                    refill_amount = self.R(block+10)
                    refill_amount = refill_amount*self.expected_active_vaults
                    print(f"refilling at block {block + 10} by {refill_amount} to fill reserve to {self.required_reserve(block+10)}")
                    self.wt.refill(refill_amount)

                    # snapshot coin pool after refill Tx
                    if "coin_pool" in subplots:
                        amounts = [coin['amount'] for coin in self.wt.fbcoins]
                        pool_after_refill.append([block, amounts])

                    # snapshot vault excesses before CF Tx
                    if "vault_excesses" in subplots:
                        vault_requirement = self.wt.fee_reserve_per_vault(block)
                        excesses = []
                        for vault in self.wt.vaults:
                            excess = sum(
                                [coin['amount'] for coin in vault['fee_reserve']]) - vault_requirement
                            if excess > 0:
                                excesses.append(excess)
                        vault_excess_before_cf.append([block, excesses])

                    cf_fee = self.wt.consolidate_fanout(block + 10)
                    print(f"consolidate-fanout Tx at block {block+10} with fee: {cf_fee}")

                    # snapshot coin pool after CF Tx confirmation
                    if "coin_pool" in subplots:
                        amounts = [coin['amount'] for coin in self.wt.fbcoins]
                        pool_after_cf.append([block+7, amounts])

                    # snapshot vault excesses after CF Tx
                    if "vault_excesses" in subplots:
                        excesses = []
                        for vault in self.wt.vaults:
                            excess = sum(
                                [coin['amount'] for coin in vault['fee_reserve']]) - vault_requirement
                            if excess > 0:
                                excesses.append(excess)
                        vault_excess_after_cf.append([block, excesses])

                    for i in range(0,self.expected_active_vaults):
                        print(f"processing delegation at block {block + 10}")
                        vaultID = self.vaults_df['vaultID'][vault_count]
                        vault_count += 1
                        amount = 10e10 # 100 BTC
                        self.wt.process_delegation(vaultID, amount, block + 10)


            if "balance" in subplots:
                balances.append([block, self.wt.balance(),
                             self.required_reserve(block)])
            if "risk_status" in subplots:
                status = self.wt.risk_status(block)
                if (status['vaults_at_risk'] != 0) or (status['delegation_requires'] != 0):
                    risk_status.append(status)
            if "operations" in subplots or "cumulative_ops"in subplots:
                costs.append([block, refill_fee, cf_fee, cancel_fee])

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
            for frame in pool_after_refill:
                tuples = list(zip([frame[0] for i in frame[1]], frame[1]))
                pool_df = DataFrame(tuples, columns=['block', 'amount'])
                pool_df.plot.scatter(x='block', y='amount', color='r',
                                     alpha=0.1, s=5, ax=axes[plot_num], label="After Refill")
            for frame in pool_after_cf:
                tuples = list(zip([frame[0] for i in frame[1]], frame[1]))
                pool_df = DataFrame(tuples, columns=['block', 'amount'])
                pool_df.plot.scatter(x='block', y='amount', color='g',
                                     alpha=0.1, s=5, ax=axes[plot_num], label="After CF")
            for frame in pool_after_spend:
                tuples = list(zip([frame[0] for i in frame[1]], frame[1]))
                pool_df = DataFrame(tuples, columns=['block', 'amount'])
                pool_df.plot.scatter(x='block', y='amount', color='b', alpha=0.1,
                                     s=5, ax=axes[plot_num], label="After Spend Attempt")
            handles, labels = axes[plot_num].get_legend_handles_labels()
            if subplots[1] == "operations":  # FIX, find index
                handles, _labels = axes[1].get_legend_handles_labels()
            labels = set(labels)
            axes[plot_num].legend(handles, labels, loc='upper right')
            axes[plot_num].set_title("Feebump Coin Pool")
            axes[plot_num].set_ylabel("Coin Amount (Satoshis)", labelpad=15)
            axes[plot_num].set_xlabel("Block", labelpad=15)
            plot_num += 1

        if "vault_excesses" in subplots:
            for frame in vault_excess_after_cf:
                tuples = list(zip([frame[0] for i in frame[1]], frame[1]))
                excesses_df = DataFrame(tuples, columns=['block', 'amount'])
                # , label="After CF")
                excesses_df.plot.scatter(
                    x='block', y='amount', color='r', ax=axes[plot_num])
            for frame in vault_excess_after_delegation:
                tuples = list(zip([frame[0] for i in frame[1]], frame[1]))
                excesses_df = DataFrame(tuples, columns=['block', 'amount'])
                # , label="After Delegation")
                excesses_df.plot.scatter(
                    x='block', y='amount', color='g', ax=axes[plot_num])
            for frame in vault_excess_before_cf:
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
    config = {
        "n_stk": 7, "n_man": 3, "reserve_strat": "CUMMAX95Q90", "estimate_strat": "ME30",
        "O_version": 0, "I_version": 1, "feerate_src": "tx_fee_history.csv", 
        "estimate_smart_feerate_src": "fee_estimates_fine.csv", "weights_src": "tx_weights.csv",
        "block_datetime_src": "block_height_datetime.csv"
    }
    fname = "TestReport"

    sim = WTSim(config, fname)

    start_block = FIRST_BLOCK_IN_DATA 
    end_block = 681000

    # "vault_excesses", "coin_pool_age", "coin_pool", "risk_status"]
    subplots = ["balance", "operations", "cumulative_ops"]
    sim.plot_simulation(start_block, end_block, subplots)

    # sim.plot_strategic_values(start_block, end_block, "ME30", "CUMMAX95Q90")

    # sim.plot_cf_cost_per_refill_amount(start_block, end_block, 1, 5)
