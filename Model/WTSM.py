""" 
TODO:
* simulate requirement for cancel feebump, then implement feebump algo
* Use integers for all values with units in satoshis
* Add random time interval between balance low & re-fill trigger (to simulate slow stakeholder),
  to investigate time-at-risk. 
* Consider ordering of outputs in fee generator for CF Tx
* Make good documentation
* Remove possibility for inconsistencies in progress of blocks with WTSim

Plotting:
* plot the amount wasted by overpaying fees due to inadequate availability of coins
* plot O and FeeRate
"""

import hashlib
import time
from random import randint, random, choice
from pandas import read_csv, DataFrame, option_context, Timedelta, to_datetime
from matplotlib import pyplot as plt
import numpy as np


class WTSM():
    """Watchtower state machine."""

    def __init__(self, config):
        self.n_stk = config['n_stk']
        self.n_man = config['n_man']
        # vaults = [{"id": str, "amount": int, "fee_reserve": [fbcoin]}, ...]
        self.vaults = []
        # fbcoins = [{"idx": int, "amount": int, "allocation": Option<vaultID>, "processed": Option<block_num>}, ...]
        self.fbcoins = []
        self.fbcoin_count = 0

        self.feerate_df = read_csv(
            config['feerate_src'], index_col="Block")
        self.estimate_smart_feerate_df = read_csv(
            config['estimate_smart_feerate_src'], parse_dates=True, index_col="DateTime")

        self.weights_df = read_csv(config['weights_src'], sep=";")
        # Set (n_stk, n_man) as multiindex for weights dataframe that gives transaction size in WUs
        self.weights_df.set_index(['n_stk', 'n_man'], inplace=True)

        self.block_datetime_df = read_csv(
            config['block_datetime_src'], index_col="Block")

        # analysis strategy over historical feerates for fee_reserve
        self.reserve_strat = config['reserve_strat']
        # analysis strategy over historical feerates for Vm
        self.estimate_strat = config['estimate_strat']

        self.num_Vb_coins = 7

        self.O_version = config['O_version']
        self.I_version = config['I_version']

    def _get_block_datetime(self, block_num):
        """Return datetime associated with given block number."""
        return self.block_datetime_df['DateTime'][block_num]

    def _remove_coin(self, coin):
        self.fbcoins.remove(coin)
        for vault in self.vaults:
            try:
                vault['fee_reserve'].remove(coin)
            # Not in that vault
            except(ValueError):
                continue

        # FIXME: Does this version perform better or worse? 
        # for vault in self.vaults:
        #     if coin in vault['fee_reserve']:
        #         vault['fee_reserve'].remove(coin)
        #         break # Coin is unique, stop looping once found and removed

    def _estimate_smart_feerate(self, block_height):
        # Simulator provides current block_height. Data source for estimateSmartFee indexed by datetime.
        # Convert block_height to datetime and find the previous (using 'ffill') datum for associated
        # block_height.
        target = "block1-2"
        datetime = self._get_block_datetime(block_height)
        loc = self.estimate_smart_feerate_df.index.get_loc(key=datetime, method="ffill", tolerance="0000-00-00 04:00:00")
        # If data isnan or less than or equal to 0, return value error
        estimate = self.estimate_smart_feerate_df[f"{target}"][loc]
        if np.isnan(estimate) or (estimate <= 0):
            raise(ValueError(f"No estimate smart feerate data at block {block_height}"))
        else:
            return estimate

    def _feerate_reserve_per_vault(self, block_height):
        """Return feerate reserve per vault (satoshi/vbyte). The value is determined from a
           statistical analysis of historical feerates, using one of the implemented strategies
           chosen with the self.reserve_strat parameter. 
        """
        thirtyD = 144*30 # 30 days in blocks
        ninetyD = 144*90 # 90 days in blocks
        if self.reserve_strat not in self.feerate_df:
            if self.reserve_strat == '95Q30':
                self.feerate_df['95Q30'] = self.feerate_df['MeanFeerate'].rolling(thirtyD, min_periods=144).quantile(
                    quantile=0.95, interpolation='linear')

            elif self.reserve_strat == '95Q90':
                self.feerate_df['95Q90'] = self.feerate_df['MeanFeerate'].rolling(ninetyD, min_periods=144).quantile(
                    quantile=0.95, interpolation='linear')

            elif self.reserve_strat == 'CUMMAX95Q90':
                self.feerate_df['CUMMAX95Q90'] = self.feerate_df['MeanFeerate'].rolling(ninetyD, min_periods=144).quantile(
                    quantile=0.95, interpolation='linear').cummax()

            else:
                raise ValueError("Strategy not implemented")

        return self.feerate_df[self.reserve_strat][block_height]

    def _feerate(self, block_height):
        """Return a current feerate estimate (satoshi/vbyte). The value is determined from a
           statistical analysis of historical feerates, using one of the implemented strategies
           chosen with the self.estimate_strat parameter. 
        """
        thirtyD = 144*30 # 30 days in blocks
        if self.estimate_strat not in self.feerate_df:
            if self.estimate_strat == 'MA30':
                self.feerate_df['MA30'] = self.feerate_df['MeanFeerate'].rolling(
                    thirtyD, min_periods=144).mean()

            elif self.estimate_strat == 'ME30':
                self.feerate_df['ME30'] = self.feerate_df['MeanFeerate'].rolling(
                    thirtyD, min_periods=144).median()
            else:
                raise ValueError("Strategy not implemented")

        return self.feerate_df[self.estimate_strat][block_height]

    def _feerate_to_fee(self, feerate, tx_type, n_fb_inputs):
        """Convert feerate (satoshi/vByte) into transaction fee (satoshi).

        Keyword arguments:
        feerate - the feerate to be converted
        tx_type - 'cancel', 'emergency', or 'unemergency'
        n_fb_inputs - number of feebump inputs included in the tx's size
        """
        if tx_type not in ['cancel', 'emergency', 'unemergency']:
            raise ValueError("Invalid tx_type")
        # feerate in satoshis/vbyte, weights in WU == 4vbytes, so feerate*weight/4 gives satoshis
        return round(feerate*(int(self.weights_df[tx_type][self.n_stk, self.n_man]+n_fb_inputs*self.weights_df['feebump'][self.n_stk, self.n_man])/4), 0)

    def fee_reserve_per_vault(self, block_height):
        reserve = self._feerate_to_fee(
            self._feerate_reserve_per_vault(block_height), 'cancel', 0)
        Vm = self.Vm(block_height)
        # FIXME: Is buf_factor necessary? Consider the excess for re-fills and
        # the guarantees for the output set of the CF Txs
        buf_factor = 1
        return max(reserve, Vm)

    def Vm(self, block_height):
        """Amount for the main feebump coin
        """
        feerate = self._feerate(block_height)
        Vm = self._feerate_to_fee(
            feerate, 'cancel', 0) + feerate*(272.0/4)
        if Vm < 0:
            raise ValueError(f"Vm = {Vm} for block {block_height}. Shouldn't be negative.")
        return Vm

    def Vb(self, block_height):
        """Amount for a backup feebump coin
        """
        reserve = self.fee_reserve_per_vault(block_height)
        reserve_rate = self._feerate_reserve_per_vault(block_height)
        t1 = (reserve - self.Vm(block_height))/self.num_Vb_coins
        t2 = reserve_rate*(272.0/4) + self._feerate_to_fee(10, 'cancel', 0)
        return max(t1, t2)

    def O(self, block_height):
        """O is the coin distribution to create with a CF TX.

           The sum over O should be equal to the fee reserve per vault.
           There should be at least 1 Vm sized coin in O.
        """
        # Strategy 0
        # O = [Vm, Vb, Vb, ...Vb]  with 1 + self.num_Vb_coins elements
        if self.O_version == 0:
            Vm = self.Vm(block_height)
            Vb = self.Vb(block_height)
            O = [Vb for i in range(self.num_Vb_coins)]
            O.insert(0, Vm)
            return O

        # Strategy 1
        # O = [Vm, M*Vm, (M^2)Vm, (M^3)Vm, ....]
        if self.O_version == 1:
            reserve = self.fee_reserve_per_vault(block_height)
            Vm = self.Vm(block_height)
            M = 1.2  # Factor increase per coin
            # FIXME: What is a better way than to hard code this range?
            O = [Vm*(M**i) for i in range(0, 15)]
            cumO = []
            for x in O:
                cumO.append(x+sum(cumO))
            for (x, y) in list(zip(O, cumO)):
                if y > reserve:
                    cumO.remove(y)
                    O.remove(x)
            diff = reserve - sum(O)
            # If the difference isn't significant (i.e. it's smaller than Vm), add the diff to the final
            # coin. Else, create a new coin of amount equal to the difference.
            if diff < O[0]:
                O[-1] += diff
            else:
                O.append(diff)
            return O

        # Strategy 2
        # O = [Vm, Vm, Vm, Vm, ..., Vm*]
        if self.O_version == 2:
            reserve = self.fee_reserve_per_vault(block_height)
            Vm = self.Vm(block_height)
            O = []
            while sum(O) <= reserve:
                O.append(Vm)
            if len(O) > 1:
                O.pop()
            O[-1] += reserve - sum(O)
            assert(sum(O) == reserve)
            return O

    def balance(self):
        return sum([coin['amount'] for coin in self.fbcoins])

    def under_requirement(self, fee_reserve, block_height):
        """Returns the amount under requirement for the given fee_reserve. 
        """
        required_reserve = self.fee_reserve_per_vault(block_height)
        total = sum([coin['amount'] for coin in fee_reserve])
        if total >= required_reserve:
            return 0
        else:
            # Satoshis should be integer amounts
            return int(required_reserve - total)

    def is_negligible(self, coin, block_height):
        """A coin is considered negligible if its amount is less than the minimum
           of Vm and the fee required to bump a Cancel transaction by 10 sats per vByte
           in the worst case (when the fee rate is equal to the reserve rate).
           Note: t1 is same as the lower bound of Vb.
        """
        # FIXME: What is a reasonable factor of a 'negligible coin'?
        reserve_rate = self._feerate_reserve_per_vault(block_height)
        t1 = reserve_rate*(272.0/4) + self._feerate_to_fee(10, 'cancel', 0)
        t2 = self.Vm(block_height)
        minimum = min(t1, t2)
        if coin['amount'] <= minimum:
            return True
        else:
            return False

    def refill(self, amount):
        """Refill the WT by generating a new feebump coin worth 'amount', with no allocation."""
        utxo = {'idx': self.fbcoin_count, 'amount': amount,
                'allocation': None, 'processed': None}
        self.fbcoin_count += 1
        self.fbcoins.append(utxo)

    def I0(self, block_height):
        """Select coins to consume as inputs for the tx transaction,
           remove them from P and V.

           Return: total amount of consumed inputs, number of inputs 
        """
        num_inputs = 0
        # Take all fbcoins, get their total amount,
        # and remove them from self.fbcoins
        total = 0
        # loop over copy of the list since the remove() method changes list indexing
        for coin in list(self.fbcoins):
            total += coin['amount']
            self._remove_coin(coin)
            num_inputs += 1

        return total, num_inputs

    def I1(self, block_height):
        """Select coins to consume as inputs for the tx transaction,
           remove them from P and V.

           Return: total amount of consumed inputs, number of inputs 
        """
        num_inputs = 0
        # Take all fbcoins that haven't been processed, get their total amount,
        # and remove them from self.fbcoins
        total_unprocessed = 0
        # loop over copy of the list since the remove() method changes list indexing
        for coin in list(self.fbcoins):
            if coin['processed'] == None:
                total_unprocessed += coin['amount']
                self._remove_coin(coin)
                num_inputs += 1

        # Take all fbcoins that are negligible, get their total amount, and remove
        # them from self.fbcoins and their associated vault's fee_reserve
        total_negligible = 0
        for coin in list(self.fbcoins):
            if self.is_negligible(coin, block_height):
                total_negligible += coin['amount']
                self._remove_coin(coin)
                num_inputs += 1

        # Only consolidate old coins during low fee periods, defined as when the
        # current feerate is less than 1/x of the feerate for the reserve per vault. Otherwise,
        # only do fan-out.
        feerate = self._feerate(block_height)
        reserve_rate = self._feerate_reserve_per_vault(block_height)
        old_age = 12*7*144  # 12 weeks worth of blocks
        total_old = 0
        x = 10  # FIXME: find appropriate value
        if feerate < reserve_rate/x:
            for coin in list(self.fbcoins):
                if (block_height - coin['processed'] > old_age) and (coin['allocation'] == None):
                    total_old += coin['amount']
                    self._remove_coin(coin)
                    num_inputs += 1

        total_to_process = total_unprocessed + total_negligible + total_old
        return total_to_process, num_inputs

    def consolidate_fanout(self, block_height):
        """Simulate the WT creating a consolidate-fanout (CF) tx which aims to 1) create coins from 
           new re-fills that enable accurate feebumping and 2) to consolidate negligible feebump coins
           if the current feerate is "low". 

           Note that negligible coins that are consolidated will be removed from their 
           associated vault's fee_reserve. So this process will diminish the vault's fee
           reserve until the new coins are confirmed and re-allocated.

           The 'processed' value of a fbcoin states the block number in which the CF Tx occurred 
           that created this coin. The 'processed' value is used to determine if coins have been 
           processed yet, and their 'age'. 
           FIXME: Should "old" coins be consolidated? 

           CF transactions help maintain coin sizes that enable accurate fee-bumping.
        """
        # save initial state to recover if failure
        fbcoins_copy = list(self.fbcoins)
        vaults_copy = list(self.vaults)

        # Set target values for our coin creation amounts
        O = self.O(block_height)

        # Select and consume inputs with I(t), returning the total amount and the 
        # number of inputs.
        if self.I_version == 0:
            total_to_process, num_inputs = self.I0(block_height)
        elif self.I_version == 1:
            total_to_process, num_inputs = self.I1(block_height)
        
        # Counter for number of outputs of the CF Tx
        num_outputs = 0

        # Now create a distribution of new coins
        num_new_reserves = total_to_process//(sum(O))

        if num_new_reserves == 0:
            print(f"        CF Tx failed sice num_new_reserves = 0 (not accounting for expected fee")
            # Not enough in available coins to fanout to 1 complete fee_reserve, so return
            # to initial state and return 0 (as in, 0 fee paid)
            self.fbcoins = fbcoins_copy
            self.vaults = vaults_copy
            return 0

        for i in range(0, int(num_new_reserves)):
            for x in O:
                self.fbcoins.append(
                    {'idx': self.fbcoin_count, 'amount': x, 'allocation': None, 'processed': block_height})
                self.fbcoin_count += 1
                num_outputs += 1

        # Compute fee for CF Tx
        try:
            feerate = self._estimate_smart_feerate(block_height)
        except(ValueError, KeyError):
            feerate = self._feerate(block_height)

        P2WPKH_INPUT_vBytes = 67.75
        P2WPKH_OUTPUT_vBytes = 31
        cf_tx_fee = (10.75 + num_outputs*P2WPKH_OUTPUT_vBytes +
                     num_inputs*P2WPKH_INPUT_vBytes)*feerate

        # If there is any remainder, use it first to pay the fee for this transaction
        remainder = total_to_process - (num_new_reserves*sum(O))

        # Check if remainder would cover the fee for the tx, if so, add the remainder-fee to the final new coin
        if remainder > cf_tx_fee:
            self.fbcoins[-1]['amount'] += (remainder-cf_tx_fee)
            return cf_tx_fee
        else:
            if num_new_reserves == 1:
                print(f"        CF Tx failed sice num_new_reserves = 0 (accounting for expected fee")
                # Not enough in available coins to fanout to 1 complete fee_reserve, when accounting
                # for the fee, so return to initial state and return 0 (as in, 0 fee paid)
                self.fbcoins = fbcoins_copy
                self.vaults = vaults_copy
                return 0

            # Otherwise, implicitly consume the entire remainder, and use some
            # of the value in the CF Tx's outputs to pay for the rest of the fee
            cf_tx_fee -= remainder
            # Scan through new outputs (the most recent num_outputs coins in self.fbcoins)
            # and use them to pay for the tx (starts with last coin of O(t) in the latest reserve, 
            # and works backwards). If consumed entirely, the fee no longer needs to account for 
            # that output's size.
            outputs = list(self.fbcoins[:-num_outputs-1:-1])

            # FIXME: sorting methods will be different depending on O
            # outputs = sorted(list(self.fbcoins[:-num_outputs-1:-1]), key=lambda coin: coin['amount'], reverse=True)
            # outputs = sorted(list(self.fbcoins[:-num_outputs-1:-1]), key=lambda coin: coin['amount'])

            for coin in outputs:
                # This coin is sufficient
                if coin['amount'] >= cf_tx_fee:
                    # Update the amount in the actual self.fbcoins list, not in the copy
                    for c in self.fbcoins:
                        if c == coin:
                            c['amount'] -= cf_tx_fee
                            cf_tx_fee = 0
                            break  # coins are unique, can stop when the correct one is found
                    break
                # The coin can't cover the fee, but could cover if it was entirely consumed and
                # the tx size is reduced by removing the output
                elif cf_tx_fee > coin['amount'] >= cf_tx_fee - P2WPKH_OUTPUT_vBytes*feerate:
                    self.fbcoins.remove(coin)
                    cf_tx_fee = 0
                    num_outputs -= 1
                    # FIXME: Currently overpays slightly, is that ok? or better to add difference to
                    # one of the other fbcoins?
                    break
                # The coin can't cover the fee even if the tx's size is reduced by not including
                # this coin as an output
                elif cf_tx_fee - P2WPKH_OUTPUT_vBytes*feerate > coin['amount']:
                    self.fbcoins.remove(coin)
                    cf_tx_fee -= coin['amount']
                    cf_tx_fee -= P2WPKH_OUTPUT_vBytes*feerate
                    num_outputs -= 1

            if cf_tx_fee > 0:
                raise(RuntimeError(f"The fee for the consolidate-fanout transaction is too high to pay for even if it creates 0 outputs"))

            # note: cf_tx_fee used above to track how much each output contributed to the required fee,
            # so it is recomputed here to return the actual fee paid
            cf_tx_fee = (10.75 + num_outputs*P2WPKH_OUTPUT_vBytes +
                         num_inputs*P2WPKH_INPUT_vBytes)*feerate
            return cf_tx_fee

    def process_delegation(self, vaultID, amount, block_height):
        """Stakeholder attempts to delegate a new vault to WT. WT computes if it can accept 
           delegation, raises an error if it cannot. 
        """
        total_unallocated = round(sum(
            [coin['amount'] for coin in self.fbcoins if (coin['allocation'] == None) & (coin['processed'] != None)]), 0)
        required_reserve = self.fee_reserve_per_vault(block_height)

        Vm = self.Vm(block_height)
        print(f"    Fee Reserve per Vault: {required_reserve}, Vm = {Vm}")
        if int(required_reserve) > int(total_unallocated):
            raise RuntimeError(f"Watchtower doesn't ackknowledge delegation for vault {vaultID} since total un-allocated and processed fee-reserve is insufficient")

        # WT begins allocating feebump coins to this new vault and finally updates the vault's fee_reserve
        else:
            fee_reserve = []
            tolerances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            Vm_found = False
            while sum([coin['amount'] for coin in fee_reserve]) < required_reserve:
                if Vm_found == False:
                    for tol in tolerances:
                        try:
                            fbcoin = next(coin for coin in self.fbcoins if (coin['allocation'] == None) & (
                                (1+tol)*Vm >= coin['amount'] >= ((1-tol)*Vm)))
                            fbcoin.update({'idx': fbcoin['idx'], 'amount': fbcoin['amount'],
                                           'allocation': vaultID, 'processed': fbcoin['processed']})
                            fee_reserve.append(fbcoin)
                            Vm_found = True
                            print(f"    Vm = {fbcoin['amount']} coin found with tolerance {tol}")
                            break
                        except(StopIteration):
                            print(f"    No coin found for Vm = {Vm} with tolerance {tol}")
                            continue
                available = [coin for coin in self.fbcoins if (
                    coin['allocation'] == None) & (coin['processed'] != None)]
                if available == []:
                    raise(RuntimeError(f"No available coins for delegation"))
                # Scan through remaining coins (ignoring Vm-sized first)
                for tol in tolerances[::-1]:
                    try:
                        # FIXME: Is there advantage to choosing randomly?
                        fbcoin = next(coin for coin in available if (
                            (1+tol)*Vm < coin['amount']) or (coin['amount'] < (1-tol)*Vm))
                        fbcoin.update({'idx': fbcoin['idx'], 'amount': fbcoin['amount'],
                                       'allocation': vaultID, 'processed': fbcoin['processed']})
                        fee_reserve.append(fbcoin)
                        print(f"    Coin of size {fbcoin['amount']} added to the fee_reserve")
                        break
                    except(StopIteration):
                        print(f"    No coin found with size other than Vm = {Vm} with tolerance {tol}")

                # Allocate additional Vm coins if no other available coins
                all_Vm = all((1+tolerances[0])*Vm >= coin['amount']
                             >= (1-tolerances[0])*Vm for coin in available)
                if all_Vm:
                    print(f"    All coins found were Vm-sized at block {block_height}")
                    fbcoin = next(coin for coin in available)
                    fbcoin.update({'idx': fbcoin['idx'], 'amount': fbcoin['amount'],
                                   'allocation': vaultID, 'processed': fbcoin['processed']})
                    fee_reserve.append(fbcoin)

            new_reserve_total = sum([coin['amount'] for coin in fee_reserve])
            assert(new_reserve_total >= required_reserve)
            print(f"    Reserve for vault {vaultID} has excess of {new_reserve_total-required_reserve}")

            # Successful new delegation and allocation!
            self.vaults.append(
                {"id": vaultID, "amount": amount, "fee_reserve": fee_reserve})

    def top_up_delegations(self, block_height):
        """When the fee_reserve of a vault becomes insufficient, additional feebump coins should
           be allocated to that vault until it has the required fee reserve.

           This process should occur after a CF or refill tx.
        """
        # Optimistically assumes that the coin_pool is sufficiently filled to top up all
        # delegations, but raises RuntimeError if assumption is false.

        Vm = self.Vm(block_height)

        # Determine which vaults have insufficient fee_reserves (based on current fee-market)
        # and allocate the appropriate fbcoins.
        for vault in self.vaults:
            top_up_amount = self.under_requirement(
                vault['fee_reserve'], block_height)
            if top_up_amount == 0:
                continue
            else:
                print(f"  {vault['id']} under requirement by {top_up_amount}, topping up...")
                Vm_found = False
                reserve_top_up = []
                tolerances = [0.1, 0.2, 0.3, 0.4, 0.5]

                # Is there a Vm sized coin in the vault?
                for tol in tolerances:
                    try:
                        Vm_coin = next(coin for coin in vault['fee_reserve'] if (coin['allocation'] == None) & (
                            (1+tol)*Vm >= coin['amount'] >= ((1-tol)*Vm)))
                        Vm_found = True
                        break
                    except(StopIteration):
                        print(f"    No coin found for Vm = {Vm} with tolerance {tol}")
                        continue

                # Allocate coins to the vault, with "smart" process
                while sum([coin['amount'] for coin in reserve_top_up]) < top_up_amount:
                    # If not, try to find a Vm-sized coin and allocate it to the vault
                    if Vm_found == False:
                        for tol in tolerances:
                            try:
                                # Doesn't need to be 'processed' if it is the correct size already
                                fbcoin = next(coin for coin in self.fbcoins if (coin['allocation'] == None) & (
                                    (1+tol)*Vm >= coin['amount'] >= ((1-tol)*Vm)))
                                fbcoin.update({'idx': fbcoin['idx'], 'amount': fbcoin['amount'],
                                               'allocation': vault['id'], 'processed': fbcoin['processed']})
                                reserve_top_up.append(fbcoin)
                                Vm_found = True
                                print(f"    Vm = {fbcoin['amount']} coin found with tolerance {tol}")
                                break # from for loop
                            except(StopIteration):
                                print(f"    No coin found for Vm = {Vm} with tolerance {tol}")
                                continue # for loop
                    diff = sum([coin['amount']
                                for coin in reserve_top_up]) - top_up_amount
                    if diff <= 0:
                        break # out of while loop

                    available = [coin for coin in self.fbcoins if (
                        coin['allocation'] == None) & (coin['processed'] != None)]
                    if available == []:
                        raise(RuntimeError(f"No available coins for delegation"))

                    # Try to find smallest coin that covers the diff
                    try:
                        fbcoin = min([coin for coin in available if coin['amount'] >= diff], key=lambda c: c['amount'])
                        fbcoin.update({'idx': fbcoin['idx'], 'amount': fbcoin['amount'],
                                       'allocation': vault['id'], 'processed': fbcoin['processed']})
                        reserve_top_up.append(fbcoin)
                        print(f"    Coin of size {fbcoin['amount']} allocated to the vault")
                        break  # out of while loop
                    # If no coins cover the diff, allocate the largest coin and try again
                    except(ValueError):
                        fbcoin = max(available, key=lambda c: c['amount'])
                        fbcoin.update({'idx': fbcoin['idx'], 'amount': fbcoin['amount'],
                                       'allocation': vault['id'], 'processed': fbcoin['processed']})
                        reserve_top_up.append(fbcoin)
                        print(f"    Coin of size {fbcoin['amount']} allocated to the vault")
                        continue  # while loop

                vault['fee_reserve'].extend(reserve_top_up)

                required_reserve = self.fee_reserve_per_vault(block_height)
                new_reserve_total = sum([coin['amount']
                                         for coin in vault['fee_reserve']])
                if new_reserve_total < required_reserve:
                    raise(RuntimeError(f"Not enough available coins to top up vault {vault['id']}"))
                print(f"    Reserve for vault {vault['id']} has excess of {new_reserve_total-required_reserve}")

    def process_cancel(self, vaultID, block_height):
        """The cancel must be updated with a fee (the large Vm allocated to it).
           If this fee is unsuccessful at pushing the cancel through, additional small coins may
           be added from the fee_reserve.
        """
        vault = next(
            (vault for vault in self.vaults if vault['id'] == vaultID), None)
        if vault == None:
            raise RuntimeError(f"No vault found with id {vaultID}")

        try:
            init_fee = self._estimate_smart_feerate(block_height)
        except(ValueError, KeyError):
            init_fee = self._feerate(block_height)

        cancel_fb_inputs = []

        # Strat 1: randomly select coins until the fee is met
        # Performs moderately bad in low-stable fee market and ok in volatile fee market
        # while init_fee > 0:
        #     coin = choice(vault['fee_reserve'])
        #     init_fee -= coin['amount']
        #     cancel_fb_inputs.append(coin)
        #     vault['fee_reserve'].remove(coin)
        #     self.fbcoins.remove(coin)
        #     if vault['fee_reserve'] == []:
        #         raise RuntimeError(f"Fee reserve for vault {vault['id']} was insufficient to process cancel tx")

        # Strat 2: select smallest coins first
        # FIXME: Performs bad in low-stable feemarket and good in volatile fee market
        # while init_fee > 0:
        # smallest_coin = min(
        #     vault['fee_reserve'], key=lambda coin: coin['amount'])
        # cancel_fb_inputs.append(smallest_coin)
        # vault['fee_reserve'].remove(smallest_coin)
        # self.fbcoins.remove(smallest_coin)
        # init_fee -= smallest_coin['amount']
        # if vault['fee_reserve'] == []:
        #     raise RuntimeError(f"Fee reserve for vault {vault['id']} was insufficient to process cancel tx")

        # Strat 3: select smallest coin to cover init_fee, if no coin, remove largest and try again.
        while init_fee > 0:
            if vault['fee_reserve'] == []:
                raise RuntimeError(f"Fee reserve for vault {vault['id']} was insufficient to process cancel tx")
            
            # sort in increasing order of amount
            reserve = sorted(vault['fee_reserve'],
                             key=lambda coin: coin['amount'])
            try:
                fbcoin = next(
                    coin for coin in reserve if coin['amount'] > init_fee)
                vault['fee_reserve'].remove(fbcoin)
                self.fbcoins.remove(fbcoin)
                init_fee -= fbcoin['amount']
                cancel_fb_inputs.append(fbcoin)
            except(StopIteration):
                fbcoin = reserve[-1]
                vault['fee_reserve'].remove(fbcoin)
                self.fbcoins.remove(fbcoin)
                init_fee -= fbcoin['amount']
                cancel_fb_inputs.append(fbcoin)

        return cancel_fb_inputs

    def finalize_cancel(self, vaultID):
        """Once the cancel is confirmed, any remaining fbcoins allocated to vaultID 
           become unallocated. The vault with vaultID is removed from vaults.
        """
        for coin in self.fbcoins:
            if coin['allocation'] == vaultID:
                coin['allocation'] = None
        for vault in list(self.vaults):
            if vault['id'] == vaultID:
                self.vaults.remove(vault)

    def process_spend(self, vaultID):
        """Once a vault is consumed with a spend, the fee-reserve that was allocated to it 
           becomes un-allocated and the vault is removed from the set of vaults. 
        """
        self.vaults = [
            vault for vault in self.vaults if vault['id'] != vaultID]

        for coin in self.fbcoins:
            if coin['allocation'] == vaultID:
                coin['allocation'] = None

    def risk_status(self, block_height):
        """Return a summary of the risk status for the set of vaults being watched.
        """
        # For cancel
        under_requirement = []
        for vault in self.vaults:
            y = self.under_requirement(vault['fee_reserve'], block_height)
            if y != 0:
                under_requirement.append(y)
        # For delegation
        available = [coin for coin in self.fbcoins if coin['allocation']==None]
        delegation_requires = sum(self.O(block_height)) - sum([coin['amount'] for coin in available])
        if delegation_requires < 0:
            delegation_requires = 0
        return {"block": block_height, "num_vaults": len(self.vaults), "vaults_at_risk": len(under_requirement), "severity": sum(under_requirement), "delegation_requires": delegation_requires}        