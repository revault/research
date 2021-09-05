""" 
TODO:
* simulate requirement for cancel feebump, then implement feebump algo
* Use integers for all values with units in satoshis
* Add random time interval between balance low & re-fill trigger (to simulate slow stakeholder),
  to investigate time-at-risk. 
* Make good documentation
* Remove possibility for inconsistencies in progress of blocks with WTSim   
    - could break with certain DELEGATION_PERIODs. 
"""

import logging
import numpy as np

from pandas import read_csv
from utils import TX_OVERHEAD_SIZE, P2WPKH_INPUT_SIZE, P2WPKH_OUTPUT_SIZE, MAX_TX_SIZE


class StateMachine:
    """Watchtower state machine."""

    def __init__(self, config):
        self.n_stk = config["n_stk"]
        self.n_man = config["n_man"]
        # vaults = [{"id": str, "amount": int, "fee_reserve": [fbcoin]}, ...]
        self.vaults = []
        # fbcoins = [{"idx": int, "amount": int, "allocation": Option<vaultID>, "processed": Option<block_num>}, ...]
        self.fbcoins = []
        self.fbcoin_count = 0

        self.feerate_df = read_csv(config["feerate_src"], index_col="Block")
        self.estimate_smart_feerate_df = read_csv(
            config["estimate_smart_feerate_src"], parse_dates=True, index_col="DateTime"
        )

        self.weights_df = read_csv(config["weights_src"], sep=";")
        # Set (n_stk, n_man) as multiindex for weights dataframe that gives transaction size in WUs
        self.weights_df.set_index(["n_stk", "n_man"], inplace=True)

        self.block_datetime_df = read_csv(
            config["block_datetime_src"], index_col="Block"
        )

        # analysis strategy over historical feerates for fee_reserve
        self.reserve_strat = config["reserve_strat"]
        # analysis strategy over historical feerates for Vm
        self.estimate_strat = config["estimate_strat"]

        self.O_version = config["O_version"]
        self.I_version = config["I_version"]

        self.O_0_factor = 7  # num of Vb coins
        self.O_1_factor = 2  # multiplier M

        # avoid unnecessary search by caching fee reserve per vault
        self.frpv = (None, None)  # block, value

    def _get_block_datetime(self, block_num):
        """Return datetime associated with given block number."""
        return self.block_datetime_df["DateTime"][block_num]

    def _remove_coin(self, coin):
        self.fbcoins.remove(coin)
        for vault in self.vaults:
            try:
                vault["fee_reserve"].remove(coin)
            # Not in that vault
            except (ValueError):
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
        loc = self.estimate_smart_feerate_df.index.get_loc(
            key=datetime, method="ffill", tolerance="0000-00-00 04:00:00"
        )
        # If data isnan or less than or equal to 0, return value error
        estimate = self.estimate_smart_feerate_df[f"{target}"][loc]
        if np.isnan(estimate) or (estimate <= 0):
            raise (
                ValueError(f"No estimate smart feerate data at block {block_height}")
            )
        else:
            return estimate

    def _feerate_reserve_per_vault(self, block_height):
        """Return feerate reserve per vault (satoshi/vbyte). The value is determined from a
        statistical analysis of historical feerates, using one of the implemented strategies
        chosen with the self.reserve_strat parameter.
        """
        if self.frpv[0] == block_height:
            return self.frpv[1]

        else:
            thirtyD = 144 * 30  # 30 days in blocks
            ninetyD = 144 * 90  # 90 days in blocks
            if self.reserve_strat not in self.feerate_df:
                if self.reserve_strat == "95Q30":
                    self.feerate_df["95Q30"] = (
                        self.feerate_df["MeanFeerate"]
                        .rolling(thirtyD, min_periods=144)
                        .quantile(quantile=0.95, interpolation="linear")
                    )
                elif self.reserve_strat == "95Q90":
                    self.feerate_df["95Q90"] = (
                        self.feerate_df["MeanFeerate"]
                        .rolling(ninetyD, min_periods=144)
                        .quantile(quantile=0.95, interpolation="linear")
                    )
                elif self.reserve_strat == "CUMMAX95Q90":
                    self.feerate_df["CUMMAX95Q90"] = (
                        self.feerate_df["MeanFeerate"]
                        .rolling(ninetyD, min_periods=144)
                        .quantile(quantile=0.95, interpolation="linear")
                        .cummax()
                    )
                else:
                    raise ValueError("Strategy not implemented")

            self.frpv = (
                block_height,
                self.feerate_df[self.reserve_strat][block_height],
            )
            return self.frpv[1]

    def _feerate(self, block_height):
        """Return a current feerate estimate (satoshi/vbyte). The value is determined from a
        statistical analysis of historical feerates, using one of the implemented strategies
        chosen with the self.estimate_strat parameter.
        """
        thirtyD = 144 * 30  # 30 days in blocks
        if self.estimate_strat not in self.feerate_df:
            if self.estimate_strat == "MA30":
                self.feerate_df["MA30"] = (
                    self.feerate_df["MeanFeerate"]
                    .rolling(thirtyD, min_periods=144)
                    .mean()
                )

            elif self.estimate_strat == "ME30":
                self.feerate_df["ME30"] = (
                    self.feerate_df["MeanFeerate"]
                    .rolling(thirtyD, min_periods=144)
                    .median()
                )
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
        if tx_type not in ["cancel", "emergency", "unemergency"]:
            raise ValueError("Invalid tx_type")
        # feerate in satoshis/vbyte, weights in WU == 4vbytes, so feerate*weight/4 gives satoshis
        return round(
            feerate
            * (
                int(
                    self.weights_df[tx_type][self.n_stk, self.n_man]
                    + n_fb_inputs * self.weights_df["feebump"][self.n_stk, self.n_man]
                )
                / 4
            ),
            0,
        )

    def fee_reserve_per_vault(self, block_height):
        reserve = self._feerate_to_fee(
            self._feerate_reserve_per_vault(block_height), "cancel", 0
        )
        Vm = self.Vm(block_height)
        # FIXME: Is buf_factor necessary? Consider the excess for re-fills and
        # the guarantees for the output set of the CF Txs
        buf_factor = 1
        return max(reserve, Vm)

    def Vm(self, block_height):
        """Amount for the main feebump coin"""
        feerate = self._feerate(block_height)
        Vm = self._feerate_to_fee(feerate, "cancel", 0) + feerate * (272.0 / 4)
        if Vm <= 0:
            raise ValueError(
                f"Vm = {Vm} for block {block_height}. Shouldn't be non-positive."
            )
        return Vm

    def Vb(self, block_height):
        """Amount for a backup feebump coin"""
        reserve = self.fee_reserve_per_vault(block_height)
        reserve_rate = self._feerate_reserve_per_vault(block_height)
        t1 = (reserve - self.Vm(block_height)) / self.O_0_factor
        t2 = reserve_rate * (272.0 / 4) + self._feerate_to_fee(10, "cancel", 0)
        return max(t1, t2)

    def fb_coins_dist(self, block_height):
        """The coin distribution to create with a CF TX.

        O(t) in the paper.
        The sum over O should be equal to the fee reserve per vault.
        There should be at least 1 Vm sized coin in O.
        """
        # Strategy 0
        # O = [Vm, Vb, Vb, ...Vb]  with 1 + self.O_0_factor elements
        if self.O_version == 0:
            Vm = self.Vm(block_height)
            Vb = self.Vb(block_height)
            return [Vm] + [Vb for i in range(self.O_0_factor)]

        # Strategy 1
        # O = [Vm, MVm, 2MVm, 3MVm, ...]
        if self.O_version == 1:
            frpv = int(self.fee_reserve_per_vault(block_height))
            Vm = self.Vm(block_height)
            M = self.O_1_factor  # Factor increase per coin
            O = [Vm]
            while sum(O) < frpv:
                O.append(int((len(O)) * M * Vm))
            diff = sum(O) - frpv
            # find the minimal subset sum of elements that is greater than diff, and remove them
            subset = []
            while sum(subset) < diff:
                subset.append(O.pop())
            excess = sum(subset) - diff
            if excess >= Vm:
                O.append(excess)
            else:
                O[-1] += excess
            return O

    def balance(self):
        return sum([coin["amount"] for coin in self.fbcoins])

    def under_requirement(self, fee_reserve, block_height):
        """Returns the amount under requirement for the given fee_reserve."""
        required_reserve = self.fee_reserve_per_vault(block_height)
        total = sum([coin["amount"] for coin in fee_reserve])
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
        t1 = reserve_rate * (272.0 / 4) + self._feerate_to_fee(10, "cancel", 0)
        t2 = self.Vm(block_height)
        minimum = min(t1, t2)
        if coin["amount"] <= minimum:
            return True
        else:
            return False

    def refill(self, amount):
        """Refill the WT by generating a new feebump coin worth 'amount', with no allocation."""
        utxo = {
            "idx": self.fbcoin_count,
            "amount": amount,
            "allocation": None,
            "processed": None,
        }
        self.fbcoin_count += 1
        self.fbcoins.append(utxo)

    # FIXME: should return the picked coins
    def grab_coins_0(self, block_height):
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
            total += coin["amount"]
            self._remove_coin(coin)
            num_inputs += 1

        return total, num_inputs

    # FIXME: should return the picked coins
    def grab_coins_1(self, block_height):
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
            if coin["processed"] == None:
                total_unprocessed += coin["amount"]
                self._remove_coin(coin)
                num_inputs += 1

        # Take all fbcoins that are negligible, get their total amount, and remove
        # them from self.fbcoins and their associated vault's fee_reserve
        total_negligible = 0
        for coin in list(self.fbcoins):
            if self.is_negligible(coin, block_height):
                total_negligible += coin["amount"]
                self._remove_coin(coin)
                num_inputs += 1

        # Only consolidate old coins during low fee periods, defined as when the
        # current feerate is less than 1/x of the feerate for the reserve per vault. Otherwise,
        # only do fan-out.
        feerate = self._feerate(block_height)
        reserve_rate = self._feerate_reserve_per_vault(block_height)
        old_age = 12 * 7 * 144  # 12 weeks worth of blocks
        total_old = 0
        x = 10  # FIXME: find appropriate value
        if feerate < reserve_rate / x:
            for coin in list(self.fbcoins):
                if (block_height - coin["processed"] > old_age) and (
                    coin["allocation"] == None
                ):
                    total_old += coin["amount"]
                    self._remove_coin(coin)
                    num_inputs += 1

        total_to_consume = total_unprocessed + total_negligible + total_old
        return total_to_consume, num_inputs

    def grab_coins_2(self, block_height):
        """Select coins to consume as inputs for the tx transaction,
        remove them from P and V.
        - unprocessed
        - unallocated and not in O(t) and old
        - negligible

        Return: total amount of consumed inputs, number of inputs
        """
        fb_coins = self.fb_coins_dist(block_height)
        num_inputs = 0

        # Take all fbcoins that haven't been processed, get their total amount,
        # and remove them from self.fbcoins
        total_unprocessed = 0
        # loop over copy of the list since the remove() method changes list indexing
        for coin in list(self.fbcoins):
            if coin["processed"] == None:
                total_unprocessed += coin["amount"]
                self._remove_coin(coin)
                num_inputs += 1

        # Take all unallocated, old, and not in fb_coins, get their total amount,
        # and remove them from self.fbcoins
        total_unallocated = 0
        old_age = 12 * 7 * 144  # 13 weeks worth of blocks
        for coin in list(self.fbcoins):
            if coin["allocation"] == None and coin["amount"] not in fb_coins:
                if block_height - coin["processed"] > old_age:
                    total_unallocated += coin["amount"]
                    self._remove_coin(coin)
                    num_inputs += 1

        # Take all fbcoins that are negligible, get their total amount, and remove
        # them from self.fbcoins and their associated vault's fee_reserve
        total_negligible = 0
        for coin in list(self.fbcoins):
            if self.is_negligible(coin, block_height):
                total_negligible += coin["amount"]
                self._remove_coin(coin)
                num_inputs += 1

        total_to_consume = total_unprocessed + total_unallocated + total_negligible
        return total_to_consume, num_inputs

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
        fb_coins = self.fb_coins_dist(block_height)

        # Select and consume inputs with I(t), returning the total amount and the
        # number of inputs.
        if self.I_version == 0:
            total_to_consume, num_inputs = self.grab_coins_0(block_height)
        elif self.I_version == 1:
            total_to_consume, num_inputs = self.grab_coins_1(block_height)
        elif self.I_version == 2:
            total_to_consume, num_inputs = self.grab_coins_2(block_height)

        # Counter for number of outputs of the CF Tx
        num_outputs = 0

        # Now create a distribution of new coins
        num_new_reserves = total_to_consume // (sum(fb_coins))

        if num_new_reserves == 0:
            logging.debug(
                "        CF Tx failed sice num_new_reserves = 0 (not accounting for expected fee)"
            )
            # Not enough in available coins to fanout to 1 complete fee_reserve, so return
            # to initial state and return 0 (as in, 0 fee paid)
            self.fbcoins = fbcoins_copy
            self.vaults = vaults_copy
            return 0

        for i in range(0, int(num_new_reserves)):
            for x in fb_coins:
                self.fbcoins.append(
                    {
                        "idx": self.fbcoin_count,
                        "amount": x,
                        "allocation": None,
                        "processed": block_height,
                    }
                )
                self.fbcoin_count += 1
                num_outputs += 1

        # Compute fee for CF Tx
        try:
            feerate = self._estimate_smart_feerate(block_height)
        except (ValueError, KeyError):
            feerate = self._feerate(block_height)

        cf_tx_fee = (
            TX_OVERHEAD_SIZE
            + num_outputs * P2WPKH_OUTPUT_SIZE
            + num_inputs * P2WPKH_INPUT_SIZE
        ) * feerate

        # If there is any remainder, use it first to pay the fee for this transaction
        remainder = total_to_consume - (num_new_reserves * sum(fb_coins))

        # Check if remainder would cover the fee for the tx, if so, add the remainder-fee to the final new coin
        if remainder > cf_tx_fee:
            self.fbcoins[-1]["amount"] += remainder - cf_tx_fee
            return cf_tx_fee
        else:
            if num_new_reserves == 1:
                logging.debug(
                    f"        CF Tx failed sice num_new_reserves = 0 (accounting for expected fee)"
                )
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
            if self.O_version == 0:
                outputs = list(self.fbcoins[: -num_outputs - 1 : -1])
            # Take from largest first
            if self.O_version == 1:
                outputs = sorted(
                    list(self.fbcoins[: -num_outputs - 1 : -1]),
                    key=lambda coin: coin["amount"],
                    reverse=True,
                )

            for coin in outputs:
                # This coin is sufficient
                if coin["amount"] >= cf_tx_fee:
                    # Update the amount in the actual self.fbcoins list, not in the copy
                    for c in self.fbcoins:
                        if c == coin:
                            c["amount"] -= cf_tx_fee
                            cf_tx_fee = 0
                            break  # coins are unique, can stop when the correct one is found
                    break
                # The coin can't cover the fee, but could cover if it was entirely consumed and
                # the tx size is reduced by removing the output
                elif (
                    cf_tx_fee
                    > coin["amount"]
                    >= cf_tx_fee - P2WPKH_OUTPUT_SIZE * feerate
                ):
                    self.fbcoins.remove(coin)
                    cf_tx_fee = 0
                    num_outputs -= 1
                    # FIXME: Currently overpays slightly, is that ok? or better to add difference to
                    # one of the other fbcoins?
                    break
                # The coin can't cover the fee even if the tx's size is reduced by not including
                # this coin as an output
                elif cf_tx_fee - P2WPKH_OUTPUT_SIZE * feerate > coin["amount"]:
                    self.fbcoins.remove(coin)
                    cf_tx_fee -= coin["amount"]
                    cf_tx_fee -= P2WPKH_OUTPUT_SIZE * feerate
                    num_outputs -= 1

            if cf_tx_fee > 0:
                raise (
                    RuntimeError(
                        "The fee for the consolidate-fanout transaction is too high to pay for even if it creates 0 outputs"
                    )
                )

            # note: cf_tx_fee used above to track how much each output contributed to the required fee,
            # so it is recomputed here to return the actual fee paid
            cf_size = (
                10.75
                + num_outputs * P2WPKH_OUTPUT_SIZE
                + num_inputs * P2WPKH_INPUT_SIZE
            )
            if cf_size > MAX_TX_SIZE:
                raise (
                    RuntimeError(
                        "The consolidate_fanout transactino is too large! Please be smarter when constructing it."
                    )
                )
            cf_tx_fee = cf_size * feerate
            return cf_tx_fee

    def allocate(self, vaultID, amount, block_height):
        """WT allocates coins to a (new/existing) vault if there is enough
        available coins to meet the requirement.
        """
        # Recovery state
        fbcoins_copy = list(self.fbcoins)
        vaults_copy = list(self.vaults)

        try:
            # If vault already exists, de-allocate its current fee reserve first
            vault = next(vault for vault in self.vaults if vault["id"] == vaultID)
            if self.under_requirement(vault["fee_reserve"], block_height) == 0:
                return
            else:
                logging.debug(
                    f"  Allocation transition to an existing vault {vaultID} at block {block_height}"
                )
                for coin in vault["fee_reserve"]:
                    coin["allocation"] = None
                self.vaults.remove(vault)
        except (StopIteration):
            logging.debug(
                f"  Allocation transition to new vault {vaultID} at block {block_height}"
            )

        total_unallocated = round(
            sum(
                [
                    coin["amount"]
                    for coin in self.fbcoins
                    if (coin["allocation"] == None) & (coin["processed"] != None)
                ]
            ),
            0,
        )
        required_reserve = self.fee_reserve_per_vault(block_height)

        Vm = self.Vm(block_height)
        logging.debug(f"    Fee Reserve per Vault: {required_reserve}, Vm = {Vm}")
        if int(required_reserve) > int(total_unallocated):
            self.fbcoins = fbcoins_copy
            self.vaults = vaults_copy
            raise RuntimeError(
                f"Watchtower doesn't ackknowledge delegation for vault {vaultID} since total un-allocated and processed fee-reserve is insufficient"
            )

        # WT begins allocating feebump coins to this new vault and finally updates the vault's fee_reserve
        else:
            fee_reserve = []
            tolerances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            search_Vm = True
            while sum([coin["amount"] for coin in fee_reserve]) < required_reserve:
                if search_Vm == True:
                    for tol in tolerances:
                        try:
                            fbcoin = next(
                                coin
                                for coin in self.fbcoins
                                if (coin["allocation"] == None)
                                & ((1 + tol) * Vm >= coin["amount"] >= ((1 - tol) * Vm))
                            )
                            fbcoin.update(
                                {
                                    "idx": fbcoin["idx"],
                                    "amount": fbcoin["amount"],
                                    "allocation": vaultID,
                                    "processed": fbcoin["processed"],
                                }
                            )
                            fee_reserve.append(fbcoin)
                            search_Vm = False
                            logging.debug(
                                f"    Vm = {fbcoin['amount']} coin found with tolerance {tol*100}%, added to fee reserve"
                            )
                            break
                        except (StopIteration):
                            if tol == tolerances[-1]:
                                logging.debug(
                                    f"    No coin found for Vm = {Vm} with tolerance {tol*100}%"
                                )
                                search_Vm = False
                            continue
                available = [
                    coin
                    for coin in self.fbcoins
                    if (coin["allocation"] == None) & (coin["processed"] != None)
                ]
                if available == []:
                    self.fbcoins = fbcoins_copy
                    self.vaults = vaults_copy
                    raise (RuntimeError(f"No available coins for delegation"))
                # Scan through remaining coins (ignoring Vm-sized first)
                for tol in tolerances[::-1]:
                    try:
                        # FIXME: Is there advantage to choosing randomly?
                        fbcoin = next(
                            coin
                            for coin in available
                            if ((1 + tol) * Vm < coin["amount"])
                            or (coin["amount"] < (1 - tol) * Vm)
                        )
                        fbcoin.update(
                            {
                                "idx": fbcoin["idx"],
                                "amount": fbcoin["amount"],
                                "allocation": vaultID,
                                "processed": fbcoin["processed"],
                            }
                        )
                        fee_reserve.append(fbcoin)
                        logging.debug(
                            f"    Coin of size {fbcoin['amount']} added to the fee reserve"
                        )
                        break
                    except (StopIteration):
                        if tol == tolerances[::-1][-1]:
                            logging.debug(
                                f"    No coin found with size other than Vm = {Vm} with tolerance {tol*100}%"
                            )

                # Allocate additional Vm coins if no other available coins
                all_Vm = all(
                    (1 + tolerances[0]) * Vm
                    >= coin["amount"]
                    >= (1 - tolerances[0]) * Vm
                    for coin in available
                )
                if all_Vm:
                    logging.debug(
                        f"    All coins found were Vm-sized at block {block_height}"
                    )
                    fbcoin = next(coin for coin in available)
                    fbcoin.update(
                        {
                            "idx": fbcoin["idx"],
                            "amount": fbcoin["amount"],
                            "allocation": vaultID,
                            "processed": fbcoin["processed"],
                        }
                    )
                    fee_reserve.append(fbcoin)

            new_reserve_total = sum([coin["amount"] for coin in fee_reserve])
            assert new_reserve_total >= required_reserve
            logging.debug(
                f"    Reserve for vault {vaultID} has excess of {new_reserve_total-required_reserve}"
            )

            # Successful new delegation and allocation!
            self.vaults.append(
                {"id": vaultID, "amount": amount, "fee_reserve": fee_reserve}
            )

    def process_cancel(self, vaultID, block_height):
        """The cancel must be updated with a fee (the large Vm allocated to it).
        If this fee is unsuccessful at pushing the cancel through, additional small coins may
        be added from the fee_reserve.
        """
        vault = next((vault for vault in self.vaults if vault["id"] == vaultID), None)
        if vault == None:
            raise RuntimeError(f"No vault found with id {vaultID}")

        try:
            init_fee = self._estimate_smart_feerate(block_height)
        except (ValueError, KeyError):
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
            if vault["fee_reserve"] == []:
                raise RuntimeError(
                    f"Fee reserve for vault {vault['id']} was insufficient to process cancel tx"
                )

            # sort in increasing order of amount
            reserve = sorted(vault["fee_reserve"], key=lambda coin: coin["amount"])
            try:
                fbcoin = next(coin for coin in reserve if coin["amount"] > init_fee)
                vault["fee_reserve"].remove(fbcoin)
                self.fbcoins.remove(fbcoin)
                init_fee -= fbcoin["amount"]
                cancel_fb_inputs.append(fbcoin)
            except (StopIteration):
                fbcoin = reserve[-1]
                vault["fee_reserve"].remove(fbcoin)
                self.fbcoins.remove(fbcoin)
                init_fee -= fbcoin["amount"]
                cancel_fb_inputs.append(fbcoin)

        return cancel_fb_inputs

    def finalize_cancel(self, vaultID):
        """Once the cancel is confirmed, any remaining fbcoins allocated to vaultID
        become unallocated. The vault with vaultID is removed from vaults.
        """
        for coin in self.fbcoins:
            if coin["allocation"] == vaultID:
                coin["allocation"] = None
        for vault in list(self.vaults):
            if vault["id"] == vaultID:
                self.vaults.remove(vault)

    def process_spend(self, vaultID):
        """Once a vault is consumed with a spend, the fee-reserve that was allocated to it
        becomes un-allocated and the vault is removed from the set of vaults.
        """
        self.vaults = [vault for vault in self.vaults if vault["id"] != vaultID]

        for coin in self.fbcoins:
            if coin["allocation"] == vaultID:
                coin["allocation"] = None

    def risk_status(self, block_height):
        """Return a summary of the risk status for the set of vaults being watched."""
        # For cancel
        under_requirement = []
        for vault in self.vaults:
            y = self.under_requirement(vault["fee_reserve"], block_height)
            if y != 0:
                under_requirement.append(y)
        # For delegation
        available = [coin for coin in self.fbcoins if coin["allocation"] == None]
        delegation_requires = sum(self.fb_coins_dist(block_height)) - sum(
            [coin["amount"] for coin in available]
        )
        if delegation_requires < 0:
            delegation_requires = 0
        return {
            "block": block_height,
            "num_vaults": len(self.vaults),
            "vaults_at_risk": len(under_requirement),
            "severity": sum(under_requirement),
            "delegation_requires": delegation_requires,
        }
