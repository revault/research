""" 
TODO:
* simulate requirement for cancel feebump, then implement feebump algo
* Use integers for all values with units in satoshis
* Add random time interval between balance low and re-fill trigger (to simulate slow stakeholder),
  to investigate time-at-risk. 
* Make good documentation
* Remove possibility for inconsistencies in progress of blocks with WTSim   
    - could break with certain DELEGATION_PERIODs. 
"""

import logging
import numpy as np

from pandas import read_csv
from utils import (
    P2WPKH_INPUT_SIZE,
    P2WPKH_OUTPUT_SIZE,
    cf_tx_size,
    MAX_TX_SIZE,
    CANCEL_TX_WEIGHT,
    TX_OVERHEAD_SIZE,
)


class CfError(RuntimeError):
    """An error arising during the creation of the Consolidate-Fanout tx"""

    def __init__(self, message):
        self.message = message


class StateMachine:
    """Watchtower state machine."""

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
    ):
        self.n_stk = n_stk
        self.n_man = n_man
        # vaults = [{"id": str, "amount": int, "fee_reserve": [fbcoin]}, ...]
        self.vaults = []
        # fbcoins = [{"idx": int, "amount": int, "allocation": Option<vault_id>, "processed": Option<block_num>}, ...]
        self.fbcoins = []
        self.fbcoin_id = 0

        self.hist_df = read_csv(
            hist_feerate_csv, parse_dates=True, index_col="block_height"
        )

        # analysis strategy over historical feerates for fee_reserve
        self.reserve_strat = reserve_strat
        # analysis strategy over historical feerates for Vm
        self.estimate_strat = estimate_strat

        self.O_version = o_version
        self.I_version = i_version
        self.allocate_version = allocate_version

        self.O_0_factor = 7  # num of Vb coins
        self.O_1_factor = 2  # multiplier M
        self.I_2_tol = 0.2

        # avoid unnecessary search by caching fee reserve per vault, Vm, feerate
        self.frpv = (None, None)  # block, value
        self.Vm_cache = (None, None)  # block, value
        self.feerate = (None, None)  # block, value

    # FIXME: a FeebumpCoin and FeebumpCoinPool class would be neat
    def _fbcoin_id(self):
        self.fbcoin_id += 1
        return self.fbcoin_id

    def _add_coin(self, amount, allocation=None, processed=None):
        assert isinstance(amount, int)
        self.fbcoins.append(
            {
                "idx": self._fbcoin_id(),
                "amount": amount,
                "allocation": allocation,
                "processed": processed,
            }
        )

    # FIXME: have a by index version..
    def _remove_coin(self, coin):
        self.fbcoins.remove(coin)
        if coin["allocation"] is None:
            return

        # FIXME have a mapping instead of inefficiently looping..
        for vault in self.vaults:
            if coin in vault["fee_reserve"]:
                vault["fee_reserve"].remove(coin)
                return

    def _estimate_smart_feerate(self, block_height):
        # FIXME: why always 1-2 ??
        # If data isnan or less than or equal to 0, return value error
        estimate = self.hist_df["Est 1block"][block_height]
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
            if self.reserve_strat not in self.hist_df:
                if self.reserve_strat == "95Q30":
                    self.hist_df["95Q30"] = (
                        self.hist_df["mean_feerate"]
                        .rolling(thirtyD, min_periods=144)
                        .quantile(quantile=0.95, interpolation="linear")
                    )
                elif self.reserve_strat == "95Q90":
                    self.hist_df["95Q90"] = (
                        self.hist_df["mean_feerate"]
                        .rolling(ninetyD, min_periods=144)
                        .quantile(quantile=0.95, interpolation="linear")
                    )
                elif self.reserve_strat == "CUMMAX95Q90":
                    self.hist_df["CUMMAX95Q90"] = (
                        self.hist_df["mean_feerate"]
                        .rolling(ninetyD, min_periods=144)
                        .quantile(quantile=0.95, interpolation="linear")
                        .cummax()
                    )
                else:
                    raise ValueError("Strategy not implemented")

            self.frpv = (
                block_height,
                self.hist_df[self.reserve_strat][block_height],
            )
            return self.frpv[1]

    def _feerate(self, block_height):
        """Return a current feerate estimate (satoshi/vbyte). The value is determined from a
        statistical analysis of historical feerates, using one of the implemented strategies
        chosen with the self.estimate_strat parameter.
        """
        if self.feerate[0] == block_height:
            return self.feerate[1]

        thirtyD = 144 * 30  # 30 days in blocks
        if self.estimate_strat not in self.hist_df:
            if self.estimate_strat == "MA30":
                self.hist_df["MA30"] = (
                    self.hist_df["mean_feerate"]
                    .rolling(thirtyD, min_periods=144)
                    .mean()
                )

            elif self.estimate_strat == "ME30":
                self.hist_df["ME30"] = (
                    self.hist_df["mean_feerate"]
                    .rolling(thirtyD, min_periods=144)
                    .median()
                )
            else:
                raise ValueError("Strategy not implemented")

        self.feerate = (block_height, self.hist_df[self.estimate_strat][block_height])
        return self.feerate[1]

    # FIXME: remove tx_type!!
    def _feerate_to_fee(self, feerate, tx_type, n_fb_inputs):
        """Convert feerate (satoshi/vByte) into transaction fee (satoshi).

        Keyword arguments:
        feerate - the feerate to be converted
        tx_type - 'cancel', 'emergency', or 'unemergency'
        n_fb_inputs - number of feebump inputs included in the tx's size
        """
        if tx_type not in ["cancel", "emergency", "unemergency"]:
            raise ValueError("Invalid tx_type")
        # feerate is in satoshis/vbyte
        cancel_tx_size_no_fb = (CANCEL_TX_WEIGHT[self.n_stk][self.n_man] + 3) // 4
        cancel_tx_size = cancel_tx_size_no_fb + n_fb_inputs * P2WPKH_INPUT_SIZE
        return int(cancel_tx_size * feerate)

    def fee_reserve_per_vault(self, block_height):
        return self._feerate_to_fee(
            self._feerate_reserve_per_vault(block_height), "cancel", 0
        )

    def Vm(self, block_height):
        """Amount for the main feebump coin"""
        if self.Vm_cache[0] == block_height:
            return self.Vm_cache[1]

        feerate = self._feerate(block_height)
        Vm = int(
            self._feerate_to_fee(feerate, "cancel", 0) + feerate * P2WPKH_INPUT_SIZE
        )
        if Vm <= 0:
            raise ValueError(
                f"Vm = {Vm} for block {block_height}. Shouldn't be non-positive."
            )
        self.Vm_cache = (block_height, Vm)
        return Vm

    def Vb(self, block_height):
        """Amount for a backup feebump coin"""
        reserve = self.fee_reserve_per_vault(block_height)
        reserve_rate = self._feerate_reserve_per_vault(block_height)
        t1 = (reserve - self.Vm(block_height)) / self.O_0_factor
        t2 = reserve_rate * P2WPKH_INPUT_SIZE + self._feerate_to_fee(10, "cancel", 0)
        return int(max(t1, t2))

    def fb_coins_dist(self, block_height):
        """The coin distribution to create with a CF TX.

        O(t) in the paper.
        The sum over O should be equal to the fee reserve per vault.
        There should be at least 1 Vm sized coin in O.
        """
        # Strategy 0
        # dist = [Vm, Vb, Vb, ...Vb]  with 1 + self.O_0_factor elements
        if self.O_version == 0:
            Vm = self.Vm(block_height)
            Vb = self.Vb(block_height)
            return [Vm] + [Vb for i in range(self.O_0_factor)]

        # Strategy 1
        # dist = [Vm, MVm, 2MVm, 3MVm, ...]
        if self.O_version == 1:
            frpv = self.fee_reserve_per_vault(block_height)
            Vm = self.Vm(block_height)
            M = self.O_1_factor  # Factor increase per coin
            dist = [Vm]
            while sum(dist) < frpv:
                dist.append(int((len(dist)) * M * Vm))
            diff = sum(dist) - frpv
            # find the minimal subset sum of elements that is greater than diff, and remove them
            subset = []
            while sum(subset) < diff:
                subset.append(dist.pop())
            excess = sum(subset) - diff
            assert isinstance(excess, int)
            if excess >= Vm:
                dist.append(excess)
            else:
                dist[-1] += excess
            return dist

    def unallocated_balance(self):
        return sum(
            [coin["amount"] for coin in self.fbcoins if coin["allocation"] == None]
        )

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
        of Vm and the fee required to bump a Cancel transaction by 1 sat per vByte
        in the worst case (when the fee rate is equal to the reserve rate).
        Note: t1 is same as the lower bound of Vb.
        """
        assert isinstance(coin["amount"], int)
        reserve_rate = self._feerate_reserve_per_vault(block_height)
        t1 = reserve_rate * P2WPKH_INPUT_SIZE + self._feerate_to_fee(1, "cancel", 0)
        t2 = self.Vm(block_height)
        minimum = min(t1, t2)
        if coin["amount"] <= minimum:
            return True
        else:
            return False

    def refill(self, amount):
        """Refill the WT by generating a new feebump coin worth 'amount', with no allocation."""
        assert isinstance(amount, int)
        self._add_coin(amount)

    def grab_coins_0(self, block_height):
        """Select coins to consume as inputs for the CF transaction,
        remove them from P and V.

        This version grabs all the existing feebump coins.

        Return: total amount of consumed inputs, number of inputs
        """
        num_inputs = 0
        total = 0

        # FIXME: delete the list, don't delete each elem
        # loop over copy of the list since the remove() method changes list indexing
        for coin in list(self.fbcoins):
            total += coin["amount"]
            self._remove_coin(coin)
            num_inputs += 1

        return total, num_inputs

    def grab_coins_1(self, block_height):
        """Select coins to consume as inputs for the CF transaction,
        remove them from P and V.

        This version grabs all the coins that either haven't been processed yet
        or are negligible.

        Return: total amount of consumed inputs, number of inputs
        """
        num_inputs = 0
        total_unprocessed = 0

        # FIXME: don't loop 2 times! Also may be more efficient to re-create the list
        # instead of removing that much.
        # loop over copy of the list since the remove() method changes list indexing
        for coin in list(self.fbcoins):
            if coin["processed"] is None:
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

        total_to_consume = total_unprocessed + total_negligible
        return total_to_consume, num_inputs

    def grab_coins_2(self, block_height):
        """Select coins to consume as inputs for the CF transaction,
        remove them from P and V.

        This version grabs all coins that are unprocessed, all
        unallocated coins that are not in the target coin distribution
        with a tolerance of X% (where X% == self.I_2_tol*100), and
        if the fee-rate is low, all negligible coins

        Return: total amount of consumed inputs, number of inputs
        """
        fb_coins = self.fb_coins_dist(block_height)
        num_inputs = 0

        # Take all fbcoins that haven't been processed, get their total amount,
        # and remove them from self.fbcoins
        total_unprocessed = 0
        # loop over copy of the list since the remove() method changes list indexing
        for coin in list(self.fbcoins):
            if coin["processed"] is None:
                total_unprocessed += coin["amount"]
                self._remove_coin(coin)
                num_inputs += 1

        # Take all unallocated fbcoins that aren't in O(t) within the tolerance, and
        # remove them from self.fbcoins
        total_diverged = 0
        tol = self.I_2_tol
        dist = set(self.fb_coins_dist(block_height))  # set removes repeated values
        for coin in [fbcoin for fbcoin in self.fbcoins if fbcoin["allocation"] is None]:
            coin_ok = False
            for x in dist:
                if (1 - tol) * x <= coin["amount"] <= (1 + tol) * x:
                    coin_ok = True
                    break
            if coin_ok:
                continue
            else:
                total_diverged += coin["amount"]
                self._remove_coin(coin)
                num_inputs += 1

        low_feerate = self._feerate(block_height) <= 5
        if low_feerate:
            vault_diminished = dict()
            # Take all fbcoins that are negligible, get their total amount, and remove
            # them from self.fbcoins and their associated vault's fee_reserve
            total_negligible = 0
            for coin in list(self.fbcoins):
                if self.is_negligible(coin, block_height):
                    total_negligible += coin["amount"]
                    self._remove_coin(coin)
                    num_inputs += 1
                    if coin["allocation"] is not None:
                        try:
                            vault_diminished[coin["allocation"]] += coin["amount"]
                        except (KeyError):
                            vault_diminished[coin["allocation"]] = coin["amount"]
            if vault_diminished != {}:
                logging.debug(
                    f"    I2 consumed from the following vaults: \n{vault_diminished}"
                )

        total_to_consume = total_unprocessed + total_diverged

        return total_to_consume, num_inputs

    def grab_coins_3(self, height):
        """Select coins to consume as inputs of the CF transaction.

        This version grabs all coins that were just refilled. In addition it grabs
        some fb coins for consolidation:
            - Allocated ones that it's not safe to keep (those that would not bump the
              Cancel tx feerate at the reserve (max) feerate).
            - Unallocated ones ones we would not create, if the current feerate is low.

        Returns the total amount to consolidate and the number of coins to consolidate.
        """
        to_keep = []
        n_to_consolidate = 0
        total_to_consolidate = 0

        reserve_feerate = self._feerate_reserve_per_vault(height)
        dust_threshold = reserve_feerate * P2WPKH_INPUT_SIZE + self._feerate_to_fee(
            1, "cancel", 0
        )
        # FIXME: this should use the next 3 blocks feerate
        low_feerate = self._feerate(height) <= 5
        min_fbcoin_value = self.min_fbcoin_value(height)
        for coin in self.fbcoins:
            if (
                (coin["processed"] is None)
                or (coin["allocation"] is not None and coin["amount"] < dust_threshold)
                or (
                    coin["allocation"] is None
                    and low_feerate
                    and coin["amount"] < min_fbcoin_value
                )
            ):
                n_to_consolidate += 1
                total_to_consolidate += coin["amount"]
                # FIXME: index!!
                for v in self.vaults:
                    if coin in v["fee_reserve"]:
                        v["fee_reserve"].remove(coin)
                        break
            else:
                to_keep.append(coin)

        self.fbcoins = to_keep
        return total_to_consolidate, n_to_consolidate

    def min_fbcoin_value(self, height):
        """The minimum value for a feebumping coin we create is one that allows
        to pay for its inclusion at the maximum feerate AND increase the Cancel
        tx fee by at least 5sat/vbyte.
        """
        feerate = self._feerate_reserve_per_vault(height)
        return int(feerate * P2WPKH_INPUT_SIZE + self._feerate_to_fee(5, "cancel", 0))

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

        CF transactions help maintain coin sizes that enable accurate fee-bumping.
        """
        # save initial state to recover if failure
        fbcoins_copy = list(self.fbcoins)
        vaults_copy = list(self.vaults)
        # Set target values for our coin creation amounts
        target_coin_dist = self.fb_coins_dist(block_height)

        # Select and consume inputs with I(t), returning the total amount and the
        # number of inputs.
        if self.I_version == 0:
            total_to_consume, num_inputs = self.grab_coins_0(block_height)
        elif self.I_version == 1:
            total_to_consume, num_inputs = self.grab_coins_1(block_height)
        elif self.I_version == 2:
            total_to_consume, num_inputs = self.grab_coins_2(block_height)
        elif self.I_version == 3:
            total_to_consume, num_inputs = self.grab_coins_3(block_height)
        else:
            raise CfError("Unknown algorithm version for coin consolidation")

        try:
            feerate = self._estimate_smart_feerate(block_height)
        except (ValueError, KeyError):
            feerate = self._feerate(block_height)

        # FIXME this doesn't re-create enough coins? If we consolidated some.

        # Keep track of the CF tx size and fees as we add outputs
        cf_size = TX_OVERHEAD_SIZE + P2WPKH_INPUT_SIZE * num_inputs
        cf_tx_fee = int(cf_size * feerate)
        # The cost of a distribution for a single vault in the CF tx
        dist_size = P2WPKH_OUTPUT_SIZE * len(target_coin_dist)
        dist_fees = int(dist_size * feerate)
        dist_cost = sum(target_coin_dist) + dist_fees
        # Add new distributions of coins to the CF until we can't afford it anymore
        num_new_reserves = 0
        consumed = 0
        while True:
            consumed += dist_cost
            if consumed > total_to_consume:
                break

            # Don't create a new set of outputs if we can't pay the fees for it
            if total_to_consume - consumed <= cf_tx_fee:
                break
            cf_size += dist_size
            cf_tx_fee += int(dist_size * feerate)
            if cf_size > MAX_TX_SIZE:
                raise CfError("The consolidate_fanout transaction is too large!")

            num_new_reserves += 1
            for x in target_coin_dist:
                self._add_coin(x, processed=block_height)

        if num_new_reserves == 0:
            logging.debug(
                "        CF Tx failed sice num_new_reserves = 0 (not accounting for expected fee)"
            )
            # Not enough in available coins to fanout to 1 complete fee_reserve, so return
            # to initial state and return 0 (as in, 0 fee paid)
            self.fbcoins = fbcoins_copy
            self.vaults = vaults_copy
            return 0

        remainder = total_to_consume - (num_new_reserves * sum(target_coin_dist))
        assert isinstance(remainder, int)
        assert (
            remainder >= cf_tx_fee
        ), "We must never try to create more fb coins than we can afford to"

        # If we have more than we need for the CF fee..
        remainder -= cf_tx_fee
        if remainder > 0:
            # .. First try to opportunistically add more fb coins
            added_coin_value = int(self.min_fbcoin_value(block_height) * 1.3)
            output_fee = int(P2WPKH_OUTPUT_SIZE * feerate)
            # The number of coins this large we can add
            added_coins_count = remainder / (added_coin_value + output_fee)
            if added_coins_count >= 1:
                # For a bias toward a lower number of larger outputs, truncate
                # the number of coins added and add the excess value to the outputs
                added_coins_count = int(added_coins_count)
                outputs_fee = added_coin_value + output_fee
                added_coin_value = int((remainder - outputs_fee) / added_coins_count)
                for _ in range(added_coins_count):
                    self._add_coin(added_coin_value, processed=block_height)
                    cf_tx_fee += outputs_fee
                return cf_tx_fee

            # And fallback to distribute the excess across the created fb coins
            num_outputs = num_new_reserves * len(target_coin_dist)
            increase = (remainder) // num_outputs
            for c in self.fbcoins[len(self.fbcoins) - num_outputs :]:
                c["amount"] += increase

        return cf_tx_fee

    def _allocate_0(self, vault_id, amount, block_height):
        """WT allocates coins to a (new/existing) vault if there is enough
        available coins to meet the requirement.
        """
        # Recovery state
        fbcoins_copy = list(self.fbcoins)
        vaults_copy = list(self.vaults)

        try:
            # If vault already exists, de-allocate its current fee reserve first
            vault = next(vault for vault in self.vaults if vault["id"] == vault_id)
            if self.under_requirement(vault["fee_reserve"], block_height) == 0:
                return
            else:
                logging.debug(
                    f"  Allocation transition to an existing vault {vault_id} at block {block_height}"
                )
                for coin in vault["fee_reserve"]:
                    coin["allocation"] = None
                self.vaults.remove(vault)
        except (StopIteration):
            logging.debug(
                f"  Allocation transition to new vault {vault_id} at block {block_height}"
            )

        total_unallocated = round(
            sum(
                [
                    coin["amount"]
                    for coin in self.fbcoins
                    if (coin["allocation"] is None) and (coin["processed"] != None)
                ]
            ),
            0,
        )
        required_reserve = self.fee_reserve_per_vault(block_height)

        Vm = self.Vm(block_height)
        logging.debug(f"    Fee Reserve per Vault: {required_reserve}, Vm = {Vm}")
        if required_reserve > total_unallocated:
            self.fbcoins = fbcoins_copy
            self.vaults = vaults_copy
            raise RuntimeError(
                f"Watchtower doesn't acknowledge delegation for vault {vault_id} since total un-allocated and processed fee-reserve is insufficient"
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
                                if (coin["allocation"] is None)
                                and (
                                    (1 + tol) * Vm >= coin["amount"] >= ((1 - tol) * Vm)
                                )
                            )
                            fbcoin["allocation"] = vault_id
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
                    if (coin["allocation"] is None) and (coin["processed"] != None)
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
                        fbcoin["allocation"] = vault_id
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
                    fbcoin["allocation"] = vault_id
                    fee_reserve.append(fbcoin)

            new_reserve_total = sum([coin["amount"] for coin in fee_reserve])
            assert new_reserve_total >= required_reserve
            logging.debug(
                f"    Reserve for vault {vault_id} has excess of {new_reserve_total-required_reserve}"
            )

            # Successful new delegation and allocation!
            assert isinstance(amount, int)
            self.vaults.append(
                {"id": vault_id, "amount": amount, "fee_reserve": fee_reserve}
            )

    def _allocate_1(self, vault_id, amount, block_height):
        """WT allocates coins to a (new/existing) vault if there is enough
        available coins to meet the requirement.
        """
        # Recovery state
        fbcoins_copy = list(self.fbcoins)
        vaults_copy = list(self.vaults)

        try:
            # If vault already exists and is under requirement, de-allocate its current fee
            # reserve first
            vault = next(vault for vault in self.vaults if vault["id"] == vault_id)
            if self.under_requirement(vault["fee_reserve"], block_height) == 0:
                return
            else:
                logging.debug(
                    f"  Allocation transition to an existing vault {vault_id} at block {block_height}"
                )
                for coin in vault["fee_reserve"]:
                    coin["allocation"] = None
                self.vaults.remove(vault)
        except (StopIteration):
            logging.debug(
                f"  Allocation transition to new vault {vault_id} at block {block_height}"
            )

        total_unallocated = round(
            sum(
                [
                    coin["amount"]
                    for coin in self.fbcoins
                    if (coin["allocation"] is None) and (coin["processed"] != None)
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
                f"Watchtower doesn't acknowledge delegation for vault {vault_id} since total un-allocated and processed fee-reserve is insufficient"
            )

        # WT begins allocating coins to this vault and finally updates the vault's fee_reserve
        else:
            fee_reserve = []
            dist = self.fb_coins_dist(block_height)
            tolerances = [0.05, 0.1, 0.2, 0.3]
            while sum([coin["amount"] for coin in fee_reserve]) < required_reserve:
                # Optimistically search for coins in O with small tolerance
                for tol in tolerances:
                    not_found = []
                    for x in dist:
                        try:
                            fbcoin = next(
                                coin
                                for coin in self.fbcoins
                                if (coin["allocation"] is None)
                                and ((1 - tol) * x <= coin["amount"] <= (1 + tol) * x)
                            )
                            fbcoin["allocation"] = vault_id
                            fee_reserve.append(fbcoin)
                            logging.debug(
                                f"    Coin = {fbcoin['amount']} found with tolerance {tol*100}%, added to fee reserve"
                            )
                        except (StopIteration):
                            logging.debug(
                                f"    No coin found with amount = {x} with tolerance {tol*100}%"
                            )
                            not_found.append(x)
                            continue
                    # If there was any failure, try again with a wider tolerance for remaining not found amounts
                    dist = not_found
                    # All coins found with some tolerance
                    if not_found == []:
                        break

                # Now handle the remaining requirement
                diff = required_reserve - sum([coin["amount"] for coin in fee_reserve])
                available = [
                    coin
                    for coin in self.fbcoins
                    if (coin["allocation"] is None) and (coin["processed"] != None)
                ]
                # sort in increasing order of amount
                available = sorted(available, key=lambda coin: coin["amount"])
                # Try to fill the remaining requirement with smallest available coin that covers the diff
                try:
                    fbcoin = next(coin for coin in available if coin["amount"] > diff)
                    fbcoin["allocation"] = vault_id
                    fee_reserve.append(fbcoin)
                    logging.debug(
                        f"    Coin = {fbcoin['amount']} found to cover remaining requirement and added to fee reserve"
                    )
                    continue
                # Otherwise, take the largest available coin and repeat
                except (StopIteration):
                    fbcoin = available[-1]
                    fbcoin["allocation"] = vault_id
                    fee_reserve.append(fbcoin)
                    logging.debug(
                        f"    Coin = {fbcoin['amount']} found as largest available and added to fee reserve"
                    )
                    continue

            new_reserve_total = sum([coin["amount"] for coin in fee_reserve])
            assert new_reserve_total >= required_reserve
            logging.debug(
                f"    Reserve for vault {vault_id} has excess of {new_reserve_total-required_reserve}"
            )

            # Successful new delegation and allocation!
            self.vaults.append(
                {"id": vault_id, "amount": amount, "fee_reserve": fee_reserve}
            )

    def allocate(self, vault_id, amount, block_height):

        if self.allocate_version == 0:
            self._allocate_0(vault_id, amount, block_height)

        elif self.allocate_version == 1:
            self._allocate_1(vault_id, amount, block_height)

    def process_cancel(self, vault_id, block_height):
        """The cancel must be updated with a fee (the large Vm allocated to it).
        If this fee is unsuccessful at pushing the cancel through, additional small coins may
        be added from the fee_reserve.
        """
        vault = next((vault for vault in self.vaults if vault["id"] == vault_id), None)
        if vault is None:
            raise RuntimeError(f"No vault found with id {vault_id}")

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
                assert isinstance(fbcoin["amount"], int)
                self._remove_coin(fbcoin)
                init_fee -= fbcoin["amount"]
                cancel_fb_inputs.append(fbcoin)
            except (StopIteration):
                fbcoin = reserve[-1]
                self._remove_coin(fbcoin)
                init_fee -= fbcoin["amount"]
                cancel_fb_inputs.append(fbcoin)

        return cancel_fb_inputs

    def finalize_cancel(self, vault_id):
        """Once the cancel is confirmed, any remaining fbcoins allocated to vault_id
        become unallocated. The vault with vault_id is removed from vaults.
        """
        for coin in self.fbcoins:
            if coin["allocation"] == vault_id:
                coin["allocation"] = None
        for vault in list(self.vaults):
            if vault["id"] == vault_id:
                self.vaults.remove(vault)

    def process_spend(self, vault_id):
        """Once a vault is consumed with a spend, the fee-reserve that was allocated to it
        becomes un-allocated and the vault is removed from the set of vaults.
        """
        self.vaults = [vault for vault in self.vaults if vault["id"] != vault_id]

        for coin in self.fbcoins:
            if coin["allocation"] == vault_id:
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
        available = [coin for coin in self.fbcoins if coin["allocation"] is None]
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
