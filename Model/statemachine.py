""" 
TODO:
* simulate requirement for cancel feebump, then implement feebump algo
* Add random time interval between balance low and re-fill trigger (to simulate slow stakeholder),
  to investigate time-at-risk. 
* Make good documentation
* Remove possibility for inconsistencies in progress of blocks with WTSim   
    - could break with certain DELEGATION_PERIODs. 
"""

import itertools
import logging
import numpy as np
from copy import deepcopy
from enum import Enum
from pandas import read_csv
from transactions import CancelTx, ConsolidateFanoutTx
from utils import (
    P2WPKH_INPUT_SIZE,
    P2WPKH_OUTPUT_SIZE,
    cf_tx_size,
    MAX_TX_SIZE,
    CANCEL_TX_WEIGHT,
    TX_OVERHEAD_SIZE,
    FB_DUST_THRESH,
)


class CfError(RuntimeError):
    """An error arising during the creation of the Consolidate-Fanout tx"""

    def __init__(self, message):
        self.message = message


class AllocationError(RuntimeError):
    """We don't have enough unallocated coins to pay for an entire vault reserve"""

    def __init__(self, required_reserve, unallocated_balance):
        self.message = (
            f"Required reserve: {required_reserve}, unallocated "
            f"balance: {unallocated_balance}"
        )


class ProcessingState(Enum):
    """The state of feebump coin"""

    UNPROCESSED = 0
    PENDING = 1
    CONFIRMED = 2


class VaultState(Enum):
    """Whether a vault is being"""

    READY = 0
    SPENDING = 1
    CANCELING = 2


class Vault:
    """A vault the WT is watching for."""

    def __init__(self, _id, amount, status=VaultState.READY):
        assert isinstance(amount, int) and isinstance(_id, int)
        self.id = _id
        self.amount = amount
        # The feebump coins that were allocated to this vault
        self.fb_coins = {}
        # status used to track whether vault should be considered during other state transitions
        self.status = status

    def __repr__(self):
        return f"Vault(id={self.id}, amount={self.amount}, coins={self.fb_coins})"

    def allocated_coins(self):
        return self.fb_coins.values()

    def allocate_coin(self, coin):
        assert coin.id not in self.fb_coins
        assert coin.is_confirmed()
        self.fb_coins[coin.id] = coin

    def deallocate_coin(self, coin):
        del self.fb_coins[coin.id]

    def deallocate_all_coins(self):
        self.fb_coins = {}

    def reserve_balance(self):
        return sum(c.amount for c in self.fb_coins.values())

    def set_status(self, status):
        assert isinstance(status, VaultState)
        self.status = status

    def is_available(self):
        return self.status == VaultState.READY


class FeebumpCoin:
    """A coin in the WT wallet that will eventually be used to feebump."""

    def __init__(
        self, _id, amount, processing_state=ProcessingState.UNPROCESSED, fan_block=None
    ):
        assert fan_block is None or processing_state == ProcessingState.CONFIRMED
        assert isinstance(amount, int) and isinstance(_id, int)
        self.id = _id
        self.amount = amount
        self.processing_state = processing_state
        # If confirmed, the block at which it was created by the CF tx,
        # tracked because of grab_coins_1
        self.fan_block = fan_block

    def __repr__(self):
        return f"Coin(id={self.id}, amount={self.amount}, fan_block={self.fan_block})"

    def is_confirmed(self):
        """Whether this coin was fanned out and confirmed"""
        if self.processing_state == ProcessingState.CONFIRMED:
            assert self.fan_block is not None
            return True
        return False

    def is_unconfirmed(self):
        """Whether this coin was fanned out but not yet confirmed"""
        if self.processing_state == ProcessingState.PENDING:
            assert self.fan_block is None
            return True
        return False

    def is_unprocessed(self):
        """Whether this coin is a new refill coin"""
        return self.processing_state == ProcessingState.UNPROCESSED

    def increase_amount(self, value_increase):
        self.amount += value_increase

    def confirm(self, height):
        self.processing_state = ProcessingState.CONFIRMED
        self.fan_block = height


class CoinPool:
    """A set of feebump coins that the WT operates."""

    def __init__(self):
        # A map from the coin id to the coin
        self.coins = {}
        # A map from the coin id to the id of the vault it's allocated to
        self.allocation_map = {}
        # A counter to generate unique ids for coins
        self.coin_id = 0

    def new_coin_id(self):
        self.coin_id += 1
        return self.coin_id

    def n_coins(self):
        return len(self.coins)

    def list_coins(self):
        return self.coins.values()

    def balance(self):
        return sum(c.amount for c in self.coins.values())

    def is_allocated(self, coin):
        return coin.id in self.allocation_map

    def coin_allocation(self, coin):
        return self.allocation_map[coin.id]

    def unallocated_coins(self):
        """Return coins that were fanned out but not yet allocated."""
        return [
            c
            for c in self.coins.values()
            if c.is_confirmed() and c.id not in self.allocation_map
        ]

    def allocate_coin(self, coin, vault):
        assert isinstance(coin, FeebumpCoin) and isinstance(vault, Vault)
        assert coin.id not in self.allocation_map
        assert coin.is_confirmed()
        self.allocation_map[coin.id] = vault.id

    def deallocate_coin(self, coin):
        del self.allocation_map[coin.id]

    def add_coin(
        self,
        amount,
        processing_state=ProcessingState.UNPROCESSED,
        fan_block=None,
        allocated_vault_id=None,
    ):
        coin_id = self.new_coin_id()
        self.coins[coin_id] = FeebumpCoin(coin_id, amount, processing_state, fan_block)
        if allocated_vault_id is not None:
            assert isinstance(allocated_vault_id, int)
            self.allocation_map[coin_id] = allocated_vault_id
        return self.coins[coin_id]

    def confirm_coin(self, coin, fan_height):
        self.coins[coin.id].confirm(fan_height)

    def remove_coin(self, coin):
        """Remove a coin from the pool by value"""
        if self.is_allocated(coin):
            self.deallocate_coin(coin)
        del self.coins[coin.id]


class StateMachine:
    """Watchtower state machine."""

    def __init__(
        self,
        n_stk,
        n_man,
        locktime,
        hist_feerate_csv,
        reserve_strat,
        estimate_strat,
        o_version,
        i_version,
        allocate_version,
        cancel_coin_selec,
    ):
        self.n_stk = n_stk
        self.n_man = n_man
        self.locktime = locktime
        self.vaults = {}
        self.coin_pool = CoinPool()
        # List of relevant unconfirmed transactions: [Tx, Tx, Tx,...]
        self.mempool = []

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
        self.cancel_coin_selection = cancel_coin_selec

        self.vb_coins_count = 8
        self.vm_factor = 1.2  # multiplier M
        self.I_2_tol = 0.3

        # avoid unnecessary search by caching fee reserve per vault, Vm, feerate
        self.frpv = (None, None)  # block, value
        self.Vm_cache = (None, None)  # block, value
        self.feerate = (None, None)  # block, value

    def list_vaults(self):
        return list(self.vaults.values())

    def list_available_vaults(self):
        return [v for v in self.list_vaults() if v.is_available()]

    def list_coins(self):
        return list(self.coin_pool.list_coins())

    def unconfirmed_transactions(self):
        return self.mempool

    def remove_coin(self, coin):
        if self.coin_pool.is_allocated(coin):
            vault_id = self.coin_pool.coin_allocation(coin)
            self.vaults[vault_id].deallocate_coin(coin)
        self.coin_pool.remove_coin(coin)

    def grab_coins(self, f):
        """Grab coins from the pool according to a filter."""
        coins = []

        for coin in list(self.coin_pool.list_coins()):
            if f(coin):
                coins.append(coin)

        return coins

    def remove_coins(self, coins):
        """Remove all these coins from the pool."""
        for c in coins:
            self.remove_coin(c)

    def remove_vault(self, vault):
        for coin in vault.allocated_coins():
            self.coin_pool.deallocate_coin(coin)
        del self.vaults[vault.id]

    def _feerate_reserve_per_vault(self, block_height):
        """Return feerate reserve per vault (satoshi/vbyte). The value is determined from a
        statistical analysis of historical feerates, using one of the implemented strategies
        chosen with the self.reserve_strat parameter.

        Note how we assume the presigned feerate to be 0. It's 88 "For Real"
        (in practical-revault).
        """
        if self.frpv[0] == block_height:
            return self.frpv[1]

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

        self.frpv = (block_height, self.hist_df[self.reserve_strat][block_height])
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

            elif self.estimate_strat == "95Q1":
                self.hist_df["95Q1"] = (
                    self.hist_df["mean_feerate"]
                    .rolling(144, min_periods=72)
                    .quantile(quantile=0.95, interpolation="linear")
                )

            else:
                raise ValueError("Strategy not implemented")

        self.feerate = (block_height, self.hist_df[self.estimate_strat][block_height])
        return self.feerate[1]

    def next_block_feerate(self, height):
        """Value of `estimatesmartfee 1 CONSERVATIVE`.

        When estimates aren't available, falls back to the maximum of the last
        3 block median.
        """
        try:
            return int(self.hist_df["est_1block"][height])
        except ValueError:
            try:
                # FIXME: we can do better than that.
                return int(
                    max(
                        self.hist_df["mean_feerate"][h]
                        for h in range(height - 2, height + 1)
                    )
                )
            except ValueError:
                # Mean feerate might be NA if there was no tx in this block
                return 0

    def cancel_vbytes(self):
        """Size of the Cancel transaction without any feebump input"""
        return (CANCEL_TX_WEIGHT[self.n_stk][self.n_man] + 3) // 4

    def cancel_tx_fee(self, feerate, n_fb_inputs):
        """Get the Cancel tx fee at this feerate for this number of fb txins."""
        cancel_tx_size = self.cancel_vbytes() + n_fb_inputs * P2WPKH_INPUT_SIZE
        return int(cancel_tx_size * feerate)

    def fee_reserve_per_vault(self, block_height):
        return self.cancel_tx_fee(self._feerate_reserve_per_vault(block_height), 0)

    def is_tx_confirmed(self, tx, height):
        """We consider a transaction to have been confirmed in this block if its
        feerate was above the min feerate in this block."""
        min_feerate = self.hist_df["min_feerate"][height]
        min_feerate = 0 if min_feerate == "NaN" else float(min_feerate)
        return tx.feerate() > min_feerate

    def Vm(self, block_height):
        """Amount for the main feebump coin"""
        if self.Vm_cache[0] == block_height:
            return self.Vm_cache[1]

        feerate = self._feerate(block_height)
        vm = int(self.cancel_tx_fee(feerate, 0) + feerate * P2WPKH_INPUT_SIZE)
        assert vm > 0
        self.Vm_cache = (block_height, vm)
        return vm

    def Vb(self, block_height):
        """Amount for a backup feebump coin"""
        reserve = self.fee_reserve_per_vault(block_height)
        reserve_rate = self._feerate_reserve_per_vault(block_height)
        vb = reserve / self.vb_coins_count
        min_vb = self.cancel_tx_fee(5, 0)
        return int(reserve_rate * P2WPKH_INPUT_SIZE + max(vb, min_vb))

    def coins_dist_reserve(self, block_height):
        """The coin amount distribution used to cover up to the worst case.

        These coins are needed to be able to Cancel up the reserve feerate, but
        not usually optimal during normal operations.
        """
        if self.O_version == 0:
            vb = self.Vb(block_height)
            return [vb] * self.vb_coins_count

        # Strategy 1
        # dist = [Vm, MVm, 2MVm, 3MVm, ...]
        if self.O_version == 1:
            reserve_feerate = self._feerate_reserve_per_vault(block_height)
            fbcoin_cost = int(reserve_feerate * P2WPKH_INPUT_SIZE)
            frpv = self.fee_reserve_per_vault(block_height)
            Vm = self.Vm(block_height)
            M = self.vm_factor  # Factor increase per coin
            dist = [Vm]
            while sum(dist) < frpv - len(dist) * fbcoin_cost:
                dist.append(int((len(dist)) * M * Vm + fbcoin_cost))
            diff = sum(dist) - frpv - int(len(dist) * fbcoin_cost)
            # find the minimal subset sum of elements that is greater than diff, and remove them
            subset = []
            while sum(subset) < diff:
                subset.append(dist.pop())
            excess = sum(subset) - diff
            assert isinstance(excess, int)
            if excess >= Vm + fbcoin_cost:
                dist.append(excess + fbcoin_cost)
            else:
                dist[-1] += excess
            return dist

    def coins_dist_bonus(self, block_height):
        """The coin amount distribution used to reduce overpayments.

        These coins can't be used for feebumping in worst case scenario (at
        reserve feerate) but still useful to reduce overpayments.
        """
        vm = self.Vm(block_height)
        vb = self.Vb(block_height)
        dist = [vm]
        while vm * self.vm_factor < vb:
            vm *= self.vm_factor
            dist.append(int(vm))
        return dist

    def fb_coins_dist(self, block_height):
        """The coin amount distribution to target for a vault reserve.

        This is the concatenation of the reserve distribution and the bonus
        distribution.
        """
        return self.coins_dist_reserve(block_height) + self.coins_dist_bonus(
            block_height
        )

    def unallocated_balance(self):
        return sum(
            [
                coin.amount
                for coin in self.coin_pool.list_coins()
                if not self.coin_pool.is_allocated(coin)
            ]
        )

    def balance(self):
        return self.coin_pool.balance()

    def under_requirement(self, vault, block_height):
        """Returns whether a given vault wouldn't be able to bump at reserve feerate."""
        required_reserve = sum(self.coins_dist_reserve(block_height))
        min_coin_value = self.min_fbcoin_value(block_height)
        usable_balance = sum(
            [c.amount for c in vault.fb_coins.values() if c.amount >= min_coin_value]
        )
        return usable_balance < required_reserve

    def is_negligible(self, coin, block_height):
        """A coin is considered negligible if its amount is less than the minimum
        of Vm and the fee required to bump a Cancel transaction by 1 sat per vByte
        in the worst case (when the fee rate is equal to the reserve rate).
        Note: t1 is same as the lower bound of Vb.
        """
        reserve_rate = self._feerate_reserve_per_vault(block_height)
        t1 = reserve_rate * P2WPKH_INPUT_SIZE + self.cancel_tx_fee(1, 0)
        t2 = self.Vm(block_height)
        minimum = min(t1, t2)
        if coin.amount <= minimum:
            return True
        else:
            return False

    def refill(self, amount):
        """Refill the WT by generating a new feebump coin worth 'amount', with no allocation."""
        assert isinstance(amount, int)
        self.coin_pool.add_coin(amount)

    def grab_coins_0(self, block_height):
        """Select coins to consume as inputs for the CF transaction,
        remove them from P and V.

        This version grabs all the existing feebump coins.
        """
        return self.coin_pool.list_coins()

    def grab_coins_1(self, block_height):
        """Select coins to consume as inputs for the CF transaction,
        remove them from P and V.

        This version grabs all the coins that either haven't been processed yet
        or are negligible.
        """
        return self.grab_coins(
            lambda coin: not coin.is_unconfirmed()
            and (coin.is_unprocessed() or self.is_negligible(coin, block_height))
        )

    def grab_coins_2(self, block_height):
        """Select coins to consume as inputs for the CF transaction,
        remove them from P and V.

        This version grabs all coins that are unprocessed, all
        unallocated coins that are not in the target coin distribution
        with a tolerance of X% (where X% == self.I_2_tol*100), and
        if the fee-rate is low, all negligible coins
        """
        dist = set(self.fb_coins_dist(block_height))
        low_feerate = self._feerate(block_height) <= 5

        def coin_in_dist(coin_value, dist, tolerance):
            for x in dist:
                if (1 - tolerance) * x <= coin_value <= (1 + tolerance) * x:
                    return True
            return False

        def coin_filter(coin):
            if coin.is_unconfirmed():
                return False
            if coin.is_unprocessed():
                return True

            if not self.coin_pool.is_allocated(coin):
                if not coin_in_dist(coin.amount, dist, self.I_2_tol):
                    return True

            if low_feerate and coin.amount < FB_DUST_THRESH:
                return True

            return False

        return self.grab_coins(coin_filter)

    def grab_coins_3(self, height):
        """Select coins to consume as inputs of the CF transaction.

        This version grabs all coins that were just refilled. In addition it grabs
        some fb coins for consolidation:
            - Allocated ones that it's not safe to keep (those that would not bump the
              Cancel tx feerate at the reserve (max) feerate).
        """
        dust = min(FB_DUST_THRESH, self.Vm(height))
        vm_min = self.Vm(height) * 0.9

        def coin_filter(coin):
            if coin.is_unconfirmed():
                return False
            if coin.is_unprocessed():
                return True

            if coin.amount < dust:
                return True
            if not self.coin_pool.is_allocated(coin) and coin.amount < vm_min:
                return True

            return False

        return self.grab_coins(coin_filter)

    def min_fbcoin_value(self, height):
        """The absolute minimum value for a feebumping coin.

        Pays for its inclusion at the maximum feerate AND increases the Cancel
        tx fee by at least 2sat/vbyte.
        """
        feerate = self._feerate_reserve_per_vault(height)
        return int(feerate * P2WPKH_INPUT_SIZE + self.cancel_tx_fee(2, 0))

    def min_acceptable_fbcoin_value(self, height):
        """The minimum value for a feebumping coin we create is one that allows
        to pay for its inclusion at the maximum feerate AND increase the Cancel
        tx fee by at least 5sat/vbyte.
        """
        feerate = self._feerate_reserve_per_vault(height)
        return int(feerate * P2WPKH_INPUT_SIZE + self.cancel_tx_fee(5, 0))

    # FIXME: eventually we should allocate as many outputs as we can, even if
    # it only represents part of a reserve. It would really lower the number
    # of allocation failures.
    def broadcast_consolidate_fanout(self, block_height):
        """
        FIXME: Instead of removing coins, add them as inputs to a CF Tx. Don't remove any coins
        if the vault status is "Canceling" (or "Spending"??). Instead of adding coins to the coin_pool,
        add them as outputs to the CF Tx. Add the CF Tx to the mempool.


        Simulate the WT creating a consolidate-fanout (CF) tx which aims to 1) create coins from
        new re-fills that enable accurate feebumping and 2) to consolidate negligible feebump coins
        if the current feerate is "low".

        Note that negligible coins that are consolidated will be removed from their
        associated vault's fee_reserve. So this process will diminish the vault's fee
        reserve until the new coins are confirmed and re-allocated.

        CF transactions help maintain coin sizes that enable accurate fee-bumping.
        """
        # Set target values for our coin creation amounts
        dist_reserve = self.coins_dist_reserve(block_height)
        dist_bonus = self.coins_dist_bonus(block_height)

        # Select and consume inputs with I(t), returning the coins
        if self.I_version == 0:
            coins = self.grab_coins_0(block_height)
        elif self.I_version == 1:
            coins = self.grab_coins_1(block_height)
        elif self.I_version == 2:
            coins = self.grab_coins_2(block_height)
        elif self.I_version == 3:
            coins = self.grab_coins_3(block_height)
        else:
            raise CfError("Unknown algorithm version for coin consolidation")

        feerate = self.next_block_feerate(block_height)

        # FIXME this doesn't re-create enough coins? If we consolidated some.
        added_coins = []
        # Keep track of the CF tx size and fees as we add outputs
        cf_size = cf_tx_size(n_inputs=len(coins), n_outputs=0)
        cf_tx_fee = int(cf_size * feerate)
        # The cost of a distribution for a single vault in the CF tx
        dist_rese_size = P2WPKH_OUTPUT_SIZE * len(dist_reserve)
        dist_rese_fees = int(dist_rese_size * feerate)
        dist_rese_cost = sum(dist_reserve) + dist_rese_fees
        dist_bonu_size = P2WPKH_OUTPUT_SIZE * len(dist_bonus)
        dist_bonu_fees = int(dist_bonu_size * feerate)
        dist_bonu_cost = sum(dist_bonus) + dist_bonu_fees
        # Add new distributions of coins to the CF until we can't afford it anymore
        total_to_consume = sum(c.amount for c in coins)
        num_new_reserves = 0
        num_new_bonuses = 0
        consumed = 0
        while True:
            # First do the reserve
            consumed += dist_rese_cost
            if consumed > total_to_consume:
                break
            # Don't create a new set of outputs if we can't pay the fees for it
            if total_to_consume - consumed <= cf_tx_fee:
                break
            cf_size += dist_rese_size
            cf_tx_fee += int(dist_rese_size * feerate)
            if cf_size > MAX_TX_SIZE:
                raise CfError("The consolidate_fanout transaction is too large!")

            num_new_reserves += 1
            for x in dist_reserve:
                added_coins.append(
                    self.coin_pool.add_coin(x, processing_state=ProcessingState.PENDING)
                )

            # Then if we still have enough do the bonus
            consumed += dist_bonu_cost
            if consumed > total_to_consume:
                continue
            # Don't create a new set of outputs if we can't pay the fees for it
            if total_to_consume - consumed <= cf_tx_fee:
                break
            cf_size += dist_bonu_size
            cf_tx_fee += int(dist_bonu_size * feerate)
            if cf_size > MAX_TX_SIZE:
                raise CfError("The consolidate_fanout transaction is too large!")

            num_new_bonuses += 1
            for x in dist_bonus:
                added_coins.append(
                    self.coin_pool.add_coin(x, processing_state=ProcessingState.PENDING)
                )

        if num_new_reserves == 0:
            logging.debug(
                "        CF Tx failed sice num_new_reserves = 0 (not accounting for"
                " expected fee)"
            )
            # Not enough in available coins to fanout to 1 complete fee_reserve, so
            # return 0 (as in, 0 fee paid)
            return 0

        remainder = (
            total_to_consume
            - (num_new_reserves * sum(dist_reserve))
            - (num_new_bonuses * sum(dist_bonus))
        )
        assert isinstance(remainder, int)
        assert (
            remainder >= cf_tx_fee
        ), "We must never try to create more fb coins than we can afford to"

        # If we have more than we need for the CF fee..
        remainder -= cf_tx_fee
        if remainder > 0:
            # .. First try to opportunistically add more fb coins
            added_coin_value = int(self.min_acceptable_fbcoin_value(block_height) * 1.3)
            output_fee = int(P2WPKH_OUTPUT_SIZE * feerate)
            # The number of coins this large we can add
            added_coins_count = remainder / (added_coin_value + output_fee)
            if added_coins_count >= 1:
                # For a bias toward a lower number of larger outputs, truncate
                # the number of coins added and add the excess value to the outputs
                added_coins_count = int(added_coins_count)
                outputs_fee = added_coin_value + output_fee * added_coins_count
                added_coin_value = int((remainder - outputs_fee) / added_coins_count)
                for _ in range(added_coins_count):
                    added_coins.append(
                        self.coin_pool.add_coin(
                            added_coin_value, processing_state=ProcessingState.PENDING
                        )
                    )
                cf_tx_fee += outputs_fee
            else:
                # And fallback to distribute the excess across the created fb coins
                num_outputs = num_new_reserves * len(
                    dist_reserve
                ) + num_new_bonuses * len(dist_bonus)
                increase = remainder // num_outputs
                for coin in added_coins:
                    coin.increase_amount(increase)

        self.remove_coins(coins)
        self.mempool.append(ConsolidateFanoutTx(block_height, coins, added_coins))
        return cf_tx_fee

    def finalize_consolidate_fanout(self, tx, height):
        """Confirm cosnolidate_fanout tx and update the coin pool."""
        if self.is_tx_confirmed(tx, height):
            for coin in tx.txouts:
                self.coin_pool.confirm_coin(coin, height)
            self.mempool.remove(tx)
            return True
        return False

    # FIXME: cleanup this function..
    def _allocate_0(self, vault_id, amount, block_height):
        """WT allocates coins to a (new/existing) vault if there is enough
        available coins to meet the requirement.
        """
        dist_req = self.coins_dist_reserve(block_height)
        dist_bonus = self.coins_dist_bonus(block_height)
        min_coin_value = self.min_fbcoin_value(block_height)
        # FIXME: don't deepcopy
        # Recovery state
        coin_pool_copy = deepcopy(self.coin_pool)
        vaults_copy = deepcopy(self.vaults)

        try:
            # If vault already exists and is under requirement, de-allocate its current fee
            # reserve first
            vault = next(v for v in self.list_vaults() if v.id == vault_id)
            if not self.under_requirement(vault, block_height):
                return
            else:
                logging.debug(
                    f"  Allocation transition to an existing vault {vault_id} at block"
                    f" {block_height}"
                )
                self.remove_vault(vault)
        except (StopIteration):
            logging.debug(
                f"  Allocation transition to new vault {vault_id} at block"
                f" {block_height}"
            )

        # We only require to allocate up to the required reserve, the rest is a bonus
        # to avoid overpayments.
        usable = [
            c.amount
            for c in self.coin_pool.unallocated_coins()
            if c.amount >= min_coin_value
        ]
        total_usable = sum(usable)
        required_reserve = sum(dist_req)

        logging.debug(
            f"    Fee Reserve per Vault: {required_reserve}, "
            f"Usable unallocated coins amounts: {usable} "
            f"Unallocated coins: {self.coin_pool.unallocated_coins()}"
        )
        if required_reserve > total_usable:
            self.coin_pool = coin_pool_copy
            self.vaults = vaults_copy
            raise AllocationError(required_reserve, total_usable)

        vault = Vault(vault_id, amount)
        tolerances = [0.05, 0.1, 0.2, 0.3]
        # First optimistically search for coins in the required reserve with
        # small tolerance.
        for tol in tolerances:
            not_found = []
            for x in dist_req:
                try:
                    fbcoin = next(
                        coin
                        for coin in self.coin_pool.unallocated_coins()
                        if ((1 - tol) * x <= coin.amount <= (1 + tol) * x)
                    )
                    self.coin_pool.allocate_coin(fbcoin, vault)
                    vault.allocate_coin(fbcoin)
                    logging.debug(
                        f"    {fbcoin} found with tolerance {tol*100}%, added to"
                        " fee reserve"
                    )
                except (StopIteration):
                    logging.debug(
                        f"    No coin found with amount = {x} with tolerance"
                        f" {tol*100}%"
                    )
                    not_found.append(x)
                    continue
            # If there was any failure, try again with a wider tolerance for
            # remaining not found amounts
            dist_req = not_found
            # All coins found with some tolerance
            if not_found == []:
                break

        # If we couldn't find large enough coins close to the dist, complete
        # with coins off the dist  but make sure they increase the fee at the
        # worst case feerate.
        for coin in self.coin_pool.unallocated_coins():
            if coin.amount >= min_coin_value:
                self.coin_pool.allocate_coin(coin, vault)
                vault.allocate_coin(coin)
                logging.debug(f"    {coin} found to complete")
                if vault.reserve_balance() >= required_reserve:
                    break

        assert (
            vault.reserve_balance() >= required_reserve
        ), f"Was checked before searching, {vault.reserve_balance()} vs {required_reserve}"

        # Now we have enough coins for the required reserve we can look for
        # coins in the bonus reserve
        for x in dist_bonus:
            for coin in self.coin_pool.unallocated_coins():
                if x * 0.85 <= coin.amount <= x * 1.15:
                    self.coin_pool.allocate_coin(coin, vault)
                    vault.allocate_coin(coin)
                    break

        logging.debug(
            f"    Reserve for vault {vault.id} has excess of"
            f" {vault.reserve_balance() - required_reserve}"
        )
        # Successful new delegation and allocation!
        self.vaults[vault.id] = vault

    def allocate(self, vault_id, amount, block_height):
        if self.allocate_version == 0:
            self._allocate_0(vault_id, amount, block_height)

    def broadcast_cancel(self, vault_id, block_height):
        """Construct and broadcast the cancel tx.

        FIXME: Instead of removing selected coins, add them as inputs
        to a cancel tx. Add the cancel tx to the mempool. Set the vault's status
        to "canceling".

        The cancel must be updated with a fee (the large Vm allocated to it).
        If this fee is unsuccessful at pushing the cancel through, additional small coins may
        be added from the fee_reserve.
        """
        vault = next(
            (vault for vault in self.list_vaults() if vault.id == vault_id), None
        )
        if vault is None:
            raise RuntimeError(f"No vault found with id {vault_id}")
        # FIXME: i think this doesn't hold
        assert vault.is_available(), "FIXME"

        feerate = self.next_block_feerate(block_height)
        needed_fee = self.cancel_tx_fee(feerate, 0)

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

        cancel_fb_inputs = []
        if self.cancel_coin_selection == 0:
            cancel_fb_inputs = self.cancel_coin_selec_0(vault, needed_fee, feerate)
        elif self.cancel_coin_selection == 1:
            cancel_fb_inputs = self.cancel_coin_selec_1(vault, needed_fee, feerate)

        vault.set_status(VaultState.CANCELING)
        self.mempool.append(
            CancelTx(block_height, vault.id, self.cancel_vbytes(), cancel_fb_inputs)
        )

        return cancel_fb_inputs

    def cancel_coin_selec_0(self, vault, needed_fee, feerate):
        """Select smallest coin to cover init_fee, if no coin, remove largest and try again."""
        coins = []
        collected_fee = 0

        while collected_fee < needed_fee:
            if vault.allocated_coins() == []:
                raise RuntimeError(
                    f"Fee reserve for vault {vault.id} was insufficient to process"
                    " cancel tx"
                )

            # sort in increasing order of amount
            reserve = sorted(vault.allocated_coins(), key=lambda coin: coin.amount)
            try:
                fbcoin = next(
                    coin
                    for coin in reserve
                    if coin.amount - feerate * P2WPKH_INPUT_SIZE
                    >= needed_fee - collected_fee
                )
                self.remove_coin(fbcoin)
                coins.append(fbcoin)
                break
            except (StopIteration):
                # If we exhausted the reserve, stop there with the entire reserve
                # allocated to the Cancel.
                # FIXME: we usually have tons of unallocated coins, can we take some
                # from there out of emergency?
                if len(reserve) == 0:
                    logging.error(
                        "Not enough fbcoins to pay for Cancel fee."
                        f"Needed: {needed_fee}, got {collected_fee}"
                    )
                    break
                # Otherwise, take the largest coin and continue.
                # FIXME: this could select a coin that *decreases* the fee!
                fbcoin = reserve[-1]
                self.remove_coin(fbcoin)
                collected_fee += fbcoin.amount - feerate * P2WPKH_INPUT_SIZE
                coins.append(fbcoin)

        return coins

    def cancel_coin_selec_1(self, vault, needed_fee, feerate):
        """Select the combination that results in the smallest overpayment"""
        coins = []

        best_combination = None
        min_fee_added = None
        max_paying_combination = None
        max_fee_added = 0
        allocated_coins = vault.allocated_coins()
        print(allocated_coins)
        for candidate in itertools.chain.from_iterable(
            itertools.combinations(allocated_coins, r)
            for r in range(1, len(allocated_coins) + 1)
        ):
            added_fees = sum(
                [c.amount - P2WPKH_INPUT_SIZE * feerate for c in candidate]
            )
            # In any case record the combination paying the most fees, as a
            # best effort if we can't afford the whole fee needed.
            if added_fees > max_fee_added:
                max_paying_combination = candidate
                max_fee_added = added_fees
            # Record the combination overpaying the least
            if added_fees < needed_fee:
                continue
            if min_fee_added is not None and added_fees >= min_fee_added:
                continue
            best_combination = candidate
            min_fee_added = added_fees

        combination = best_combination
        if combination is None:
            # FIXME: we usually have tons of unallocated coins, can we take some
            # from there out of emergency?
            combination = max_paying_combination
        assert combination is not None
        for coin in combination:
            self.remove_coin(coin)
            coins.append(coin)

        return coins

    def finalize_cancel(self, tx, height):
        """Once the cancel is confirmed, any remaining fbcoins allocated to vault_id
        become unallocated. The vault with vault_id is removed from vaults.
        """
        if self.is_tx_confirmed(tx, height):
            self.remove_vault(self.vaults[tx.vault_id])
            self.mempool.remove(tx)
        else:
            self.maybe_replace_cancel(height, tx)

    def maybe_replace_cancel(self, height, tx):
        """Broadcasts a replacement cancel transaction if feerate increased.

        Updates the coin_pool and the associated vault.
        """
        vault = self.vaults[tx.vault_id]
        new_feerate = self.next_block_feerate(height)

        logging.debug(
            f"Checking if we need feebump Cancel tx for vault {vault.id}."
            f" Tx feerate: {tx.feerate()}, next block feerate: {new_feerate}"
        )
        if new_feerate > tx.feerate():
            new_fee = self.cancel_tx_fee(new_feerate, 0)
            # Bitcoin Core policy is set to 1, take some leeway by setting
            # it to 2.
            min_fee = tx.fee + self.cancel_tx_fee(2, len(tx.fbcoins))
            needed_fee = max(new_fee, min_fee)

            # Unreserve the previous feebump coins and remove the previous tx
            for coin in tx.fbcoins:
                self.coin_pool.add_coin(
                    coin.amount, coin.processing_state, coin.fan_block, vault.id
                )
                self.coin_pool.allocate_coin(coin, vault)
            self.mempool.remove(tx)

            # Push a new tx with coins selected to meet the new fee
            if self.cancel_coin_selection == 0:
                coins = self.cancel_coin_selec_0(vault, needed_fee, new_feerate)
            elif self.cancel_coin_selection == 1:
                coins = self.cancel_coin_selec_1(vault, needed_fee, new_feerate)
            self.mempool.append(CancelTx(height, vault.id, self.cancel_vbytes(), coins))

    def spend(self, vault_id, height):
        """Handle a broadcasted Spend transaction.

        We don't track the confirmation status of the Spend transaction and assume
        instant confirmation.
        The model always assume a spend sequence results in a succesful Spend
        confirmation. Technically we should wait for CSV blocks to remove the
        vault from the mapping but the approximation is good enough for the data
        we are interested in (it does not affect the availability of feebump coins).
        """
        self.remove_vault(self.vaults[vault_id])

    # TODO: re-think or get rid of
    # def risk_status(self, block_height):
    # """Return a summary of the risk status for the set of vaults being watched."""
    # # For cancel
    # under_requirement = []
    # for _, vault in self.vaults:
    # y = self.under_requirement(vault, block_height)
    # if y != 0:
    # under_requirement.append(y)
    # # For delegation
    # available = self.coin_pool.unallocated_coins()
    # delegation_requires = sum(self.fb_coins_dist(block_height)) - sum(
    # [coin.amount for coin in available]
    # )
    # if delegation_requires < 0:
    # delegation_requires = 0
    # return {
    # "block": block_height,
    # "num_vaults": len(self.vaults),
    # "vaults_at_risk": len(under_requirement),
    # "severity": sum(under_requirement),
    # "delegation_requires": delegation_requires,
    # }


# FIXME: eventually have some small pytests
if __name__ == "__main__":
    sm = StateMachine(
        n_stk=5,
        n_man=5,
        hist_feerate_csv="historical_fees.csv",
        reserve_strat="CUMMAX95Q90",
        estimate_strat="ME30",
        o_version=1,
        i_version=2,
        allocate_version=1,
    )

    sm.refill(500000)
    block = 400000
    sm.consolidate_fanout(block)
    coins = list(sm.coin_pool.list_coins())
    block += 6
    sm.allocate(vault_id=1, amount=200000, block_height=block)
    sm.process_spend(vault_id=1)
    block += 6
    sm.allocate(vault_id=2, amount=200000, block_height=block)
    sm.process_cancel(vault_id=2, block_height=block)
    sm.finalize_cancel(vault_id=2)
