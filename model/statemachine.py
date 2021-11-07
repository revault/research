""" 
TODO:
* Add random time interval between balance low and re-fill trigger (to simulate slow stakeholder),
  to investigate time-at-risk. 
* Make good documentation 
"""

import bisect
import logging
import math

from enum import Enum
from pandas import read_csv
from transactions import CancelTx, ConsolidateFanoutTx
from utils import (
    P2WPKH_INPUT_SIZE,
    P2WPKH_OUTPUT_SIZE,
    cf_tx_size,
    MAX_TX_SIZE,
    CANCEL_TX_WEIGHT,
    FB_DUST_THRESH,
    MIN_BUMP_WORST_CASE,
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
    """Whether a vault is being spent or canceled"""

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
        # tracked because of cf_coin_selec_1
        self.fan_block = fan_block

    def __repr__(self):
        return (
            f"Coin(id={self.id}, amount={self.amount}, fan_block={self.fan_block},"
            f" state={self.processing_state})"
        )

    def __lt__(a, b):
        return a.amount < b.amount

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
        coin_id=None,
    ):
        coin_id = coin_id if coin_id is not None else self.new_coin_id()
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
        fallback_est_strat,
        cf_coin_selec,
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
        self.fallback_est_strat = fallback_est_strat

        self.cf_coin_selec = cf_coin_selec
        self.cancel_coin_selection = cancel_coin_selec

        # FIXME: make these configurable by env vars?
        self.vb_coins_count = 6
        self.vm_factor = 1.2  # multiplier M
        self.I_2_tol = 0.3

        # avoid unnecessary search by caching fee reserve per vault, Vm, feerate
        self.frpv = (None, None)  # block, value
        self.vm_cache = (None, None)  # block, value
        self.feerate = (None, None)  # block, value

        # Prepare the rolling stats
        thirty_days = 144 * 30
        ninety_days = 144 * 90
        logging.debug("Preparing the fee estimation data.")

        if self.fallback_est_strat == "MA30":
            self.hist_df["MA30"] = (
                self.hist_df["mean_feerate"]
                .rolling(thirty_days, min_periods=144)
                .mean()
            )
        elif self.fallback_est_strat == "ME30":
            self.hist_df["ME30"] = (
                self.hist_df["mean_feerate"]
                .rolling(thirty_days, min_periods=144)
                .median()
            )
        elif self.fallback_est_strat == "85Q1H":
            self.hist_df["85Q1H"] = (
                self.hist_df["mean_feerate"]
                .rolling(6, min_periods=1)
                .quantile(quantile=0.85, interpolation="linear")
            )
        else:
            raise ValueError("Estimate strategy not implemented")

        if self.reserve_strat == "95Q30":
            self.hist_df["95Q30"] = (
                self.hist_df["mean_feerate"]
                .rolling(thirty_days, min_periods=144)
                .quantile(quantile=0.95, interpolation="linear")
            )
        elif self.reserve_strat == "95Q90":
            self.hist_df["95Q90"] = (
                self.hist_df["mean_feerate"]
                .rolling(ninety_days, min_periods=144)
                .quantile(quantile=0.95, interpolation="linear")
            )
        elif self.reserve_strat == "CUMMAX95Q90":
            self.hist_df["CUMMAX95Q90"] = (
                self.hist_df["mean_feerate"]
                .rolling(ninety_days, min_periods=144)
                .quantile(quantile=0.95, interpolation="linear")
                .cummax()
            )
        elif self.reserve_strat == "CUMMAX95Q1":
            self.hist_df["CUMMAX95Q1"] = (
                self.hist_df["mean_feerate"]
                .rolling(144, min_periods=14)
                .quantile(quantile=0.95, interpolation="linear")
                .cummax()
            )
        else:
            raise ValueError("Reserve strategy not implemented")

        logging.debug("Done processing the fee estimation data.")

    def list_vaults(self):
        return list(self.vaults.values())

    def vaults_count(self):
        return len(self.vaults)

    def list_available_vaults(self):
        return [v for v in self.list_vaults() if v.is_available()]

    def list_coins(self):
        return list(self.coin_pool.list_coins())

    def unconfirmed_transactions(self):
        return self.mempool

    def allocate_coin(self, coin, vault):
        self.coin_pool.allocate_coin(coin, vault)
        vault.allocate_coin(coin)

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

    def feerate_reserve_per_vault(self, block_height):
        """Return feerate reserve per vault (satoshi/vbyte). The value is determined from a
        statistical analysis of historical feerates, using one of the implemented strategies
        chosen with the self.reserve_strat parameter.

        Note how we assume the presigned feerate to be 0. It's 88 "For Real"
        (in practical-revault).
        """
        if self.frpv[0] == block_height:
            return self.frpv[1]

        self.frpv = (block_height, self.hist_df[self.reserve_strat][block_height])
        return self.frpv[1]

    def fallback_feerate(self, block_height):
        """Return a block chain based feerate estimate (satoshi/vbyte).

        The value is determined from a statistical analysis of historical feerates,
        using one of the implemented strategies chosen with the self.fallback_est_strat
        parameter.
        """
        if self.feerate[0] == block_height:
            return self.feerate[1]

        self.feerate = (
            block_height,
            self.hist_df[self.fallback_est_strat][block_height],
        )
        return self.feerate[1]

    def next_block_feerate(self, height):
        """Value of `estimatesmartfee 1 CONSERVATIVE`.

        When estimates aren't available, use the user-provided fallback method.
        """
        try:
            return int(self.hist_df["est_1block"][height])
        except ValueError:
            return self.fallback_feerate(height)

    def cancel_vbytes(self):
        """Size of the Cancel transaction without any feebump input"""
        return math.ceil(CANCEL_TX_WEIGHT[self.n_stk][self.n_man] / 4)

    def cancel_tx_fee(self, feerate, n_fb_inputs):
        """Get the Cancel tx fee at this feerate for this number of fb txins."""
        cancel_tx_size = self.cancel_vbytes() + n_fb_inputs * P2WPKH_INPUT_SIZE
        return int(cancel_tx_size * feerate)

    def fee_reserve_per_vault(self, block_height):
        """The fee needed to bump the Cancel tx at the reserve feerate."""
        return self.cancel_tx_fee(self.feerate_reserve_per_vault(block_height), 0)

    def is_tx_confirmed(self, tx, height):
        """We consider a transaction to have been confirmed in this block if its
        feerate was above the min feerate in this block."""
        min_feerate = self.hist_df["min_feerate"][height]
        min_feerate = 0 if min_feerate == "NaN" else float(min_feerate)
        return tx.feerate() > min_feerate

    def Vm(self, block_height):
        """Starting value for the low-value feebump coins.

        The low value fb coins are not part of the critical reserve but used to reduce
        overpayments.
        """
        if self.vm_cache[0] == block_height:
            return self.vm_cache[1]

        # We use the block chain based estimate so it can be deterministically
        # computed by the operator's wallet.
        feerate = self.fallback_feerate(block_height)
        vm = int(self.cancel_tx_fee(feerate, 0) + feerate * P2WPKH_INPUT_SIZE)
        assert vm > 0
        self.vm_cache = (block_height, vm)
        return vm

    def min_acceptable_fbcoin_value(self, height):
        """The minimum value for a feebumping coin we create is one that allows
        to pay for its inclusion at the maximum feerate AND increase the Cancel
        tx fee by at least 5sat/vbyte.
        """
        feerate = self.feerate_reserve_per_vault(height)
        return int(
            feerate * P2WPKH_INPUT_SIZE + self.cancel_tx_fee(MIN_BUMP_WORST_CASE, 0)
        )

    def Vb(self, block_height):
        """Value of the large feebump coins.

        The high-value fb coins are here to ensure we can bump the Cancel to the
        reserve feerate, therefore they are lower-bounded to at least pay for their
        own fee at the reserve feerate.
        In addition we make sure they at least bump the feerate by 5sat/vbyte.
        """
        reserve = self.fee_reserve_per_vault(block_height)
        reserve_rate = self.feerate_reserve_per_vault(block_height)
        vb = reserve / self.vb_coins_count
        min_vb = self.min_acceptable_fbcoin_value(block_height)
        return int(reserve_rate * P2WPKH_INPUT_SIZE + max(vb, min_vb))

    def coins_dist_reserve(self, block_height):
        """The coin amount distribution used to cover up to the worst case.

        These coins are needed to be able to Cancel up the reserve feerate, but
        not usually optimal during normal operations.
        """
        vb = self.Vb(block_height)
        return [vb] * self.vb_coins_count

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
        of Vm and the fee required to bump a Cancel transaction by 5 sat per vByte
        in the worst case (when the fee rate is equal to the reserve rate).
        Note: t1 is same as the lower bound of Vb.
        """
        # FIXME: it really is useless with the current dist. Vm is always lower than Vb.
        reserve_rate = self.feerate_reserve_per_vault(block_height)
        t1 = reserve_rate * P2WPKH_INPUT_SIZE + self.cancel_tx_fee(
            MIN_BUMP_WORST_CASE, 0
        )
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

    def cf_coin_selec_0(self, block_height):
        """Select coins to consume as inputs for the CF transaction,
        remove them from P and V.

        This version grabs all the feebump coins available.
        """

        def coin_filter(coin):
            if coin.is_unconfirmed():
                return False

            if (
                self.coin_pool.is_allocated(coin)
                and self.vaults[self.coin_pool.coin_allocation(coin)].status
                != VaultState.READY
            ):
                return False

            return True

        return self.grab_coins(coin_filter)

    def cf_coin_selec_1(self, block_height):
        """Select coins to consume as inputs for the CF transaction,
        remove them from P and V.

        This version grabs all the coins that either haven't been processed yet
        or are negligible.
        """

        def coin_filter(coin):
            if coin.is_unconfirmed():
                return False

            if (
                self.coin_pool.is_allocated(coin)
                and self.vaults[self.coin_pool.coin_allocation(coin)].status
                != VaultState.READY
            ):
                return False

            return coin.is_unprocessed() or self.is_negligible(coin, block_height)

        return self.grab_coins(coin_filter)

    def cf_coin_selec_2(self, block_height):
        """Select coins to consume as inputs for the CF transaction,
        remove them from P and V.

        This version grabs all coins that are unprocessed, all
        unallocated coins that are not in the target coin distribution
        with a tolerance of X% (where X% == self.I_2_tol*100), and
        if the fee-rate is low, all negligible coins
        """
        dist = set(self.fb_coins_dist(block_height))
        low_feerate = self.next_block_feerate(block_height) <= 5

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
            if (
                self.coin_pool.is_allocated(coin)
                and self.vaults[self.coin_pool.coin_allocation(coin)].status
                != VaultState.READY
            ):
                return False

            if not self.coin_pool.is_allocated(coin):
                if not coin_in_dist(coin.amount, dist, self.I_2_tol):
                    return True

            if low_feerate and coin.amount < FB_DUST_THRESH:
                return True

            return False

        return self.grab_coins(coin_filter)

    def cf_coin_selec_3(self, height):
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
            if (
                self.coin_pool.is_allocated(coin)
                and self.vaults[self.coin_pool.coin_allocation(coin)].status
                != VaultState.READY
            ):
                return False

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
        feerate = self.feerate_reserve_per_vault(height)
        return int(feerate * P2WPKH_INPUT_SIZE + self.cancel_tx_fee(2, 0))

    # FIXME: eventually we should allocate as many outputs as we can, even if
    # it only represents part of a reserve. It would really lower the number
    # of allocation failures.
    def broadcast_consolidate_fanout(self, block_height):
        """
        Simulate the WT creating a consolidate-fanout (CF) tx which aims to 1) create coins from
        new re-fills that enable accurate feebumping and 2) consolidate negligible feebump coins
        if the current feerate is "low".

        Note that negligible coins that are consolidated will be removed from their
        associated vault's fee_reserve. So this process will diminish the vault's fee
        reserve until the new coins are confirmed and re-allocated (FIXME: actually we
        don't currently re-create coins that are consolidated).

        CF transactions help maintain coin sizes that enable accurate fee-bumping.
        """
        # Set target values for our coin creation amounts
        dist_reserve = self.coins_dist_reserve(block_height)
        dist_bonus = self.coins_dist_bonus(block_height)

        # Select coins to be consolidated
        if self.cf_coin_selec == 0:
            coins = self.cf_coin_selec_0(block_height)
        elif self.cf_coin_selec == 1:
            coins = self.cf_coin_selec_1(block_height)
        elif self.cf_coin_selec == 2:
            coins = self.cf_coin_selec_2(block_height)
        elif self.cf_coin_selec == 3:
            coins = self.cf_coin_selec_3(block_height)
        else:
            raise CfError("Unknown algorithm version for coin consolidation")

        # FIXME: we shouldn't use the next block feerate, rather something more economical.
        feerate = self.next_block_feerate(block_height)

        # FIXME this doesn't re-create enough coins if we consolidated some.
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
        # The cost of a change output should we need to add one
        change_size = P2WPKH_OUTPUT_SIZE
        change_fee = P2WPKH_OUTPUT_SIZE * feerate
        # Add new distributions of coins to the CF until we can't afford it anymore
        total_to_consume = sum(c.amount for c in coins)
        num_new_reserves = 0
        num_new_bonuses = 0
        consumed = 0
        while True:
            # First do the reserve
            if consumed + dist_rese_cost > total_to_consume:
                break
            # Don't create a new set of outputs if we can't pay the fees for it
            if total_to_consume - consumed - dist_rese_cost <= cf_tx_fee:
                break
            # Don't create a too large tx, instead add a change output (always
            # smaller than dist_rese_size) to be processed by a latter CF tx.
            if cf_size + dist_rese_size > MAX_TX_SIZE:
                assert cf_size + change_size <= MAX_TX_SIZE, "No room for change output"
                added_coins.append(
                    self.coin_pool.add_coin(
                        total_to_consume - consumed - change_fee,
                        processing_state=ProcessingState.UNPROCESSED,
                    )
                )
                cf_size += change_size
                cf_tx_fee += change_fee
                break
            consumed += dist_rese_cost
            cf_size += dist_rese_size
            cf_tx_fee += int(dist_rese_size * feerate)

            num_new_reserves += 1
            for x in dist_reserve:
                added_coins.append(
                    self.coin_pool.add_coin(x, processing_state=ProcessingState.PENDING)
                )

            # Then if we still have enough do the bonus
            if consumed + dist_bonu_cost > total_to_consume:
                continue
            # Don't create a new set of outputs if we can't pay the fees for it
            if total_to_consume - consumed - dist_bonu_cost <= cf_tx_fee:
                break
            # Don't create a too large tx, instead add a change output (always
            # smaller than dist_rese_size) to be processed by a latter CF tx.
            if cf_size + dist_bonu_size > MAX_TX_SIZE:
                assert cf_size + change_size <= MAX_TX_SIZE, "No room for change output"
                added_coins.append(
                    self.coin_pool.add_coin(
                        total_to_consume - consumed - change_fee,
                        processing_state=ProcessingState.UNPROCESSED,
                    )
                )
                cf_size += change_size
                cf_tx_fee += change_fee
                break
            consumed += dist_bonu_cost
            cf_size += dist_bonu_size
            cf_tx_fee += int(dist_bonu_size * feerate)

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

        # If a change output was appended to the outputs, it contains the
        # remainder.
        contains_change = (
            added_coins[-1].processing_state == ProcessingState.UNPROCESSED
        )
        if not contains_change:
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
                added_coin_value = int(
                    self.min_acceptable_fbcoin_value(block_height) * 1.3
                )
                output_fee = int(P2WPKH_OUTPUT_SIZE * feerate)
                # The number of coins this large we can add
                added_coins_count = remainder / (added_coin_value + output_fee)
                if added_coins_count >= 1:
                    # For a bias toward a lower number of larger outputs, truncate
                    # the number of coins added and add the excess value to the outputs
                    added_coins_count = int(added_coins_count)
                    outputs_fee = added_coin_value + output_fee * added_coins_count
                    added_coin_value = int(
                        (remainder - outputs_fee) / added_coins_count
                    )
                    for _ in range(added_coins_count):
                        added_coins.append(
                            self.coin_pool.add_coin(
                                added_coin_value,
                                processing_state=ProcessingState.PENDING,
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
                if coin.is_unconfirmed():
                    self.coin_pool.confirm_coin(coin, height)
            self.mempool.remove(tx)
            return True
        return False

    def allocate(self, vault_id, amount, block_height):
        """WT allocates coins to a (new/existing) vault if there is enough
        available coins to meet the requirement.
        """
        dist_req = self.coins_dist_reserve(block_height)
        dist_bonus = self.coins_dist_bonus(block_height)
        min_coin_value = self.min_fbcoin_value(block_height)
        # If the vault exists and is under reserve, don't remove it immediately
        # to make sure we won't fail after modifying the internal state.
        remove_vault = False

        # If vault already exists and is under requirement, de-allocate its current fee
        # reserve first
        vault = self.vaults.get(vault_id)
        if vault is not None:
            # FIXME: the caller should check that?
            if not self.under_requirement(vault, block_height):
                return
            else:
                logging.debug(
                    f"  Allocation transition to an existing vault {vault_id} at block"
                    f" {block_height}"
                )
                remove_vault = True
        else:
            logging.debug(
                f"  Allocation transition to new vault {vault_id} at block"
                f" {block_height}"
            )

        # We only require to allocate up to the required reserve, the rest is a bonus
        # to avoid overpayments.
        usable = [] if not remove_vault else list(vault.allocated_coins())
        usable += [
            c.amount
            for c in self.coin_pool.unallocated_coins()
            if c.amount >= min_coin_value
        ]
        total_usable = sum(usable)
        required_reserve = sum(dist_req)

        if required_reserve > total_usable:
            raise AllocationError(required_reserve, total_usable)
        if remove_vault:
            self.remove_vault(vault)
        # NOTE: from now on we MUST NOT fail (or crash the program if we do as
        # we won't recover from the modified state).

        self.vaults[vault_id] = Vault(vault_id, amount)
        vault = self.vaults[vault_id]
        # First optimistically search for coins in the required reserve with
        # small tolerance.
        tolerances = [0.05, 0.1, 0.2, 0.3]
        for tol in tolerances:
            not_found = []
            for x in dist_req:
                try:
                    fbcoin = next(
                        coin
                        for coin in self.coin_pool.unallocated_coins()
                        if ((1 - tol / 2) * x <= coin.amount <= (1 + tol) * x)
                    )
                    self.allocate_coin(fbcoin, vault)
                    logging.debug(
                        f"    {fbcoin} found with tolerance {tol*100}%, added to"
                        f" fee reserve. Distribution value: {x}"
                    )
                except (StopIteration):
                    logging.debug(
                        f"    No coin found with amount = {x} with tolerance {tol*100}%"
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
        # with coins off the dist but make sure they increase the fee at the
        # worst case feerate.
        for coin in self.coin_pool.unallocated_coins():
            if coin.amount >= min_coin_value:
                self.allocate_coin(coin, vault)
                logging.debug(f"    {coin} found to complete")
                if vault.reserve_balance() >= required_reserve:
                    break

        assert vault.reserve_balance() >= required_reserve, (
            f"Was checked before searching, {vault.reserve_balance()} vs"
            f" {required_reserve}"
        )

        # Now we have enough coins for the required reserve we can look for
        # coins in the bonus reserve
        for x in dist_bonus:
            for coin in self.coin_pool.unallocated_coins():
                if x * 0.7 <= coin.amount <= x * 1.3:
                    self.allocate_coin(coin, vault)
                    break

        logging.debug(
            f"    Reserve for vault {vault.id} has excess of"
            f" {vault.reserve_balance() - required_reserve}"
        )

    def broadcast_cancel(self, vault_id, block_height):
        """Construct and broadcast the cancel tx.

        We supplement the Cancel tx fees with coins from the pool in order for it
        to meet the next block feerate.
        """
        vault = self.vaults[vault_id]

        feerate = self.next_block_feerate(block_height)
        needed_fee = self.cancel_tx_fee(feerate, 0)

        cancel_fb_inputs = []
        if self.cancel_coin_selection == 0:
            cancel_fb_inputs = self.cancel_coin_selec_0(vault, needed_fee, feerate)
        elif self.cancel_coin_selection == 1:
            cancel_fb_inputs = self.cancel_coin_selec_1(vault, needed_fee, feerate)

        vault.set_status(VaultState.CANCELING)
        self.mempool.append(
            CancelTx(
                block_height,
                vault.id,
                self.cancel_vbytes() + len(cancel_fb_inputs) * P2WPKH_INPUT_SIZE,
                cancel_fb_inputs,
            )
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
            fbcoin_cost = feerate * P2WPKH_INPUT_SIZE
            try:
                fbcoin = next(
                    coin
                    for coin in reserve
                    if coin.amount - fbcoin_cost >= needed_fee - collected_fee
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
                # Otherwise, take the largest coin that bump the fee (if there is
                # one) and continue.
                fbcoin = reserve[-1]
                if fbcoin.amount <= fbcoin_cost + self.cancel_tx_fee(feerate, 0):
                    logging.error(
                        f"Not enough coins to cover for the Cancel fee of {needed_fee}"
                        f" at feerate {feerate}. Collected {collected_fee} sats."
                    )
                    break
                self.remove_coin(fbcoin)
                collected_fee += fbcoin.amount - feerate * P2WPKH_INPUT_SIZE
                coins.append(fbcoin)

        return coins

    def cancel_coin_selec_1(self, vault, needed_fee, feerate):
        """Select the combination of fee-bumping coins that results in the
        smallest overpayment possible.

        The UTxO pool is laid out with large coins covering up to the reserve
        and smaller coins used for a finer grained coin selection to avoid
        overpayments.
        First try to find the number of Vb (large) coins needed to cover for
        the most part of the fees, then fill the gap with Vm (small) coins.
        """
        txin_cost = P2WPKH_INPUT_SIZE * feerate
        allocated_coins = sorted(vault.allocated_coins())
        # All vb coins are always larger than vm coins (or at least we assume so)
        vm_coins, vb_coins = (
            allocated_coins[: -self.vb_coins_count],
            allocated_coins[-self.vb_coins_count :],
        )
        # We often end up with more Vb coins which we would consider to be Vm coins
        # above. Try to fix this based on their value.
        while True:
            # Sanity hard stop
            if len(vm_coins) <= 6:
                break
            coin = vm_coins.pop(-1)
            if coin.amount >= vb_coins[-1].amount * 0.80:
                bisect.insort(vb_coins, coin)
            else:
                vm_coins.append(coin)
                break

        def coin_sum(coins):
            return sum(c.amount for c in coins)

        def select_coins(coins):
            for coin in coins:
                logging.debug(f"        removing coin: {coin} to add to new cancel tx")
                self.remove_coin(coin)
            return coins

        # First check if the needed amount is very low, in which case we don't
        # even need a Vb coin.
        if vb_coins[0].amount > needed_fee + txin_cost:
            # TODO: the number of vm coins is always low, we could do an exhaustive search.
            for i in range(1, len(vm_coins)):
                if coin_sum(vm_coins[:i]) >= needed_fee + txin_cost * i:
                    return select_coins(vm_coins[:i])
            return select_coins([vb_coins[0]])

        # Then gather enough vb coins
        picked_vb_coins = []
        paid = 0
        for i in range(1, len(vb_coins)):
            coin = vb_coins[-i]
            if needed_fee - paid < coin.amount - txin_cost:
                break
            picked_vb_coins.append(coin)
            paid += coin.amount - txin_cost

        # And finally fill the gap with small Vm coins. Note we go through the
        # list in reverse order as the Vm coins amount is increasing.
        # TODO: figure out why we have less overpayments by going in increasing order
        # TODO: the number of vm coins is always low, we could do an exhaustive search.
        rem_fee = needed_fee - paid
        for i in range(len(vm_coins)):
            if coin_sum(vm_coins[:i]) >= rem_fee + txin_cost * i:
                return select_coins(picked_vb_coins + vm_coins[:i])

        # All Vm coins couldn't fill the gap? Fall back to use only Vb coins
        if len(picked_vb_coins) < len(vb_coins):
            for i in range(1, len(vb_coins)):
                if coin_sum(vb_coins[:i]) > needed_fee + txin_cost * i:
                    return select_coins(vb_coins[:i])

        logging.error(
            f"Not enough reserve to pay for cancel fee ({needed_fee} sats) at feerate"
            f" {feerate}"
        )
        return select_coins(vb_coins + vm_coins)

    def finalize_cancel(self, tx, height):
        """Once the cancel is confirmed, any remaining fbcoins allocated to vault_id
        become unallocated. The vault with vault_id is removed from vaults.
        """
        if self.is_tx_confirmed(tx, height):
            self.remove_vault(self.vaults[tx.vault_id])
            self.mempool.remove(tx)
            return True
        else:
            logging.debug("    Confirmation failed...")
            self.maybe_replace_cancel(height, tx)
            return False

    def maybe_replace_cancel(self, height, tx):
        """Broadcasts a replacement cancel transaction if feerate increased.

        Updates the coin_pool and the associated vault.
        """
        vault = self.vaults[tx.vault_id]
        new_feerate = self.next_block_feerate(height)

        logging.debug(
            f"    Checking if we need to replace Cancel tx for vault {vault.id}."
            f" Current feerate: {tx.feerate()}, next block feerate: {new_feerate}"
        )
        if new_feerate > tx.feerate():
            logging.debug("    Replacing Cancel tx...")
            new_fee = self.cancel_tx_fee(new_feerate, 0)
            # Bitcoin Core policy is set to 1, take some leeway by setting
            # it to 2.
            min_fee = tx.fee + self.cancel_tx_fee(2, len(tx.fbcoins))
            needed_fee = max(new_fee, min_fee)

            # Add the feebump coins back to the pool and remove the previous tx
            for coin in tx.fbcoins:
                coin = self.coin_pool.add_coin(
                    coin.amount,
                    coin.processing_state,
                    coin.fan_block,
                    vault.id,
                    coin_id=coin.id,
                )
                vault.allocate_coin(coin)
            self.mempool.remove(tx)

            # Push a new tx with coins selected to meet the new fee
            if self.cancel_coin_selection == 0:
                cancel_fb_inputs = self.cancel_coin_selec_0(
                    vault, needed_fee, new_feerate
                )
            elif self.cancel_coin_selection == 1:
                cancel_fb_inputs = self.cancel_coin_selec_1(
                    vault, needed_fee, new_feerate
                )

            self.mempool.append(
                CancelTx(
                    height,
                    vault.id,
                    self.cancel_vbytes() + len(cancel_fb_inputs) * P2WPKH_INPUT_SIZE,
                    cancel_fb_inputs,
                )
            )

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
