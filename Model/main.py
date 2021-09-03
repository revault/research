import logging
import random

from simulation import Simulation

if __name__ == "__main__":
    random.seed(21000000)
    # FIXME: make it configurable through command line
    logging.basicConfig(level=logging.DEBUG)

    # FIXME: make it configurable through command line
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
    logging.info(f"Using config {config}")
    fname = "TestReport"

    sim = Simulation(config, fname)

    start_block = 200000
    end_block = 681000

    # "operations", "coin_pool_age", "coin_pool", "risk_status"
    subplots = ["balance", "vault_excesses", "cumulative_ops", "overpayments"]
    sim.plot_simulation(start_block, end_block, subplots)

    sim.plot_strategic_values(
        start_block, end_block, "ME30", "CUMMAX95Q90", O_version=1
    )
