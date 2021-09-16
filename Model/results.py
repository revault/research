import logging
import multiprocessing as mp
import os
import pandas as pd
import random
import sys

from functools import partial
from simulation import Simulation


def sim_process(prng_seed, val=None, study_type=None, config_map=None):
    logging.basicConfig(level=logging.INFO)
    req_types = [
        "N_STK",
        "N_MAN",
        "HIST_CSV",
        "RESERVE_STRAT",
        "ESTIMATE_STRAT",
        "I_VERSION",
        "EXPECTED_ACTIVE_VAULTS",
        "REFILL_PERIOD",
        "REFILL_EXCESS",
        "SPEND_RATE",
        "INVALID_SPEND_RATE",
        "CATASTROPHE_RATE",
    ]
    if study_type not in req_types:
        logging.error(
            "Study requires a type from: EXPECTED_ACTIVE_VAULTS,"
            " REFILL_EXCESS, REFILL_PERIOD, REFILL_EXCESS, DELEGATION_PERIOD,"
            " INVALID_SPEND_RATE, CATASTROPHE_RATE, N_STK, N_MAN, HIST_CSV,"
            " RESERVE_STRAT, ESTIMATE_STRAT, I_VERSION."
        )
        sys.exit(1)

    logging.info(
        f"""Using config:\n
            N_STK = {config_map["N_STK"]}
            N_MAN = {config_map["N_MAN"]}
            LOCKTIME = {config_map["LOCKTIME"]}
            HIST_CSV = {config_map["HIST_CSV"]}
            RESERVE_STRAT = {config_map["RESERVE_STRAT"]}
            ESTIMATE_STRAT = {config_map["ESTIMATE_STRAT"]}
            I_VERSION = {config_map["I_VERSION"]}
            EXPECTED_ACTIVE_VAULTS = {config_map["EXPECTED_ACTIVE_VAULTS"]}
            REFILL_PERIOD = {config_map["REFILL_PERIOD"]}
            REFILL_EXCESS = {config_map["REFILL_EXCESS"]}
            SPEND_RATE = {config_map["SPEND_RATE"]}
            INVALID_SPEND_RATE = {config_map["INVALID_SPEND_RATE"]}
            CATASTROPHE_RATE = {config_map["CATASTROPHE_RATE"]}
        """
    )

    config_map[study_type] = val
    logging.info(f"Simulating with {study_type} = {val}, prng_seed = {prng_seed}\n")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    fname = os.path.join(results_dir, f"{study_type}-{val}-Report-{prng_seed}")
    start_block = 350000
    end_block = 681000
    random.seed(prng_seed)
    sim = Simulation(
        int(config_map["N_STK"]),
        int(config_map["N_MAN"]),
        int(config_map["LOCKTIME"]),
        config_map["HIST_CSV"],
        config_map["RESERVE_STRAT"],
        config_map["ESTIMATE_STRAT"],
        int(config_map["I_VERSION"]),
        int(config_map["CANCEL_COIN_SELECTION"]),
        int(config_map["EXPECTED_ACTIVE_VAULTS"]),
        int(config_map["REFILL_EXCESS"] * config_map["EXPECTED_ACTIVE_VAULTS"]),
        int(config_map["REFILL_PERIOD"]),
        int(config_map["SPEND_RATE"]),
        float(config_map["INVALID_SPEND_RATE"]),
        float(config_map["CATASTROPHE_RATE"]),
        with_balance=True,
        with_divergence=True,
        with_cum_op_cost=True,
        with_overpayments=True,
    )
    try:
        sim.run(start_block, end_block)
        if sim.end_block == end_block:
            (report, report_df) = sim.plot()
            with open(f"{fname}.txt", "w+", encoding="utf-8") as f:
                f.write(report)
            return report_df

    except (RuntimeError):
        # FIXME: return empty report_df?
        logging.info(f"RuntimeError: Simulating with {study_type} = {val} failed.")


def multiprocess_run(range_seed, val, study_type, config_map):
    # FIXME: what if process fails?
    assert len(range_seed) >= 2
    cores = len(range_seed)
    with mp.Pool(processes=cores) as pool:
        dfs = pool.map(
            partial(sim_process, val=val, study_type=study_type, config_map=config_map),
            range_seed,
        )
        results_df = dfs[0]
        for df in dfs[1:]:
            results_df = pd.concat([results_df, df], axis=0)
        return results_df


if __name__ == "__main__":
    mp.set_start_method("spawn")

    # Set the default config
    config_map = {
        "N_STK": 5,
        "N_MAN": 3,
        "LOCKTIME": 12,
        "HIST_CSV": "../block_fees/historical_fees.csv",
        "RESERVE_STRAT": "CUMMAX95Q90",
        "ESTIMATE_STRAT": "ME30",
        "I_VERSION": 2,
        "CANCEL_COIN_SELECTION": 0,
        "EXPECTED_ACTIVE_VAULTS": 5,
        "REFILL_PERIOD": 144 * 7,
        "REFILL_EXCESS": 5,
        "SPEND_RATE": 1,
        "INVALID_SPEND_RATE": 0.1,
        "CATASTROPHE_RATE": 0.005,
        "CANCEL_COIN_SELECTION": 0,
    }

    # Set the study parameters
    study_type = "O_VERSION"
    val_range = [0, 1]
    sim_repeats = 10
    cores = 10

    report_name = f"study-{study_type}"
    range_seed = list(range(21000000, 21000000 + cores))

    # Generate results
    for val in val_range:
        config_map[study_type] = val
        report = (
            f"{report_name}\nnumber of simulations:"
            f" {sim_repeats*cores}\nseeds used:"
            f" {range_seed[0]}-{range_seed[0]+(sim_repeats*cores)}\nconfig:"
            f" {config_map}\n\nResults:\n"
        )

        sim_results = []
        for i in range(0, sim_repeats):
            range_seed = [s + cores for s in range_seed]
            result = multiprocess_run(range_seed, val, study_type, config_map)
            sim_results.append(result)

        stats_df = sim_results[0]
        for df in sim_results:
            stats_df = pd.concat([stats_df, df], axis=0)

        for col in stats_df.columns:
            report += f"{col} mean:    {stats_df[col].mean()}\n"
            report += f"{col} std dev: {stats_df[col].std()}\n"

        with open(f"{report_name}-{val}.txt", "w+", encoding="utf-8") as f:
            f.write(report)
