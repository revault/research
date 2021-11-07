import logging
import multiprocessing as mp
import os
from pandas import DataFrame
import pandas as pd
import random
import sys
from functools import partial
from simulation import Simulation
import json
from main import main

# Set the study parameters
NUM_CORES = int(os.getenv("NUM_CORES", 1))
REPEATS_PER_CORE = int(os.getenv("REPEATS_PER_CORE", 1))
STUDY_TYPE = os.getenv("STUDY_TYPE", None)
VAL_RANGE = json.loads(os.getenv("VAL_RANGE", None))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def sim_process(prng_seed, val, config):
    # set sim specific env vars
    config["PRNG_SEED"] = f"{prng_seed}"
    config["REPORT_FILENAME"] = os.path.join(
        RESULTS_DIR, f"report_{STUDY_TYPE}_{val}-PRNG_{prng_seed}"
    )

    logging.info(f"Simulating with {STUDY_TYPE} = {val}, prng_seed = {prng_seed}\n")

    try:
        return main(conf=config, return_results=True)
    except (RuntimeError):
        # FIXME: return empty report_df?
        logging.info(f"RuntimeError: Simulating with {study_type} = {val} failed.")


def multiprocess_run(val, config, range_seed):
    # FIXME: what if process fails?
    assert len(range_seed) >= 2
    cores = len(range_seed)
    print(
        f"Multiprocess run with {STUDY_TYPE}: {val} and prng seed range: {range_seed}"
    )

    with mp.Pool(processes=cores) as pool:
        dfs = pool.map(partial(sim_process, val=val, config=config), range_seed)
        results_df = dfs[0]
        for df in dfs[1:]:
            results_df = pd.concat([results_df, df], axis=0)
        return results_df


if __name__ == "__main__":

    assert (
        os.path.basename(os.getcwd()).split("/")[-1] == "model"
    ), "The script currently uses relative paths, unfortunately"

    mp.set_start_method("spawn")

    req_vars = ["NUM_CORES", "REPEATS_PER_CORE", "STUDY_TYPE", "VAL_RANGE"]

    if any(v is None for v in req_vars):
        logging.error(
            "Need all these environment variables to be set: NUM_CORES,"
            " REPEATS_PER_CORE, STUDY_TYPE, VAL_RANGE."
        )
        sys.exit(1)

    valid_types = [
        "N_STK",
        "N_MAN",
        "LOCKTIME",
        "RESERVE_STRAT",
        "FALLBACK_EST_STRAT",
        "CF_COIN_SELECTION",
        "CANCEL_COIN_SELECTION",
        "NUMBER_VAULTS",
        "REFILL_EXCESS",
        "UNVAULT_RATE",
        "INVALID_SPEND_RATE",
        "CATASTROPHE_RATE",
        "DELEGATE_RATE",
    ]

    if STUDY_TYPE not in valid_types:
        logging.error(f"STUDY_TYPE must be set to one of: {valid_types}")
        sys.exit(1)

    report_name = f"study-{STUDY_TYPE}"

    config = {
        "REPORT_FILENAME": None,
        "PLOT_FILENAME": None,
        "PROFILE_FILENAME": os.getenv("PROFILE_FILENAME", None),
        "N_STK": os.getenv("N_STK", 7),
        "N_MAN": os.getenv("N_MAN", 3),
        "LOCKTIME": os.getenv("LOCKTIME", 24),
        "HIST_CSV": os.getenv("HIST_CSV", "../block_fees/historical_fees.csv"),
        "RESERVE_STRAT": os.getenv("RESERVE_STRAT", "CUMMAX95Q90"),
        "FALLBACK_EST_STRAT": os.getenv("FALLBACK_EST_STRAT", "85Q1H"),
        "CF_COIN_SELECTION": os.getenv("CF_COIN_SELECTION", 3),
        "CANCEL_COIN_SELECTION": os.getenv("CANCEL_COIN_SELECTION", 1),
        "NUMBER_VAULTS": os.getenv("NUMBER_VAULTS", 10),
        "REFILL_EXCESS": os.getenv("REFILL_EXCESS", 2),
        "REFILL_PERIOD": os.getenv("REFILL_PERIOD", 1008),
        "UNVAULT_RATE": os.getenv("UNVAULT_RATE", 0.5),
        "INVALID_SPEND_RATE": os.getenv("INVALID_SPEND_RATE", 0.01),
        "CATASTROPHE_RATE": os.getenv("CATASTROPHE_RATE", 0.001),
        "DELEGATE_RATE": os.getenv("DELEGATE_RATE", None),
        "PLOT_BALANCE": True,
        "PLOT_CUM_OP_COST": True,
        "PLOT_RISK_TIME": False,
        "PLOT_DIVERGENCE": True,
        "PLOT_OP_COST": False,
        "PLOT_OVERPAYMENTS": False,
        "PLOT_RISK_STATUS": False,
        "PLOT_FB_COINS_DIST": False,
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "DEBUG").upper(),
        "PRNG_SEED": os.getenv("PRNG_SEED", 21000000),
    }

    # Generate results
    report_rows = []
    for val in VAL_RANGE:
        range_seed = list(range(config['PRNG_SEED'], config['PRNG_SEED'] + NUM_CORES))
        config[STUDY_TYPE] = val
        report = (
            f"{report_name}\nnumber of simulations:"
            f" {REPEATS_PER_CORE*NUM_CORES}\nseeds used:"
            f" {range_seed[0]}-{range_seed[0]+(REPEATS_PER_CORE*NUM_CORES)}\nconfig:"
            f" {config}\n\nResults:\n"
        )

        sim_results = []
        for i in range(0, REPEATS_PER_CORE):
            result = multiprocess_run(val, config, range_seed)
            sim_results.append(result)
            range_seed = [s + NUM_CORES for s in range_seed]

        stats_df = sim_results[0]
        for df in sim_results:
            stats_df = pd.concat([stats_df, df], axis=0)

        row = [val]
        for col in stats_df.columns:
            report += f"{col} mean:    {stats_df[col].mean()}\n"
            report += f"{col} std dev: {stats_df[col].std()}\n"
            row.append(stats_df[col].mean())
            row.append(stats_df[col].std())
        report_rows.append(row)
        with open(
            f"{RESULTS_DIR}/{report_name}-{val}.txt", "w+", encoding="utf-8"
        ) as f:
            f.write(report)

        # Save the csv at each val in case of failure
        report_df = DataFrame(
            report_rows,
            columns=[
                STUDY_TYPE,
                "mean_balance_mean",
                "mean_balance_std_dev",
                "cum_ops_cost_mean",
                "cum_ops_cost_std_dev",
                "cum_cancel_fee_mean",
                "cum_cancel_fee_std_dev",
                "cum_cf_fee_mean",
                "cum_cf_fee_std_dev",
                "cum_refill_fee_mean",
                "cum_refill_fee_std_dev",
                "time_at_risk_mean",
                "time_at_risk_std_dev",
                "mean_recovery_time_mean",
                "mean_recovery_time_std_dev",
                "median_recovery_time_mean",
                "median_recovery_time_std_dev",
                "max_recovery_time_mean",
                "max_recovery_time_std_dev",
                "delegation_failure_count_mean",
                "delegation_failure_count_std_dev",
                "delegation_failure_rate_mean",
                "delegation_failure_rate_std_dev",
                "max_cancel_conf_time_mean",
                "max_cancel_conf_time_std_dev",
                "max_cf_conf_time_mean",
                "max_cf_conf_time_std_dev",
                "max_risk_coef_mean",
                "max_risk_coef_std_dev",
            ],
        )
        report_df.set_index(f"{STUDY_TYPE}", inplace=True)
        report_df.to_csv(f"{RESULTS_DIR}/{report_name}")
