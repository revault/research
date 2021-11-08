import logging
import os
import random
import sys

from simulation import Simulation


def main(conf, return_results=False, show_plot=False):
    random.seed(conf["PRNG_SEED"])

    if conf["LOG_LEVEL"] in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        logging.basicConfig(level=conf["LOG_LEVEL"])
    else:
        logging.basicConfig(level=logging.DEBUG)
        logging.error("Invalid log level provided, setting as DEBUG instead.")

    req_vars = [
        conf["N_STK"],
        conf["N_MAN"],
        conf["LOCKTIME"],
        conf["HIST_CSV"],
        conf["RESERVE_STRAT"],
        conf["FALLBACK_EST_STRAT"],
        conf["CF_COIN_SELECTION"],
        conf["CANCEL_COIN_SELECTION"],
        conf["NUMBER_VAULTS"],
        conf["REFILL_EXCESS"],
        conf["REFILL_PERIOD"],
        conf["UNVAULT_RATE"],
        conf["INVALID_SPEND_RATE"],
        conf["CATASTROPHE_RATE"],
    ]
    if any(v is None for v in req_vars):
        logging.error(
            "Need all these environment variables to be set: N_STK, N_MAN, LOCKTIME,"
            " HIST_CSV, RESERVE_STRAT, FALLBACK_EST_STRAT, CF_COIN_SELECTION,"
            " CANCEL_COIN_SELECTION, NUMBER_VAULTS, REFILL_EXCESS,"
            " REFILL_PERIOD, UNVAULT_RATE, INVALID_SPEND_RATE, CATASTROPHE_RATE."
        )
        sys.exit(1)

    plot_types = [
        conf["PLOT_BALANCE"],
        conf["PLOT_CUM_OP_COST"],
        conf["PLOT_DIVERGENCE"],
        conf["PLOT_OP_COST"],
        conf["PLOT_OVERPAYMENTS"],
        conf["PLOT_RISK_STATUS"],
        conf["PLOT_FB_COINS_DIST"],
    ]
    if len([plot for plot in plot_types if plot is True]) < 2:
        logging.error(
            "Must generate at least two plot types to run simulation. Plot types are:"
            " PLOT_BALANCE, PLOT_CUM_PLOT_OP_COST, PLOT_DIVERGENCE, PLOT_OP_COST,"
            " PLOT_OVERPAYMENTS, PLOT_RISK_STATUS, or PLOT_FB_COINS_DIST."
        )
        sys.exit(1)

    logging.info(f"Configuration:\n{conf}")

    sim = Simulation(
        int(conf["N_STK"]),
        int(conf["N_MAN"]),
        int(conf["LOCKTIME"]),
        conf["HIST_CSV"],
        conf["RESERVE_STRAT"],
        conf["FALLBACK_EST_STRAT"],
        int(conf["CF_COIN_SELECTION"]),
        int(conf["CANCEL_COIN_SELECTION"]),
        int(conf["NUMBER_VAULTS"]),
        int(conf["REFILL_EXCESS"]),
        int(conf["REFILL_PERIOD"]),
        float(conf["UNVAULT_RATE"]),
        float(conf["INVALID_SPEND_RATE"]),
        float(conf["CATASTROPHE_RATE"]),
        float(conf["DELEGATE_RATE"]) if conf["DELEGATE_RATE"] is not None else None,
        with_balance=conf["PLOT_BALANCE"],
        with_divergence=conf["PLOT_DIVERGENCE"],
        with_op_cost=conf["PLOT_OP_COST"],
        with_cum_op_cost=conf["PLOT_CUM_OP_COST"],
        with_risk_time=conf["PLOT_RISK_TIME"],
        with_overpayments=conf["PLOT_OVERPAYMENTS"],
        with_risk_status=conf["PLOT_RISK_STATUS"],
        with_fb_coins_dist=conf["PLOT_FB_COINS_DIST"],
    )

    start_block = 350000
    end_block = 681000

    if conf["PROFILE_FILENAME"] is not None:
        import pstats
        from pstats import SortKey
        import cProfile

        cProfile.run("sim.run(start_block, end_block)", f"{conf['PROFILE_FILENAME']}")
        p = pstats.Stats(f"{conf['PROFILE_FILENAME']}")
        p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()

    else:
        sim.run(start_block, end_block)

        (report, report_df) = sim.plot(conf['PLOT_FILENAME'], show_plot)
        logging.info(f"Report\n{report}")

        if conf["REPORT_FILENAME"] is not None:
            with open(f"{conf['REPORT_FILENAME']}.txt", "w+", encoding="utf-8") as f:
                f.write(report)

        if return_results:
            return report_df


if __name__ == "__main__":
    assert (
        os.path.basename(os.getcwd()).split("/")[-1] == "model"
    ), "The script currently uses relative paths, unfortunately"

    configuration = {
        "REPORT_FILENAME": os.getenv("REPORT_FILENAME", None),
        "PLOT_FILENAME": os.getenv("PLOT_FILENAME", None),
        "PROFILE_FILENAME": os.getenv("PROFILE_FILENAME", None),
        "N_STK": os.getenv("N_STK", 7),
        "N_MAN": os.getenv("N_MAN", 3),
        "LOCKTIME": os.getenv("LOCKTIME", 24),
        "HIST_CSV": os.getenv("HIST_CSV", "../block_fees/historical_fees.csv"),
        "RESERVE_STRAT": os.getenv("RESERVE_STRAT", "CUMMAX95Q90"),
        "FALLBACK_EST_STRAT": os.getenv("FALLBACK_EST_STRAT", "85Q1H"),
        "CF_COIN_SELECTION": os.getenv("CF_COIN_SELECTION", 1),
        "CANCEL_COIN_SELECTION": os.getenv("CANCEL_COIN_SELECTION", 1),
        "NUMBER_VAULTS": os.getenv("NUMBER_VAULTS", 10),
        "REFILL_EXCESS": os.getenv("REFILL_EXCESS", 2),
        "REFILL_PERIOD": os.getenv("REFILL_PERIOD", 1008),
        "UNVAULT_RATE": os.getenv("UNVAULT_RATE", 0.5),
        "INVALID_SPEND_RATE": os.getenv("INVALID_SPEND_RATE", 0.01),
        "CATASTROPHE_RATE": os.getenv("CATASTROPHE_RATE", 0.001),
        "DELEGATE_RATE": os.getenv("DELEGATE_RATE", None),
        "PLOT_BALANCE": bool(int(os.getenv("PLOT_BALANCE", 1))),
        "PLOT_CUM_OP_COST": bool(int(os.getenv("PLOT_CUM_OP_COST", 1))),
        "PLOT_RISK_TIME": bool(int(os.getenv("PLOT_RISK_TIME", 0)))
        and bool(int(os.getenv("PLOT_CUM_OP_COST", 1))),
        "PLOT_DIVERGENCE": bool(int(os.getenv("PLOT_DIVERGENCE", 0))),
        "PLOT_OP_COST": bool(int(os.getenv("PLOT_OP_COST", 0))),
        "PLOT_OVERPAYMENTS": bool(int(os.getenv("PLOT_OVERPAYMENTS", 0))),
        "PLOT_RISK_STATUS": bool(int(os.getenv("PLOT_RISK_STATUS", 0))),
        "PLOT_FB_COINS_DIST": bool(int(os.getenv("PLOT_FB_COINS_DIST", 0))),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "DEBUG").upper(),
        "PRNG_SEED": os.getenv("PRNG_SEED", 21000000),
    }

    main(configuration, return_results=False, show_plot=True)
