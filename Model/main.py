import logging
import os
import random
import sys

from simulation import Simulation

REPORT_FILENAME = os.getenv("REPORT_FILENAME", None)
PLOT_FILENAME = os.getenv("PLOT_FILENAME", None)
PROFILE_FILENAME = os.getenv("PROFILE_FILENAME", None)
N_STK = os.getenv("N_STK", None)
N_MAN = os.getenv("N_MAN", None)
LOCKTIME = os.getenv("LOCKTIME", None)
HIST_CSV = os.getenv("HIST_CSV", None)
RESERVE_STRAT = os.getenv("RESERVE_STRAT", None)
ESTIMATE_STRAT = os.getenv("ESTIMATE_STRAT", None)
I_VERSION = os.getenv("I_VERSION", None)
CANCEL_COIN_SELECTION = os.getenv("CANCEL_COIN_SELECTION", None)
NUMBER_VAULTS = os.getenv("NUMBER_VAULTS", None)
REFILL_EXCESS = os.getenv("REFILL_EXCESS", None)
REFILL_PERIOD = os.getenv("REFILL_PERIOD", None)
# Unvault rate per day
UNVAULT_RATE = os.getenv("UNVAULT_RATE", None)
# Invalid rate per spend
INVALID_SPEND_RATE = os.getenv("INVALID_SPEND_RATE", None)
# Catastrophe rate per day
CATASTROPHE_RATE = os.getenv("CATASTROPHE_RATE", None)
# Delegate rate per day (if scale_is not fixed)
DELEGATE_RATE = os.getenv("DELEGATE_RATE", None)

if __name__ == "__main__":
    random.seed(21000000)
    # FIXME: make it configurable through command line
    logging.basicConfig(level=logging.DEBUG)

    # note: fee_estimates_fine.csv starts on block 415909 at 2016-05-18 02:00:00

    req_vars = [
        N_STK,
        N_MAN,
        LOCKTIME,
        HIST_CSV,
        RESERVE_STRAT,
        ESTIMATE_STRAT,
        I_VERSION,
        CANCEL_COIN_SELECTION,
        NUMBER_VAULTS,
        REFILL_EXCESS,
        REFILL_PERIOD,
        UNVAULT_RATE,
        INVALID_SPEND_RATE,
        CATASTROPHE_RATE,
    ]
    if any(v is None for v in req_vars):
        logging.error(
            "Need all these environment variables to be set: N_STK, N_MAN, LOCKTIME,"
            " HIST_CSV, RESERVE_STRAT, ESTIMATE_STRAT, I_VERSION,"
            " CANCEL_COIN_SELECTION, NUMBER_VAULTS, REFILL_EXCESS,"
            " REFILL_PERIOD, UNVAULT_RATE, INVALID_SPEND_RATE, CATASTROPHE_RATE."
        )
        sys.exit(1)
    logging.info(f"Config: {', '.join(v for v in req_vars)}")
    sim = Simulation(
        int(N_STK),
        int(N_MAN),
        int(LOCKTIME),
        HIST_CSV,
        RESERVE_STRAT,
        ESTIMATE_STRAT,
        int(I_VERSION),
        int(CANCEL_COIN_SELECTION),
        int(NUMBER_VAULTS),
        int(REFILL_EXCESS),
        int(REFILL_PERIOD),
        float(UNVAULT_RATE),
        float(INVALID_SPEND_RATE),
        float(CATASTROPHE_RATE),
        float(DELEGATE_RATE) if DELEGATE_RATE is not None else None,
        with_balance=True,
        # with_fb_coins_dist=True,
        with_cum_op_cost=True,
        with_divergence=True,
        with_overpayments=True,
    )

    start_block = 350000
    end_block = 681000

    if PROFILE_FILENAME is not None:
        import pstats
        from pstats import SortKey
        import cProfile

        cProfile.run('sim.run(start_block, end_block)', f'{PROFILE_FILENAME}')
        p = pstats.Stats(f"{PROFILE_FILENAME}")
        stats = p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()

    else:
        sim.run(start_block, end_block)

        report = sim.plot(PLOT_FILENAME, True)[0]
        logging.info(f"Report\n{report}")

        if REPORT_FILENAME is not None:
            with open(f"{REPORT_FILENAME}.txt", "w+", encoding="utf-8") as f:
                f.write(report)

        # sim.plot_fee_history(start_block,end_block, PLOT_FILENAME, True)
