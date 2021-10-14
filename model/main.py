import logging
import os
import random
import sys

from simulation import Simulation

REPORT_FILENAME = os.getenv("REPORT_FILENAME", None)
PLOT_FILENAME = os.getenv("PLOT_FILENAME", None)
PROFILE_FILENAME = os.getenv("PROFILE_FILENAME", None)
N_STK = os.getenv("N_STK", 7)
N_MAN = os.getenv("N_MAN", 3)
LOCKTIME = os.getenv("LOCKTIME", 24)
HIST_CSV = os.getenv("HIST_CSV", "../block_fees/historical_fees.csv")
RESERVE_STRAT = os.getenv("RESERVE_STRAT", "CUMMAX95Q90")
ESTIMATE_STRAT = os.getenv("ESTIMATE_STRAT", "85Q1H")
I_VERSION = os.getenv("I_VERSION", 3)
CANCEL_COIN_SELECTION = os.getenv("CANCEL_COIN_SELECTION", 1)
NUMBER_VAULTS = os.getenv("NUMBER_VAULTS", 20)
REFILL_EXCESS = os.getenv("REFILL_EXCESS", 2)
REFILL_PERIOD = os.getenv("REFILL_PERIOD", 1008)
# Unvault rate per day
UNVAULT_RATE = os.getenv("UNVAULT_RATE", 1)
# Invalid rate per unvault
INVALID_SPEND_RATE = os.getenv("INVALID_SPEND_RATE", 0.01)
# Catastrophe rate per day
CATASTROPHE_RATE = os.getenv("CATASTROPHE_RATE", 0.001)
# Delegate rate per day (if not running at fixed scale)
DELEGATE_RATE = os.getenv("DELEGATE_RATE", None)

# Plot types
BALANCE = os.getenv("BALANCE", "true").lower() == "true"
CUM_OP_COST = os.getenv("CUM_OP_COST", "true").lower() == "true"
RISK_TIME = (
    os.getenv("RISK_TIME", "false").lower() == "true" if CUM_OP_COST is True else False
)
DIVERGENCE = os.getenv("DIVERGENCE", "false").lower() == "true"
OP_COST = os.getenv("OP_COST", "false").lower() == "true"
OVERPAYMENTS = os.getenv("OVERPAYMENTS", "false").lower() == "true"
RISK_STATUS = os.getenv("RISK_STATUS", "false").lower() == "true"
FB_COINS_DIST = os.getenv("FB_COINS_DIST", "false").lower() == "true"

if __name__ == "__main__":
    random.seed(21000000)
    # FIXME: make it configurable through command line
    logging.basicConfig(level=logging.DEBUG)

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

    plot_types = [
        BALANCE,
        CUM_OP_COST,
        DIVERGENCE,
        OP_COST,
        OVERPAYMENTS,
        RISK_STATUS,
        FB_COINS_DIST,
    ]
    if len([plot for plot in plot_types if plot is True]) < 2:
        logging.error(
            "Must generate at least two plot types to run simulation. Plot types are:"
            " BALANCE, CUM_OP_COST, DIVERGENCE, OP_COST, OVERPAYMENTS, RISK_STATUS,"
            " or FB_COINS_DIST."
        )
        sys.exit(1)

    logging.info(f"Config: {', '.join(str(v) for v in req_vars)}")
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
        with_balance=BALANCE,
        with_divergence=DIVERGENCE,
        with_op_cost=OP_COST,
        with_cum_op_cost=CUM_OP_COST,
        with_risk_time=RISK_TIME,
        with_overpayments=OVERPAYMENTS,
        with_risk_status=RISK_STATUS,
        with_fb_coins_dist=FB_COINS_DIST,
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