import logging
import os
import random
import sys

from simulation import Simulation

def study_cf_outputs_0(range_factor, range_refill_excess):
    N_STK = 5
    N_MAN = 5
    HIST_CSV = "historical_fees.csv"
    RESERVE_STRAT = "CUMMAX95Q90"
    ESTIMATE_STRAT = "ME30"
    O_VERSION = 0
    I_VERSION = 1
    EXPECTED_ACTIVE_VAULTS = 20
    REFILL_PERIOD = 144*7
    DELEGATION_PERIOD = 144
    INVALID_SPEND_RATE = 0.1
    CATASTROPHE_RATE = 0.005

    logging.info(f"""Using config:\n
            N_STK = {N_STK}
            N_MAN = {N_MAN}
            HIST_CSV = {HIST_CSV}
            RESERVE_STRAT = {RESERVE_STRAT}
            ESTIMATE_STRAT = {ESTIMATE_STRAT}
            O_VERSION = {O_VERSION}
            I_VERSION = {I_VERSION}
            EXPECTED_ACTIVE_VAULTS = {EXPECTED_ACTIVE_VAULTS}
            REFILL_PERIOD = {REFILL_PERIOD}
            DELEGATION_PERIOD = {DELEGATION_PERIOD}
            INVALID_SPEND_RATE = {INVALID_SPEND_RATE}
            CATASTROPHE_RATE = {CATASTROPHE_RATE}
        """)

    for re in RE:
        for w in W:
            logging.info(f"Simulating with w: {w} and refill_excess: {re}*E\n")
            fname = f"Results/O0-W{w}-RE{re}E-Report"
            start_block = 200000
            end_block = 681000
            sim = Simulation(
                int(N_STK),
                int(N_MAN),
                HIST_CSV,
                RESERVE_STRAT,
                ESTIMATE_STRAT,
                int(O_VERSION),
                int(I_VERSION),
                int(EXPECTED_ACTIVE_VAULTS),
                int(re*EXPECTED_ACTIVE_VAULTS),
                int(REFILL_PERIOD),
                int(DELEGATION_PERIOD),
                float(INVALID_SPEND_RATE),
                float(CATASTROPHE_RATE),
                with_balance=True,
                with_vault_excess=True,
                with_cum_op_cost=True,
                with_overpayments=True,
            )
            sim.refill_excess = re*sim.expected_active_vaults
            sim.wt.O_0_factor = w
            try:
                sim.run(start_block, end_block)
                report = sim.plot(output=fname)
                logging.info(f"Report\n{report}")

                with open(f"{fname}.txt", "w+", encoding="utf-8") as f:
                    f.write(report)
            except(RuntimeError):
                logging.info(f"RuntimeError: Simulating with w: {w} and refill_excess: {re}*E failed.")


def study_cf_outputs_1(range_factor, range_refill_excess):
    N_STK = 5
    N_MAN = 5
    HIST_CSV = "historical_fees.csv"
    RESERVE_STRAT = "CUMMAX95Q90"
    ESTIMATE_STRAT = "ME30"
    O_VERSION = 1
    I_VERSION = 1
    EXPECTED_ACTIVE_VAULTS = 20
    REFILL_PERIOD = 144*7
    DELEGATION_PERIOD = 144
    INVALID_SPEND_RATE = 0.1
    CATASTROPHE_RATE = 0.005

    logging.info(f"""Using config:\n
            N_STK = {N_STK}
            N_MAN = {N_MAN}
            HIST_CSV = {HIST_CSV}
            RESERVE_STRAT = {RESERVE_STRAT}
            ESTIMATE_STRAT = {ESTIMATE_STRAT}
            O_VERSION = {O_VERSION}
            I_VERSION = {I_VERSION}
            EXPECTED_ACTIVE_VAULTS = {EXPECTED_ACTIVE_VAULTS}
            REFILL_PERIOD = {REFILL_PERIOD}
            DELEGATION_PERIOD = {DELEGATION_PERIOD}
            INVALID_SPEND_RATE = {INVALID_SPEND_RATE}
            CATASTROPHE_RATE = {CATASTROPHE_RATE}
        """)

    for re in RE:
        for m in M:
            logging.info(f"Simulating with m: {m} and refill_excess: {re}*E\n")
            fname = f"Results/O1-M{m}-RE{re}E-Report"
            start_block = 200000
            end_block = 681000
            sim = Simulation(
                int(N_STK),
                int(N_MAN),
                HIST_CSV,
                RESERVE_STRAT,
                ESTIMATE_STRAT,
                int(O_VERSION),
                int(I_VERSION),
                int(EXPECTED_ACTIVE_VAULTS),
                int(re*EXPECTED_ACTIVE_VAULTS),
                int(REFILL_PERIOD),
                int(DELEGATION_PERIOD),
                float(INVALID_SPEND_RATE),
                float(CATASTROPHE_RATE),
                with_balance=True,
                with_vault_excess=True,
                with_cum_op_cost=True,
                with_overpayments=True,
                with_fb_coins_dist=True,
            )
            sim.refill_excess = re*sim.expected_active_vaults
            sim.wt.O_1_factor = m
            try:
                sim.run(start_block, end_block)
                report = sim.plot(output=fname)
                logging.info(f"Report\n{report}")

                with open(f"{fname}.txt", "w+", encoding="utf-8") as f:
                    f.write(report)
            except(RuntimeError):
                logging.info(f"RuntimeError: Simulating with m: {m} and refill_excess: {re}*E failed.")


def study_cf_inputs(range_E, range_RE, O_version):
    N_STK = 5
    N_MAN = 5
    HIST_CSV = "historical_fees.csv"
    RESERVE_STRAT = "CUMMAX95Q90"
    ESTIMATE_STRAT = "ME30"
    REFILL_PERIOD = 144*7
    DELEGATION_PERIOD = 144
    INVALID_SPEND_RATE = 0.1
    CATASTROPHE_RATE = 0.005

    logging.info(f"""Using config:\n
            N_STK = {N_STK}
            N_MAN = {N_MAN}
            HIST_CSV = {HIST_CSV}
            RESERVE_STRAT = {RESERVE_STRAT}
            ESTIMATE_STRAT = {ESTIMATE_STRAT}
            O_VERSION = {O_version}
            REFILL_PERIOD = {REFILL_PERIOD}
            DELEGATION_PERIOD = {DELEGATION_PERIOD}
            INVALID_SPEND_RATE = {INVALID_SPEND_RATE}
            CATASTROPHE_RATE = {CATASTROPHE_RATE}
        """)

    for re in range_RE:
        for E in range_E:
            for I_version in [0, 1, 2]:
                logging.info(f"""Simulating with:
                    I_version {I_version}
                    refill_excess: {re}*E
                    expected_active_vaults: {E}""")
                fname = f"Results/I{I_version}-E{E}-RE{re}E-O{O_version}-Report"
                start_block = 200000
                end_block = 681000
                sim = Simulation(
                    int(N_STK),
                    int(N_MAN),
                    HIST_CSV,
                    RESERVE_STRAT,
                    ESTIMATE_STRAT,
                    int(O_version),
                    int(I_version),
                    int(E),
                    int(re*E),
                    int(REFILL_PERIOD),
                    int(DELEGATION_PERIOD),
                    float(INVALID_SPEND_RATE),
                    float(CATASTROPHE_RATE),
                    with_balance=True,
                    with_vault_excess=True,
                    with_cum_op_cost=True,
                    with_overpayments=True,
                )
                sim.wt.O_1_factor = 1.75  # FIXME use best value
                sim.wt.O_0_factor = 6  # FIXME use best value
                try:
                    sim.run(start_block, end_block)
                    report = sim.plot(output=fname)
                    logging.info(f"Report\n{report}")

                    with open(f"{fname}.txt", "w+", encoding="utf-8") as f:
                        f.write(report)
                except(RuntimeError):
                    logging.info(f"""Simulating FAILED with:
                    I_version {I_version}
                    refill_excess: {re}*E
                    expected_active_vaults: {E}""")


if __name__ == "__main__":
    random.seed(21000000)
    logging.basicConfig(level=logging.DEBUG)

    W = [2,3,4,5,6]
    RE = [3,5,7]
    study_cf_outputs_0(W,RE)

    M = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2]
    RE = [3, 5, 7]
    study_cf_outputs_1(M, RE)

    E = [5,10,50]
    RE = [3,5,7]
    O_version = 1
    study_cf_inputs(E, RE, O_version)
