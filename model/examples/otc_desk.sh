#!/usr/bin/env bash

# This instanciates an OTC desk owned by 5 stakeholders and managed by 3 of them using a 2of3 multisig.
# The managers batch payouts every evening before leaving the office, allowing for a large 6-hours Unvault timelock.
# Stakeholders check their watchtower(s) for refill every month.
# The spending policies in place are clear (no spendings outside business hours, up to 2 unvaults per day) and therefore the
# rate of invalid unvault attempts is very low (around once every 3 months).
# Stakeholders collectively delegate new vaults to managers on a weekly basis. We assume a starting pool of 20 vaults.

NAME="OTC_DESK" \
REPORT_FILENAME=$NAME"_report" \
PLOT_FILENAME=$NAME"_plot" \
N_STK=5 \
N_MAN=3 \
LOCKTIME=36 \
NUMBER_VAULTS=20 \
REFILL_EXCESS=5 \
REFILL_PERIOD=$((144 * 31)) \
UNVAULT_RATE="0.71" \
DELEGATE_RATE="0.71" \
INVALID_SPEND_RATE="0.01" \
PLOT_BALANCE=1 \
PLOT_CUM_OP_COST=1 \
PLOT_RISK_TIME=1 \
PLOT_DIVERGENCE=1 \
PLOT_OVERPAYMENTS=1 \
python3 main.py
