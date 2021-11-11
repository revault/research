#!/usr/bin/env sh

# This is a copy of the OTC_DESK example with a much higher invalid spend rate (indicative
# of a misconfigured watchtower policy). Useful to manually test the efficiency of the
# Cancel coin selection!

NAME="MISCONFIGURED" \
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
INVALID_SPEND_RATE="0.1" \
CF_COIN_SELECTION=2 \
PLOT_BALANCE=1 \
PLOT_CUM_OP_COST=1 \
PLOT_RISK_TIME=1 \
PLOT_DIVERGENCE=1 \
PLOT_OVERPAYMENTS=1 \
PLOT_FB_COINS_DIST=1 \
python3 main.py
