#!/usr/bin/env bash

# This instanciates an OTC desk owned by 5 stakeholders and managed by 3 of them using a 2of3 multisig.
# The managers batch payouts every evening before leaving the office, allowing for a large 6-hours Unvault timelock.
# Stakeholders check their watchtower(s) for refill every month.
# The spending policies in place are clear (no spendings outside business hours, up to 2 unvaults per day) and therefore the
# rate of invalid unvault attempts is very low (around once every 3 months).
# We generate comparative vaults for a study on keeping a constant pool of vaults at the value of 2, 4, 8 and 16.
# We simulate in parallel on 3 CPUs at a time, with 10 repeats of the simulation per core.

N_STK=5 \
N_MAN=3 \
LOCKTIME=36 \
REFILL_PERIOD=$((144 * 31)) \
UNVAULT_RATE="0.71" \
INVALID_SPEND_RATE="0.01" \
LOG_LEVEL="ERROR" \
NUM_CORES="3" \
REPEATS_PER_CORE="10" \
STUDY_TYPE="NUMBER_VAULTS" \
VAL_RANGE='[2,4,8,16]' \
python3 results.py
