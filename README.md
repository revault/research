# Watchtower Wallet and Operational Environment Simulator

In a Revault deployment a watchtower (WT) is relied upon to enforce restrictive policies on funds which have been delegated to managers by stakeholders. To compute the operational costs and identify risks we have constructed a model of the WT wallet and its operating environment, and we simulate this for various types of Revault user.

The WT wallet model is a statemachine with a few atomic state transitions. Readers are referred to the research paper for a detailed discussion. 

# Running the simulator

The simulator requires all of the following environment variables to be set. 

| ENV VAR | Meaning | Value type |
| --- | --- | --- |
| PLOT_FILENAME | Name of graphical plot of results | `str` |
| REPORT_FILENAME | Name of text-based results report | `str` |
| N_STK | Number of stakeholders | `int in (1,10)` |
| N_MAN | Number of managers | `int in (1,10)` |
| LOCKTIME | Relative locktime length for unvault | `int > 0` |
| RESERVE_STRAT | Strategy for defining feerate reserve per vault | `CUMMAX95Q90` or `CUMMAX95Q1` |
| ESTIMATE_STRAT | Fall-back strategy for fee-rate estimation if estimateSmartFee fails | `ME30` 
| I_VERSION | Input selection version for consolidate-fanout transaction |`0`, `1`, `2` or `3`|
| CANCEL_COIN_SELECTION | Coin selection version for cancel transaction |`0` or `1`|
| HIST_CSV | Path to fee history data | `../block_fees/historical_fees.csv` |
| NUMBER_VAULTS | Number of vaults to initialize simulation with | `int in (1,500)`|
| REFILL_EXCESS | Excess number of vaults to prepare for with each refill | `int` |
| REFILL_PERIOD | Interval between refill attempts | `int > 144` |
| DELEGATE_RATE | Probability per day to trigger new vault registration | `float in (0,1)`|
| UNVAULT_RATE | Probability per day to trigger an unvault | `float in (0,1)`|
| INVALID_SPEND_RATE | Probability per unvault to trigger a cancel instead of a spend | `float in (0,1)`|
| CATASTROPHE_RATE | Probability per block to trigger a catastrophe| `float in (0,1)`|

So as an example, one can simulate by navigating to `Model` and running the command:

`PLOT_FILENAME=plot REPORT_FILENAME=out LOCKTIME=24 N_STK=5 N_MAN=3 RESERVE_STRAT=CUMMAX95Q90 ESTIMATE_STRAT=ME30 I_VERSION=3 HIST_CSV=../block_fees/historical_fees.csv NUMBER_VAULTS=5 REFILL_EXCESS=1 REFILL_PERIOD=2016 DELEGATE_RATE=6 UNVAULT_RATE=6 INVALID_SPEND_RATE="0.01" CATASTROPHE_RATE="0.0001" CANCEL_COIN_SELECTION=1 python3 main.py`

To configure which results to plot, you can set the following variables in the `Simulation` instance constructed in `main.py`:

```python
        with_balance=False,
        with_divergence=False,
        with_op_cost=False,
        with_cum_op_cost=False,
        with_overpayments=False,
        with_risk_status=False,
        with_risk_time=False,
        with_fb_coins_dist=False,
```

Note that at least two result types are necessary to plot. 

To configure the simulation to run with or without a fixed scale (number of vaults), then set the `with_scale_fixed` variable in the `Simulation` instance constructed in `main.py`. When `with_scale_fixed=True` the `DELEGATE_RATE` environment variable is not used. In that case, there is always a new vault registration for each unvault. `DELEGATE_RATE` is only used when `with_scale_fixed=False` to simulate a more dynamic and realistic operation. 