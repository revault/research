# Watchtower Wallet and Operational Environment Simulator

In a [Revault deployment](https://github.com/revault/practical-revault) a watchtower (WT) is relied upon to enforce restrictive policies on funds which have been delegated to managers by stakeholders. To compute the operational costs and identify risks we have constructed a model of the WT wallet and its operating environment, and we simulate this for various types of Revault user.

The WT wallet model is a statemachine with a few atomic state transitions. Readers are referred to the research paper for a detailed discussion.

The WT achieves its goal to enforce policies by broadcasting a cancel transaction when an unvault attempt breaks the policy. The WT must pay the transaction fee for the cancel transaction at the time of broadcast. The cancel transaction is prepared by stakeholders with `ANYONECANPAY|ALL` signatures to enable the WT to add inputs to bump the transaction fee. The WT must maintain a pool of confirmed coins in order to accurately supplement cancel transaction fees, and it manages its pool of coins with self-paying consolidate-fanout transactions. The operational costs for a WT thus consist of cancel transaction fees, consolidate-fanout transaction fees, and refill transaction fees (paid by the operator).

Note that although the [cancel transaction](https://github.com/revault/practical-revault/blob/master/transactions.md#cancel_tx)
is in practice pre-signed with a `88sat/vb` feerate the simulation assumes it
needs to pay for the whole fee, as a worst case scenario.

# Running the simulator

## Configuration

The simulator can be configured by setting the following environment variables. 

| ENV VAR | Meaning | Value type | Default value |
| --- | --- | --- | --- |
| PLOT_FILENAME | Name of graphical plot of results | `str` | `None`|
| REPORT_FILENAME | Name of text-based results report | `str` |`None`|
| N_STK | Number of stakeholders | `int in (1,10)` |`7`|
| N_MAN | Number of managers | `int in (1,10)` |`3`|
| LOCKTIME | Relative locktime length for unvault | `int > 0` |`24`|
| RESERVE_STRAT | Strategy for defining feerate reserve per vault | `CUMMAX95Q90` or `CUMMAX95Q1` |`CUMMAX95Q90`|
| FALLBACK_EST_STRAT | Fall-back strategy for fee-rate estimation if estimateSmartFee fails | `ME30` or `85Q1H`| `85Q1H`|
| CF_COIN_SELECTION | Coin selection version for consolidate-fanout transaction |`0`, `1`, `2` or `3`|`3`|
| CANCEL_COIN_SELECTION | Coin selection version for cancel transaction |`0` or `1`|`1`|
| HIST_CSV | Path to fee history data | `str` | [`block_fees/historical_fees.csv`](block_fees/historical_fees.csv) |
| NUMBER_VAULTS | Number of vaults to initialize simulation with | `int` in `(1,500)`|`20`|
| REFILL_EXCESS | Excess number of vaults to prepare for with each refill | `int` |`2`|
| REFILL_PERIOD | Interval between refill attempts | `int > 144` |`1008`|
| DELEGATE_RATE | Probability per day to trigger new vault registration | `float` in `(0,1)`|`None`|
| UNVAULT_RATE | Probability per day to trigger an unvault | `float` in `(0,1)`|`1`|
| INVALID_SPEND_RATE | Probability per unvault to trigger a cancel instead of a spend | `float` in `(0,1)`|`0.01`|
| CATASTROPHE_RATE | Probability per block to trigger a catastrophe| `float` in `(0,1)`|`0.001`|

If `DELEGATE_RATE` is not set, the simulation will run at a fixed scale where there is a new vault registration for each unvault. If `DELEGATE_RATE` is set the simulation will register new vaults stochastically, simulating a more dynamic and realistic operation. 

To control which results to plot, you can set the following environment variables:

| ENV VAR | Plot content | Value type | Default value |
| --- | --- | --- | --- |
|PLOT_BALANCE|total balance, un-allocated balance, required reserve against time|`0` or `1`|`1`|
|PLOT_DIVERGENCE| minimum, maximum and mean divergence of vault reserves from requirement|`0` or `1`|`0`|
|PLOT_CUM_OP_COST|cumulative operation cost for cancel, consolidate-fanout and refill transactions|`0` or `1`|`1`|
|PLOT_RISK_TIME|highlights cumulative operations cost plot with time-at-risk|`0` or `1`|`0`|
|PLOT_OP_COST| cost per operation for cancel, consolidate-fanout and refill transactions|`0` or `1`|`0`|
|PLOT_OVERPAYMENTS|cumulative and individual cancel transaction fee overpayments compared to current fee-rate estimate|`0` or `1`|`0`|
|PLOT_RISK_STATUS||risk coefficient against time|`0` or `1`|`0`|
|PLOT_FB_COINS_DIST|coin pool distribution (sampled every 10,000 blocks)|`0` or `1`|`0`|

Note that at least two plot types are required to run the simulation.

## Dependencies

We use [`pandas`](https://pandas.pydata.org/) for data analysis and [`matplotlib`](https://matplotlib.org/) for plotting the results.
Before running the simulation, you need to install these dependencies.
```
cd model/
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

## Examples

You can run the `main.py` script with the defaults (by not specifying a configuration) or try one
of the available examples. For instance the `otc_desk.sh` script simulates the usage of Revault for
a typical bitcoin retailer:
```
# From the `model` directory
./examples/otc_desk.sh
```

From there explore by tweaking the config as detailed above and report bugs/inconsistencies to our
[bug tracker](https://github.com/revault/watchtower_paper/issues)!


## Monte Carlo simulations

To get make a real comparison between factors that effect risk and operational costs, running a single simulation can be mis-leading as the variance between results can be quite large. To get the precision on comparative results, you can use the monte-carlo method. By varying the pseudo-random number generator seed, you can simulate "alternative histories" where the operational sequencing (which are triggered stochastically) is different. With `results.py` you can run `NUM_CORES` independent simulations in parallel with each on a separate CPU. You can also specify a number of repeats of the parallel simulations to run with `REPEATS_PER_CORE`. You must specify `STUDY_TYPE` as one of:

```
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
```

You must also set a range of values `VAL_RANGE` for the chosen study type. These environment variables are necessary in addition to a subset of those required to run `main.py`. The environment variables that correspond to valid study types must all be set (or else a default value is used). The plot environment variables aren't configurable with `results.py` because the intention is not to generate plots, but instead to aggregate results per value (in `VAL_RANGE`) and output the results into a file at `results/report_{STUDY_TYPE}_{value}-PRNG_{prng_seed}.csv`. These aggregate results can then be interpreted or plotted for interpretation. The data that is generated consists of the mean and standard deviation (std dev) of aggregate data over all simulations at a given value in `VAL_RANGE`.

|Data name|Meaning|
|---|---|
|mean_balance_mean|  mean of mean balances across the span of a simulation|
|mean_balance_std_dev|  std dev of mean balances across the span of a simulation|
|cum_ops_cost_mean|  mean cumulative operational costs|
|cum_ops_cost_std_dev|  std dev of cumulative operational costs|
|cum_cancel_fee_mean|  mean cumulative cancel transaction fees|
|cum_cancel_fee_std_dev|  std dev of cumulative cancel transaction fees|
|cum_cf_fee_mean|  mean cumulative consolidate-fanout transaction fees|
|cum_cf_fee_std_dev|  std dev of cumulative consolidate-fanout transaction fees|
|cum_refill_fee_mean|  mean cumulative refill transaction fees|
|cum_refill_fee_std_dev|  std dev of cumulative refill transaction fees|
|time_at_risk_mean|  mean of total time at risk during a simulation|
|time_at_risk_std_dev|  std dev of total time at risk during a simulation|
|mean_recovery_time_mean|  mean time to recover from an at-risk state|
|mean_recovery_time_std_dev|  std dev of time to recover from an at-risk state|
|median_recovery_time_mean|  mean of median of time to recover from an at-risk state|
|median_recovery_time_std_dev|  std_dev of median time to recover from an at-risk state|
|max_recovery_time_mean|  mean of max time to recover during a simulation|
|max_recovery_time_std_dev|  std dev of max time to recover during a simulation|
|delegation_failure_count_mean| mean of total number of delegation failures during a simulation|
|delegation_failure_count_std_dev|  std dev of total number of delegation failures during a simulation|
|delegation_failure_rate_mean|  mean of rate of delegation failures during a simulation (percentage)|
|delegation_failure_rate_std_dev|  std dev of rate of delegation failures during a simulation (percentage)|
|max_cancel_conf_time_mean| mean of maximum confirmation time for a cancel transaction during a simulation|
|max_cancel_conf_time_std_dev|  std dev of maximum confirmation time for a cancel transaction during a simulation|
|max_cf_conf_time_mean|  mean of maximum confirmation time for a consolidate-fanout transaction during a simulation|
|max_cf_conf_time_std_dev|  std dev of maximum confirmation time for a consolidate-fanout transaction during a simulation|
|max_risk_coef_mean|  mean of the maximum of the risk coefficient during a simulation|
|max_risk_coef_std_dev|  std dev of the maximum of the risk coefficient during a simulation|

An example of how to use this is given with the `monte_carlo.sh` script in the examples.

```
# From the `model` directory
./examples/monte_carlo.sh
```

Note that this could take a long time if `REPEATS_PER_CORE` is high.