# Watchtower Wallet and Operational Environment Simulator

In a Revault deployment a watchtower (WT) is relied upon to enforce restrictive policies on funds which have been delegated to managers by stakeholders. To compute the operational costs and identify risks we have constructed a model of the WT wallet and its operating environment, and we simulate this for various types of Revault user.

The WT wallet model is a statemachine with a few atomic state transitions. Readers are referred to the research paper for a detailed discussion. 

The WT achieves its goal to enforce policies by broadcasting a cancel transaction when an unvault attempt breaks the policy. The WT must pay the transaction fee for the cancel transaction at the time of broadcast. The cancel transaction is prepared by stakeholders with `ANYONECANPAY|ALL` signatures to enable the WT to add inputs to bump the transaction fee. The WT must maintain a pool of confirmed coins in order to accurately supplement cancel transaction fees, and it manages its pool of coins with self-paying consolidate-fanout transactions. The operational costs for a WT thus consist of cancel transaction fees, consolidate-fanout transaction fees, and refill transaction fees (paid by the operator).

# Running the simulator

The simulator can be configured by setting the following environment variables. 

| ENV VAR | Meaning | Value type |Default value
| --- | --- | --- |
| PLOT_FILENAME | Name of graphical plot of results | `str` | `None`|
| REPORT_FILENAME | Name of text-based results report | `str` |`None`|
| N_STK | Number of stakeholders | `int in (1,10)` |`7`|
| N_MAN | Number of managers | `int in (1,10)` |`3`|
| LOCKTIME | Relative locktime length for unvault | `int > 0` |`24`|
| RESERVE_STRAT | Strategy for defining feerate reserve per vault | `CUMMAX95Q90` or `CUMMAX95Q1` |`CUMMAX95Q90`|
| ESTIMATE_STRAT | Fall-back strategy for fee-rate estimation if estimateSmartFee fails | `ME30` or `85Q1H`| `85Q1H`|
| I_VERSION | Input selection version for consolidate-fanout transaction |`0`, `1`, `2` or `3`|`3`|
| CANCEL_COIN_SELECTION | Coin selection version for cancel transaction |`0` or `1`|`1`|
| HIST_CSV | Path to fee history data | `str` | `../block_fees/historical_fees.csv`|
| NUMBER_VAULTS | Number of vaults to initialize simulation with | `int in (1,500)`|`20`|
| REFILL_EXCESS | Excess number of vaults to prepare for with each refill | `int` |`2`|
| REFILL_PERIOD | Interval between refill attempts | `int > 144` |`1008`|
| DELEGATE_RATE | Probability per day to trigger new vault registration | `float in (0,1)`|`None`|
| UNVAULT_RATE | Probability per day to trigger an unvault | `float in (0,1)`|`1`|
| INVALID_SPEND_RATE | Probability per unvault to trigger a cancel instead of a spend | `float in (0,1)`|`0.01`|
| CATASTROPHE_RATE | Probability per block to trigger a catastrophe| `float in (0,1)`|`0.001`|

If `DELEGATE_RATE` is not set, the simulation will run at a fixed scale where there is a new vault registration for each unvault. If `DELEGATE_RATE` is set the simulation will register new vaults stochastically, simulating a more dynamic and realistic operation. 

So as an example, one can simulate by navigating to `Model` and running the command:

`PLOT_FILENAME=plot REPORT_FILENAME=out LOCKTIME=12 N_STK=5 N_MAN=2 RESERVE_STRAT=CUMMAX95Q90 ESTIMATE_STRAT=ME30 I_VERSION=3 HIST_CSV=../block_fees/historical_fees.csv NUMBER_VAULTS=5 REFILL_EXCESS=1 REFILL_PERIOD=2016 DELEGATE_RATE=6 UNVAULT_RATE=6 INVALID_SPEND_RATE="0.01" CATASTROPHE_RATE="0.0001" CANCEL_COIN_SELECTION=1 python3 main.py`

To control which results to plot, you can set the following environment variables:

| ENV VAR | Plot content | Value type |Default value
| --- | --- | --- | --- |
|BALANCE|total balance, un-allocated balance, required reserve against time|boolean|True|
|DIVERGENCE| minimum, maximum and mean divergence of vault reserves from requirement|boolean|False|
|CUM_OP_COST|cumulative operation cost for cancel, consolidate-fanout and refill transactions|boolean|True|
|RISK_TIME|highlights CUM_OP_COST plot with time-at-risk|boolean|True|
|OP_COST| cost per operation for cancel, consolidate-fanout and refill transactions|boolean|False|
|OVERPAYMENTS|cumulative and individual cancel transaction fee overpayments compared to current fee-rate estimate|boolean|False|
|RISK_STATUS||risk coefficient against time|boolean|False|
|FB_COINS_DIST|coin pool distribution (sampled every 10,000 blocks)|boolean|False|

Note that at least two plot types are required to run the simulation. 