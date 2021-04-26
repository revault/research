# Watchtower

## Contents

- The backbone (bitcoind connection, messages handling)
- The stakeholder interaction (signature verification, storage, storage assurance, ACK, emergency trigger and response)
- Spend policy (API, enforcement)
- Fee-bumping wallet reserves management
- Fee-bumping algorithm
- Overall algorithm(s) for event based processing of all of the above
- Set-up procedure for the WT by the stakeholder
    
## Backbone

- Insert more formal description of watchtowerd and its interfaces (with bitcoind and plugins)
<!--The watchtower runs a full bitcoin node, and watchtowerd communicates with bitcoind through RPC to handle blockchain and wallet functionality. Watchtowerd handles communication with its stakeholder and with the coordinator. Watchtowerd will expose an interface that signals an event at .... 
There will be various plugins that use these events as hooks, do some computation and state transitions, and respond to the watchtower. For example, a spend policy plugin that handles volume constraints maintains a database of unvault and spend transactions and their date-times and amounts. Upon a block connection event it will scan for new unvault transactions, compute whether the amount during the last time window (say, a week) plus the amount for the new unvault transactions would exceeded the policy limit. If it does not exceed the limit, the plugin responds with a "checks passed" message, otherwise, it responds with a "checks failed" message. This design is modular and extensible, and allows specific deployments to operate with as many or as little WT features as desired. -->   

## Stakeholder interaction

### Routine interaction

Receive signature from stakeholder (Sig message)
- Reconstruction of revocation transaction(s)  (EmerTx, CancelTx, Unvault-EmerTx)
- Signature verification 
    - which of EmerTx, CancelTx, or Unvault-EmerTx -- if any -- is the signature for?
- Signature storage 
    - SQLite
    - What's the Schema?
    
Respond with SigAck message
- ack = true if verification and storage succeed
- else ack = false

### Emergency interaction    
    
- Listen on multiple channels (Noise over TCP, secure email, etc) with stakeholder 
    - Also monitoring blocks & mempool for other WTs' broadcasting emergency TXs
- Receive and verify emergency trigger message
- Respond to emergency by broadcasting all EmerTx and Unvault-EmerTx

### Re-fill WT wallet

Let's begin by specifying the onchain output descriptors which are used in a Revault deployment. The deposit descriptor, informally the "main wallet" which is "storing the co-owned coins". The fee-bumping descriptor, a single P2WPKH per watchtower which they can spend from in order to fee-bump revocation transactions. This descriptor is the one stakeholders need to pay to in order to "refill" their watchtower. The stakeholders' WTs each have their own wallet (software), the _WT wallet_, to unlock fee-bump outputs.

In order to function properly, a WT wallet must have access to a _nice_ (discussed in [Fee-bump output set structure]) utxo pool. Re-fill transactions create this pool, and themselves incur a small fee. 

There are two approaches to the problem of how to maintain sufficient balance in each of the WT wallets, distinguished by the source of funding. The WT wallet can either be funded from the main wallet, or by an external source. Initially, both approaches seem valid, but the following discussion shows how this choice effects the protection of assets, stakeholders' trust assumptions, the incentives of participants, and the user experience.

#### WT wallets funded from the main wallet

As the stakeholders are responsible for the secure operation of their own WT, they should ensure that the WT wallet is sufficiently funded. This means they must initiate the re-fill process. The process involves determining the balance and UTxO set of the WT wallet (from inspecting the blockchain-- to avoid unnecessary communication with the WT), constructing the Refill Tx(s) (which spends part of a deposit UTxO), adding their signature to the Tx(s), gathering consent and signatures from other stakeholders for the Refill Tx(s) (which may require them each verifying the initiating stakeholder's WT's wallet state), and broadcasting the Refill Tx. 

Note that most of the complexity of this process will be automated, and the user experience of this process will be something like:
- Notifaction that WT balance is low
- Click re-fill WT
- Plug in HW, and visually verify that the HW displays a reasonable Tx (amount, descriptor)
    if OK:
    - Click sign on the HW
    - Click "share signature with other stakeholders" [other stakeholder process begins]
    - Wait for other stakeholders
        if all accept:
        - Broadcast
        else:
        - stop? who denied? how to proceed?

[Other stakeholder process]:
- Notification that stakeholder _x_ is requesting re-fill for their WT
- click "accept" or "deny"
    if "accept":
    - plug in HW, and visually verify that the HW displays a reasonable Tx (amount, descriptor)
        if OK:
        - Click sign on the HW
        - Click share signature with other stakeholders
        else:
        - stop/ respond with NACK?
    if "deny": 
    - stop/ respond with NACK?

This could be a separate process from the routine signing, or it could be bundled into routine signing when one of the stakeholders requires it.

It should _not_ be classed as a 'bypass Tx', but should be specified as another type of revault Tx to avoid normalizing this type of transaction for stakeholders. 

Should there be some verification by each stakeholder that the WT wallet is running low? This amounts to an automated check using blockchain data and knowing the public key of the WT in question.
- Should the verification be carried out by stakeholders' laptops, or stakeholders' WTs? 
    - Stakeholders' laptops are less secure (are far easier to compromise), but this would then require no communication with the WT. 
    - If relying on the WTs to perform the verification, then it is more difficult to compromise, but it requires communication with the WT. In any case, if stakeholders' laptops are compromised, they may ACK a NACK from the WT and show the user that the verification passed, even if the WT were to sign their messages. 
- Can have both:
    - track watchtower's funds in the stakeholders's revaultd for the happy case (preventing new acknowledgements when funds are low)
    - watchtower refusing to ACK if funds are too low (should only happen if the wallet bypassed the restriction imposed by the wallet daemon, ie if they are running a custom software for some shady purpose)
- This is ok in the case where the fee market doesn't increase rapidly, causing the WT wallet to have insufficient funds without a change to the number of vaults being guarded. 


What countermeasures can be used to ensure that Re-fill Txs cannot be used to steel funds?
- Already depends on visual verification by ALL stakeholders
- Can be automatically checked by all stakeholders' wallet software (e.g. type-checked and/or amount checked)
- Can be automatically checked by all stakeholders' HW (e.g. type-checked and/or white-listed address checked)

What incentives are there for individual stakeholders with this setup?
- Since WTs are funded from co-owned funds, a stakeholder can set up a strict spend policy, and essentially have the other stakeholders pay for the enforcement of that policy even if their spend policy would not have dissallowed a spend. 
- Stakeholders are not incentivised to delay their WT's broadcast of Cancel Tx, since they are not paying for it out of their own pocket

Counter arguments to re-filling WT wallets directly from the main wallet:
- Funds on each WT wallet are negligible when compared with the funds at stake (e.g. a $100M in the vaults but only $1k in each WT wallet).
- Introducing complexity for this so small divergence of incentives is likely to do more harm than good (more complex software => more bugs, more complex routine => less user friendly, more reliance on human verification => less human verification after some time)
- This may not be feasible at all (low number of vaults)


#### WT wallets funded from an external source

To refill their WT wallets from an external source (e.g. buying BTC on an exchange), there should be a two transaction process. The withdrawal transaction from the exchange (or from their personal wallet, or otherwise), and the Refill Tx generating outputs for the WT wallet.
Either the stakeholders should each have their own individual _refill wallet_ to craft Refill Txs, or the WT should craft its own Refill Txs. This is necessary as an intermediate step (e.g. between the exchange and the WT wallet) to ensure that Refill Txs are constructed properly, with a well structured output set and with sufficient amounts. No guarantees can be made about the formatting of Txs from the external source, so this should not be relied upon. Also, the feebumping algorithm depends on _confirmed_ UTxOs, and so Refill Txs are broadcast in advanced. 
   
The main user experience benefit of this approach is that there needn't be agreement among stakeholders to refill a watchtower. Stakeholders act independently to secure the operation of their own WT. Roughly, the UX is like this:
    
[Refill wallet operated by Stakeholder]
- Notifaction that WT balance is low
If re-fill wallet is low:
    - Buy BTC on exchange or OTC
    - Withdraw BTC to refill wallet
- Click re-fill WT
- Plug in HW, and visually verify that the HW displays a reasonable Tx (amount, descriptor)
    if OK:
    - Click sign on the HW
    - Broadcast
    else:
    - stop? how to proceed? Laptop compromised

[WT crafts its own refill Txs]
- Notifaction that WT balance is low
- Buy BTC on exchange or OTC
- Withdraw BTC to WT wallet
- Notification that WT balance is refilled
        

    
The case where the refill wallet is operated by the stakeholder has worse UX (more steps). 
Directly filling the WT wallet is much simpler, as the Refill Txs can be automated. 

In terms of the security of funds in transit, the stakeholder-operated refill wallet approach relies on both the HW and the WT wallet not being compromised, whereas the WT wallet approach doesn't rely on the stakeholders HW being secure. 

There is a risk that the stakeholder is unable to aquire BTC from an external source. They should have multiple options. If they fail to refill the WT wallet, they will eventually fail in enforcing their own spend policy. Moreover, the "WT wallet low" warning notification comes before a "WT wallet cannot accept more delegations" type of message. 
    
From a whole system point of view, there is an incentive for stakeholders to be greedy with respect to WT fees. This could mean delaying the broadcast of Cancel Txs, to allow another stakeholder to incurr the cost. We expect the nominal amount of bitcoin to be reserved for WT wallets to be negligible compared to the amount at stake within the main wallet. Therefor this slight incentive to be greedy should not be a major concern. It can be alleviated by the company compensating stakeholders. 
    

Concluding remarks:
- The precident of stakeholders signing 'bypass-like' Txs does not exist if WT wallets are funded from an external source.
- Coupling the security of the protocol with the availability of external sources of BTC might be unwise, unless the stakeholders ensure they have multiple options
- The user experience of buying BTC from an external source intuitively seems better than requiring multi-party signing routine

  
## Spend Policy 

### API

We need a policy API. Some policies may refer to explicit bitcoin policy descriptors, some to blockchain data (e.g. height) and others may refer to data external to the blockchain. The risk to successful enforcement of particular spend policy depends on the reliability of the required data source.
- What external data oracles are necessary and reliable? 
- Note that reliance on external data oracles has a larger attack surface than Bitcoin network & blockchain data, given the extensive verifiability of blockchain data associated with proof-of-work.

The stakeholders will specify spending policies that watchtowers enforce. These policies may contain arbitrary combinations of constraints: 
- white-listed or black-listed xpubs, public keys, or addresses
- amount, frequency or volume constraints
- time constraints
- ...? 

### Enforcement

Once specified in an appropriate set of API calls, the watchtower (watchtowerd + plugins) must be able to interpret the policy and enforce its constraints. Through monitoring blocks, its mempool, internal revault network events (Spend Tx advertisements and Stakeholder Sig messages), and any external channels with data oracles, the watchtower must determine whether a Spend Tx adheres to the policy or not. If the Spend Tx adheres to the policy, then the watchtower doesn't need to respond. Otherwise, it responds by broadcasting the associated Cancel Tx.

## Fee-bumping wallet reserves management

A watchtower needs to be prepared for the situation where the fee of the revocation transaction (or set of revocation transactions) that it will broadcast is inadequate given the state of the fee market. For this reason, signatures given for the revocation transactions use SIGHASH_ANYONECANPAY | SIGHASH_ALL flags, allowing additional inputs to be added before broadcasting the transaction, increasing the fee. In preparation, Feebump Txs are signed, broadcast and confirmed, with outputs that are only consumable by the Watchtower. The output amounts should be varied, distributed in such a way that there is _always_ enough for any fee-market state, but (secondary) without overspending. 

To achieve this, several algorithms are required to:
1. Estimate fee-reserve per "vault UTxO" [def 1]
2. Produce fee-bump outputs with 'well-structured' distribution of amounts (e.g. 10, 20, 50, 100, 200, 500, 1000, 5000 sats)
3. Estimate fee-bump amount at spend time (using bitcoin core, or modified bitcoin core?)
4. Coin-selection algorithm (using bitcoin core?)
5. Bump the fee if the initial estimation failed.

[def 1] vault UTxO means either a deposit UTxO locked to the deposit descriptor or an unvault UTxO locked to the unvault descriptor

Before discussing the details of these algorithms, let's consider their use at a higher level. Given an estimate of the fee-reserve per revocation transaction (which is dependent on the transaction type) we can compute the _total revocation reserve_. This is the total amount kept in reserve by a WT to perform its functions of cancelling transactions and enacting the emergency procedure. The total revocation reserve is a conservative amount that is beyond what is expected to be used. It is computed by the WT upon each block connection and by the stakeholders' wallet to notify the stakeholder when a re-fill is necessary. In addition to maintaining sufficient total revocation reserve, the amount should be distributed appropriately across a number of fee-bump outputs with varying amounts. A 'well-structured' fee-bump output set helps in bumping-fees with precision, without overspending. The WT will use a fee-estimation algorithm at the time of broadcast, an algorithm that takes into account the recent history of the fee-market state and estimates a conservative fee value to ensure that the transaction will be confirmed within X blocks. Here the bitcoin-core algorithm will be used. Given an estimate for the fee-bump amount, the coin-selection algorithm will select which fee-bump outputs to use [bitcoin core presumably not optimal here]. Finally, if the initial estimation is insufficient (e.g. the transaction was not included by the target block), the transaction fee should be bumped. In the case of a C, the Tx must be mined before the U CSV. In the case of an E or UE, the Tx fee may be bumped by other WTs [Explore feasibility of this mechanism]. 

Note that while feebumping is expected for Cancel Txs (which may happen regularly), it is not relied upon nor expected for Emergency Txs. Requiring each WT to hold a fee-reserve for each Emergency Tx is extremely expensive when compared to preparing the emergency Tx with very high fees during the initial signing phase. This becomes more apparent as the system size scales up (more watchtowers) and the number of vault UTxOs increases. The scaling relation is `total emergency fee-reserve ~ number of watchtowers * number of vault UTxOs`.

### Total Revocation Reserve

Let V(t) be the set of vault UTxOs at time t (as a blockheight). Let the elements of V be uniquely labelled by an integer, i. 
```
    V(t) = {v_1, v_2, ..., v_i, ...}
```
The elements v_i may refer to either deposit UTxOs or unvault UTxOs. Let the fee-reserve per vault UTxO at time t be r(v_i, t).

The rationale for maintaining a fee-reserve _PER VAULT_ is that otherwise, WTs could be abused by rogue managers triggering invalid spend attempts until the fee-reserves are drained, and then bypass the spend policy enforcement with a fraudulent Spend Tx. WTs must be able to respond to a mass broadcast of invalid spend attempts for each vault.  

Note that r(v_i, t) is a function of the max weight of the Cancel Tx, w_C. w_C is fixed per deployment, based on the participant set, and it refers to the largest possible Cancel Tx weight that can be relayed, e.g. with the max number of feebump inputs, which is about 100k WU. It is also a function of the _feerate-reserve estimate_, f_C(t), which is determined from analysing the fee market with a given strategy. f_C(t) is frequently recomputed. For example, if we decide on a strategy that uses a 30 day moving average fee-rate (sats/WU), the fee-reserve per vault UTxO is given by
```
    r(v_i, t) = f_C(t)*w_C = 30 moving average * w_C
```
Let the _total revocation reserve_, R, be
```
    R(t) = SUM_{v_i \in V(t)}(r(v_i, t))
```
This is total value of bitcoin required by a WT to be able to enforce it's spending policy. It's the sum over all v_i in V(t) of r(v_i, t). 

Next, consider how V(t), and thus R(t), changes in time. A new element is added to V when a new deposit UTxO is confirmed. An element v_j that initially refers to a deposit UTxO remains in V if that deposit UTxO is consumed by an unvault transaction, but from then on refers to the new unvault UTxO.  An element is removed from V whenever a C, E, UE, or S is confirmed.

\\ Feebump outputs pay to 1-of-W WTs where each of the WTs is controlled by the same stakeholder \\

This feature would help increase resilience of the stakeholder's WT without increasing the `R(t)`, since only one WT is required to and is able to spend the same feebump outputs.

### Fee-reserve estimation in Revault 

GOAL: Maintain sufficient reserves for spend policy enforcement and emergency scenario 

\\ Initial notes and questions \\

- Numerical analysis of the historical fee-market to determine what fee/byte is appropriate
    - blockchain data
    - mempool data?
    - price/ price volatility data
- Relevant statistics
    - Avg can be highly skewed
    - Median (50%-ile) less skewed by outliers, but perhaps its not conservative enough to target top 50%-ile
    - Outlier detection & exclusion (e.g. truncation or winsorising) or estimate value based on 95% to 100%-ile of successful historic fee values
- Time-frame
    - Consider only data since SegWit activation date? 
    - Use similar moving average exponential decay as bitcoin-core estimateSmartFee?
    - Use worst months in history?

- Multiply the reserves for the expected fee-value as a buffer against extreme volatility [double? triple?]

- How regularly do we want to perform the analysis? 
    - Frequent re-calculation provides cover in case the fee-market trends upwards over time and the reserve level needs to increase to maintain the buffer
    - If it's found that the fee-market has trended upwards, notify stakeholder that reserves are low
    - Performing analysis continually is fine, but want to avoid over-sensitivity to single blocks with spiked fees. The estimation should be relatively smooth to avoid over-burdening the stakeholders with warnings.

#### Data sources and risks

Blockchain data
    - Query bitcoind and construct a data set (RPC/ZMQ)
    - External data source (no need)
Mempool data
    - Gather mempool data over time to construct an in-house data set (RPC/ZMQ)
    - External data source for historic mempool data (e.g. blockchain.info, bitcoin_logger [https://bitcoindevkit.org/blog/2021/01/fee-estimation-for-light-clients-part-1/])
    - What data does bitcoind store for its estimateSmartFee? counts in buckets & targets. 
        - Does it store this data? Is it persisted through crash/ restart?
        - Can I get GET this data? (not with current RPC. Maybe with a new RPC?) 
Price/ Volatility data
    - External data source(s!) for historic _and current_ price
    
- What is the risk of using blockchain data for the fee-reserve estimation? 
    - Hypothetical attack: a miner wants to drive up the fee-reserve of revault-deployments, and so they mine blocks which pay themselves huge fees, skewing the estimation to be much higher. They do not need to broadcast the block they mine to the mempool first, and can avoid accidentally paying the fees to another miner. This doesn't benefit the miner. The estimated fee at spend time _is_ a function of mempool data, which is not abusable by miners. The miner gains nothing from driving up the fee-reserve estimation of watchtowers. 
    - Hypothetical attack: a competitor custodian wants to drive up the fee-reserve of revault-deployments, and so they bribe a miner to perform the above attack. The consequent effect on the fee-reserve estimate depends on the actual fee-reserve estimation algorithm. Criteria for an algorithm to be resistant to such an attack: 
        1. Large data set (attackers must bribe miners to skew the fees of _many_ blocks to have significant effect)
        2. Outlier detection and exclusion
    - Note also that if an attacker invests significant resources into this attack, their attempt could always be thwarted by adjusting the alorithm by increasing its resistance or by using an alternative algorithm that includes mempool data. In this case their investment into the attack is wasted.
        
- What is the risk of using external data source for _live_ data aquisition (price/ price volatility)?
    - Oracle problem; relying on data feed from several sources and doing consistency validation is tricky
    
- What is the risk of local mempool tampering for our bitcoin node?
    - Must rely on the mempool for fee estimation (if using estimateSmartFee)
    - Connected to large and diverse set of peers, risk is minimised.
    - However, eclipse attacks are possible 
    
- What is the risk of using external data source for _historic_ data aquisition (price/ price volatility/ mempool data)?
    - Data can be scrutinized before being used, so tampering can be countered.
    - Process could be time-consuming and difficult to automate. Might be valid as a one-time approach. Resultant data must be managed securely.
    
Since the WT needs to continually generate fee-reserve estimates to keep up-to-date with fee-market, the estimation algorithm needs live and historic data as input. This almost definitely rules out the use of price/ price volatility data as the reconciliation process for a multi-oracle data source would be a pain-in-the-ass. Given that the watchtower operates a full bitcoin node, the blockchain data is given, and live mempool data is given. Historic mempool data _should_ be possible to gather and validate.
    
#### Data analysis

The WT needs a systematic way of collecting data and making estimations of required fee-reserves. The simplest approach would rely on blockchain data only. A more complicated approach would integrate both blockchain and historic mempool data (as in bitcoin core's estimateSmartFee). In some deployment contexts it could be feasible to correlate the fee-reserve estimate with a "target" (an expected number of blocks before being mined) of a day (144 blocks) or a week (1008 blocks). In other contexts, the target could be unkown, and too large for it to be useful at all. 

A simple initial algorithm that applies generically, then, would be based only on blockchain data. Here we have the actual fee rate of every tx in the complete history of the blockchain. 

Then, how should we process the data and perform the analysis? 

We may wish to define some criteria by which to sanitize this data;
- Determine an appropriate time-frame
    - Should txs before and after SegWit be considered equal?
    - Should newer txs be weighted more heavily? 
- What are characteristics of "outliers"? Should we exclude these from the data, or choose statistics that are naturally resistant to them even when included?
- We would like for the estimate to be relatively smooth; a volatile estimator would lead to bad UX through requiring high frequency of re-fills or confusing the stakeholder by generating warnings one day and being OK the next.

Idea #1: define the 95%-ile of fee-rates for a rolling window over the previous 30 days, call it 95Q30. Anything above that stands a strong chance of being accepted, unless there is high volatility in the fee-market. In which case, does it make sense to double the MA95 and use that as the fee-reserve estimate? 
    - A time-frame of 30 days is more characteristic of the recent fee-market than of historic high-fee periods. 
    - by choosing a lower bound of 95%-ile, the top 5%-ile of "extreme outliers" are excluded
    - An average over 30 days would be smooth

Idea #2: Consider the month(s) which historically showed the highest demand for block-space, and determine the 95%-ile fee-rates for that period. Call it MAX95Q.
    - This time-frame represents the "worst" that the fee-rate market has ever been. If our algorithm would have performed well then, it will likely perform well now. 
    - If the recent month becomes the "worst" then the estimator adjusts to a new time-frame. If the fee-market subsequently calms down, the fee-estimator stays the same. 
    - by choosing a lower bound of 95%-ile, the top 5%-ile of "extreme outliers" are excluded
    - The estimator is only volatile if the fee-market is growing constantly, in which case the estimatore _should_ be adjusting.  
    
### Fee-bump estimation at spend time

#### bitcoin core

[1] https://gist.github.com/morcos/d3637f015bc4e607e1fd10d8351e9f41
[2] https://blog.iany.me/2020/08/bitcoin-core-fee-estimate-algorithm/
[3] https://bitcointechtalk.com/whats-new-in-bitcoin-core-v0-15-part-2-41b6d0493136

GOAL: Be as conservative and useful for the average person as possible

- Not forward looking
- Historic mempool data
    - using this mitigates miner manipulation of fees, since to manipulate fees they must broadcast transactions to the mempool
    - buckets (exponentially spaced) and targets (blocks before confirmation)
    - "Only transactions which had no unconfirmed parents at the time they entered the mempool are considered." [1]
- Historic fee data 
    - exponentially decaying moving average, with half life of 18, 144, 1008 blocks [v0.15]
    - counters decay by 0.962, 0.9952, 0.99931 every block [v0.15]
- confirmation fraction is used to calculate estimate by scanning buckets with;
    - greater all passed, or
    - less all failed
    - max(60% threshold at target/2, 85% threshold at target, 95% threshold at 2*target)
- The estimator monitors the following types of event:
    - A new block is added to the chain
    - A new transaction is added to the mempool
    - A transaction is removed from pool and the reason is not that it has been added to the chain.
    
estimateSmartFee() https://bitcoincore.org/en/doc/0.21.0/rpc/util/estimatesmartfee/

https://github.com/bitcoin/bitcoin/blob/master/src/policy/fees.cpp

#### alternatives

https://gist.github.com/DavidVorick/ba17fb7db50b9270d1c2131ff425422b
https://blog.bitgo.com/the-challenges-of-bitcoin-transaction-fee-estimation-e47a64a61c72

#### Revault

GOAL: Conservative estimate based on current fee-market state. Fee-bump algorithm in-case the initial estimation fails.

Use bitcoin core's estimateSmartFee
    - Target is always 1 for Cancel Txs
    - Target can be configurable for Spend Txs
    - Conservative estimates always used
    
Fee-bump algorithm to ensure that cancel Txs are processed _before_ CSV limit. 
\\ Simple linear fee bumping \\
    - Divide time into 1 block intervals with range ` t = (1,2,...,CSV) ` 
    - Let the feerate estimates be ` f = (f_1, f_2, ..., f_CSV) `
    - Let ` (f_i+1 - f_i) = (f_C-f*)/ CSV ` where f* is the estimateSmartFee value
    - Must assume that `f* < f_C` always, and that `(f_i+1 - f_i)` is not too small to fail RBF rules
    - Set `f_1 == f*` and `f_CSV = f_C`
    - Thus, ` f_i+1 = f_i + (f_C-f*)/ CSV  ` is solvable for each i

\\ Exponential decaying fee bumping \\
    - Divide time into 1 block intervals with range ` t = (1,2,...,CSV) `
    - Let the feerate estimates be ` f = (f_1, f_2, ..., f_CSV) `
    - Let f* be the estimateSmartFee value
    let d be the decay rate (between 0 and 1) of the spacing between consecutive fees
    want 
    `f_2-f_1 = d (f_C-f*)`, 
    `f_3-f_2 = d ( f_2 - f_1 ) = d^2 (f_C-f*)`,
    `f_i+1 - f+i = d^i (f_C-f*)`
    with the constraints that `f_1 == f*` and `f_CSV = f_C`
    
    `f_i+1 = f_i + d^i (f_C-f*)`
    
    ```
    for example, CSV = 4
    
    f_1 = f* 
    f_2 = f_1 + d (f_C-f*)
    f_3 = f_2 + d^2 (f_C-f*)
    f_4 = f_3 + d^3 (f_C-f*) = (f_2 + d^2 (f_C-f*)) + d^3 (f_C-f*) = ((f_1 + d(f_C-f*)) + d^2 (f_C-f*)) + d^3 (f_C-f*)
        = f* + d(f_C-f*) + d^2 (f_C-f*) + d^3 (f_C-f*) == f_C
        
        -> d + d^2 + d^3 = 1 -> d ~ 0.54369
        
    for CSV = 5
    
    f_5 = f_4 + d^4 (f_C-f*)
        -> d + d^2 + d^3 + d^4 = 1 -> d ~ 0.604729
        
    for CSV = 11, d ~ 0.556641
    
    ```
but we don't want d to be fixed, we want a configurable rate. So, we need to have a free parameter. So instead of saying f_CSV = f_C, let's say f_CSV < f_C.
    In this case the above values for d represent the max rate of decay, and we can define d* as our actual chosen rate of decay such that 0 < d* < d.
    ```    
    in the case of CSV = i

    d +  d^2  + ... + d^i-1  = (f_C-f*)/(f_C-f*)
    
    d* + d*^2 + ... + d*^i-1 = (f_i-f*)/(f_C-f*)
    
    if d* < d, then (f_i-f*)/(f_C-f*) < (f_C-f*)/(f_C-f*) -> f_i < f_C 
    ```
This results in increase by an exponentially decaying amount until the feerate is equal to or less than the max cap, the feerate estimate f_C.
        - The fee-bump _must_ be monotonically increasing (to enable RBF)
        - The fee-bump is capped by the fee-reserve for the transaction f_C
        - If the fee estimate from estimateSmartFee fails the first time, then fee-market has spiked, so increasing the fee significantly is likely necessary. There must however be some reserve kept back to enable bumping again.
            - what is the game-theory for the mining network in terms of purposefully not including Cancel Txs in the first block after it enters the mempool to trigger revault deployments to significantly bump the fee? Well, it's a competitive market, so I _guess_ the nash equilibrium falls on a miner processing the transaction for the initial fee rather than wait and allow a competitor to take it in the next block.
        - Unfortunately for high CSV value, the fee-bump near the CSV will be insignificant (e.g. d = 0.5, d^25 ~ 3x10^-8, which would result in less than 1 sats). 
        
<!--\\ Quadratic decaying fee bumping \\  UNFINISHED       
    - Divide time into 1 block intervals with range ` t = (1,2,...,CSV) ` 
    - Let the fee-rates be ` f = (f_1, f_2, ..., f_CSV) `
    - Let f* be the estimateSmartFee value
          `f_i+1 - f_i = (1/i) (f_C-f*)`
          `(f_i+1 - f_i)/i = i(f_C-f*)`
          
          f = at^2 + bt + C   with a < 0 
          
          
    - Must assume that `f* < f_C` always, and that `(f_i+1 - f_i)` is not too small to fail RBF rules
    - Set `f_1 == f*` and `f_CSV = f_C`
    
    e.g. let CSV = 4
    
    f_1 = f*
    f_2 = f_1 + (f_C-f*)
    f_3 = f_2 + (1/2)(f_C-f*)
    f_4 = f_3 + (1/3)(f_C-f*) 
        = f_2 + (1/2)(f_C-f*) + (1/3)(f_C-f*) 
        = f* + (f_C-f*) + (1/2)(f_C-f*) + (1/3)(f_C-f*)
        == f_C
        
        -> (1 + 1/2 + 1/3)(f_C-f*) = f_C-f*-->

### Fee computation

Note that for each transaction type, there is a deterministic _range_ of possible sizes when fully signed. Note that the computation is specific to the participants set. We have a script to generate this data as a csv file, and can graphically analyse with python & pandas.
    
### Fee-bump output set structure

Stakeholders will re-fill the fee-wallet of their WT. Feebump Txs should construct outputs according to a distribution which allows fees to be appended to Cancel Txs and then subsequently bumped again if the initial feerate was insufficient. There needs to be some criteria by which we constrain this problem statement, to better specify what 'well distributed' output set structure is. 

Criteria
- The total amount of each feebump output should be at least R(t).
- Coins should be grouped per-vault in a way that adds up to r(v_i,t).
- Coins should be distributed by the expected algorithm for fee-bumping.
    If simple linear feebumping is used, coins should be distributed as 
    (f*, (f_C-f*)/ CSV, (f_C-f*)/ CSV, (f_C-f*)/ CSV, (f_C-f*)/ CSV, (f_C-f*)/ CSV, ...)
- Evolving with the fee market
    
- The difficulty with the above is that we don't have f* at the time filling the fee wallet. Call our initial estimate f', and the estimate that we need at spend time f*. We aim for the difference, f'-f*, to be small. If f'-f* >> 0, we can always create more change. If f'-f* << 0, we can always add additional inputs. Thus we should have a buffer of additional smaller inputs (e.g. of the size (f_C-f*)/ CSV). 
- Is it sufficient to generate f' with estimateSmartFee at the time of calculating R(t)? This could lead to a bunch of "warning, WT wallet low" notifications for the stakeholder in the beginning. Perhaps it's better to choose something like the median of the previous 30 days of feerates? 

## Overall algorithm(s) for event based processing of all of the above

...

## Set-up procedure for the WT by the stakeholder

...

## Additional notes

It's important to note that a stakeholder who is not willing to trust other stakeholders is completely reliant on their WT(s) to enforce their spending policy. The risk that their WT fails or is corrupted can be mitigated by operating more WT(s), or outsourcing _additional_ WT(s), if the cost of doing so is not prohibitive.  

## Research Questions

- How does requiring emergency as well as revault functionality effect the wallet reserves management?
- What is the cost structure for a WT (either outsourced or operated by a stakeholder)
- The risk model for a stakeholder is weaker than the risk model for the custody operation as a whole. Both should be considered. 
- Is a Feebump Tx output pool (where each output is locked to 1-of-N among the stakeholders) a tennable idea? The risk is that a compromised watchtower can lead to an empty feebump pool. Probably not acceptable.  
- Should a WT broadcast a C if there is an extended halt after the Unvault Tx is confirmed, but no Spend Tx is broadcast?
- How does a revault deployment's costs scale with more WTs?
- How do multiple revault deployments scale given the blockspace limitation? 


# WT system
        
Fee-bump algorithm to ensure that emergency Txs are processed ASAP.
    - A function for the system of WTs not, an individual WT.
    - Idea is to have watchtowers monitor for E or UE Txs and offer support for pushing it through the network by bumping the fee. 
    - This also raises the business question of, do we want to sell this as a support package?
