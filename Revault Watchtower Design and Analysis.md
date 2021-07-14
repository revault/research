# Revault Watchtower Design and Analysis

## Contents

1. Introduction
    - Revault custody
    - WTs in Revault custody
        + Policy Enforcement
        + Why A1CP fee method vs CPFP and consequences
    - Related work
        + Other L2 applications, their WT and fee methods
    - Objective
        + WT wallet problem statement: Watchtowers monitor their interfaces to determine if and how they should respond to events by broadcasting pre-signed transactions and supplementing their fee. Successful response is predicated on availability of UTxOs that can be used to pay the fees to ensure the txs are processed on time. The primary complicating factor is the inherent unpredictability and non-linearity of the fee-market. Thus, how can we maintain the security of operations by ensuring the availability of UTxOs without grossly overpaying transaction fees and without burdening WT operators?   

2. Methodology & scope
    - No emergency, operational phase, single WT
    - WT wallet numerical model
    - Simulation using historic fee data & mock events triggering a WT response
    - Comparison of operational costs, time at risk, stakeholder interaction

3. System formalism
    - Fee market formalism (Is it _really_ unpredicatable and non-linear? How to show this?)
    - Formal WT wallet model (UTxO notation, transaction types & sizes, equations for coin-sizing and fee_reserves, functions; re-fill, consolidate-fanout, allocation, re-allocation, cancel/spend)

3. Algorithm Design
    - constraints (e.g. deterministic -> re-fill without communication, down-time tolerance, no failure modes, etc.)
    - goals: minimised operational fees, minimised stakeholder activity, minimised time-at-risk
    - strategies for coin-sizes (accurate fee paying), fee reserve per vault, allocation to vault 

4. Results
    - Show results for different deployment types with different operating expectations. 
    - Show results for different strategies/ algorithms
    - Compare results

5. Conclusion



