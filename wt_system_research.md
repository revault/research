# Watchtower System

GOAL: Investigate the risks, countermeasures and costs of custodial functions associated with the system of watchtowers (spend policy enforcement, emergency response) to aid design choices for a practically secure watchtower implementation. 

This risk model will be constructed with several assumptions that limit the scope of analysis to the system of watchtowers (WTs) in the operation phase of custody. The initialization phase wherein the WTs are setup and configured is left for future work. The broader system risk model of custodial operations with Revault is available as a library of attack trees.

- Perhaps (time-permitting) if there is not enough depth to this risk model and analysis, we can consider the setup too.
- Do we want to explore updates/ security patches/ policy changes?

We can make statements about custodial functions of the shared system of WTs (spend policy enforcement, emergency response), and about the custodial functions of an individual stakeholder's WTs. Is our goal to delineate these, clearly stating the risks, countermeasures and costs for individual stakeholders, as well as to the group?

- Simple case: each stakeholder wants to enforce the same spend policy
- Complicated case: each stakeholder has their own variation of spend policy to enforce, and doesn't want to rely on the others' policy enforcement.

Both of these cases can be addressed. 


## Assumptions

Cryptographic assumptions
- Key generation ceremony secure
- Cold private keys (stored on HWs) are secure
    - Hot private keys are vulnerable to remote theft 
- Authenticated and Encrypted channels are secure between revault entities, but can be intercepted/ dropped for traffic analysis and DoS
    - Unless noise keys are stolen
- Cryptographically secure hash functions
- Secure digital signatures

Blockchain assumptions
- Persistence 
- Liveness *with exceptions*
    - in "weak privacy model" we shouldn't assume liveness, but instead compute the cost of censorship
    - in a "malleable tx model" we shouldn't assume liveness, but instead analyse the risks of malleability attacks
    - in a "centralized p2p routing model" we shouldn't assume liveness, but instead analyse the risks of eclipse and partition attacks  

Revault protocol assumptions
- Vault UTxOs are publicly visible and linked to the custodian (this is "weak privacy model")
- WTs are secure against arbitrary physical compromise and are always online
- WTs hardware and OS not corrupted
    
Deployment assumptions
- The initialization process was secure and safe; private keys and backups were correctly and confidentially constructed for each participant, software and hardware integrity were verified, relevant public key information for both the wallet and communication was shared among participants leading to a correct configuration for the wallet clients, watchtowers, anti-replay oracles and the coordinator.
- The software development life-cycle of Revault is secure, such that the servers of honest participants use an implementation that adheres to the protocol specification.

## Individual WT

WT functions
1. Enforce Spend Policy
    - Requires fee-reserves
    - Requires secure data sources for policy checking
    - Requires synchronized fully validating bitcoind node
    - Requires storage of Cancel sigs, unvault descriptor
2. Emergency Response
    - Requires secure communication with stakeholder (on multiple channels)
    - Requires storage of emergency sigs, deposit descriptor, unvault descriptor
    - May require synchronized fully validating bitcoind node (if mass emergency broadcast feature)

When introducing Revault WTs, enumerate mempool acceptance rules, discuss RBF validity checks, standardness checks. Discuss the limited malleability with A1CP|ALL feebumping mechanism. Explicitly state the interfaces exposed by the WT and the ways in which its decisions are triggered. 

Note that an outsourced watchtower may be required only to enforce spend policy, and not to handle emergency response. 

## System Model

UTxO notation, entity labelling (distinguished by operator), physical security environment, fee-reserves per WT, wallet re-fill source.

## Questions

### Spend Policy Enforcement Risk Enumeration

What is the attack process to compromise a stakeholder's WT? What is the attack process to compromise the system of watchtowers? (See attack trees for attacks unrelated to fee management and policy configuration) What interfaces are exposed by the WT? How can these be manipulated to compromise its functionality? 
    - risks with policy data sources
    - risks with Eclipe attacks and similar through P2P network
    - risks with censorship
    - risks with fee-market spiking
    - risks with Tx malleability
    - risks with key management (hot noise and bitcoin keys)
    - risks with WT wallet re-fill
  
Attacker profiles: Miners, Insiders, Competitors, Crime Syndicate, Nation state
  
#### Policy Data Sources
Discussed in watchtower.md
    
#### Network layer attacks

Question: Which countermeasures for eclipse and p2p network attacks have not yet been integrated into bitcoin core? 
    - https://bitcoinops.org/en/topics/eclipse-attacks/
    - https://inthemesh.com/archive/bitcoin-eclipse-attacks/
    - https://inthemesh.com/archive/bitcoin-eclipse-attacks-2/

Given the existence, prevalence and feasibility of eclipse and partition attacks (through P2P network messages and underlying internet infrastructure), what countermeasures should WT and systems of WTs use in addition to those in bitcoin core? What configuration options for bitcoin core should be set? 

What is a good way to enumerate p2p network risks? 
    - Internet routing infrastructure attacks (BGP hijacking, ISP-level adversary)
    - Eclipse attacks using P2P messaging protocol
    - Targetted vs broad network partition
    - packet dropping/ tampering/ injection/ evesdropping
    - Time dilation, deanonymisation, DoS, malicious fork

What about 'unreachable nodes'? 
[Improving Bitcoin Transaction Propagation by Leveraging Unreachable Nodes]
   
#### Miner attacks
   
##### Censorship
  
Block template production (that is, choice of which transactions are included in blocks) is highly centralized due to the dominant mining pool protocols used today. There may be 1% of mining pools (by hashrate) using updated mining protocols that enable decentralized block template production (Blockstream pool with BetterHash [1] and Braiins with Stratum V2 [2]). There are about 10 dominant block producers [3] (F2Pool, Poolin, AntPool, Binance, BTC.com, ViaBTC, Huobi, Canoe, BTC.top, Slush) with >90% of total network hash-rate. 

How can we model the risk that miners collude or are bribed to censor Revault Txs for X blocks? What are the steps, incentives and costs involved in this attack? To get some insight into the problem, consider the following related questions.

Does the attacker need physical or synthetic control of the hash-rate?
- As discussed in [6], physical hash-rate attacks that rely on purchase or manufacture of ASICs and then running them are very expensive. Alternatively, simultaneous take-over of enough individual mining farms to gain even 51% of hash-rate is an extremely difficult coordination problem likely involving 100+ operations. Let us presume that this is not a viable attack for the payoff of stealing from a Revault deployment. 
- Synthetic hash-rate attacks that rely on corrupting the centralized entities that produce block templates are much cheaper. For a subtle censorship attack on a Revault deployment, the attacker is not interested in controling the majority hash-rate. The attacker is interested in enforcing a censor on cancel or emergency Txs by coordinating with about 10-15 mining pool operators. 

What Revault Txs are actually identifiable?
- Deposit UTxOs (as NofNs, not necessarily unique to revault) become publicly visible when an Unvault Tx or Emergency is broadcast. Unvault UTxOs (unique to revault deployment) become known when the Spend Tx, Cancel Tx, or Unvault Emergency Tx is broadcast. So a third party can publicly identify Txs that spend unvault UTxOs. They can distinguish Cancel and Unvault-Emergency Txs from Spend Txs through the witness script (either it spends through the NofN path or the MofN).
- An insider (e.g. a corrupted manager) has complete information on the set of Deposit UTxOs and the corresponding pre-signed transactions. Insiders are the most capable of enacting censorship attacks by colluding with centralized block producers since they can specifically point to the transactions that should be censored (Cancel, Emergency, Unvault Emergency) in order to broadcast a malicious spend transaction. Managers must collude to engage in this attack (to steal via the MofN pathway). Managers could equally attempt to steal directly from the deposit UTxO by attacking stakeholders in a "mutany" type of attack. 
- Outsiders have two clear attack pathways: a) try to corrupt managers and steal via malicious Spend Tx, or b) link custodian's identity with deposit UTxOs, and collude with miners to censor Emergency and Unvault Emergency Txs while launching an attack on the N stakeholders.


What is the cost of bribery to convince a centralized block producing entity (mining pool operator) to censor a transaction in the next block?
- Assume that Revault Txs are identifiable, and that addresses have been publicly linked to the custodian by an attacker
- Assume that mining pools have the tools to censor transactions from the block they are mining (e.g. BlockSeer [4])
- A rational profit-seeking miner must be compensated for the opportunity cost of not including the transaction in their block. That cost consists of 
    1) the fee that would have been claimed by mining the transaction (although in the case where there is an abundance of transactions in the queue for processing, a substitue transaction could be included which compensates part of the fee) 
    2) the time-cost of excess processing on the block template before generating PoW
    3) the higher-order effects on the utility and value of the bitcoin network due to censorship attacks
    4) Cost of reputation loss with their customers (miners)

The sum of these costs must be compensated, proportional to the probability of the miner successfully winning the mining race for the next block. 

Let's say the fee that would otherwise be claimed by the miner for the censored Tx is `f`. A mining pool operator might be convinced to not include a Tx that pays fee `f` if they are paid out-of-band a sum higher than `f`. 

Miners compete against each other in a race. Time costs for block template construction must be minimized to optimize their likelihood of success. Introducing additional computation or communication before block template must be accounted for. This is difficult to analyse in a general way. For a single transaction the costs is likely minimal. If pool operators have many censorship contracts active this could become significant. Apparently miners already mine empty blocks during the validation/ template creation time, which reduces the expected cost of this too. 

To understand the 3rd point, consider the extreme attack scenario where all miners are bribed to not include any transactions. They would mine empty blocks (at least as long as the block reward is worthwhile), which diminishes the utility of bitcoin as a transaction system, and threatens their long-term profit. In the future, when the block reward has become negligible, the incentives change somewhat, but it is not clear what miner strategies will emerge as dominant. The attack would be very transparent, and there would likely be significant consequences to users' perception bitcoin and its price. This demonstrates the existence of scalability limits for this attack, but does not mean that the attack couldn't happen covertly in specific cases, for the right price. Moreover, it's not obvious whether the market would react positively or negatively since there are likely pro-censorship businesses who would prefer a regulated bitcoin.
- note, the off-chain feemarket is opaque and thus may be highly inefficient. Off-chain order books would need to emerge for efficiency. But then the competition between off-chain and on-chain fee markets becomes a tractable problem, and the bidding war can play itself out. This leads me to think that censorship requires some asymmetry of information; that the briber acts in partial secrecy. If the person being censored knows they are under attack, they can increase their feerate. Perhaps its in miners' interest to publicise censor attempts since the competition would raise the fees being paid (for processing the tx or not). On the other hand, if the axiom of censorship resistance is truly fundamental to Bitcoin's success, miners must only act covertly in censorship attacks. This type of off-chain fee order book might be difficult to do in a verifiable way. Off-chain fee markets could be based on out-of-band fee payments using the Lightning Network, or more conventional insurance contracts for block space.
- This tension in utility of bitcoin network weakening as a result of over-censorship is a higher-order cost that the miner may demand as compensation (though it is unquantifiable). It might mean that censorship could become a highly profible opaque market for miners to participate in.

The 4th point is very difficult to estimate without historic data (can we figure out miner pool-hopping stats for when pool operators act selfishly and opposed to the censorship-resistance property of bitcoin?). Can we safely assume that public knowledge of pool operators censoring transactions will cause significant number of miners to pool hop? 

It is impossible to compute an actual opportunity cost, but some mining pool operators may decide there is a worthwhile price for the attack. 

Consider now how the censorship attack can scale in effectiveness. To _guarantee_ that a transaction will be censored from the next block, the attacker would need to set bribery contracts with 100% of block producers (notice the actual hash-rate is not important here, only the decentralization of block producers).
- Of course, if the attacker had majority control of the hash-rate and could engage in blockchain reorganisation attacks, that would also enable them to launch an effective attack even without 100% of block producers. However, it's important to distinguish this as a different type of attack due to; its cost, deeper systemic consequences on throughput and immutability, and the transparency of the attack [5]. It's one thing to subtly censor transactions by not including them in a block and not propagating them through the P2P network. It's another thing to attempt to reorg the chain. The former is less of a systemic attack, and less keanly monitored as far as I can tell. Either we should find or build a tool to monitor this. Can the rate of censorship be realistically measured? Can specific instances of censorship be caught? How much 'plausible deniability' is there? See [7] for initial work in this direction. 

While that may seem like an unrealistic target, roughly 90% of the network hash-rate is operated by about 12 mining pools. So, to have a 90% likelihood of successful attack requires setting contracts with those 12, and them acting in accordance with the contract. 
- I don't think it's illegal to create those contracts

Now consider that an attacker needs to censor the same transaction for each block until the relative timelock is over. Let's say the CSV value is X. The attacker would need censorship contracts with 12 mining pools for each of X blocks. This would be far more likely to be noticed by a monitoring tool. The likelyhood of success would be 0.9^X, since each independent block has 90% chance of succesful censorship. Let's look at a range of likelihoods for a successful attack:
If X = 12, 90% pool operators (by hash-rate), Attack likelihood = 0.28 
If X = 24, 90% pool operators (by hash-rate), Attack likelihood = 0.080
If X = 36, 90% pool operators (by hash-rate), Attack likelihood = 0.023
If X = 108, 90% pool operators (by hash-rate), Attack likelihood = 0.00001
If X = 12, 50% pool operators (by hash-rate), Attack likelihood = 0.00024
If X = 24, 50% pool operators (by hash-rate), Attack likelihood = 0.000000060

This rough analysis can give us insight into the minimum CSV value to resist this type of attack. 

On the one hand, it is clear that privacy is very important for a Revault deployment. This is a weakness in the secure operation of a revault deployment that shows that we should support decentralization of block production. It shows that we should support enhancements to script privacy (e.g. taproot). 

Is it any different than with other custody solutions? Well, yes. There is more reliance on timely transaction processing. It affects the delegation element of the process most. Insiders are most capable of theft.
- Is there more to compare here? What if the set of managers is equivalent to a typical custody setup (2of3). 

*Conclusion*
This analysis provides a logical way to determine minimum viable CSV length by only assuming the capacity of an attacker to negotiate bribery contracts with a percentage of mining pool operators. Furthermore, it demonstrates the value of public tracking of censorship, and of improving script privacy and mining pool protocols.

  
[1] https://blockstream.com/2019/08/08/en-mining-launch/
[2] https://braiins.com/bitcoin-mining-stack-upgrade
[3] https://coin.dance/blocks/thisweek
[4] https://cointelegraph.com/news/slippery-slope-as-new-bitcoin-mining-pool-censors-transactions
[5] https://forkmonitor.info/nodes/btc
[6] https://braiins.com/blog/how-much-would-it-cost-to-51-attack-bitcoin
[7] https://blog.bitmex.com/bitcoin-miner-transaction-fee-gathering-capability/

Other related resources:
https://github.com/libbitcoin/libbitcoin-system/wiki/Censorship-Resistance-Property
https://github.com/libbitcoin/libbitcoin-system/wiki/Side-Fee-Fallacy
https://github.com/libbitcoin/libbitcoin-system/wiki/Axiom-of-Resistance
https://medium.com/@greerre/the-economics-driving-bitcoins-inevitable-centralization-94da3a5d1f63
https://www.blockchain.com/charts/pools

##### Miner Incentive structure changes

Generally, how to consider the effects of underlying incentive structure changes as the block reward deminishes. What new risks emerge, if any? 

[Purge Attacks]
[Towards Overcoming the Undercutting Problem]

#### Fee Market Spiking
- see watchtower.md

[1] https://bramcohen.medium.com/how-wallets-can-handle-transaction-fees-ff5d020d14fb
    
    
#### Tx Malleability
Tx malleability for Cancel, Emergency and Unvault-Emergency Txs is strictly limited to enable feebumping
- Witness malleability will be analysed
- standardness malleability will be analysed [1,2]
- inter-Tx and intra-Tx input malleability will be analysed

[1] https://medium.com/summa-technology/the-bitcoin-non-standard-6103330af98c
[2] https://blog.kaiko.com/an-in-depth-guide-into-how-the-mempool-works-c758b781c608

### Security Analysis

What tolerance to arbitrary malicious attacks does the shared system of WTs have? What tolerance to arbitrary malicious attacks does an individual stakeholder's WT system have? 
    - Statement like: WT functionality is secure under the assumption of a single honest WT.
        - Consider the notion of "shared spend policy enforcement", where the spend policy of different WTs overlaps. This is secure under the assumption of a single honest WT.
        - Spend policy enforcement performed by Y WTs is secure under the assumption that any 1 of Y WTs is honest. 
        - Emergency Response is secure under the assumption that any 1 of N stakeholder-WT pair is honest (not corrupted/ attacked). 
    - Talk about the trade-off of theft vs DoS 
        - robustness to fraudulent spends -> easier to DoS
        - robustness to theft from vault -> easier to severely DoS
        - DoS by cancel isn't as costly as DoS by emergency. DoS by emergency should have some mitigation. How to tune this? Perhaps stakeholders can explicitly state something like: It's ok for 50% of cancels to have not been necessary, but no more. It's ok for 1% of emergencies to be triggered unnecessarily, but no more. Or better yet; I'm happy for any one of the stakeholders or their WTs to trigger a cancel, but not a manager. I'm happy for any two of the stakeholders (and their WT) to trigger an emergency, but no less. 

### Operational Costs

What are the costs per WT, and for the WT system as a whole? 
- See watchtower.md for fee-reserve calculations
- What are the HW and operating costs per WT? 

For an individual stakeholder's WT system, fee-management can be simplified with Feebump Txs that pay to 1ofK addresses controlled by their K watchtowers. The WTs have redundancy and are non interactive, without additional fee-reserve overhead. 

Incentive structure and incentive mechanisms for outsourced watchtowers (See related work with LN WTs)?

### Emergency Response Risk Enumeration

WT system feebump algorithm to ensure that emergency Txs are processed ASAP.
    - A function for the system of WTs not, an individual WT.
    - Idea is to have watchtowers monitor for E or UE Txs and offer support for pushing it through the network by bumping the fee. 
    - What is the risk of broadcasting a malleable E or UE? 
    - This also raises the business question of, do we want to sell this as a support package?

Is it worth considering a non-interactive 2-party (or multi-party) protocol for emergency response? To avoid severe DoS attacks? Even with minimized exposed interfaces, how feasible is it to compromise a WT to DoS a revault deployment? Would a threshold security setup here be worthwhile (v2)? 
    
#### Emergency Response Strategy

We say that each stakeholder has unilateral capability to trigger an emergency response. However, we haven't considered what constitutes an emergency scenario. This is a difficult problem, as being too sensistive to threats makes the operation susceptible to serious DoS attacks, while being too relaxed about threats makes theft easier. Is a death-threat in the mail enough to DoS a multi-million doller operation? 

Presumably, A multi-step emergency response process is needed where stakeholders are made aware of incidents, gather and share incident knowledge, and react appropriately. But what should this look like? How can we tune the response sensistivity to the threat-level for this process?    

Assume that each of the N stakeholders and their WTs have access to emergency signatures. Before triggering an emergency, what is a sufficient signal? What types of signal are acceptable? What is the weight of each signal? How is this signal transmitted to and among stakeholders? How does public knowledge of the types of signal affect the attacker strategy? 

Incident:
- WT down
- Stakeholder unresponsive
- stakeholder HW missing/ faulty
- Stakeholder threatened remotely/ physically

For majority of our security assessment, we may enumerate risks associated with the controlled environment of operations, and pre-emptively act by introducing countermeasures. However, serious attacks will cause chaotic outcomes, and will force stakeholders to think on their feet as they face uncertainty and disorder. While part of the emergency response can be pre-defined through Standard Operating Procedures, there will be a need for real-time creative strategising to respond to the situation at hand. Moreover, the human dimension of fear or anger in response to violence and danger cannot be ignored. 

Should there be training for acting under duress? Should the emergency response process be standardised or does this give too much information to attackers? Should we as a company provide tools for the response?     
