# Risk analysis of Revault watchtower system

## Contents

1. Introduction
	- Revault custody
	- WTs in Revault custody
	- Methodology & scope
	- Related work
	- Objective
	
2. System formalism
	- Formal WT system model (UTxO notation, entity labels (distinguished by operator), physical security environment, fee-reserves per WT, wallet re-fill source, spend-policy enforcement structure)
		- This should make it easy to discuss the security properties pertaining to individual stakeholders and the system as a whole. 
	- WT interfaces and decision process (summary of the spec)
	- Summary of WT system interactions (e.g. distributed feebumping, massive broadcast)

3. Security model
	1. Assumptions
	2. template: 
		- Attack vector, 
		- Risk (feasibility & impact), 
		- Countermeasures (effect on feasibility & impact, evaluation of additional risk), 
		- Design implication
	3. Attack Surface Enumeration
		- application-centric/ asset-centric war gaming & review of public research
	    1. policy data sources
	    2. tx relay jamming
	   		- mempool flooding
			- mempool partition
			- eclipse 
			- pinning
			- standardness malleability
			- witness malleability
			- ? miner harvesting ?
	    3. censorship
	    	- Incentive structure regime
	    4. Tx tampering (before broadcast)
	    5. Key (hot noise and bitcoin keys) and Signature management
	    6. Fee-reserve management (fee market spiking, WT wallet failures)


	4. Cancel functionality attack tree:
	    1. policy data sources
	    2. tx relay jamming
	    3. censorship
	    4. Tx tampering (before and after broadcast)
	    5. Key management (hot noise and bitcoin keys)
	    6. Fee-reserve management (fee market spiking, WT wallet failures)

	5. Emergency functionality attack tree:
	    1. Trigger signal
	    2. tx relay jamming
	    3. censorship
	    4. Tx tampering (before and after broadcast)
	    5. Key management (hot noise and bitcoin keys)
	    6. Fee-reserve management (fee market spiking, WT wallet failures)

4. Conclusion
