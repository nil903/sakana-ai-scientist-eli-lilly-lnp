## Name

eli_lilly_lnp_1

## Title

Oxidized MC3 tail ketone introduces new hydrogen-bonding motifs with siRNA in LNP-like environments

## Short Hypothesis

Adding a ketone group to MC3 tails (dienone oxidation product) increases tail-mediated H-bonding to siRNA, but these interactions remain weaker/less persistent than canonical headgroup (protonated amine)–RNA phosphate contacts.

## Related Work

- All-atom MD studies of ionizable lipids with nucleic acids in LNP-mimetic systems
- Oxidative degradation pathways of unsaturated lipid tails and potential impact on formulation stability
- Hydrogen bonding and contact analyses (H-bond occupancy/lifetime, RDFs) for lipid–RNA interaction characterization

## Abstract

Ionizable lipids such as DLin-MC3-DMA (MC3) are central to clinically deployed siRNA lipid nanoparticles (LNPs), yet their bis-allylic unsaturated tails are susceptible to oxidation. Oxidation can yield conjugated dienone byproducts bearing ketone functional groups, potentially introducing new polar interaction sites with encapsulated siRNA. We propose matched all-atom molecular dynamics simulations comparing native MC3 and an oxidized MC3-dienone species in otherwise identical LNP-like aqueous environments. We will quantify tail-ketone–RNA hydrogen bonding and compare it to established headgroup–RNA interactions, assess lipid–RNA proximity via radial distribution functions, and test whether additional interaction modes emerge for siRNA containing 2'-OH groups relative to fully 2'-O-methyl / 2'-fluoro modified siRNA.

## Experiments

- System construction: build two matched systems (native MC3 vs oxidized MC3-dienone) with identical lipid counts, protonation state of ionizable headgroups, water/ions representative of physiological buffer, and the provided ds-siRNA with phosphorothioate + 2'-OMe/2'-F modifications.
- Variant system: construct an additional siRNA variant containing 2'-OH groups (unmodified ribose) to test whether oxidized lipid introduces additional binding modes compared to fully 2'-O-methyl modified siRNA.
- MD protocol: equilibrate then run production all-atom MD (multiple replicates/seeds).
- H-bond analysis: compute H-bond counts, occupancies, and lifetimes for (a) tail-ketone···RNA interactions and (b) headgroup-amine···RNA interactions; compare magnitudes.
- RDF analysis: compute RDFs between lipid headgroup atoms vs RNA phosphates, and between tail-ketone oxygen vs RNA functional groups; compare peak positions/heights between native and oxidized systems.
- Mode discovery: cluster contact patterns (lipid group ↔ RNA group) to identify distinct interaction modes; compare populations between siRNA chemistries.

## Risk Factors And Limitations

- Force-field parametrization of oxidized MC3-dienone may be nontrivial; parameterize consistently (e.g., CGenFF/GAFF-style) and validate geometry/partial charges.
- Sampling limitations: interaction modes may require long timescales; mitigate with multiple replicates and convergence checks.
- LNP composition realism: simplified systems may not capture full LNP heterogeneity; report assumptions and run sensitivity checks (salt, protonation).

