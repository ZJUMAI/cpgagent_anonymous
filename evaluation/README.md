# Corrected Lung evaluation references

The 30 references use `NSCLC_2010@2010` and decision date `2010-12-31`.
`references/manifest.json` binds every rubric and trajectory by SHA-256.

These references correct sex extraction, case-specific TNM actions, and a
guideline-version mismatch. Historical paper Dynamic/Combined scores were not
recomputed, so they are not exactly reproducible with this corrected set.
