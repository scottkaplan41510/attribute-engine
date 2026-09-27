---
name: write-and-log
description: Write new copy variations from the winning attributes, log them with a control set, and later check whether they performed better. Use when someone asks to generate copy from their analysis results or to validate whether data-driven copy worked.
---

# Write and log copy variations

1. Make sure `attribute-engine` has run on the user's data, so there are
   significant winning attributes in `output/results_round2.csv`. If none are
   significant, say so and stop. There is nothing to write from.
2. The final step of `attribute-engine` writes generated copy (uses the winners)
   and control copy (uses the losers) into `output/creative_log.csv`. Show the
   user the new copy and which attributes Jev confirmed.
3. Tell the user to run the new copy, then fill in the metric column in the log.
4. When they have results, run `attribute-engine-validate` and explain the
   outcome in plain English: did generated copy beat the control? Say
   "correlates with", never "causes", and mention the sample size.
