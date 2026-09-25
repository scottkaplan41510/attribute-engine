---
name: attribute-analysis
description: Run the attribute engine on a CSV of content and a metric, then summarize which attributes correlate with performance. Use when someone asks which parts of their copy, ads, emails or subject lines correlate with a metric.
---

# Attribute analysis

1. Confirm the CSV has a `copy` column and a `conversion_rate` column. If the
   metric has another name, rename the column.
2. Confirm `.env` has `AI_GATEWAY_API_KEY`.
3. Show the user the attributes in `src/attribute_engine/attributes.yaml` and ask if they want to add
   or change any before running.
4. Run `attribute-engine --input <file>`. When discovery proposes attributes,
   show them to the user and let them choose which to keep.
5. Summarize `output/report.md` in plain English. Say "correlates with", never
   "causes". Say the sample size and that small samples only catch big effects.
