The first scoring (the "original labels" results) showed misses where the gold label contradicted both prompts' own guide lines. That prompted a label audit. The correction was then applied by rule to every gold row in four classes, not only to the rows the model missed:

- **Payroll → Utilities** (the Utilities guide line lists "payroll"). Changed: rows 17, 18, 75, 76 (Gusto Payroll). Already Utilities: rows 33, 34 (ADP Payroll).
- **Professional services → Utilities** (the Utilities guide line lists "professional services"). Changed: rows 46 (Certified Court Reporting), 47 (Expert Witness Retainer), 141 (Brightline Consulting Partners). Already Utilities: rows 35 (Contract Attorney Payment), 48 (Process Server LLC).
- **Food and ingredient suppliers → Shopping** (the Shopping guide line covers "retail and supplies"). Changed: rows 63–66 (Sysco Foods), 67–68 (US Foods), 71–72 (Local Seafood Co), 73–74 (Craft Beverage Distributor). Already Shopping: rows 69–70 (Restaurant Depot).
- **Waste services → Other** (both prompts' Other guide line lists "waste services"). Changed: row 22 (County Landfill, yard waste disposal), first as a consistency fix to match row 23 (County Landfill, debris disposal), an identical transaction type; then row 83 (Waste Management, commercial trash and grease disposal), in a follow-up applying the same class. Already Other: row 23.

Deliberately unchanged: row 61 (LegalZoom Filing) stays Other. It's a filing fee, consistent with row 58 (County Clerk Filing Fee).

19 labels changed in total (row numbers count data rows, header excluded). No prompt text changed. Both versions were re-scored from the cached model replies with 0 model calls, so they're compared on the same corrected labels. Because the audit followed the first results, the corrected scores aren't a blind measurement; the original-label scores above are kept for that reason.
