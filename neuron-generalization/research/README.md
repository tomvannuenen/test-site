# Research records

Commit everything here. Together with `outputs/<run>/` this is the audit trail.

- `hypotheses/<cid>.json`: H1/H2 records written from discovery examples only.
  `hypotheses/FROZEN.json` holds their SHA-256 hashes. Never edit a frozen record;
  write a new versioned one and log the deviation.
- `test_sets/<cid>.yaml`: authored falsifiable items, written after freezing and before
  running the model on them. Locked (hashed) when the blind sheet is created.
- `blind_sheets/<cid>.yaml`: held-out contexts with activations hidden. Label `h1`/`h2`
  for every item without running the model.
- `deviations.md`: any departure from PROTOCOL.md, with dates and reasons.
