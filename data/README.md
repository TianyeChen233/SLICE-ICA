# Aggregate results

- `results/manuscript_reported_results.json`: manuscript-reported classification,
  component-ablation, backbone, and five-seed summaries. Component-ablation
  entries are per-metric arithmetic means across runs.
- `results/pooled_external_authoritative.json`: author-provided pooled results
  for MSC01-MSC10 and the HCP held-out data, not MSC-only performance.
- `fairness/group_counts_and_aggregates.json`: group counts and pairwise
  aggregate summaries, using the shared hybrid proxy-label reference.

These are supplied result summaries, separate from outputs produced by executing
the code. Missing per-run quantities remain null. They are not inferred from
other experiments. Source imaging, individual demographic records, identity maps,
credentials, model weights and participant-level API transcripts are not included.
