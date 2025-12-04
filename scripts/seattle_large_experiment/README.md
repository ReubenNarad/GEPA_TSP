Seattle large GEPA experiment (n≈400 travel-time weights on the Seattle road network).

Files:
- `reflector_context.txt`: context passed to the reflector (non-Euclidean, travel-time edges, bold structural changes, build on best-so-far).
- `02_run_gepa_seattle_large.sh`: heavy GEPA run (default steps=20, reflection batch=3, repeats=3, baseline repeats=5, timeout=60s). Defaults to the generated Seattle travel-time metadata (`out/metadata_seattle_time_experiment.json`, split `structured_seattle_time_val`); override `META`/`--split` as needed.
- `03_plot_gepa_seattle_large.sh`: plot the latest run.
- `04_eval_seattle_large_test.sh`: evaluate baseline + best candidate on structured_seattle_time_test.

Prompt considerations:
- Student prompt is already Seattle-specific in `prompt_scratchpad.md`; this experiment adds reflector context and best/baseline metrics to feedback for richer guidance.
- Encourage larger structural edits (queue policy, limited retries per start, flipper trigger/eviction, batching policies) and discourage cosmetic rephrasings.
