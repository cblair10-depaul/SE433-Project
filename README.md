# MiniSWEAgent + SWT-Bench Integration
## Quick Start

```bash
# 1. Load and inspect SWT-Bench instances
.\.venv\Scripts\python.exe load_swt_bench_instances.py

# 2. Run mini-swe-agent on instances
.\.venv\Scripts\python.exe run_swt_bench_complete.py

# 3. Evaluate with SWT-Bench
cd ..\swt-bench
.\.venv\Scripts\activate
python -m src.main --dataset_name princeton-nlp/SWE-bench_Lite ^
    --predictions_path ../MiniSWEAgent/swebench_predictions/predictions.jsonl ^
    --filter_swt --max_workers 4 --run_id mini-swe-agent-test
```

## Output Files

| File | Purpose |
|------|---------|
| `swebench_predictions/predictions.jsonl` | JSONL format for SWT-Bench evaluation |
| `swebench_predictions/preds.json` | JSON list format (also accepted by SWT-Bench) |

