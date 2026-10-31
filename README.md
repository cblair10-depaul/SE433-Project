# MiniSWEAgent + SWT-Bench Integration
## Quick Start

### Environment Setup
1. Install Docker Desktop
2. Go to platform.deepseek.com and sign up for an account, prefil your account with some $$$ and get an API key
3. Create a .env file at the root of this directory and put your Deepseek API key in this format
```
DEEPSEEK_API_KEY=whatever-key-from-your-account-here
```

```bash
# 1. Load and inspect SWT-Bench instances
.\.venv\Scripts\python.exe load_swt_bench_instances.py

# 2. Run mini-swe-agent on instances (first 5 by default)
.\.venv\Scripts\python.exe run_swt_bench_complete.py
# ...or on specific instances, or on all 433 (main.py takes the same arguments)
.\.venv\Scripts\python.exe run_swt_bench_complete.py django__django-11055 sympy__sympy-20590
.\.venv\Scripts\python.exe run_swt_bench_complete.py --all
# Predictions are saved after each instance, and a rerun skips the ones already saved.
# Named instances always rerun; --fresh discards everything saved and starts over.

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

