"""Load SWT-Bench instances and prepare for mini-swe-agent processing"""
import json
from pathlib import Path
from typing import Dict, List, Any

# Add swt-bench to path
import sys
sys.path.insert(0, str(Path(r"C:\Users\ccase\PycharmProjects\swt-bench")))

from datasets import load_dataset


def load_swt_bench_instances(dataset_name: str = "princeton-nlp/SWE-bench_Lite",
                             split: str = "test",
                             instance_ids: List[str] = None,
                             limit: int = None) -> List[Dict[str, Any]]:
    print(f"Loading dataset: {dataset_name}")
    
    try:
        # Load from HuggingFace
        if dataset_name.lower() in {"swe-bench", "swebench", "swe_bench"}:
            dataset_name = "princeton-nlp/SWE-bench"
        elif dataset_name.lower() in {"swe-bench-lite", "swebench-lite", "lite"}:
            dataset_name = "princeton-nlp/SWE-bench_Lite"
        elif dataset_name.lower() in {"swe-bench-verified", "swebench-verified", "verified"}:
            dataset_name = "princeton-nlp/SWE-bench_Verified"
        
        dataset = load_dataset(dataset_name, split=split)
        
        instances = list(dataset)
        
        if instance_ids:
            instance_ids_set = set(instance_ids)
            instances = [i for i in instances if i["instance_id"] in instance_ids_set]
        
        if limit:
            instances = instances[:limit]
        
        print(f"Loaded {len(instances)} instances")
        
        return instances
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return []


def format_swt_bench_task(instance: Dict[str, Any]) -> str:
    return instance.get("problem_statement", "")


def save_predictions(predictions: List[Dict[str, Any]], output_dir: Path) -> None:
    # Save as JSONL (for SWT-Bench evaluation)
    jsonl_path = output_dir / "predictions.jsonl"
    with open(jsonl_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    
    # Save as JSON dict (for SWE-bench format)
    json_path = output_dir / "preds.json"
    preds_dict = {p["instance_id"]: p for p in predictions}
    with open(json_path, "w") as f:
        json.dump(preds_dict, f, indent=2)
    
    print(f"Saved {len(predictions)} predictions:")
    print(f"  - JSONL: {jsonl_path}")
    print(f"  - JSON: {json_path}")


def main():
    # Example: Load specific instances
    instance_ids = [
        "sympy__sympy-20590",
        "django__django-11055",
    ]

    print("SWT-Bench Instance Loader for mini-swe-agent")

    instances = load_swt_bench_instances(
        dataset_name="princeton-nlp/SWE-bench_Lite",
        split="test",
        instance_ids=instance_ids,
        limit=None
    )
    
    if not instances:
        print("No instances loaded. Please check your instance IDs or dataset name.")
        return

    for i, instance in enumerate(instances, 1):
        task = format_swt_bench_task(instance)
        print(f"--- Task {i} ({instance['instance_id']}) ---")
        print(task)


if __name__ == "__main__":
    main()
