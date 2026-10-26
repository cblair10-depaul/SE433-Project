import sys
import json
import shutil
import subprocess
from pathlib import Path, PurePosixPath

from datasets import load_dataset
from minisweagent import Agent
from minisweagent.config import get_config_from_spec, builtin_config_dir
from minisweagent.utils.serialize import recursive_merge
from minisweagent.agents import get_agent
from minisweagent.models import get_model
from minisweagent.environments import get_environment

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config_local_model.yaml"
OUTPUT_DIR = PROJECT_DIR / "swebench_predictions"
REPO_CACHE_DIR = PROJECT_DIR / "workspaces" / "repo_cache"
WORKSPACE_DIR = PROJECT_DIR / "workspaces" / "instances"

TEST_DIR_NAMES = {"test", "tests", "testing"}


def load_config():
    config_spec = [
        str(builtin_config_dir / "mini.yaml"),
        str(CONFIG_PATH),
    ]
    configs = [get_config_from_spec(spec) for spec in config_spec]
    return recursive_merge(*configs)


def format_task(instance: dict) -> str:
    return instance["problem_statement"]


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def prepare_workspace(instance: dict) -> Path:
    repo = instance["repo"]
    cache = REPO_CACHE_DIR / repo.replace("/", "__")
    if not cache.exists():
        print(f"  Cloning {repo} (first time only)...")
        cache.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--quiet", f"https://github.com/{repo}.git", str(cache)],
            check=True,
        )

    workspace = WORKSPACE_DIR / instance["instance_id"]
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", "--no-checkout", str(cache), str(workspace)],
        check=True,
    )
    git("checkout", "--quiet", instance["base_commit"], cwd=workspace)
    return workspace


def is_test_file(path: str, tracked_at_base: bool) -> bool:
    p = PurePosixPath(path)
    if p.suffix != ".py":
        return False
    in_test_dir = any(part in TEST_DIR_NAMES for part in p.parts[:-1])
    test_name = p.name.startswith("test_") or p.name.endswith("_test.py") or p.name == "tests.py"
    if not (in_test_dir or test_name):
        return False
    return len(p.parts) > 1 or tracked_at_base


def extract_test_patch(workspace: Path, base_commit: str) -> str:
    git("add", "-A", cwd=workspace)
    tracked = set(git("ls-tree", "-r", "--name-only", base_commit, cwd=workspace).splitlines())
    changed = git("diff", "--cached", "--name-only", base_commit, cwd=workspace).splitlines()

    test_files = [f for f in changed if is_test_file(f, f in tracked)]
    skipped = [f for f in changed if f not in test_files]
    if skipped:
        print(f"  Excluded non-test changes from patch: {skipped}")
    if not test_files:
        return ""
    return git("diff", "--cached", base_commit, "--", *test_files, cwd=workspace)


def run_agent_on_instance(agent: Agent, task: str, instance: dict, workspace: Path) -> dict:
    instance_id = instance["instance_id"]
    print(f"  Running agent on {instance_id} in {workspace}...")

    result = agent.run(task)
    print(f"  Exit status: {result.get('exit_status', 'unknown')}")

    model_patch = extract_test_patch(workspace, instance["base_commit"])

    return {
        "instance_id": instance_id,
        "model_name_or_path": "deepseek/deepseek-flash",
        "model_patch": model_patch
    }


def run_workflow():
    print("SWT-Bench Integration for mini-swe-agent")

    config = load_config()
    model_name = config.get("model", {}).get("model_name", "unknown")
    print(f"Model: {model_name}")

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {OUTPUT_DIR}")

    if len(sys.argv) > 1:
        instance_ids = sys.argv[1:]
    else:
        print("No instance IDs provided. Using first instance from SWE-bench Lite for testing.")
        print("To specify instances: python run_swt_bench_complete.py instance_id1 instance_id2 ...")
        instance_ids = None

    print("Loading SWE-bench Lite dataset...")
    try:
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        dataset_dict = {inst["instance_id"]: inst for inst in dataset}

        if instance_ids is None:
            # Use first 1 instances as default test
            instance_ids = list(dataset_dict.keys())[:5]
            print(f"Using instances: {instance_ids}")
        else:
            # Filter to only requested instances
            missing = [id for id in instance_ids if id not in dataset_dict]
            if missing:
                print(f"Warning: Some instances not found: {missing}")
                instance_ids = [id for id in instance_ids if id in dataset_dict]

            if not instance_ids:
                print("Error: No valid instance IDs")
                return

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("To run with specific instances, provide them as arguments:")
        print("python run_swt_bench_complete.py django__django-11055 sympy__sympy-20590")
        return

    print("Setting up agent...")
    model_obj = get_model(config=config.get("model", {}))

    print(f"\nProcessing {len(instance_ids)} instances...")

    predictions = []

    for instance_id in instance_ids:
        print(f"[{len(predictions)+1}/{len(instance_ids)}] {instance_id}")

        try:
            instance = dataset_dict[instance_id]

            workspace = prepare_workspace(instance)
            env_config = {**config.get("environment", {}), "cwd": str(workspace)}
            env_obj = get_environment(env_config, default_type="local")

            task = format_task(instance)

            # Create fresh agent for this instance (Don't cross contexts for better results)
            agent = get_agent(
                model=model_obj,
                env=env_obj,
                config=config.get("agent", {}),
                default_type="default"
            )

            prediction = run_agent_on_instance(agent, task, instance, workspace)
            predictions.append(prediction)

            print(f"  [OK] Generated patch ({len(prediction['model_patch'])} chars)")

        except Exception as e:
            error_msg = str(e).replace('✗', '[ERROR]')
            print(f"  [ERROR] {error_msg}")
            import traceback
            traceback.print_exc()

            predictions.append({
                "instance_id": instance_id,
                "model_name_or_path": model_name,
                "model_patch": ""
            })

    print("Saving predictions...")

    jsonl_path = OUTPUT_DIR / "predictions.jsonl"
    with open(jsonl_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    print(f"    JSONL: {jsonl_path}")

    json_path = OUTPUT_DIR / "preds.json"
    with open(json_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"    JSON: {json_path}")

    print("SUMMARY")
    print(f"Instances processed: {len(predictions)}")
    print(f"Successfully generated patches: {sum(1 for p in predictions if p['model_patch'])}")
    print(f"Failed instances: {sum(1 for p in predictions if not p['model_patch'])}")
    print(f"\nPredictions saved to: {OUTPUT_DIR}")

    print("NEXT STEPS")
    print("1. Run SWT-Bench evaluation:")
    print("   cd ~/Documents/swt-bench")
    print("   source .venv/bin/activate")
    print(f"   python -m src.main --dataset_name princeton-nlp/SWE-bench_Lite \\")
    print(f"       --predictions_path {jsonl_path} \\")
    print(f"       --filter_swt --max_workers 4 --run_id mini-swe-agent-run")
    print("2. Check evaluation results in: run_instance_swt_logs/")


if __name__ == "__main__":
    run_workflow()
