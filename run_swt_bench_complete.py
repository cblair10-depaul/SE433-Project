import os
import sys
import json
import re
import shlex
import shutil
import stat
import subprocess
from datetime import datetime
from pathlib import Path, PurePosixPath

from datasets import load_dataset
from dotenv import load_dotenv
from minisweagent import Agent
from minisweagent.config import get_config_from_spec, builtin_config_dir
from minisweagent.utils.serialize import recursive_merge
from minisweagent.agents import get_agent
from minisweagent.models import get_model
from minisweagent.environments import get_environment
from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name

from callgraph_host import build_callgraph

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")
CONFIG_PATH = PROJECT_DIR / "config_local_model.yaml"
OUTPUT_DIR = PROJECT_DIR / "swebench_predictions"
REPO_CACHE_DIR = PROJECT_DIR / "workspaces" / "repo_cache"
WORKSPACE_DIR = PROJECT_DIR / "workspaces" / "instances"
TRAJECTORY_DIR = PROJECT_DIR / "workspaces" / "trajectories"
CALLGRAPH_DIR = PROJECT_DIR / "workspaces" / "callgraphs"
CALLGRAPH_QUERY_SCRIPT = PROJECT_DIR / "callgraph_query.py"
CONTAINER_CALLGRAPH_DIR = "/callgraph"

DATASET_NAME = "eth-sri/SWT-bench_Verified_bm25_27k_zsp"  # SWT-bench Verified (433 instances)

TEST_DIR_NAMES = {"test", "tests", "testing"}
# unittest/pytest/django asserts, pytest.raises, and sympy's raises()
ASSERTION_PATTERN = re.compile(r"\bassert\b|\.assert[A-Z]\w*\(|\braises\(|\.fail\(")
SCRATCH_NAME_PATTERN = re.compile(r"repro|scratch|debug", re.IGNORECASE)


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


def _clear_readonly_and_retry(func, path, exc):
    # Git marks object files read-only; on Windows that blocks deletion.
    os.chmod(path, stat.S_IWRITE)
    func(path)


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
        shutil.rmtree(workspace, onexc=_clear_readonly_and_retry)
    workspace.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", "--no-checkout", str(cache), str(workspace)],
        check=True,
    )
    git("checkout", "--quiet", instance["base_commit"], cwd=workspace)
    return workspace


def pull_image(image: str) -> None:
    exists = subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0
    if not exists:
        print(f"  Pulling {image} (first time only)...")
        subprocess.run(["docker", "pull", "--quiet", image], check=True)


def prepare_callgraph(instance: dict) -> Path:
    """Runs pyan3 on the host against a checkout at the instance's base commit."""
    workspace = prepare_workspace(instance)
    out_dir = CALLGRAPH_DIR / instance["instance_id"]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    n_files, n_edges = build_callgraph(workspace, out_dir)
    shutil.copy(CALLGRAPH_QUERY_SCRIPT, out_dir / "query.py")
    print(f"  Built call graph: {n_files} files, {n_edges} edges")
    return out_dir


def copy_into_container(env, host_dir: Path, container_dir: str) -> None:
    subprocess.run(
        ["docker", "cp", f"{host_dir}{os.sep}.", f"{env.container_id}:{container_dir}"],
        check=True, capture_output=True,
    )


def remove_container(env) -> None:
    # DockerEnvironment.cleanup() uses bash syntax that cmd.exe can't run on Windows,
    # so the container would keep running until its 2h sleep ends.
    if env.container_id:
        subprocess.run(["docker", "rm", "-f", env.container_id], capture_output=True)
        env.container_id = None


def container_git(env, *args: str) -> str:
    result = env.execute({"command": shlex.join(["git", "--no-pager", *args])})
    if result["returncode"] != 0:
        raise RuntimeError(f"git {args[0]} failed in container: {result['output']}")
    return result["output"]


def is_test_file(path: str, tracked_at_base: bool) -> bool:
    p = PurePosixPath(path)
    if p.suffix != ".py":
        return False
    in_test_dir = any(part in TEST_DIR_NAMES for part in p.parts[:-1])
    test_name = p.name.startswith("test_") or p.name.endswith("_test.py") or p.name == "tests.py"
    if not (in_test_dir or test_name):
        return False
    return len(p.parts) > 1 or tracked_at_base


def is_scratch_file(env, path: str, submitted: bool) -> bool:
    """A new test file is scratch if it asserts nothing, or if the agent stopped before its
    cleanup step (e.g. hit the step limit) and the file is named like a repro/debug script."""
    content = env.execute({"command": shlex.join(["cat", path])})["output"]
    if not ASSERTION_PATTERN.search(content):
        return True
    return not submitted and bool(SCRATCH_NAME_PATTERN.search(PurePosixPath(path).name))


def extract_test_patch(env, base_commit: str, submitted: bool) -> str:
    container_git(env, "add", "-A")
    tracked = set(container_git(env, "ls-tree", "-r", "--name-only", base_commit).splitlines())
    changed = container_git(env, "diff", "--cached", "--name-only", base_commit).splitlines()

    test_files = [f for f in changed if is_test_file(f, f in tracked)]
    skipped = [f for f in changed if f not in test_files]
    if skipped:
        print(f"  Excluded non-test changes from patch: {skipped}")
    # Only new files can be scratch; edits to the repo's existing test files are always kept
    scratch = [f for f in test_files if f not in tracked and is_scratch_file(env, f, submitted)]
    if scratch:
        print(f"  Excluded scratch test files from patch: {scratch}")
        test_files = [f for f in test_files if f not in scratch]
    if not test_files:
        return ""
    return container_git(env, "diff", "--cached", base_commit, "--", *test_files)


def run_agent_on_instance(agent: Agent, task: str, instance: dict) -> dict:
    instance_id = instance["instance_id"]
    print(f"  Running agent on {instance_id} in {agent.env.config.image}...")

    # The SWE-bench images commit their environment setup on top of base_commit (including
    # file-mode changes), so diff against the image's HEAD rather than base_commit.
    start_commit = container_git(agent.env, "rev-parse", "HEAD").strip()

    result = agent.run(task)
    exit_status = result.get("exit_status", "unknown")
    print(f"  Exit status: {exit_status}")

    model_patch = extract_test_patch(agent.env, start_commit, submitted=exit_status == "Submitted")

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
        print("No instance IDs provided. Using first instances from SWT-bench Verified for testing.")
        print("To specify instances: python run_swt_bench_complete.py instance_id1 instance_id2 ...")
        instance_ids = None

    print(f"Loading {DATASET_NAME} dataset...")
    try:
        dataset = load_dataset(DATASET_NAME, split="test")
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

        env_obj = None
        try:
            instance = dataset_dict[instance_id]

            callgraph_dir = prepare_callgraph(instance)
            image = get_swebench_docker_image_name(instance)
            pull_image(image)
            env_config = {**config.get("environment", {}), "image": image}
            env_obj = get_environment(env_config, default_type="docker")
            copy_into_container(env_obj, callgraph_dir, CONTAINER_CALLGRAPH_DIR)

            task = format_task(instance)

            # Create fresh agent for this instance (Don't cross contexts for better results)
            agent = get_agent(
                model=model_obj,
                env=env_obj,
                # output_path makes the agent save its trajectory after every step
                config={
                    **config.get("agent", {}),
                    # timestamped so re-running an instance never overwrites an earlier trajectory
                    "output_path": TRAJECTORY_DIR / f"{instance_id}.{datetime.now():%Y%m%d-%H%M%S}.traj.json",
                },
                default_type="default"
            )

            prediction = run_agent_on_instance(agent, task, instance)
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

        finally:
            if env_obj is not None:
                remove_container(env_obj)

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
    print(f"   python -m src.main --dataset_name {DATASET_NAME} \\")
    print(f"       --predictions_path {jsonl_path} \\")
    print(f"       --max_workers 4 --run_id mini-swe-agent-run")
    print("2. Check evaluation results in: run_instance_swt_logs/")


if __name__ == "__main__":
    run_workflow()
