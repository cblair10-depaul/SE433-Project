import argparse
import importlib.util
import os
import json
import re
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")
CONFIG_PATH = PROJECT_DIR / "config_local_model.yaml"
BASELINE_CONFIG_PATH = PROJECT_DIR / "baseline_agent_config.yaml"
DEFAULT_MINI_SWE_CONFIG_PATH = PROJECT_DIR / "default_swe_mini_config.yaml"
OUTPUT_DIR = PROJECT_DIR / "swebench_predictions"
BASELINE_OUTPUT_DIR = PROJECT_DIR / "swebench_predictions_baseline"
DEFAULT_MINI_SWE_OUTPUT_DIR = PROJECT_DIR / "swebench_predictions_default_mini_swe"
PREDICTIONS_JSONL_NAME = "predictions.jsonl"
PREDICTIONS_JSON_NAME = "preds.json"
REPO_CACHE_DIR = PROJECT_DIR / "workspaces" / "repo_cache"
WORKSPACE_DIR = PROJECT_DIR / "workspaces" / "instances"
TRAJECTORY_DIR = PROJECT_DIR / "workspaces" / "trajectories"
CFG_DIR = PROJECT_DIR / "workspaces" / "cfgs"
CFG_BUILD_SCRIPT = PROJECT_DIR / "cfg_host.py"
CFG_QUERY_SCRIPT = PROJECT_DIR / "cfg_query.py"
CONTAINER_CFG_DIR = "/cfg"
DEFAULT_WORKERS = 4
CONTAINER_NAME_PREFIX = "minisweagent-"  # set by DockerEnvironment

DATASET_NAME = "eth-sri/SWT-bench_Verified_bm25_27k_zsp"  # SWT-bench Verified (433 instances)

TEST_DIR_NAMES = {"test", "tests", "testing"}
# unittest/pytest/django asserts, pytest.raises, and sympy's raises()
ASSERTION_PATTERN = re.compile(r"\bassert\b|\.assert[A-Z]\w*\(|\braises\(|\.fail\(")
SCRATCH_NAME_PATTERN = re.compile(r"repro|scratch|debug", re.IGNORECASE)

REPO_CLONE_LOCK = threading.Lock()
OUTPUT_LOCK = threading.Lock()


def log(instance_id: str, message: str) -> None:
    with OUTPUT_LOCK:
        print(f"  [{instance_id}] {message}", flush=True)


def load_config(config_path: Path = CONFIG_PATH):
    config_spec = [
        str(builtin_config_dir / "mini.yaml"),
        str(config_path),
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
    # Held while cloning so workers on the same repo can't clone it at the same time
    with REPO_CLONE_LOCK:
        if not cache.exists():
            log(instance["instance_id"], f"Cloning {repo} (first time only)...")
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


def pull_image(image: str, instance_id: str) -> None:
    # Images are kept after use: they share layers, and later instances and reruns reuse them
    exists = subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0
    if not exists:
        log(instance_id, f"Pulling {image}...")
        subprocess.run(["docker", "pull", "--quiet", image], check=True, capture_output=True)


def load_predictions(output_dir: Path) -> dict:
    """Returns the predictions saved by earlier runs, keyed by instance ID."""
    predictions_jsonl = output_dir / PREDICTIONS_JSONL_NAME
    if not predictions_jsonl.exists():
        return {}
    with open(predictions_jsonl) as f:
        predictions = [json.loads(line) for line in f if line.strip()]
    return {pred["instance_id"]: pred for pred in predictions}


def write_atomically(path: Path, content: str) -> None:
    # Replacing in one step means an interrupted run can't leave a half-written file
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(content)
    os.replace(tmp_path, path)


def save_predictions(predictions: dict, output_dir: Path) -> None:
    preds = list(predictions.values())
    write_atomically(output_dir / PREDICTIONS_JSONL_NAME, "".join(json.dumps(pred) + "\n" for pred in preds))
    write_atomically(output_dir / PREDICTIONS_JSON_NAME, json.dumps(preds, indent=2))


def prepare_cfgs(instance: dict) -> Path:
    """Builds control flow graphs with python-graphs on the host against a checkout at the
    instance's base commit.

    The graphs only depend on that commit, so finished ones are reused. Delete CFG_DIR
    after changing cfg_host.py to rebuild them.
    """
    instance_id = instance["instance_id"]
    out_dir = CFG_DIR / instance_id
    query_script = out_dir / "query.py"
    # query.py is written last, so an interrupted build never counts as finished
    if query_script.exists():
        shutil.copy(CFG_QUERY_SCRIPT, query_script)
        log(instance_id, "Reusing saved control flow graphs")
        return out_dir

    workspace = prepare_workspace(instance)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    # In its own process because the build is CPU-bound, which threads wouldn't run in parallel
    result = subprocess.run(
        [sys.executable, str(CFG_BUILD_SCRIPT), str(workspace), str(out_dir)],
        capture_output=True, text=True, errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"control flow graph build failed: {result.stderr}")
    shutil.copy(CFG_QUERY_SCRIPT, query_script)
    log(instance_id, f"Built control flow graphs: {result.stdout.strip()}")
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


def remove_all_containers() -> None:
    result = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"name={CONTAINER_NAME_PREFIX}"],
        capture_output=True, text=True,
    )
    container_ids = result.stdout.split()
    if container_ids:
        subprocess.run(["docker", "rm", "-f", *container_ids], capture_output=True)


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


def extract_test_patch(env, base_commit: str, submitted: bool, instance_id: str) -> str:
    container_git(env, "add", "-A")
    tracked = set(container_git(env, "ls-tree", "-r", "--name-only", base_commit).splitlines())
    changed = container_git(env, "diff", "--cached", "--name-only", base_commit).splitlines()

    test_files = [f for f in changed if is_test_file(f, f in tracked)]
    skipped = [f for f in changed if f not in test_files]
    if skipped:
        log(instance_id, f"Excluded non-test changes from patch: {skipped}")
    # Only new files can be scratch; edits to the repo's existing test files are always kept
    scratch = [f for f in test_files if f not in tracked and is_scratch_file(env, f, submitted)]
    if scratch:
        log(instance_id, f"Excluded scratch test files from patch: {scratch}")
        test_files = [f for f in test_files if f not in scratch]
    if not test_files:
        return ""
    return container_git(env, "diff", "--cached", base_commit, "--", *test_files)


def run_agent_on_instance(agent: Agent, task: str, instance: dict) -> dict:
    instance_id = instance["instance_id"]
    log(instance_id, f"Running agent in {agent.env.config.image}...")

    # The SWE-bench images commit their environment setup on top of base_commit (including
    # file-mode changes), so diff against the image's HEAD rather than base_commit.
    start_commit = container_git(agent.env, "rev-parse", "HEAD").strip()

    result = agent.run(task)
    exit_status = result.get("exit_status", "unknown")
    log(instance_id, f"Exit status: {exit_status}")

    model_patch = extract_test_patch(agent.env, start_commit, submitted=exit_status == "Submitted", instance_id=instance_id)

    return {
        "instance_id": instance_id,
        "model_name_or_path": "deepseek/deepseek-flash",
        "model_patch": model_patch
    }


def process_instance(instance: dict, config: dict, model_obj) -> dict:
    """Runs on a worker thread. The model is shared: it keeps no per-run state."""
    instance_id = instance["instance_id"]
    env_obj = None
    try:
        cfg_dir = prepare_cfgs(instance)
        image = get_swebench_docker_image_name(instance)
        pull_image(image, instance_id)
        env_config = {**config.get("environment", {}), "image": image}
        env_obj = get_environment(env_config, default_type="docker")
        copy_into_container(env_obj, cfg_dir, CONTAINER_CFG_DIR)

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

        return run_agent_on_instance(agent, task, instance)

    finally:
        if env_obj is not None:
            remove_container(env_obj)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mini-swe-agent on SWT-bench Verified instances.")
    parser.add_argument("instance_ids", nargs="*", help="instances to run (default: the first 5)")
    parser.add_argument("--all", action="store_true", help="run every instance in the dataset (433)")
    parser.add_argument("--fresh", action="store_true", help="discard saved predictions instead of resuming from them")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"instances to run at the same time (default: {DEFAULT_WORKERS})")
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument("--baseline", action="store_true",
                              help=f"use {BASELINE_CONFIG_PATH.name} instead of {CONFIG_PATH.name}")
    config_group.add_argument("--default-mini-swe-config", action="store_true",
                              help=f"use {DEFAULT_MINI_SWE_CONFIG_PATH.name} instead of {CONFIG_PATH.name}")
    args = parser.parse_args()
    if args.all and args.instance_ids:
        parser.error("--all cannot be combined with instance IDs")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def check_cfg_dependencies() -> None:
    """cfg_host.py runs under this interpreter, so a missing python-graphs would otherwise
    fail every instance without saved graphs, one at a time."""
    missing = [name for name in ("python_graphs", "gast", "absl")
               if importlib.util.find_spec(name) is None]
    if missing:
        sys.exit(f"{sys.executable} is missing {', '.join(missing)}, which cfg_host.py needs.\n"
                 "Run with the project venv (.\\.venv\\Scripts\\python.exe) or "
                 "install requirements.txt into this interpreter.")


def run_workflow():
    args = parse_args()
    check_cfg_dependencies()

    print("SWT-Bench Integration for mini-swe-agent")

    if args.baseline:
        config_path, output_dir, run_id = BASELINE_CONFIG_PATH, BASELINE_OUTPUT_DIR, "mini-swe-agent-baseline"
    elif args.default_mini_swe_config:
        config_path, output_dir, run_id = (DEFAULT_MINI_SWE_CONFIG_PATH, DEFAULT_MINI_SWE_OUTPUT_DIR,
                                           "mini-swe-agent-default")
    else:
        config_path, output_dir, run_id = CONFIG_PATH, OUTPUT_DIR, "mini-swe-agent-run"
    print(f"Config: {config_path.name}")
    config = load_config(config_path)
    model_name = config.get("model", {}).get("model_name", "unknown")
    print(f"Model: {model_name}")

    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_dir}")

    if args.instance_ids:
        # A duplicate ID would have two workers sharing one workspace
        instance_ids = list(dict.fromkeys(args.instance_ids))
    else:
        if not args.all:
            print("No instance IDs provided. Using first instances from SWT-bench Verified for testing.")
            print("To specify instances: python run_swt_bench_complete.py instance_id1 instance_id2 ...")
            print("To run every instance: python run_swt_bench_complete.py --all")
        instance_ids = None

    print(f"Loading {DATASET_NAME} dataset...")
    try:
        dataset = load_dataset(DATASET_NAME, split="test")
        dataset_dict = {inst["instance_id"]: inst for inst in dataset}

        if args.all:
            instance_ids = list(dataset_dict.keys())
        elif instance_ids is None:
            # Use first 5 instances as default test
            instance_ids = list(dataset_dict.keys())[:10]
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

    predictions = {} if args.fresh else load_predictions(output_dir)
    # Instances named on the command line always run, replacing any saved prediction
    if not args.instance_ids:
        finished = [id for id in instance_ids if id in predictions]
        if finished:
            print(f"Skipping {len(finished)} instances with saved predictions (--fresh to rerun them)")
            instance_ids = [id for id in instance_ids if id not in predictions]

    print("Setting up agent...")
    model_obj = get_model(config=config.get("model", {}))

    print(f"\nProcessing {len(instance_ids)} instances with {args.workers} workers...")

    errored = []

    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = {
        executor.submit(process_instance, dataset_dict[instance_id], config, model_obj): instance_id
        for instance_id in instance_ids
    }
    pending = set(futures)
    n_finished = 0
    try:
        while pending:
            # The timeout lets Ctrl+C through; a wait without one can't be interrupted on Windows
            done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            # Results are handled only on this thread, so predictions needs no lock
            for future in done:
                instance_id = futures[future]
                n_finished += 1
                progress = f"[{n_finished}/{len(instance_ids)}]"
                try:
                    prediction = future.result()
                except Exception as e:
                    error_msg = str(e).replace('✗', '[ERROR]')
                    log(instance_id, f"{progress} [ERROR] {error_msg}")
                    with OUTPUT_LOCK:
                        traceback.print_exception(e)

                    # Not saved as a prediction, so the next run retries it
                    errored.append(instance_id)
                else:
                    predictions[instance_id] = prediction
                    save_predictions(predictions, output_dir)
                    log(instance_id, f"{progress} [OK] Generated patch ({len(prediction['model_patch'])} chars)")
    except KeyboardInterrupt:
        # Ctrl+C also kills the workers' docker commands, so their results can't be trusted.
        # Finished instances are already saved; a rerun picks up everything else.
        print("Interrupted: discarding running instances and removing their containers...", flush=True)
        executor.shutdown(wait=False, cancel_futures=True)
        remove_all_containers()
        os._exit(130)
    executor.shutdown()

    print("SUMMARY")
    print(f"Saved predictions: {len(predictions)}")
    print(f"With a patch: {sum(1 for p in predictions.values() if p['model_patch'])}")
    print(f"Without a patch: {sum(1 for p in predictions.values() if not p['model_patch'])}")
    print(f"Errored this run (will retry on next run): {errored}")
    print(f"\nPredictions saved to: {output_dir}")

    print("NEXT STEPS")
    print("1. Run SWT-Bench evaluation:")
    print("   cd ~/Documents/swt-bench")
    print("   source .venv/bin/activate")
    # Evaluate against princeton-nlp/SWE-bench_Verified + --filter_swt (= the same 433 instances).
    # DATASET_NAME's eth-sri copy stores the reference test in `patch` and the code fix in
    # `test_patch`; without --is_swt, SWT-bench reads them the SWE-bench way round and never
    # applies the code fix, so no generated test can ever be fail-to-pass.
    print("   python -m src.main --dataset_name princeton-nlp/SWE-bench_Verified --filter_swt \\")
    print(f"       --predictions_path {output_dir / PREDICTIONS_JSONL_NAME} \\")
    # Each config has its own run ID so their evaluation logs stay apart
    print(f"       --max_workers 4 --run_id {run_id}")
    print("2. Check evaluation results in: run_instance_swt_logs/")


if __name__ == "__main__":
    run_workflow()
