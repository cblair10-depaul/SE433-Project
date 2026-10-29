"""Build a call graph of a repository on the host with pyan3.

The SWE-bench Docker images run old Python versions that pyan3 does not support, so the
graph is generated here and copied into the container together with callgraph_query.py.

Output (written to out_dir):
  edges.tsv  caller<TAB>callee         one "uses" edge per line, dotted names
  defs.tsv   name<TAB>path:line        where each function/class is defined
"""
import ast
import logging
import os
import sys
import warnings
from pathlib import Path

from pyan.analyzer import CallGraphVisitor

SKIP_DIRS = {".git", "build", "dist", "docs", "doc", "examples", "benchmarks", "asv_bench",
             "tests", "test", "testing", ".tox", ".eggs", "node_modules", "__pycache__"}


def find_source_files(repo: Path) -> list[str]:
    files = []
    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if not name.endswith(".py") or name.startswith("test_") or name in {"setup.py", "conftest.py"}:
                continue
            path = Path(dirpath) / name
            # pyan3 parses with the host Python; skip files it cannot parse (e.g. Python 2 syntax)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ast.parse(path.read_bytes(), filename=str(path))
            except (SyntaxError, ValueError):
                continue
            files.append(str(path))
    return files


def build_callgraph(repo: Path, out_dir: Path) -> tuple[int, int]:
    """Returns (number of source files analyzed, number of edges)."""
    files = find_source_files(repo)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pyan")
    logger.setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        visitor = CallGraphVisitor(files, root=str(repo), logger=logger)

    edges = set()
    for caller, callees in visitor.uses_edges.items():
        if not caller.defined:
            continue
        for callee in callees:
            if callee.defined:
                edges.add((caller.get_name(), callee.get_name()))

    defs = {}
    for nodes in visitor.nodes.values():
        for node in nodes:
            if node.defined and node.filename and getattr(node.ast_node, "lineno", None):
                rel = Path(node.filename).resolve().relative_to(repo.resolve()).as_posix()
                defs[node.get_name()] = f"{rel}:{node.ast_node.lineno}"

    with open(out_dir / "edges.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.writelines(f"{a}\t{b}\n" for a, b in sorted(edges))
    with open(out_dir / "defs.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.writelines(f"{name}\t{loc}\n" for name, loc in sorted(defs.items()))
    return len(files), len(edges)


if __name__ == "__main__":
    n_files, n_edges = build_callgraph(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"{n_files} files, {n_edges} edges")
