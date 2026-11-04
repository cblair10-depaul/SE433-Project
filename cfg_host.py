"""Build control flow graphs for every function in a repository on the host with python-graphs.

The SWE-bench Docker images run old Python versions that python-graphs does not support, so
the graphs are generated here and copied into the container together with cfg_query.py.

Output (written to out_dir):
  defs.tsv          name<TAB>path:line<TAB>qualname   one line per function or method
  cfg/<path>.json   {qualname: {"line", "end", "start", "blocks"}} for each source file

Each entry of "blocks" is [block_id, statements, exits]:
  statements  [[line, code], ...] in execution order
  exits       [[kind, target], ...] where kind is "next", "true", "false" (an if/while/for
              header or match case), "match"/"nomatch" (an except clause) or "exception"
              (an exception caught inside the function), and target is a block id,
              "RETURN" or "RAISE"
"""
import ast
import json
import os
import sys
import tokenize
import warnings
from pathlib import Path

import gast
from absl import logging as absl_logging
from python_graphs import control_flow

SKIP_DIRS = {".git", "build", "dist", "docs", "doc", "examples", "benchmarks", "asv_bench",
             "tests", "test", "testing", ".tox", ".eggs", "node_modules", "__pycache__"}
MAX_CODE_LEN = 160
FUNCTION_NODES = (gast.FunctionDef, gast.AsyncFunctionDef)


class Visitor(control_flow.ControlFlowVisitor):
    """Fills in statements python-graphs 1.2.3 skips; without these, the bodies of with
    statements, async code and match statements would be missing from the graph."""

    def visit_AsyncFunctionDef(self, node, current_block):
        return self.visit_FunctionDef(node, current_block)

    def visit_AsyncFor(self, node, current_block):
        return self.visit_For(node, current_block)

    def visit_TryStar(self, node, current_block):
        return self.visit_Try(node, current_block)

    def visit_With(self, node, current_block):
        for item in node.items:
            self.add_new_instruction(current_block, item.context_expr)
        return self.visit_list(node.body, current_block)

    visit_AsyncWith = visit_With

    def visit_AnnAssign(self, node, current_block):
        if node.value is not None:
            self.add_new_instruction(current_block, node)
        return current_block

    def visit_Nonlocal(self, node, current_block):
        return current_block

    def visit_Match(self, node, current_block):
        self.add_new_instruction(current_block, node.subject)
        after_block = self.new_block(node=node, label="after_block")
        previous_test = None
        for case in node.cases:
            test_block = self.new_block(node=case, label="case_block")
            if previous_test is None:
                current_block.add_exit(test_block)
            else:
                previous_test.add_exit(test_block, branch=False)
            self.add_new_instruction(test_block, case.pattern)
            body_block = self.new_block(node=case, label="true_block")
            test_block.add_exit(body_block, branch=True)
            self.visit_list(case.body, body_block).add_exit(after_block)
            previous_test = test_block
        if previous_test is None:
            current_block.add_exit(after_block)
        else:
            previous_test.add_exit(after_block, branch=False)
        return after_block


def find_source_files(repo: Path) -> list[Path]:
    files = []
    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if name.endswith(".py") and not name.startswith("test_") and name not in {"setup.py", "conftest.py"}:
                files.append(Path(dirpath) / name)
    return sorted(files)


def iter_functions(tree):
    """Yields (qualname, node) for every function and method, including nested ones."""
    stack = [(tree, "")]
    while stack:
        node, prefix = stack.pop()
        for child in gast.iter_child_nodes(node):
            if isinstance(child, (gast.ClassDef,) + FUNCTION_NODES):
                qualname = prefix + child.name
                if isinstance(child, FUNCTION_NODES):
                    yield qualname, child
                stack.append((child, qualname + "."))
            elif not isinstance(child, (gast.expr, gast.Lambda)):
                stack.append((child, prefix))


def header_roles(function):
    """Maps id(ast node) to how the statement that owns it should be displayed."""
    roles = {}
    for node in gast.walk(function):
        if isinstance(node, (gast.If, gast.While)):
            roles[id(node.test)] = ("if" if isinstance(node, gast.If) else "while", node)
        elif isinstance(node, (gast.For, gast.AsyncFor)):
            roles[id(node.target)] = ("for", node)
            roles[id(node.iter)] = ("iter", node)
        elif isinstance(node, gast.ExceptHandler) and node.type is not None:
            roles[id(node.type)] = ("except", node)
        elif isinstance(node, (gast.With, gast.AsyncWith)):
            for item in node.items:
                roles[id(item.context_expr)] = ("with", item)
        elif isinstance(node, gast.Match):
            roles[id(node.subject)] = ("match", node)
            for case in node.cases:
                roles[id(case.pattern)] = ("case", case)
        elif isinstance(node, (gast.FunctionDef, gast.AsyncFunctionDef, gast.ClassDef)):
            for decorator in node.decorator_list:
                roles[id(decorator)] = ("decorator", node)
    return roles


def describe(node, roles, source):
    """Returns the source text shown for one instruction, or None to leave it out."""
    if isinstance(node, gast.arguments) or not hasattr(node, "lineno"):
        return None  # parameter binding at function entry
    if isinstance(node, FUNCTION_NODES):
        return "def %s(...)" % node.name
    if isinstance(node, gast.ClassDef):
        return "class %s" % node.name

    def seg(n):
        return ast.get_source_segment(source, n) or "?"

    role, owner = roles.get(id(node), (None, None))
    if role in ("if", "while"):
        code = "%s %s" % (role, seg(node))
    elif role == "for":
        code = "for %s in %s" % (seg(node), seg(owner.iter))
    elif role == "iter":
        return None  # shown with the loop header
    elif role == "except":
        code = "except %s" % seg(node)
    elif role == "with":
        code = "with %s" % seg(node) + (" as %s" % seg(owner.optional_vars) if owner.optional_vars else "")
    elif role == "match":
        code = "match %s" % seg(node)
    elif role == "case":
        code = "case %s" % seg(node) + (" if %s" % seg(owner.guard) if owner.guard else "")
    elif role == "decorator":
        return None  # runs where the function is defined, not when it is called
    elif isinstance(node, (gast.Name, gast.Tuple, gast.List)):
        return None  # "except E as name" binding
    elif isinstance(node, gast.Expr) and isinstance(node.value, gast.Constant) and isinstance(node.value.value, str):
        return None  # docstring
    else:
        code = seg(node)
    code = " ".join(code.split())
    if len(code) > MAX_CODE_LEN:
        code = code[:MAX_CODE_LEN - 3] + "..."
    return code


def function_cfg(function, source):
    visitor = Visitor()
    visitor.run(gast.Module(body=[function], type_ignores=[]))
    graph = visitor.graph

    entry = return_block = raise_block = None
    for block in graph.blocks:
        if block.node is function:
            if block.label == "<entry:%s>" % function.name:
                entry = block
            elif block.label == "<return>":
                return_block = block
            elif block.label == "<raise>":
                raise_block = block

    # Blocks reachable from the function's entry; nested functions have their own entries
    order, seen, stack = [], {entry, return_block, raise_block}, list(entry.next)
    order_start = next(iter(entry.next))
    while stack:
        block = stack.pop()
        if block in seen:
            continue
        seen.add(block)
        order.append(block)
        stack.extend(block.next)

    roles = header_roles(function)
    statements = {}
    for block in order:
        statements[block] = []
        for cf_node in block.control_flow_nodes:
            code = describe(cf_node.instruction.node, roles, source)
            if code is not None:
                statements[block].append([cf_node.instruction.node.lineno, code])

    def skip_empty(block):
        """Follows blocks that show no statements (e.g. one holding only parameter binding)."""
        hops = 0
        while (block in statements and not statements[block] and len(block.exits_from_end) == 1
               and not block.branches and not block.except_branches and hops < 100):
            block = next(iter(block.exits_from_end))
            hops += 1
        return block

    order = [b for b in order if skip_empty(b) is b]
    order.sort(key=lambda b: (statements[b][0][0] if statements[b] else float("inf")))
    ids = {block: i + 1 for i, block in enumerate(order)}
    ids[return_block], ids[raise_block] = "RETURN", "RAISE"

    def ref(block):
        return ids[skip_empty(block)]

    blocks = []
    for block in order:
        exits, covered = [], set()
        for key, target in sorted(block.branches.items(), reverse=True):
            exits.append(["true" if key else "false", ref(target)])
            covered.add(target)
        for key, target in sorted(block.except_branches.items(), reverse=True):
            exits.append(["match" if key else "nomatch", ref(target)])
            covered.add(target)
        # The end of a finally section continues normally only to reraise_branches[False];
        # its other exits are taken only when a return, raise, break or continue passed through it
        normal_after_finally = block.reraise_branches.get(False)
        for target in sorted(block.exits_from_end - covered, key=lambda b: str(ref(b))):
            kind = "finally" if normal_after_finally is not None and target is not normal_after_finally else "next"
            exits.append([kind, ref(target)])
            covered.add(target)
        for target in sorted(block.exits_from_middle - covered - {raise_block}, key=lambda b: str(ref(b))):
            exits.append(["exception", ref(target)])
        blocks.append([ids[block], statements[block], exits])
    return {"line": function.lineno, "end": getattr(function, "end_lineno", function.lineno),
            "start": ref(order_start), "blocks": blocks}


def build_cfgs(repo: Path, out_dir: Path) -> tuple[int, int, int]:
    """Returns (number of source files analyzed, functions graphed, functions skipped)."""
    absl_logging.set_verbosity(absl_logging.ERROR)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_files = n_functions = n_failed = 0
    defs = []
    for path in find_source_files(repo):
        rel = path.relative_to(repo).as_posix()
        try:
            with tokenize.open(path) as f:
                source = f.read()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree = gast.parse(source, filename=str(path))
        except (SyntaxError, ValueError, UnicodeDecodeError):
            continue  # e.g. Python 2 syntax the host Python cannot parse
        n_files += 1
        module = rel[:-3].replace("/", ".")
        if module.endswith(".__init__"):
            module = module[:-len(".__init__")]
        graphs = {}
        for qualname, function in iter_functions(tree):
            try:
                graphs[qualname] = function_cfg(function, source)
            except Exception:  # python-graphs does not handle every construct; skip that function
                n_failed += 1
                continue
            defs.append((module + "." + qualname, "%s:%d" % (rel, function.lineno), qualname))
        if graphs:
            n_functions += len(graphs)
            target = out_dir / "cfg" / (rel + ".json")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(graphs, separators=(",", ":")), encoding="utf-8")

    with open(out_dir / "defs.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.writelines("%s\t%s\t%s\n" % d for d in sorted(defs))
    return n_files, n_functions, n_failed


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    n_files, n_functions, n_failed = build_cfgs(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"{n_files} files, {n_functions} functions" + (f" ({n_failed} skipped)" if n_failed else ""))
