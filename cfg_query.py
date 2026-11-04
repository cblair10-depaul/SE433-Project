"""Query the pre-built control flow graphs inside the container.

Kept compatible with old Python versions (no f-strings) because it runs on the
SWE-bench image's interpreter.

Usage:
  python /cfg/query.py find <text>             list functions whose dotted name contains <text>
  python /cfg/query.py cfg <dotted.name>       basic blocks of the function and where each one exits to
  python /cfg/query.py paths <dotted.name>     execution paths through the function and the branch
                                               decisions each one takes
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_LINES = 300
MAX_PATHS = 40
MAX_STEPS = 200000  # bounds the search in functions with very many paths
EXIT_LABELS = {"true": "True", "false": "False", "match": "caught", "nomatch": "not caught",
               "exception": "on exception", "finally": "resuming return/raise"}


def load_defs():
    defs = {}
    with open(os.path.join(HERE, "defs.tsv")) as f:
        for line in f:
            name, loc, qualname = line.rstrip("\n").split("\t")
            defs[name] = (loc, qualname)
    return defs


def resolve(name, defs):
    if name in defs:
        return name
    matches = sorted(n for n in defs if n.endswith("." + name))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        sys.exit("No function named '%s'. Try: python %s find %s" % (name, __file__, name.split(".")[-1]))
    print("'%s' is ambiguous; use one of:" % name)
    for m in matches[:MAX_LINES]:
        print("  %s  (%s)" % (m, defs[m][0]))
    sys.exit(1)


def load_function(name, defs):
    loc, qualname = defs[name]
    path = loc.rsplit(":", 1)[0]
    with open(os.path.join(HERE, "cfg", path + ".json")) as f:
        graph = json.load(f)[qualname]
    blocks = {}
    for block_id, statements, exits in graph["blocks"]:
        blocks[block_id] = (statements, exits)
    return path, graph, blocks


def block_ref(target, blocks):
    if target in ("RETURN", "RAISE"):
        return target
    statements = blocks[target][0]
    return "B%d" % target + (" (line %d)" % statements[0][0] if statements else "")


def show_cfg(name, defs):
    path, graph, blocks = load_function(name, defs)
    order = [b[0] for b in graph["blocks"]]
    decisions = sum(1 for b in order if any(k in ("true", "false") for k, _ in blocks[b][1]))
    lines = ["%s  (%s:%d-%d)" % (name, path, graph["line"], graph["end"]),
             "%d blocks, %d decision point(s). Exceptions that may propagate out of any statement "
             "are not shown." % (len(order), decisions)]
    for block_id in order:
        statements, exits = blocks[block_id]
        lines.append("B%d" % block_id)
        for line, code in statements:
            lines.append("    %d: %s" % (line, code))
        if not statements:
            lines.append("    (empty)")
        parts = []
        for kind, target in exits:
            ref = block_ref(target, blocks)
            parts.append(ref if kind == "next" else "%s: %s" % (EXIT_LABELS[kind], ref))
        lines.append("    -> " + " | ".join(parts))
    emit(lines)


def ends_with(statements, keyword):
    if not statements:
        return False
    return statements[-1][1].split(" ", 1)[0].split("(", 1)[0] == keyword


def show_paths(name, defs):
    """Lists entry-to-exit paths. Each exit of a block is taken at most once per path, so a
    loop is shown as either skipped or entered once."""
    path, graph, blocks = load_function(name, defs)
    start = graph["start"]
    found = []
    truncated = [False]
    steps_left = [MAX_STEPS]

    def walk(block_id, taken, steps, pending):
        """pending is (target, description) for a return or raise passing through a finally
        section, which decides where that section continues."""
        steps_left[0] -= 1
        if len(found) >= MAX_PATHS or steps_left[0] <= 0:
            truncated[0] = True
            return
        statements, exits = blocks[block_id]
        ends_finally = any(kind == "finally" for kind, _ in exits)
        for index, (kind, target) in enumerate(exits):
            if (block_id, index) in taken:
                continue
            if kind == "exception" and ends_with(statements, "except"):
                continue  # evaluating the except clause itself failed; rarely worth a path
            new_pending = pending
            if kind == "finally":
                if pending is None or (pending[0] != target and target in ("RETURN", "RAISE")):
                    continue
                new_pending = None
            elif kind == "next" and ends_finally and pending is not None:
                continue
            elif kind == "match":
                new_pending = None
            elif kind in ("exception", "nomatch"):
                new_pending = ("RAISE", "exception not caught")
            elif kind == "next" and ends_with(statements, "return"):
                new_pending = ("RETURN", "%d: %s" % tuple(statements[-1]))
            elif kind == "next" and ends_with(statements, "raise"):
                new_pending = ("RAISE", "%d: %s" % tuple(statements[-1]))

            new_steps = steps
            if kind not in ("next", "finally"):
                if kind == "exception":
                    where = "exception raised in lines %d-%d" % (statements[0][0], statements[-1][0])                         if statements else "exception raised"
                else:
                    where = "%d: %s" % tuple(statements[-1]) if statements else "?"
                new_steps = steps + ["%s  [%s]" % (where, EXIT_LABELS[kind])]
            if target in ("RETURN", "RAISE"):
                if kind == "finally":
                    end = pending[1]
                elif target == "RETURN":
                    end = new_pending[1] if new_pending else "reaches the end of the function"
                else:
                    end = new_pending[1] if new_pending else "exception not caught"
                found.append((new_steps, target, "  (%s)" % end))
            else:
                walk(target, taken | {(block_id, index)}, new_steps, new_pending)

    if start in ("RETURN", "RAISE"):
        found.append(([], start, ""))
    else:
        walk(start, frozenset(), [], None)
    lines = ["%s  (%s:%d-%d)" % (name, path, graph["line"], graph["end"]),
             "%d paths%s. Each loop is either skipped or entered once." %
             (len(found), " (stopped early; more exist)" if truncated[0] else "")]
    for i, (steps, end, last) in enumerate(found, 1):
        lines.append("Path %d -> %s%s" % (i, end, last))
        for step in steps:
            lines.append("    " + step)
        if not steps:
            lines.append("    (no branch decisions)")
    emit(lines)


def emit(lines):
    for line in lines[:MAX_LINES]:
        print(line)
    if len(lines) > MAX_LINES:
        print("... output truncated at %d lines" % MAX_LINES)


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("find", "cfg", "paths"):
        sys.exit(__doc__)
    sys.setrecursionlimit(10000)
    defs = load_defs()
    command, name = sys.argv[1], sys.argv[2]
    if command == "find":
        matches = sorted(n for n in defs if name in n)
        for m in matches[:MAX_LINES]:
            print("%s  (%s)" % (m, defs[m][0]))
        if not matches:
            print("No matches for '%s'" % name)
        return
    name = resolve(name, defs)
    if command == "cfg":
        show_cfg(name, defs)
    else:
        show_paths(name, defs)


if __name__ == "__main__":
    main()
