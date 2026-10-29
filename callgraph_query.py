"""Query the pre-built call graph inside the container.

Kept compatible with old Python versions (no f-strings) because it runs on the
SWE-bench image's interpreter.

Usage:
  python /callgraph/query.py find <text>                  list functions whose name contains <text>
  python /callgraph/query.py down <dotted.name> [depth]   functions called by <dotted.name> (default depth 3)
  python /callgraph/query.py up <dotted.name> [depth]     functions that call <dotted.name> (default depth 3)
"""
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_LINES = 300


def load():
    calls, callers, defs = defaultdict(set), defaultdict(set), {}
    with open(os.path.join(HERE, "edges.tsv")) as f:
        for line in f:
            a, b = line.rstrip("\n").split("\t")
            calls[a].add(b)
            callers[b].add(a)
    with open(os.path.join(HERE, "defs.tsv")) as f:
        for line in f:
            name, loc = line.rstrip("\n").split("\t")
            defs[name] = loc
    return calls, callers, defs


def resolve(name, defs):
    if name in defs:
        return name
    matches = sorted(n for n in defs if n == name or n.endswith("." + name))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        sys.exit("No function named '%s'. Try: python %s find %s" % (name, __file__, name.split(".")[-1]))
    print("'%s' is ambiguous; use one of:" % name)
    for m in matches[:MAX_LINES]:
        print("  %s  (%s)" % (m, defs[m]))
    sys.exit(1)


def walk(start, graph, defs, max_depth):
    seen = {start}
    lines = ["%s  (%s)" % (start, defs.get(start, "?"))]
    stack = [(start, 0)]
    while stack and len(lines) < MAX_LINES:
        node, depth = stack.pop()
        if depth >= max_depth:
            continue
        for nxt in sorted(graph.get(node, ()), reverse=True):
            if nxt in seen:
                continue
            seen.add(nxt)
            lines.append("%s%s  (%s)" % ("    " * (depth + 1), nxt, defs.get(nxt, "?")))
            stack.append((nxt, depth + 1))
    for line in lines:
        print(line)
    if len(lines) >= MAX_LINES:
        print("... output truncated at %d lines; use a smaller depth" % MAX_LINES)


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("find", "down", "up"):
        sys.exit(__doc__)
    calls, callers, defs = load()
    command, name = sys.argv[1], sys.argv[2]
    if command == "find":
        matches = sorted(n for n in defs if name in n)
        for m in matches[:MAX_LINES]:
            print("%s  (%s)" % (m, defs[m]))
        if not matches:
            print("No matches for '%s'" % name)
        return
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    walk(resolve(name, defs), calls if command == "down" else callers, defs, depth)


if __name__ == "__main__":
    main()
