"""Every global a function loads must exist in the module that defines it.

Written for a bug that ran in production for weeks. tile_server_v2_.py had two
lines reading `if state in _SLURM_IN_FLIGHT`, which is app_v28.py's constant —
the server's own is IN_FLIGHT_SLURM_STATES. Both lines sat behind a guard that
only runs when a stage is being *re*submitted (`if row.get("kb_load_job_id")`),
so the module imported, the server served every endpoint, and the first
resubmission of Stage 6 got a NameError instead of the refusal the line exists
to produce. Nothing catches this: py_compile checks syntax, the import checks
module-level code, and a test that execs the function against a stub namespace
containing the wrong name passes precisely because it supplies what the module
lacks (test_pipeline_steps does exactly that, legitimately, for the UI's copy).

So this checks name resolution instead, statically, with no import — which
matters because importing tile_server_v2_ builds a FastAPI app and opens HDF5
handles. Binding is deliberately over-approximated: a name bound anywhere in a
scope counts as bound everywhere in it, because the cost of a false positive
here is a test nobody trusts, and the bug being hunted is a name bound in *no*
scope at all.
"""

import ast
import builtins
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

_BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__spec__"}

#: Every module the pipeline actually runs. Hand-listing them went stale the
#: moment two new files (malignancy.py, select_tumour_slides.py) landed
#: unchecked, so the set is derived instead — minus the superseded servers and
#: app_v2..v27, where a finding would be true and useless.
_SUPERSEDED = {"tile_server.py", "tile_server_v3.py", "tile_server_v4.py"}
_LIVE_APP = ("app_v28.py", "api_client.py", "hpc_chat_handlers_v23.py",
             "local_cache.py", "query_planner_v25.py", "plan_query.py",
             "llm_layer_v25.py", "ui_actions_v25.py")


def _targets():
    paths = [p for p in sorted((BACKEND).glob("*.py")) if p.name not in _SUPERSEDED]
    paths += sorted((BACKEND / "tests").glob("*.py"))
    paths += [ROOT / "app" / name for name in _LIVE_APP]
    return [p for p in paths if p.is_file()]


TARGETS = _targets()

_SCOPED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _bound_here(body_nodes) -> set:
    """Names bound in this scope, not descending into nested scopes.

    Comprehension and lambda targets are folded in rather than given their own
    scope: over-approximating what is bound cannot hide an undefined *global*,
    which is the only thing this file is looking for.
    """
    bound = set()
    stack = list(body_nodes)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        if isinstance(node, _SCOPED):
            # The nested scope's *name* is bound here; its body is not this scope's.
            if not isinstance(node, ast.Lambda):
                bound.add(node.name)
            continue
        stack.extend(ast.iter_child_nodes(node))
    return bound


def _params(node) -> set:
    args = node.args
    names = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names


def undefined_names(source: str, filename: str = "<probe>") -> list:
    """[(lineno, name)] for every Name load resolving to no enclosing scope."""
    tree = ast.parse(source, filename)
    findings = []

    def visit(node, body_nodes, visible, params=frozenset()):
        scope = visible | params | _bound_here(body_nodes)
        stack = list(body_nodes)
        while stack:
            child = stack.pop()
            if isinstance(child, _SCOPED):
                if isinstance(child, ast.Lambda):
                    visit(child, [child.body], scope, _params(child))
                elif isinstance(child, ast.ClassDef):
                    # A class body sees the enclosing scope but does not extend
                    # it for nested functions, which is close enough here.
                    visit(child, child.body, scope)
                else:
                    visit(child, child.body, scope, _params(child))
                    for d in child.decorator_list + list(ast.walk(child.args)):
                        stack.append(d)
                continue
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                if child.id not in scope:
                    findings.append((child.lineno, child.id))
            stack.extend(ast.iter_child_nodes(child))

    visit(tree, tree.body, _BUILTINS)
    return sorted(set(findings))


# --- the check itself -----------------------------------------------------

def test_no_module_loads_a_name_it_never_defines(_tmp=None):
    problems = []
    for path in TARGETS:
        if not path.is_file():
            continue
        for lineno, name in undefined_names(path.read_text(), str(path)):
            problems.append(f"{path.relative_to(ROOT)}:{lineno}: {name}")
    assert not problems, (
        "These names are loaded but bound in no scope, so they raise NameError "
        "the first time their branch runs:\n  " + "\n  ".join(problems))


def test_the_server_uses_its_own_in_flight_constant(_tmp=None):
    """The specific instance, pinned by name.

    Kept alongside the general check because the two modules define sets with
    almost the same contents under different names, so a copy-paste between
    them is easy to make again and reads correctly.
    """
    server = (BACKEND / "tile_server_v2_.py").read_text()
    assert "IN_FLIGHT_SLURM_STATES = {" in server
    assert "_SLURM_IN_FLIGHT" not in server, (
        "_SLURM_IN_FLIGHT is app_v28.py's constant; the server's is "
        "IN_FLIGHT_SLURM_STATES.")


# --- and that the check can fail --------------------------------------------

def test_the_checker_catches_a_borrowed_constant(_tmp=None):
    """The bug this file exists for, reduced.

    Without this the suite proves only that a checker ran, not that it can come
    out bad — and a checker whose binding rules are too generous reports clean
    on everything, this bug included.
    """
    borrowed = (
        "IN_FLIGHT = {'PENDING'}\n"
        "def refuse(row, state):\n"
        "    if row.get('job_id'):\n"
        "        if state in _SLURM_IN_FLIGHT:\n"
        "            raise ValueError(state)\n"
    )
    assert undefined_names(borrowed) == [(4, "_SLURM_IN_FLIGHT")]


def test_the_checker_does_not_flag_closures_or_comprehensions(_tmp=None):
    """The false positives that would make it unusable on this codebase.

    kb_stage's staging worker closes over engine, frame and table; app_v28 sorts
    with `key=lambda kv: kv[1]`. A checker that flags either is noise, and noise
    is how a real finding gets ignored.
    """
    fine = (
        "import os\n"
        "def outer(engine, frame, table, workers):\n"
        "    def inner(bound):\n"
        "        start, stop = bound\n"
        "        return engine, frame.iloc[start:stop], table, os.sep\n"
        "    pairs = sorted({}.items(), key=lambda kv: kv[1])\n"
        "    return [inner(b) for b in pairs if b]\n"
        "class C:\n"
        "    attr = os.sep\n"
        "    def m(self):\n"
        "        try:\n"
        "            return self.attr\n"
        "        except ValueError as e:\n"
        "            return e\n"
    )
    assert undefined_names(fine) == []


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_undefined_names_test_"))
        try:
            fn(tmp_path)
            print(f"PASS  {name}")
        except Exception as e:
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
