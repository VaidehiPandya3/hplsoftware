"""Names that resolve inside one module but not across the seam to another.

Companion to test_undefined_names.py, which catches a name bound in no scope at
all. These are the same failure shape one step out — each fires only when its
branch runs, and each has already happened here:

  * submit_cluster_assignment.py called bootstrap_container_extras() and read
    _CONTAINER_EXTRA_PACKAGES_GPU without importing either from
    submit_feature_extraction, behind `if args.bootstrap_gpu_faiss`. The module
    imported fine; --bootstrap-gpu-faiss would have raised NameError;
  * the same class reaches across the process boundary too, where a path
    api_client.py builds and the server never declares is a 404 on a screen
    nobody opens until a run gets that far.

Everything here is deliberately conservative — decorated functions, *args,
**kwargs, non-local base classes and dynamic attributes are all skipped —
because the value of this suite is that a finding is always real. The tests at
the bottom feed each check a seeded bug, since a checker too generous to report
anything would pass on every one of the bugs above.
"""

import ast
import re
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

#: Superseded servers. CLAUDE.md's rule is the highest version number, and
#: v3/v4 predate every endpoint in this file — on the cluster the *current*
#: server is deployed under the name tile_server_v4.py, which is a copy of
#: tile_server_v2_.py and not this repo's v4.
SUPERSEDED = {"tile_server.py", "tile_server_v3.py", "tile_server_v4.py"}

#: The UI modules app_v28.py actually imports. app_v2 through app_v27 are
#: history and are not checked; a finding in one of them would be true and
#: useless.
LIVE_APP = ("app_v28.py", "api_client.py", "hpc_chat_handlers_v23.py",
            "local_cache.py", "query_planner_v25.py", "plan_query.py",
            "llm_layer_v25.py", "ui_actions_v25.py")

_SCOPED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def live_sources() -> dict:
    """{module stem: source} for everything the pipeline actually runs."""
    paths = [p for p in sorted((ROOT / "backend").glob("*.py"))
             if p.name not in SUPERSEDED]
    paths += [ROOT / "app" / name for name in LIVE_APP]
    return {p.stem: p.read_text() for p in paths if p.is_file()}


# --- what a module binds at its top level ---------------------------------

def _module_bindings(tree) -> set:
    """Every name importable from this module, however conditionally bound.

    Over-approximated on purpose: a name assigned inside a try/except or an
    `if` at module level is still importable, and treating it as absent would
    make this suite cry wolf.
    """
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


def _top_level_functions(tree) -> dict:
    return {n.name: n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _top_level_classes(tree) -> dict:
    return {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}


# --- call signatures -------------------------------------------------------

def _signature_findings(fn, call, where, skip_first=False) -> list:
    """Arity and keyword problems for a plain local function.

    A decorator can change a signature arbitrarily and *args/**kwargs accept
    anything, so both are skipped rather than guessed at.
    """
    if fn.decorator_list:
        return []
    a = fn.args
    if a.vararg or a.kwarg:
        return []
    if any(isinstance(x, ast.Starred) for x in call.args):
        return []
    if any(k.arg is None for k in call.keywords):
        return []

    positional = [x.arg for x in a.posonlyargs + a.args][1 if skip_first else 0:]
    kwonly = [x.arg for x in a.kwonlyargs]
    n_defaults = len(a.defaults)
    required = positional[:len(positional) - n_defaults] if n_defaults else list(positional)
    given_kw = [k.arg for k in call.keywords]

    found = []
    for k in given_kw:
        if k not in positional and k not in kwonly:
            found.append(f"{where}: {fn.name}() got an unexpected keyword {k!r}")
    if len(call.args) > len(positional):
        found.append(f"{where}: {fn.name}() takes {len(positional)} positional "
                     f"argument(s), {len(call.args)} given")
    supplied = set(positional[:len(call.args)]) | set(given_kw)
    missing = [x for x in required if x not in supplied]
    missing += [x.arg for x, d in zip(a.kwonlyargs, a.kw_defaults)
                if d is None and x.arg not in given_kw]
    if missing:
        found.append(f"{where}: {fn.name}() missing required argument(s) {missing}")
    return found


# --- the sweep -------------------------------------------------------------

def cross_module_findings(sources: dict) -> list:
    """Findings across every module in `sources`, as readable strings."""
    trees = {name: ast.parse(src, name) for name, src in sources.items()}
    bindings = {name: _module_bindings(t) for name, t in trees.items()}
    functions = {name: _top_level_functions(t) for name, t in trees.items()}
    classes = {name: _top_level_classes(t) for name, t in trees.items()}
    findings = []

    for mod, tree in trees.items():
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for al in node.names:
                    if al.name in trees:
                        aliases[al.asname or al.name] = al.name
            elif isinstance(node, ast.ImportFrom) and node.level == 0 \
                    and node.module in trees:
                for al in node.names:
                    if al.name not in bindings[node.module]:
                        findings.append(
                            f"{mod}.py:{node.lineno}: from {node.module} import "
                            f"{al.name} — {node.module} defines no {al.name}")

        def visit(body, bound):
            """Walk one scope, resolving calls against what it can see."""
            stack = list(body)
            while stack:
                node = stack.pop()
                if isinstance(node, _SCOPED):
                    inner = set(bound)
                    if not isinstance(node, ast.ClassDef):
                        args = node.args
                        inner |= {x.arg for x in
                                  args.posonlyargs + args.args + args.kwonlyargs}
                        if args.vararg:
                            inner.add(args.vararg.arg)
                        if args.kwarg:
                            inner.add(args.kwarg.arg)
                    body_nodes = ([node.body] if isinstance(node, ast.Lambda)
                                  else node.body)
                    for sub in ast.walk(ast.Module(body=list(body_nodes),
                                                   type_ignores=[])):
                        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                            inner.add(sub.id)
                    visit(body_nodes, inner)
                    continue

                where = f"{mod}.py:{getattr(node, 'lineno', 0)}"
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                        and isinstance(node.ctx, ast.Load) \
                        and node.value.id in aliases \
                        and node.value.id not in bound:
                    target = aliases[node.value.id]
                    if node.attr not in bindings[target]:
                        findings.append(f"{where}: {node.value.id}.{node.attr} — "
                                        f"{target} defines no {node.attr}")
                if isinstance(node, ast.Call):
                    fn_node = node.func
                    if isinstance(fn_node, ast.Name) and fn_node.id not in bound:
                        if fn_node.id in functions[mod]:
                            findings.extend(_signature_findings(
                                functions[mod][fn_node.id], node, where))
                        elif fn_node.id in classes[mod]:
                            init = _top_level_functions(
                                ast.Module(body=classes[mod][fn_node.id].body,
                                           type_ignores=[])).get("__init__")
                            if init is not None:
                                findings.extend(_signature_findings(
                                    init, node, where, skip_first=True))
                    elif isinstance(fn_node, ast.Attribute) \
                            and isinstance(fn_node.value, ast.Name) \
                            and fn_node.value.id in aliases \
                            and fn_node.value.id not in bound:
                        target = aliases[fn_node.value.id]
                        if fn_node.attr in functions[target]:
                            findings.extend(_signature_findings(
                                functions[target][fn_node.attr], node, where))
                stack.extend(ast.iter_child_nodes(node))

        visit(tree.body, set())

    return sorted(dict.fromkeys(findings))


def self_attribute_findings(sources: dict) -> list:
    """self.<attr> loads bound nowhere in their own class.

    Only classes with no base (or `object`): anything inherited is invisible to
    a reader of this file, and a class that calls setattr is skipped outright.
    """
    findings = []
    for mod, src in sources.items():
        tree = ast.parse(src, mod)
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            if cls.keywords or any(not (isinstance(b, ast.Name) and b.id == "object")
                                   for b in cls.bases):
                continue
            bound, dynamic = set(), False
            for node in ast.walk(cls):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    bound.add(node.name)
                elif isinstance(node, ast.Attribute) \
                        and isinstance(node.value, ast.Name) \
                        and node.value.id == "self" \
                        and isinstance(node.ctx, (ast.Store, ast.Del)):
                    bound.add(node.attr)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    bound.add(node.id)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                        and node.func.id == "setattr":
                    dynamic = True
            if dynamic:
                continue
            for node in ast.walk(cls):
                if isinstance(node, ast.Attribute) \
                        and isinstance(node.value, ast.Name) \
                        and node.value.id == "self" \
                        and isinstance(node.ctx, ast.Load) \
                        and not node.attr.startswith("__") \
                        and node.attr not in bound:
                    findings.append(f"{mod}.py:{node.lineno}: "
                                    f"{cls.name}.self.{node.attr} is bound nowhere "
                                    f"in the class")
    return sorted(dict.fromkeys(findings))


# --- the UI/server seam ----------------------------------------------------

#: The client's own request helpers. Only their first argument is read as a
#: path, so `base_url.rstrip("/")` is not mistaken for a route.
_REQUEST_HELPERS = ("_get", "_get_json", "_post_json", "_get_image")


def _route_key(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "{}", path.rstrip("/"))


def unmatched_client_paths(server_src: str, client_src: str) -> list:
    """Paths api_client.py requests that the server declares no route for."""
    routes = {_route_key(p) for p in re.findall(
        r'@app\.(?:get|post|put|delete|patch)\(\s*[fr]?["\']([^"\']+)["\']',
        server_src)}
    findings = []
    for node in ast.walk(ast.parse(client_src, "api_client.py")):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr in _REQUEST_HELPERS
                and node.args):
            continue
        arg = node.args[0]
        if isinstance(arg, ast.JoinedStr):
            path = "".join(v.value if isinstance(v, ast.Constant) else "{}"
                           for v in arg.values)
        elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            path = arg.value
        else:
            continue
        if _route_key(path) not in routes:
            findings.append(f"api_client.py:{node.lineno}: {path!r} matches no "
                            f"@app route in tile_server_v2_.py")
    return sorted(dict.fromkeys(findings))


# --- the checks themselves -------------------------------------------------

def test_no_module_borrows_a_name_another_module_owns(_tmp=None):
    problems = cross_module_findings(live_sources())
    assert not problems, "\n  " + "\n  ".join(problems)


def test_no_class_reads_an_attribute_it_never_sets(_tmp=None):
    problems = self_attribute_findings(live_sources())
    assert not problems, "\n  " + "\n  ".join(problems)


def test_every_client_path_matches_a_server_route(_tmp=None):
    server = (BACKEND / "tile_server_v2_.py").read_text()
    client = (ROOT / "app" / "api_client.py").read_text()
    problems = unmatched_client_paths(server, client)
    assert not problems, "\n  " + "\n  ".join(problems)


def test_the_client_and_server_agree_on_the_kb_endpoints(_tmp=None):
    """The four Stage 5/6 endpoints by name, since a rename is the likely edit.

    The general check above would catch a renamed path, but only while the
    client still calls it through a helper this file knows about.
    """
    server = (BACKEND / "tile_server_v2_.py").read_text()
    for path in ("/dataset-jobs/{submission_id}/register-preview",
                 "/dataset-jobs/{submission_id}/register",
                 "/dataset-jobs/{submission_id}/kb-load-preview",
                 "/dataset-jobs/{submission_id}/kb-load",
                 "/dataset-jobs/{submission_id}/register-submit",
                 "/dataset-jobs/{submission_id}/kb-load-submit"):
        assert f'"{path}"' in server, f"{path} is no longer declared"


# --- and that each check can fail ------------------------------------------

def test_a_missing_import_is_reported(_tmp=None):
    """submit_cluster_assignment.py's bug, reduced."""
    sources = {
        "extractor": "_PACKAGES_GPU = ('faiss-gpu',)\ndef bootstrap(d):\n    return d\n",
        "submitter": ("from extractor import _PACKAGES_GPU\n"
                      "def main(args):\n"
                      "    if args.bootstrap_gpu:\n"
                      "        return bootstrap(_PACKAGES_GPU)\n"),
    }
    # The undefined-name checker owns the bare `bootstrap` call; this one owns
    # the import list and the attribute access.
    sources["submitter"] = ("import extractor\n"
                            "def main(args):\n"
                            "    if args.bootstrap_gpu:\n"
                            "        return extractor.bootstrap_container_extras(1)\n")
    found = cross_module_findings(sources)
    assert any("defines no bootstrap_container_extras" in f for f in found), found


def test_a_name_absent_from_the_target_module_is_reported(_tmp=None):
    sources = {"stage": "def stage_frame(e, t, f):\n    return 1\n",
               "loader": "from stage import sweep_stale\n"}
    found = cross_module_findings(sources)
    assert any("defines no sweep_stale" in f for f in found), found


def test_a_misspelled_keyword_argument_is_reported(_tmp=None):
    sources = {"stage": "def stage_frame(engine, table, frame, workers=4):\n    return 1\n",
               "loader": ("import stage\n"
                          "def load(engine, frame):\n"
                          "    return stage.stage_frame(engine, 't', frame, worker=4)\n")}
    found = cross_module_findings(sources)
    assert any("unexpected keyword 'worker'" in f for f in found), found


def test_a_missing_required_argument_is_reported(_tmp=None):
    sources = {"m": ("def project(embeddings, reference, mean):\n"
                     "    return 1\n"
                     "def go(e, r):\n"
                     "    return project(e, r)\n")}
    found = cross_module_findings(sources)
    assert any("missing required argument(s) ['mean']" in f for f in found), found


def test_a_self_attribute_typo_is_reported(_tmp=None):
    sources = {"c": ("class Searcher:\n"
                     "    def __init__(self, reference):\n"
                     "        self.reference = reference\n"
                     "    def search(self, q):\n"
                     "        return self.refernce\n")}
    found = self_attribute_findings(sources)
    assert any("self.refernce" in f for f in found), found


def test_a_client_path_the_server_never_declares_is_reported(_tmp=None):
    server = '@app.post("/dataset-jobs/{submission_id}/kb-load")\ndef load():\n    pass\n'
    client = ('class C:\n'
              '    def go(self, submission_id):\n'
              '        return self._post_json(f"/dataset-jobs/{submission_id}/kb_load", {})\n')
    found = unmatched_client_paths(server, client)
    assert len(found) == 1 and "kb_load" in found[0], found


def test_the_checks_tolerate_what_this_codebase_actually_does(_tmp=None):
    """Shadowing, closures, decorators and *args must not be reported.

    Each of these appears in the real modules, and any of them reported as a
    finding would make the suite something people disable.
    """
    sources = {
        "helpers": ("def stage_frame(engine, table, frame, workers=4):\n"
                    "    return workers\n"
                    "CHUNK_ROWS = 100\n"),
        "user": (
            "import helpers\n"
            "def outer(engine, frame, table):\n"
            "    def inner(bound):\n"
            "        start, stop = bound\n"
            "        return helpers.stage_frame(engine, table, frame.iloc[start:stop])\n"
            "    return inner, helpers.CHUNK_ROWS\n"
            "def shadowed(stage_frame):\n"
            "    return stage_frame(1, 2, 3, 4, 5, 6)\n"
            "def anything(*args, **kwargs):\n"
            "    return args, kwargs\n"
            "def calls_anything():\n"
            "    return anything(1, 2, 3, whatever=4)\n"),
    }
    assert cross_module_findings(sources) == []
    assert self_attribute_findings(sources) == []


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_cross_module_test_"))
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
