"""The connection URL, which six modules used to build by string interpolation.

    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

Nothing about that raises. A password containing '@' or '/' re-splits the URL
and the driver is handed a different host and database, so the error names a
machine nobody configured; a socket path in DB_HOST does the same thing with
its slashes; and the string carries the password into any message that prints
it. Every one of those produces a plausible failure somewhere other than the
cause, which is the failure mode this codebase is written against.

The tests that matter here are the ones that show the old form was wrong, not
just that the new one is right.
"""

import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from sqlalchemy.engine import make_url  # noqa: E402

import db_url  # noqa: E402

#: One of everything that means something in a URL.
AWKWARD = "p@ss/w:ord#1?x"


def test_a_password_with_punctuation_keeps_host_and_database(_tmp=None):
    url = db_url.database_url("hpl_kb_test", user="vpandya", password=AWKWARD,
                              host="hpc-login-01", port="5432")
    assert url.host == "hpc-login-01"
    assert url.database == "hpl_kb_test"
    assert url.password == AWKWARD
    assert url.port == 5432


def test_the_old_interpolation_really_did_break(_tmp=None):
    """Why the module exists, pinned as a fact rather than a claim in a comment.

    If this ever stops being true, the interpolated form is safe again and this
    file is unnecessary — which is worth knowing either way.
    """
    interpolated = make_url(
        f"postgresql+psycopg2://vpandya:{AWKWARD}@hpc-login-01:5432/hpl_kb_test")
    assert interpolated.host != "hpc-login-01"
    assert interpolated.database != "hpl_kb_test"


def test_an_empty_password_is_dropped_so_a_passfile_still_applies(_tmp=None):
    """libpq reads ~/.pgpass and $PGPASSWORD only when no password is supplied.

    This is not a nicety: home is shared to the compute nodes, so a passfile is
    one of the two ways a Slurm-backed Stage 6 can authenticate at all. An
    empty string here would count as a supplied password and skip the file.
    """
    url = db_url.database_url("hpl_kb", user="vpandya", password="",
                              host="hpc-login-01", port="5432")
    assert url.password is None
    assert "password" not in url.translate_connect_args()


def test_a_socket_path_survives_as_the_host(_tmp=None):
    """DB_HOST is a socket directory on the machine running Postgres, which is
    how the server itself connects — and what resolve_job_db_host refuses to
    hand a compute node."""
    url = db_url.database_url("hpl_kb", user="vpandya", password="",
                              host="/mnt/cephfs-lts/pgsocket", port="5432")
    assert url.host == "/mnt/cephfs-lts/pgsocket"
    assert url.translate_connect_args()["host"] == "/mnt/cephfs-lts/pgsocket"


def test_the_password_is_masked_in_anything_printable(_tmp=None):
    """app_v28.py printed the URL in the one message that runs when the
    connection fails, which is precisely when someone screenshots it."""
    url = db_url.database_url("hpl_kb", user="vpandya", password=AWKWARD,
                              host="hpc-login-01", port="5432")
    assert AWKWARD not in str(url)
    assert AWKWARD not in db_url.safe_text(url)
    assert "***" in db_url.safe_text(url)


def test_a_nonsense_port_does_not_crash_at_import(_tmp=None):
    """app_v28 builds a URL at module level, so a bad DB_PORT would be a UI that
    will not start rather than a connection that fails with a clear message."""
    url = db_url.database_url("hpl_kb", user="v", password="", host="h",
                              port="not-a-port")
    assert url.port is None


def test_the_environment_supplies_what_the_caller_omits(_tmp=None, monkeypatch=None):
    import os
    saved = {k: os.environ.get(k) for k in
             ("DB_USER", "DB_PASS", "DB_HOST", "DB_PORT", "DB_NAME")}
    try:
        os.environ.update({"DB_USER": "someone", "DB_PASS": AWKWARD,
                           "DB_HOST": "hpc-login-01", "DB_PORT": "5433",
                           "DB_NAME": "hpl_kb_test"})
        url = db_url.database_url()
        assert (url.username, url.host, url.port, url.database) == (
            "someone", "hpc-login-01", 5433, "hpl_kb_test")
        assert url.password == AWKWARD
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_every_module_builds_its_url_through_this_one(_tmp=None):
    """The point of the module is that there is one of it.

    A new f-string somewhere else reintroduces the bug for that path only, which
    is how five of the six sites came to disagree about the DB_PASS branch in
    the first place.
    """
    superseded = {"tile_server.py", "tile_server_v3.py", "tile_server_v4.py"}
    offenders = []
    for path in list(BACKEND.glob("*.py")) + [BACKEND.parent / "app" / "app_v28.py"]:
        if path.name in superseded or path.name == "db_url.py":
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if "postgresql+psycopg2://" in line and not line.lstrip().startswith("#"):
                offenders.append(f"{path.name}:{lineno}: {line.strip()[:70]}")
    assert not offenders, ("build these through db_url.database_url():\n  "
                           + "\n  ".join(offenders))


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_db_url_test_"))
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
