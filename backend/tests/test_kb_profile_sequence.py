#!/usr/bin/env python3
"""Stage 6 must not die on an id sequence that has fallen behind its table.

On 2026-09-27 the RADIOGENOMICS load (18.5M tiles) failed at its last
statement: `duplicate key value violates unique constraint
"hpl_profile_summary_pkey" — Key (id)=(1) already exists`. The aggregate
tables take `id` from a serial sequence, the INSERT names no id, and the
sequence was behind MAX(id) — what a data-only restore or a COPY with explicit
ids leaves behind. load() is one transaction, so every tile update was rolled
back with it. The SQLite fixtures cannot show this (INTEGER PRIMARY KEY picks
max+1 itself), so these drive the real function with a connection that
answers the four Postgres queries it makes.

Runs under pytest and standalone.
"""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import load_hpc_assignments as lha  # noqa: E402


class _FakePostgres:
    """One serial sequence per table, and each table's MAX(id)."""

    class _Dialect:
        def __init__(self, name):
            self.name = name

    class _Result:
        def __init__(self, value):
            self.value = value

        def scalar(self):
            return self.value

    def __init__(self, *, max_id, last_value, dialect="postgresql", serial=True):
        self.dialect = self._Dialect(dialect)
        self.max_id, self.last_value, self.serial = max_id, last_value, serial
        self.statements = []

    def execute(self, clause, params=None):
        sql = str(clause)
        self.statements.append(sql)
        if "pg_get_serial_sequence" in sql:
            return self._Result(f"public.{params['t']}_id_seq" if self.serial else None)
        if "MAX(id)" in sql:
            return self._Result(self.max_id)
        if "pg_sequence_last_value" in sql:
            return self._Result(self.last_value)
        if "setval" in sql:
            self.last_value = params["v"]
            return self._Result(params["v"])
        raise AssertionError(f"unexpected statement: {sql}")


def test_a_sequence_behind_the_table_is_advanced_before_the_insert(_tmp=None):
    conn = _FakePostgres(max_id=14042, last_value=None)   # never called: nextval gives 1
    assert lha._catch_up_id_sequence(conn, "hpl_profile_summary") == 14042
    assert conn.last_value == 14042, "the next insert would reuse id 1"


def test_a_sequence_that_is_ahead_is_left_alone(_tmp=None):
    conn = _FakePostgres(max_id=100, last_value=250)
    assert lha._catch_up_id_sequence(conn, "hpl_profile_summary") is None
    assert conn.last_value == 250, "never moved backwards"
    assert not any("setval" in s for s in conn.statements)


def test_an_empty_table_and_a_table_without_a_serial_id_need_nothing(_tmp=None):
    assert lha._catch_up_id_sequence(_FakePostgres(max_id=None, last_value=None),
                                     "hpl_profile_summary") is None
    assert lha._catch_up_id_sequence(_FakePostgres(max_id=5, last_value=None, serial=False),
                                     "hpl_profile_summary") is None


def test_sqlite_is_not_asked_postgres_questions(_tmp=None):
    conn = _FakePostgres(max_id=5, last_value=None, dialect="sqlite")
    assert lha._catch_up_id_sequence(conn, "hpl_profile_summary") is None
    assert conn.statements == []


def test_both_aggregate_tables_are_caught_up_before_either_insert(_tmp=None):
    source = inspect.getsource(lha.replace_profiles)
    catch_up = source.index("_catch_up_id_sequence(conn, table)")
    tables = source[source.rindex("for table in", 0, catch_up):catch_up]
    assert '"hpl_profile_summary"' in tables and '"hpl_profile_proportion"' in tables
    assert catch_up < source.index('_insert("hpl_profile_summary"')


# --- standalone runner ------------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        try:
            fn(Path(tempfile.mkdtemp()))
            print(f"PASS  {name}")
        except Exception as e:
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
