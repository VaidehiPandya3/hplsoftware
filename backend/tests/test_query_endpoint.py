"""/query: the endpoint's own docstring called it a stub for two commits —
"the Streamlit client still runs fetch_answer_from_db locally... in Phase 2
you move that logic here too." This tests that Phase 2 actually happened and
stays that way, without needing a live Postgres:

  * chat_answers.py's pure functions (no DB connection needed) match the
    Streamlit original's behaviour exactly — these are the ones a caller can
    least afford to have drift, since a wrong polarity flips which KB table
    a query reads.
  * the endpoint's source no longer contains the placeholder that returned
    the bare query plan and nothing else — a regression back to the stub
    would pass every other test here (the plan-building half was always
    real) while silently dropping the DB answer and the LLM explanation.

Runs standalone (no pytest) the same way every other suite in this directory
does, per CLAUDE.md.
"""

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import chat_answers  # noqa: E402

_SOURCE = (BACKEND / "tile_server_v2_.py").read_text()


def _query_endpoint_body() -> str:
    """The /query handler's own source — /query is the last @app.-decorated
    route in the file, so there is no following "\\n@app." to bound it on."""
    start = _SOURCE.index('@app.post("/query")')
    rest = _SOURCE[start + 1:]
    end = start + 1 + rest.index("\n@app.") if "\n@app." in rest else len(_SOURCE)
    return _SOURCE[start:end]


def test_classify_malignancy_polarity_positive():
    assert chat_answers.classify_malignancy_polarity("is this malignant") == "malignant"
    assert chat_answers.classify_malignancy_polarity("malignant epithelium") == "malignant"


def test_classify_malignancy_polarity_negative():
    assert chat_answers.classify_malignancy_polarity("is this non malignant") == "non"
    assert chat_answers.classify_malignancy_polarity("not malignant") == "non"
    assert chat_answers.classify_malignancy_polarity("without malignant epithelium") == "non"


def test_classify_malignancy_polarity_neither():
    assert chat_answers.classify_malignancy_polarity("what is hpc 40") is None
    assert chat_answers.classify_malignancy_polarity("") is None


def test_bool_malignant_coerces_every_spelling_the_streamlit_original_did():
    # Verbatim vocabulary from hpc_chat_handlers_v23.py's two inline
    # normalisations — this file's docstring promises to track them by hand,
    # so pin the exact set rather than trusting a paraphrase.
    for v in (True, 1, "true", "T", "1", "yes", "Y"):
        assert chat_answers._bool_malignant(v) is True, v
    for v in (False, 0, "false", "F", "0", "no", "N"):
        assert chat_answers._bool_malignant(v) is False, v


def test_bool_malignant_none_for_unrecognised_or_missing():
    assert chat_answers._bool_malignant(None) is None
    assert chat_answers._bool_malignant("maybe") is None
    # Not None: this mirrors hpc_chat_handlers_v23.py's own inline coercion
    # verbatim, including its quirk that any nonzero int is just bool(int) —
    # unlike backend/malignancy.py's stricter parse_malignant, which refuses
    # anything other than 0/1. Faithful to the original this replaces, not to
    # the stricter module.
    assert chat_answers._bool_malignant(2) is True


def test_handle_tile_rejects_empty_name_without_a_connection():
    # No `conn` argument needed for this branch — proves the empty-name guard
    # runs before any query is built, matching the Streamlit original's
    # `if not tile_name: return ...` short-circuit.
    text, images = chat_answers.handle_tile(conn=None, tile_name="  ", intents={})
    assert "specify a tile name" in text
    assert images == []


def test_handle_hpc_reports_invalid_id_without_a_connection():
    # handle_hpc calls inspect(engine) up front regardless of id validity, so
    # this needs a real (if empty) Engine — but still no `conn`, since an
    # unparseable id is caught before the loop ever executes a query.
    from sqlalchemy import create_engine

    engine = create_engine("sqlite://")
    text, images = chat_answers.handle_hpc(conn=None, ids=["not-a-number"], intents={}, engine=engine)
    assert "Invalid HPC ID" in text
    assert images == []


def test_query_endpoint_is_not_the_old_stub():
    body = _query_endpoint_body()
    # The exact sentence the stub returned instead of an answer. Its presence
    # here would mean someone reverted the endpoint to "return the plan".
    assert "In Phase 2 you move" not in body
    assert '"structured_answer": None,' not in body
    # What the real pipeline must produce that the stub never did.
    assert "final_answer" in body
    assert "chat_answers.fetch_answer_from_db" in body
    assert "should_fetch_from_db" in body


def test_query_endpoint_reuses_the_planner_rather_than_reimplementing_it():
    # The whole point of moving this server-side was reusing the one
    # implementation app_v28.py already has (query_planner_v25 and friends),
    # not writing a second query planner that could disagree with it.
    body = _query_endpoint_body()
    assert "from query_planner_v25 import build_query_plan_v25" in body
    assert "from hpc_chat_handlers_v23 import detect_entity_patterns" in body


def test_hpc_reference_maps_helper_exists_and_is_used_by_query():
    assert "_hpc_reference_maps" in _SOURCE
    assert "_hpc_reference_maps(target)" in _query_endpoint_body()


# --- standalone runner -----------------------------------------------------


def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
