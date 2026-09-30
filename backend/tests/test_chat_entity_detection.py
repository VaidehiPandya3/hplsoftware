"""Regression tests for three chat query-understanding bugs found by testing
the pipeline against realistic questions (2026-09-18) — the symptom in every
case was a specific, answerable question silently getting no real answer.

  1. build_query_plan_regex's greeting detector used plain substring
     containment ("hi" in q, "hey" in q). "which" contains "hi" and "they"
     contains "hey", so "Which slides have HPC 40?" was classified as a
     greeting and never reached the database at all.
  2. should_fetch_from_db() checked plan["intent"] before checking whether
     any entity (tile/slide/hpc/sample) had actually been found — so even
     after fixing (1), a message like "Hello, tell me about HPC 40" (a real
     greeting word, genuinely present) still discarded a named entity.
  3. detect_entity_patterns()'s after() helper matched a keyword and then
     captured whatever character(s) immediately followed with no separator
     required — so "hpcs"/"slides"/"tiles"/"samples" (ordinary English
     plurals, no ID intended) each matched the keyword and then captured
     their own trailing "s" as if it were the identifier, and "how many
     tiles does this hpc have" captured hpc="have". Both blocked the
     correct, digit/pattern-anchored fallback regexes from ever running,
     since those only fire `if not <keyword>_match`.

Runs standalone (no pytest) per CLAUDE.md's convention, same pattern as
test_query_endpoint.py.
"""

import os
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent.parent / "app"
sys.path.insert(0, str(APP))

# Deterministic and fast: these tests are about the regex/entity layer, not
# the LLM planner, and the hybrid planner would otherwise call out to Ollama
# (slow, and not guaranteed to be running wherever this suite executes).
os.environ.setdefault("USE_LLM_PLANNER", "0")

from hpc_chat_handlers_v23 import detect_entity_patterns  # noqa: E402
from llm_layer_v25 import should_fetch_from_db  # noqa: E402
from query_planner_v25 import build_query_plan_regex, build_query_plan_v25  # noqa: E402


# --- (1) greeting word-boundary ---------------------------------------------


def test_which_is_not_a_greeting():
    plan = build_query_plan_regex("which slides have hpc 40")
    assert plan["intent"] != "greeting"
    assert plan["entities"]["hpc"] == ["40"]


def test_they_is_not_a_greeting():
    plan = build_query_plan_regex("they mentioned hpc 12 earlier")
    assert plan["intent"] != "greeting"


def test_this_is_not_a_greeting():
    # "this" also contains "hi" as a substring — same failure class.
    plan = build_query_plan_regex("what is this hpc 40 about")
    assert plan["intent"] != "greeting"


def test_real_greetings_still_classify_as_greetings():
    for q in ("hi there", "hello!", "hey, how are you", "good morning"):
        assert build_query_plan_regex(q)["intent"] == "greeting", q


# --- (2) an entity always outranks a leading pleasantry ---------------------


def test_should_fetch_from_db_true_for_greeting_prefixed_real_question():
    plan = build_query_plan_v25("hello, tell me about hpc 40", slide_list=[])
    assert plan["entities"]["hpc"] == ["40"]
    assert should_fetch_from_db(plan) is True


def test_should_fetch_from_db_false_for_a_bare_greeting():
    plan = build_query_plan_v25("hi there", slide_list=[])
    assert should_fetch_from_db(plan) is False


# --- (3) plural glomming in detect_entity_patterns's after() ---------------


def test_plural_hpc_is_not_captured_as_an_id():
    assert detect_entity_patterns("how many hpcs are there")["hpc"] is None


def test_plural_slide_is_not_captured_as_an_id():
    assert detect_entity_patterns("which slides have hpc 40")["slide"] is None


def test_plural_tile_is_not_captured_as_an_id():
    assert detect_entity_patterns("how many tiles are in this slide")["tile"] is None


def test_plural_sample_is_not_captured_as_an_id():
    detected = detect_entity_patterns("what samples contain hpc 40")
    assert detected["sample"] is None
    assert detected["hpc"] == ["40"]


def test_hpc_used_generically_with_no_number_is_not_captured():
    # The concrete failure that motivated this fix: "hpc" referenced with no
    # number anywhere nearby used to capture the next English word ("have")
    # as if it were an HPC id.
    assert detect_entity_patterns("how many tiles does this hpc have")["hpc"] is None


def test_hpc_with_a_real_number_still_works_in_every_common_spelling():
    for q in ("hpc 40", "hpc40", "hpc:40", "hpc=40"):
        assert detect_entity_patterns(q)["hpc"] == ["40"], q


def test_slide_fallback_still_finds_a_real_tcga_id_after_the_plural_fix():
    detected = detect_entity_patterns("tell me about slide TCGA-55-7574-01Z-00-DX1")
    assert detected["slide"] == "tcga-55-7574-01z-00-dx1"


def test_tile_still_matches_a_real_filename_after_the_plural_fix():
    # after("tile") captures from right after "tile" (an underscore isn't a
    # letter, so the plural guard doesn't block it here), giving
    # "_182.jpeg" rather than the full "tile_182.jpeg" — a pre-existing
    # imprecision, unchanged by this fix, and still enough for
    # chat_answers.handle_tile's ILIKE fallback to find the real row. What
    # matters here is that the plural guard didn't make this *worse*.
    detected = detect_entity_patterns("what about tile_182.jpeg")
    assert detected["tile"] is not None
    assert "182" in detected["tile"] and "jpeg" in detected["tile"]


# --- standalone runner -------------------------------------------------------


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
