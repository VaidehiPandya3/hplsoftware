"""
Hybrid query planner for app_v25: NLP enrich → regex → optional LLM → validate.
"""

from __future__ import annotations

import re
from typing import Any

from llm_layer_v25 import apply_nlp_hints_to_plan, merge_plans, plan_query_with_llm
from nlp_enrich_v25 import enrich_prompt, nlp_entity_hints, validate_plan_entities


def build_query_plan_regex(query: str) -> dict[str, Any]:
    q = query.lower()

    tile = re.findall(r"(tile[_\-]?\d+|\w+\.(jpg|jpeg|png|tif))", q)
    slide_tcga = re.findall(r"(tcga[-\w]+dx\d+)", q)
    hpc = re.findall(r"hpc[\s\-]*([0-9]+)", q)
    if not hpc:
        hpc = re.findall(r"\bcomponent\s+(\d+)\b", q)
    sample = re.findall(r"sample[\s\-]*([a-z0-9]+)", q)

    non_malignant = "not malignant" in q or "non malignant" in q or "non-malignant" in q
    malignant = "malignant" in q and not non_malignant
    survival = any(k in q for k in ["survival", "cox", "hazard", "prognosis", "prognostic"])
    analytics = any(
        k in q for k in ["how many", "count", "total hpcs", "number of hpcs", "portion", "coverage", "percent"]
    )

    # Word-boundary match, not substring: plain `"hi" in q`/`"hey" in q` also
    # matched "which" (w-HI-ch) and "they" (t-HEY) — so "Which slides have
    # HPC 40?" was classified as a greeting and never reached the database at
    # all. Any short greeting word is at risk of this; \b keeps it to real
    # standalone occurrences.
    greeting = bool(re.search(r"\b(hello|hi|hey|good morning|good afternoon)\b", q))
    help_intent = any(w in q for w in ("what can you do", "help me", "how do i", "capabilities"))

    if greeting:
        intent = "greeting"
    elif help_intent:
        intent = "help"
    elif analytics:
        intent = "analytics_query"
    elif any(w in q for w in ("why", "explain", "reason")):
        intent = "explain_query"
    elif slide_tcga:
        intent = "slide_query"
    elif tile:
        intent = "tile_query"
    elif hpc:
        intent = "hpc_query"
    elif sample:
        intent = "sample_query"
    else:
        intent = "general_query"

    operations: list[str] = []
    if malignant or non_malignant:
        operations.append("malignancy_lookup")
    if survival:
        operations.append("survival_analysis")
    if any(k in q for k in ["how many", "count", "total hpcs", "number of hpcs"]):
        operations.append("count_hpcs")
    if any(k in q for k in ["portion", "coverage", "covered by malignant", "percent malignant"]):
        operations.append("slide_malignant_coverage")

    if "heatmap" in q:
        highlight_mode = "Heatmap"
    elif "inflammation" in q:
        highlight_mode = "Inflammation"
    elif "necrosis" in q:
        highlight_mode = "Necrosis"
    elif malignant or non_malignant:
        highlight_mode = "Malignant"
    elif any(w in q for w in ("adjacent", "beside", "cooccur")):
        highlight_mode = "Adjacency"
    elif hpc or "cluster" in q:
        highlight_mode = "HPC clusters"
    else:
        highlight_mode = None

    ui_actions = {
        "open_slide_viewer": bool(slide_tcga)
        or bool(hpc)
        or bool(highlight_mode)
        or malignant
        or non_malignant
        or any(w in q for w in ("open slide", "show slide", "view slide", "viewer")),
        "show_tile_preview": bool(slide_tcga or tile),
        "explore_hpc_slides": bool(hpc)
        and any(w in q for w in ("which slides", "compare", "across slides", "most tiles", "rank", "wsis")),
        "highlight_mode": highlight_mode,
        "selected_hpc": int(hpc[0]) if hpc else None,
    }

    return {
        "intent": intent,
        "entities": {
            "tile": [t[0] for t in tile],
            "slide": [s.upper() for s in slide_tcga],
            "hpc": hpc,
            "sample": sample,
        },
        "flags": {
            "malignant": malignant or None,
            "non_malignant": non_malignant or None,
            "survival": survival or None,
        },
        "operations": operations,
        "ui_actions": ui_actions,
        "confidence": 0.9,
        "planner": "regex",
    }


def build_query_plan_v25(
    query: str,
    *,
    slide_list: list[str] | None = None,
    hpc_title_map: dict[int, str] | None = None,
    valid_hpc_ids: set[int] | None = None,
    session_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    enriched = enrich_prompt(query)
    hints = nlp_entity_hints(enriched, slide_list or [], hpc_title_map)

    regex_plan = build_query_plan_regex(hints["enriched_text"])
    regex_plan = apply_nlp_hints_to_plan(regex_plan, hints)

    llm_plan = plan_query_with_llm(hints["enriched_text"], context=session_context)
    plan = merge_plans(regex_plan, llm_plan)
    plan["enriched_query"] = hints["enriched_text"]

    if slide_list is not None:
        plan = validate_plan_entities(plan, slide_list, valid_hpc_ids)

    return plan
