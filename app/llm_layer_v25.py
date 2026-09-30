"""
Ollama layer for app_v25: query planning + grounded explanation.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import ollama

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")

VALID_INTENTS = frozenset(
    {
        "slide_query",
        "tile_query",
        "hpc_query",
        "sample_query",
        "general_query",
        "greeting",
        "help",
        "analytics_query",
        "explain_query",
    }
)

VALID_HIGHLIGHT_MODES = frozenset(
    {
        "HPC clusters",
        "Inflammation",
        "Necrosis",
        "Malignant",
        "Adjacency",
        "Heatmap",
        "Survival Risk Heatmap",
    }
)

PLANNER_SYSTEM = """You are a query planner for a histopathology knowledge-base chatbot (HPC = Histomorphological Phenotype Cluster).

Given a user message, output ONLY valid JSON (no markdown) with this schema:
{
  "intent": "slide_query|tile_query|hpc_query|sample_query|general_query|greeting|help|analytics_query|explain_query",
  "entities": { "tile": [], "slide": [], "hpc": [], "sample": [] },
  "flags": { "malignant": null, "non_malignant": null, "survival": null },
  "operations": ["malignancy_lookup", "survival_analysis", "count_hpcs", "slide_malignant_coverage"],
  "ui_actions": {
    "open_slide_viewer": false,
    "show_tile_preview": false,
    "explore_hpc_slides": false,
    "highlight_mode": null,
    "selected_hpc": null
  },
  "confidence": 0.0,
  "reasoning": "one short sentence"
}

Rules:
- Put slide IDs exactly as the user wrote them (any dataset/format).
- HPC numbers in entities.hpc as strings without the prefix.
- greeting/help for hellos; analytics_query for counts/coverage; explain_query for why/compare questions.
- explore_hpc_slides true when user wants slides where an HPC is common.
- open_slide_viewer when user wants to view/open/show a slide.
- Do not invent entities not mentioned or strongly implied.
- If a "Current session context" block is given and the user refers to "this slide"/"that HPC"/
  similar without naming one explicitly, resolve it using that context (e.g. reuse the active slide).
"""

EXPLAINER_SYSTEM = """You are a scientific assistant for a histopathology AI system.

Rules:
- Be concise and accurate
- Use only facts present in the structured result / evidence
- Do not hallucinate HPC or slide facts
- Use markdown for lists when helpful
- For survival, say "associated with" not "causes"
"""


def llm_enabled() -> bool:
    return os.getenv("LLM_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")


def llm_planner_enabled() -> bool:
    return llm_enabled() and os.getenv("USE_LLM_PLANNER", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _chat(messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
    kwargs: dict[str, Any] = {"model": DEFAULT_MODEL, "messages": messages}
    if json_mode:
        kwargs["format"] = "json"
    response = ollama.chat(**kwargs)
    return (response.get("message") or {}).get("content") or ""


def _extract_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _as_str_list(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    return [s] if s else []


def _normalize_plan(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    if not raw or not isinstance(raw, dict):
        return None

    intent = str(raw.get("intent") or "general_query").strip().lower()
    if intent not in VALID_INTENTS:
        intent = "general_query"

    entities_in = raw.get("entities") if isinstance(raw.get("entities"), dict) else {}
    entities = {
        "tile": _as_str_list(entities_in.get("tile")),
        "slide": [s.upper() for s in _as_str_list(entities_in.get("slide"))],
        "hpc": _as_str_list(entities_in.get("hpc")),
        "sample": _as_str_list(entities_in.get("sample")),
    }

    flags_in = raw.get("flags") if isinstance(raw.get("flags"), dict) else {}
    flags = {
        "malignant": flags_in.get("malignant"),
        "non_malignant": flags_in.get("non_malignant"),
        "survival": flags_in.get("survival"),
    }

    operations = raw.get("operations")
    if not isinstance(operations, list):
        operations = []
    operations = [str(o) for o in operations]

    ui_in = raw.get("ui_actions") if isinstance(raw.get("ui_actions"), dict) else {}
    highlight = ui_in.get("highlight_mode")
    if highlight and str(highlight) not in VALID_HIGHLIGHT_MODES:
        highlight = None
    selected = ui_in.get("selected_hpc")
    try:
        selected_hpc = int(selected) if selected is not None else None
    except (TypeError, ValueError):
        selected_hpc = None

    ui_actions = {
        "open_slide_viewer": bool(ui_in.get("open_slide_viewer")),
        "show_tile_preview": bool(ui_in.get("show_tile_preview")),
        "explore_hpc_slides": bool(ui_in.get("explore_hpc_slides")),
        "highlight_mode": highlight,
        "selected_hpc": selected_hpc,
    }

    try:
        confidence = float(raw.get("confidence", 0.7))
    except (TypeError, ValueError):
        confidence = 0.7

    return {
        "intent": intent,
        "entities": entities,
        "flags": flags,
        "operations": operations,
        "ui_actions": ui_actions,
        "confidence": max(0.0, min(1.0, confidence)),
        "reasoning": str(raw.get("reasoning") or "").strip(),
        "planner": "llm",
    }


def _format_session_context(context: dict[str, Any] | None) -> str:
    if not context:
        return ""
    lines = []
    if context.get("active_slide"):
        lines.append(f"- Currently active/open slide: {context['active_slide']}")
    if context.get("viewer_open") is not None:
        lines.append(f"- Slide viewer open: {bool(context['viewer_open'])}")
    if context.get("selected_hpc") is not None:
        lines.append(f"- Currently selected HPC: {context['selected_hpc']}")
    if context.get("highlight_mode"):
        lines.append(f"- Current highlight mode: {context['highlight_mode']}")
    if not lines:
        return ""
    return "\n\nCurrent session context (use this to resolve 'this slide'/'that HPC' references):\n" + "\n".join(lines)


def plan_query_with_llm(
    user_query: str, *, context: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    if not llm_planner_enabled():
        return None
    try:
        content = _chat(
            [
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": user_query + _format_session_context(context)},
            ],
            json_mode=True,
        )
        return _normalize_plan(_extract_json(content))
    except Exception:
        return None


def merge_plans(regex_plan: dict[str, Any], llm_plan: dict[str, Any] | None) -> dict[str, Any]:
    if not llm_plan:
        out = dict(regex_plan)
        out["planner"] = "regex"
        return out

    merged = dict(regex_plan)
    merged["planner"] = "hybrid"
    merged["llm_reasoning"] = llm_plan.get("reasoning", "")

    llm_conf = float(llm_plan.get("confidence", 0))
    if llm_conf >= 0.55 and llm_plan.get("intent") not in ("general_query",):
        merged["intent"] = llm_plan["intent"]

    for key in ("tile", "slide", "hpc", "sample"):
        llm_vals = llm_plan["entities"].get(key) or []
        regex_vals = regex_plan["entities"].get(key) or []
        merged["entities"][key] = llm_vals if llm_vals else regex_vals

    for key in ("malignant", "non_malignant", "survival"):
        lv = llm_plan["flags"].get(key)
        rv = regex_plan["flags"].get(key)
        merged["flags"][key] = lv if lv is not None else rv

    merged["operations"] = sorted(
        set(regex_plan.get("operations") or []) | set(llm_plan.get("operations") or [])
    )

    ui = dict(regex_plan.get("ui_actions") or {})
    ui_llm = llm_plan.get("ui_actions") or {}
    for k, v in ui_llm.items():
        if k in ("highlight_mode", "selected_hpc"):
            if v is not None:
                ui[k] = v
        else:
            ui[k] = bool(ui.get(k)) or bool(v)
    merged["ui_actions"] = ui
    merged["confidence"] = max(float(regex_plan.get("confidence", 0)), llm_conf)
    return merged


def apply_nlp_hints_to_plan(plan: dict[str, Any], hints: dict[str, Any]) -> dict[str, Any]:
    out = dict(plan)
    ents = dict(plan.get("entities") or {})
    for key in ("tile", "slide", "hpc", "sample"):
        hint_vals = hints.get(key) or []
        if hint_vals and not ents.get(key):
            ents[key] = hint_vals
        elif hint_vals and ents.get(key):
            merged_vals = list(dict.fromkeys(list(ents.get(key) or []) + list(hint_vals)))
            ents[key] = merged_vals
    out["entities"] = ents
    if hints.get("slide") and not out.get("ui_actions", {}).get("open_slide_viewer"):
        ui = dict(out.get("ui_actions") or {})
        ui["open_slide_viewer"] = True
        out["ui_actions"] = ui
    if hints.get("hpc") and any(
        w in (hints.get("enriched_text") or "").lower()
        for w in ("which slides", "compare slides", "across slides", "most tiles", "rank")
    ):
        ui = dict(out.get("ui_actions") or {})
        ui["explore_hpc_slides"] = True
        out["ui_actions"] = ui
    return out


def first_hpc_id(plan: dict[str, Any]) -> int | None:
    for h in plan.get("entities", {}).get("hpc") or []:
        m = re.search(r"(\d+)", str(h))
        if m:
            return int(m.group(1))
    ui = plan.get("ui_actions") or {}
    if ui.get("selected_hpc") is not None:
        try:
            return int(ui["selected_hpc"])
        except (TypeError, ValueError):
            pass
    return None


def should_fetch_from_db(plan: dict[str, Any]) -> bool:
    # Checked before the intent short-circuit below, not after: a message
    # like "Hello, tell me about HPC 40" opens with a greeting word but names
    # a real entity, and greeting/help/general_query is exactly the intent
    # the regex planner assigns whenever a greeting word appears anywhere in
    # the message (see build_query_plan_regex) — so checking intent first
    # discarded the HPC lookup entirely. explain_answer() already prefers a
    # real structured_answer over its canned greeting reply whenever one
    # comes back, so returning True here for a named entity is enough; no
    # other change is needed.
    ents = plan.get("entities") or {}
    if any(ents.get(k) for k in ("tile", "slide", "hpc", "sample")):
        return True
    if plan.get("intent") in ("general_query", "greeting", "help"):
        return False
    if plan.get("operations"):
        return True
    if plan.get("intent") in ("analytics_query", "explain_query", "slide_query", "tile_query", "hpc_query", "sample_query"):
        return True
    flags = plan.get("flags") or {}
    return bool(flags.get("survival") or flags.get("malignant") or flags.get("non_malignant"))


def explain_answer(
    plan: dict[str, Any],
    structured_answer: str | None,
    *,
    user_query: str | None = None,
    history: list[dict[str, str]] | None = None,
    evidence: dict[str, Any] | None = None,
) -> str:
    if not llm_enabled():
        if structured_answer:
            return structured_answer
        if plan.get("intent") in ("greeting", "help", "general_query"):
            return (
                "Hello! Ask about an **HPC**, a **slide**, tiles, survival, malignancy, "
                "or say *open the slide viewer*."
            )
        return structured_answer or "I could not find an answer for that question."

    intent = plan.get("intent", "general_query")
    if structured_answer is None and intent in ("greeting", "help", "general_query"):
        q = (user_query or "").lower()
        if any(w in q for w in ("hello", "hi", "hey")):
            return (
                "Hello! I can look up **HPCs**, **slides** (any ID format), **tiles**, "
                "**survival**, and open the **WSI viewer**."
            )
        return (
            "I can answer questions about **HPCs**, **slides**, **tiles**, and **samples**, "
            "run survival and count analytics, and open viewers. Try: *Which slides have HPC 40?*"
        )

    parts = [
        f"User question:\n{user_query or '(not provided)'}",
        f"\nParsed intent: {intent}",
        f"\nEntities:\n{json.dumps(plan.get('entities', {}), indent=2)}",
        f"\nFlags:\n{json.dumps(plan.get('flags', {}), indent=2)}",
    ]
    if plan.get("llm_reasoning"):
        parts.append(f"\nPlanner note: {plan['llm_reasoning']}")
    if evidence:
        parts.append(f"\nEvidence bundle:\n{json.dumps(evidence, indent=2)}")
    parts.append(f"\nStructured database/API result:\n{structured_answer or '(none)'}")
    parts.append(
        "\nExplain this to the user in clear, simple language. "
        "Preserve key numbers and facts from the structured result."
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": EXPLAINER_SYSTEM}]
    if history:
        for msg in history[-6:]:
            role = msg.get("role")
            content = msg.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": str(content)[:4000]})
    messages.append({"role": "user", "content": "".join(parts)})

    try:
        return _chat(messages).strip() or (structured_answer or "")
    except Exception:
        return structured_answer or "Sorry, I could not generate a response right now."
