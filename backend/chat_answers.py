"""Server-side twin of app/hpc_chat_handlers_v23.py, for the FastAPI /query
endpoint — completing the "Phase 2" api_client.py's query() docstring has
described since it was written: "the Streamlit client still runs
fetch_answer_from_db locally for the full NL pipeline. In Phase 2 you move
that logic here too."

Why a twin rather than importing hpc_chat_handlers_v23.py directly: every
handler there interleaves its SQL/text logic with real `st.*` calls
(st.image, st.session_state.active_slide = ..., st.subheader, st.columns) as
side effects — safe only inside a live Streamlit script run. Calling them from
a FastAPI request handler, which has no ScriptRunContext, is relying on
undefined behaviour of a UI library used completely outside the execution
model it assumes — exactly the "looks fine, is not" failure class this
codebase is written against (see CLAUDE.md). So this file ports the SQL and
text-formatting half only, verbatim (same tables, same columns, same LIMITs),
and turns what used to be an inline st.image() into a structured "evidence"
entry the caller returns to the client — which renders it as a real <img>
against the tile server's own /tile_image/{slide_tile} endpoint instead of a
markdown blob.

Consequence: these two files' SQL must be kept in sync by hand. If you change
a query in app/hpc_chat_handlers_v23.py, change it here too (and vice versa) —
there is no test that catches drift between them, the same way there is
between api_client.py and tile_server_v2_.py's own endpoints.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from sqlalchemy import Engine, inspect, text


def classify_malignancy_polarity(q: str) -> str | None:
    q = (q or "").lower()
    q_norm = re.sub(r"[_\-]+", " ", q)
    neg_patterns = [
        r"\bnon\s*malignant\b",
        r"\bnot\s+malignant\b",
        r"\bdoes\s+not\s+(have|contain|show)\s+malignant\b",
        r"\bwithout\s+malignant\b",
        r"\black\s+of\s+malignant\b",
        r"\bno\s+malignant\b",
        r"\bnon\s*malignant\s*epithelium\b",
        r"\bnot\s+malignant\s*epithelium\b",
        r"\bdoes\s+not\s+have\s+malignant\s*epithelium\b",
        r"\bwithout\s+malignant\s*epithelium\b",
        r"\bno\s+malignant\s*epithelium\b",
    ]
    if any(re.search(p, q_norm) for p in neg_patterns):
        return "non"
    pos_patterns = [r"\bmalignant\b", r"\bmalignant\s*epithelium\b"]
    if any(re.search(p, q_norm) for p in pos_patterns):
        return "malignant"
    return None


def _bool_malignant(flag: Any) -> bool | None:
    """Same tri-state coercion hpc_chat_handlers_v23.py duplicates twice —
    not routed through backend/malignancy.py because that module *raises* on
    an unrecognised value (parse_malignant), and a raise here would 500 the
    whole chat turn over one dictionary row. describe_malignant() would work,
    but returns a label ("missing") rather than the None this needs — so this
    stays its own small function rather than half-fitting either one."""
    if flag is None:
        return None
    if isinstance(flag, (int, np.integer)):
        return bool(flag)
    if isinstance(flag, str):
        s = flag.strip().lower()
        if s in ("true", "t", "1", "yes", "y"):
            return True
        if s in ("false", "f", "0", "no", "n"):
            return False
    return None


def handle_tile(conn, tile_name: str, intents: dict) -> tuple[str, list[dict]]:
    """Returns (markdown, tile_images) — tile_images is 0 or 1 entries, since
    the original showed exactly one image via st.image() for a tile match."""
    tile_name = (tile_name or "").strip()
    if not tile_name:
        return "Please specify a tile name.", []

    provided_slide_tile = None
    if "_" in tile_name and tile_name.upper().startswith("TCGA-"):
        provided_slide_tile = tile_name.strip().upper()

    if provided_slide_tile:
        row = conn.execute(
            text("SELECT * FROM tile_registry WHERE slide_tile = :st LIMIT 1"),
            {"st": provided_slide_tile},
        ).fetchone()
    else:
        row = conn.execute(
            text(
                """
                SELECT * FROM tile_registry
                WHERE tiles = :t OR h5_source_path = :t
                LIMIT 1
                """
            ),
            {"t": tile_name},
        ).fetchone()
        if not row:
            row = conn.execute(
                text(
                    """
                    SELECT * FROM tile_registry
                    WHERE tiles ILIKE :t OR h5_source_path ILIKE :t
                    ORDER BY image_index ASC
                    LIMIT 1
                    """
                ),
                {"t": f"%{tile_name}%"},
            ).fetchone()

    if not row:
        return f"No tile found matching '{tile_name}'.", []

    info = dict(row._mapping)
    out = [f"### Tile `{tile_name}` Summary"]
    out += [f"- **{k}**: {v}" for k, v in info.items()]

    hpc_id = info.get("hpc_id")
    try:
        hpc_id_int = int(hpc_id) if hpc_id is not None else None
    except (TypeError, ValueError):
        hpc_id_int = None

    if intents.get("malignant") and hpc_id_int is not None:
        if intents.get("negative"):
            kb_row = conn.execute(
                text("SELECT * FROM hpc_non_malignant_details WHERE hpc_id = :hpc_id LIMIT 1"),
                {"hpc_id": hpc_id_int},
            ).fetchone()
            out.append(f"Tile belongs to **non-malignant epithelium** (HPC {hpc_id_int})")
        else:
            malignant_flag = _bool_malignant(
                conn.execute(
                    text("SELECT malignant FROM hpc_dictionary WHERE hpc_id = :hpc_id LIMIT 1"),
                    {"hpc_id": hpc_id_int},
                ).scalar()
            )
            if malignant_flag is True:
                kb_row = conn.execute(
                    text("SELECT * FROM hpc_malignant_details WHERE hpc_id = :hpc_id LIMIT 1"),
                    {"hpc_id": hpc_id_int},
                ).fetchone()
                out.append(f"Tile belongs to **malignant epithelium** (HPC {hpc_id_int})")
            else:
                kb_row = conn.execute(
                    text("SELECT * FROM hpc_non_malignant_details WHERE hpc_id = :hpc_id LIMIT 1"),
                    {"hpc_id": hpc_id_int},
                ).fetchone()
                out.append(f"Tile belongs to **non-malignant epithelium** (HPC {hpc_id_int})")

        if kb_row:
            out.append("**Epithelium details from KB:**")
            for k, v in dict(kb_row._mapping).items():
                out.append(f"- **{k}**: {v}")
            out.append("")

    tile_images: list[dict] = []
    slide_tile_key = info.get("slide_tile") or provided_slide_tile
    if slide_tile_key:
        slide_tile_key = str(slide_tile_key).strip().upper()
        tile_images.append({"slide_tile": slide_tile_key, "caption": f"{tile_name} ({slide_tile_key})"})

    return "\n".join(out), tile_images


def handle_slide(conn, slide_id: str, intents: dict) -> str:
    slide_id = re.sub(r"^slide\s+", "", str(slide_id), flags=re.I).strip().upper()
    out = [f"### 🧫 Slide `{slide_id}` Summary\n"]

    if intents.get("malignant"):
        want_non = bool(intents.get("negative"))
        malignant_clause = "hd.malignant = FALSE" if want_non else "hd.malignant = TRUE"

        hpc_ids = conn.execute(
            text(
                f"""
                SELECT DISTINCT hp.hpc_id
                FROM hpl_profile_proportion hp
                JOIN hpc_dictionary hd ON hp.hpc_id = hd.hpc_id
                WHERE UPPER(hp.slides) = :slide_id
                  AND {malignant_clause}
                ORDER BY hp.hpc_id
                """
            ),
            {"slide_id": slide_id},
        ).scalars().all()

        if hpc_ids:
            label = "Non-malignant" if want_non else "Malignant"
            out.append(f"⚠️ {label} HPCs: {', '.join(map(str, hpc_ids))}")
        else:
            out.append("No matching HPCs detected for this slide.")

        return "\n".join(out) + f"\n\nThe slide viewer is now updated for **{slide_id}**."

    summary = conn.execute(
        text("SELECT * FROM hpl_profile_summary WHERE UPPER(slides) = :slide_id LIMIT 1"),
        {"slide_id": slide_id},
    ).fetchone()

    if summary:
        out.append("**Slide Summary:**")
        for k, v in dict(summary._mapping).items():
            out.append(f"- **{k}**: {v}")
        out.append("")
    else:
        out.append("No slide summary found.")
        out.append("")

    proportions = conn.execute(
        text(
            """
            SELECT hpc_id, proportion, samples
            FROM hpl_profile_proportion
            WHERE UPPER(slides) = :slide_id
            ORDER BY proportion DESC
            LIMIT 5
            """
        ),
        {"slide_id": slide_id},
    ).fetchall()

    if proportions:
        out.append("**Top HPCs:**")
        for row in proportions:
            d = dict(row._mapping)
            prop = d.get("proportion", None)
            try:
                prop_txt = f"{float(prop):.5f}" if prop is not None else "NA"
            except (TypeError, ValueError):
                prop_txt = "NA"
            out.append(
                f"- **HPC {d.get('hpc_id')}** → proportion: {prop_txt}, sample: {d.get('samples')}"
            )
        out.append("")

    return "\n".join(out) + f"\n\nThe slide viewer is now updated for **{slide_id}**."


def handle_sample(conn, sample_id: str, intents: dict) -> str:
    sample_id = (sample_id or "").strip().upper()
    out = [f"### 🧬 Sample `{sample_id}` Summary"]

    if not sample_id:
        return "Please specify a valid sample ID."

    summary = conn.execute(
        text("SELECT * FROM hpl_profile_summary WHERE UPPER(samples) = :s LIMIT 1"),
        {"s": sample_id},
    ).fetchone()

    if not summary:
        summary = conn.execute(
            text("SELECT * FROM hpl_profile_summary WHERE samples ILIKE :s LIMIT 1"),
            {"s": f"%{sample_id}%"},
        ).fetchone()

    if summary:
        out += [f"- **{k}**: {v}" for k, v in dict(summary._mapping).items()]
    else:
        out.append("No sample summary found.")

    return "\n".join(out)


def handle_hpc(conn, ids: list, intents: dict, engine: Engine) -> tuple[str, list[dict]]:
    insp = inspect(engine)
    out = []
    tile_images: list[dict] = []

    for hpc_id in ids:
        try:
            hpc_id_int = int(hpc_id)
        except (TypeError, ValueError):
            out.append(f"## HPC {hpc_id} Summary\n\n⚠️ Invalid HPC ID: {hpc_id}")
            continue

        block = [f"## HPC {hpc_id_int} Summary\n"]

        base = conn.execute(
            text("SELECT * FROM hpc_dictionary WHERE hpc_id = :hpc_id LIMIT 1"),
            {"hpc_id": hpc_id_int},
        ).fetchone()

        if not base:
            block.append(f"No entry found for HPC {hpc_id_int}.\n")
            out.append("\n".join(block))
            continue

        d = dict(base._mapping)

        if intents.get("malignant"):
            want_non = bool(intents.get("negative"))
            malignant_flag = _bool_malignant(d.get("malignant"))

            if want_non:
                kb_row = conn.execute(
                    text("SELECT * FROM hpc_non_malignant_details WHERE hpc_id = :hpc_id LIMIT 1"),
                    {"hpc_id": hpc_id_int},
                ).fetchone()
                block.append("**Non-malignant epithelium details:**")
            else:
                if malignant_flag is True:
                    kb_row = conn.execute(
                        text("SELECT * FROM hpc_malignant_details WHERE hpc_id = :hpc_id LIMIT 1"),
                        {"hpc_id": hpc_id_int},
                    ).fetchone()
                    block.append("**Malignant epithelium details:**")
                else:
                    kb_row = conn.execute(
                        text("SELECT * FROM hpc_non_malignant_details WHERE hpc_id = :hpc_id LIMIT 1"),
                        {"hpc_id": hpc_id_int},
                    ).fetchone()
                    block.append("**Non-malignant epithelium details:**")

            if kb_row:
                for k, v in dict(kb_row._mapping).items():
                    block.append(f"- **{k}**: {v}")
                block.append("")
            else:
                block.append("No epithelial phenotype details found for this HPC.")

            out.append("\n".join(block))
            continue

        block.append("**Dictionary Details:**")
        for k, v in d.items():
            block.append(f"- **{k}**: {v}")
        block.append("")

        tables = insp.get_table_names()
        for table in tables:
            if table in ("hpc_dictionary", "h_latent_vectors"):
                continue
            # Stage 6's scratch tables (kb_stage.py) exist only while a load is
            # running and hold that CSV's rows, not Knowledge Bank state — see
            # the identical skip in hpc_chat_handlers_v23.handle_hpc.
            if table.startswith("hpl_stage_"):
                continue

            cols = [c["name"] for c in insp.get_columns(table)]
            id_col = "hpc_id" if "hpc_id" in cols else ("dominant_hpc" if "dominant_hpc" in cols else None)
            if not id_col:
                continue

            rows = conn.execute(
                text(f"SELECT * FROM {table} WHERE {id_col} = :hpc_id LIMIT 5"),
                {"hpc_id": hpc_id_int},
            ).fetchall()

            if not rows:
                continue

            block.append(f"**{table.replace('_', ' ').title()}:**")
            for r in rows:
                for k, v in dict(r._mapping).items():
                    block.append(f"- **{k}**: {v}")
                block.append("")

        tile_rows = conn.execute(
            text(
                """
                SELECT tiles, slide_tile, image_index
                FROM tile_registry
                WHERE hpc_id = :hpc_id
                ORDER BY image_index ASC
                LIMIT 6
                """
            ),
            {"hpc_id": hpc_id_int},
        ).fetchall()

        for row in tile_rows:
            st_key = str(row.slide_tile or "").strip().upper()
            if st_key:
                tile_images.append({"slide_tile": st_key, "caption": str(row.tiles)})

        out.append("\n".join(block))

    return "\n\n---\n\n".join(out), tile_images


def handle_analytics(conn, query: str, intent_type: str) -> str:
    q = query or ""
    polarity = classify_malignancy_polarity(q)

    if intent_type == "count_hpcs":
        result = conn.execute(
            text(
                """
                SELECT
                    SUM(CASE WHEN malignant IS TRUE  THEN 1 ELSE 0 END) AS malignant_count,
                    SUM(CASE WHEN malignant IS FALSE THEN 1 ELSE 0 END) AS non_malignant_count,
                    SUM(CASE WHEN malignant IS NULL  THEN 1 ELSE 0 END) AS unknown_count,
                    COUNT(*) AS total_hpcs
                FROM hpc_dictionary;
                """
            )
        ).fetchone()

        total = int(result.total_hpcs or 0)
        malignant = int(result.malignant_count or 0)
        non_malignant = int(result.non_malignant_count or 0)
        unknown = int(getattr(result, "unknown_count", 0) or 0)

        malignant_percent = round(100 * malignant / total, 2) if total else 0
        non_malignant_percent = round(100 * non_malignant / total, 2) if total else 0
        unknown_percent = round(100 * unknown / total, 2) if total else 0

        if polarity == "malignant":
            return f"There are **{malignant} malignant HPCs** out of {total} total ({malignant_percent}%)."
        if polarity == "non":
            return f"There are **{non_malignant} non-malignant HPCs** out of {total} total ({non_malignant_percent}%)."

        return (
            f"Total HPCs: **{total}**\n"
            f"- Malignant: {malignant} ({malignant_percent}%)\n"
            f"- Non-malignant: {non_malignant} ({non_malignant_percent}%)\n"
            f"- Unknown: {unknown} ({unknown_percent}%)"
        )

    if intent_type == "slide_malignant_coverage":
        m = re.search(r"\b(TCGA-[A-Z0-9\-]+-?DX\d+)\b", q, re.I)
        if not m:
            return "Please specify a valid slide ID."

        slide_id = m.group(1).strip().upper()

        result = conn.execute(
            text(
                """
                SELECT
                    ROUND(
                        CAST(
                            100 * SUM(CASE WHEN hd.malignant IS TRUE  THEN hp.proportion ELSE 0 END)
                            / NULLIF(SUM(hp.proportion), 0)
                            AS numeric
                        ),
                        2
                    ) AS malignant_percent,
                    ROUND(
                        CAST(
                            100 * SUM(CASE WHEN hd.malignant IS FALSE THEN hp.proportion ELSE 0 END)
                            / NULLIF(SUM(hp.proportion), 0)
                            AS numeric
                        ),
                        2
                    ) AS non_malignant_percent
                FROM hpl_profile_proportion hp
                JOIN hpc_dictionary hd ON hp.hpc_id = hd.hpc_id
                WHERE UPPER(hp.slides) = :slide_id;
                """
            ),
            {"slide_id": slide_id},
        ).fetchone()

        if not result:
            return f"No data found for slide `{slide_id}`."

        mal = result.malignant_percent
        non = result.non_malignant_percent

        if polarity in ("malignant", None):
            if mal is None:
                return f"No malignant coverage data found for slide `{slide_id}`."
            return f"Malignant epithelium covers **{mal}%** of slide `{slide_id}`."

        if non is None:
            return f"No non-malignant coverage data found for slide `{slide_id}`."
        return f"Non-malignant covers **{non}%** of slide `{slide_id}`."

    return "Sorry, I couldn't identify what kind of analytics you want."


def handle_survival(conn, query: str) -> str:
    q = query or ""
    hpc_match = re.search(r"\bhpc[-_\s]*([0-9]+)\b", q, re.I)
    tile_match = re.search(
        r"(?:\btile[_\-\s]*)?([A-Za-z0-9_\-]+\.(?:jpe?g|png|tif))\b",
        q,
        re.I,
    )
    slide_match = re.search(r"\b(TCGA-[A-Z0-9\-]+-?DX\d+)\b", q, re.I)
    slide_id = slide_match.group(1).strip().upper() if slide_match else None

    if tile_match and not hpc_match:
        tile_name = tile_match.group(1).strip()
        hpc_id_val = None
        if slide_id:
            slide_tile = f"{slide_id}_{tile_name}"
            hpc_id_val = conn.execute(
                text("SELECT hpc_id FROM tile_registry WHERE UPPER(slide_tile) = :st LIMIT 1"),
                {"st": slide_tile.upper()},
            ).scalar()

        if hpc_id_val is None:
            hpc_id_val = conn.execute(
                text("SELECT hpc_id FROM tile_registry WHERE tiles ILIKE :tile LIMIT 1"),
                {"tile": f"%{tile_name}%"},
            ).scalar()

        if hpc_id_val is None:
            return f"No HPC found for tile `{tile_name}`."

        try:
            hpc_id_int = int(hpc_id_val)
        except (TypeError, ValueError):
            return f"Tile `{tile_name}` has an invalid HPC ID mapping: {hpc_id_val}."
        hpc_match = True
    else:
        hpc_id_int = None

    if hpc_match:
        if hpc_id_int is None:
            try:
                hpc_id_int = int(hpc_match.group(1))
            except (TypeError, ValueError):
                return "Please specify a valid HPC ID."
        result = conn.execute(
            text("SELECT * FROM hpc_survival_analysis WHERE hpc_id = :hpc_id LIMIT 1"),
            {"hpc_id": hpc_id_int},
        ).fetchone()

        if not result:
            return f"No survival data found for HPC {hpc_id_int}."

        d = dict(result._mapping)

        def _fnum(v, nd=3, default="NA"):
            try:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return default
                return f"{float(v):.{nd}f}"
            except Exception:
                return default

        p = d.get("p", None)
        try:
            p_float = float(p) if p is not None else None
        except Exception:
            p_float = None

        log2_p_val = d.get("log2_p", None)
        if log2_p_val is None:
            if p_float is None:
                log2_p_txt = "NA"
            else:
                eps = 1e-300
                log2_p_txt = _fnum(-np.log2(max(p_float, eps)), nd=3)
        else:
            log2_p_txt = _fnum(log2_p_val, nd=3)

        sig_txt = "Not statistically significant."
        if p_float is not None and p_float < 0.05:
            sig_txt = "Significant association with survival (p < 0.05)."

        return (
            f"### Survival analysis for HPC {hpc_id_int}\n"
            f"- **Hazard Ratio (HR):** {_fnum(d.get('expcoef', None), 3)}\n"
            f"- **Coefficient (β):** {_fnum(d.get('coef', None), 3)}\n"
            f"- **Standard Error (SE):** {_fnum(d.get('se', None), 3)}\n"
            f"- **95% CI (HR):** ({_fnum(d.get('expcoef_lower_95', None), 3)}, {_fnum(d.get('expcoef_upper_95', None), 3)})\n"
            f"- **Z-score:** {_fnum(d.get('z', None), 3)}\n"
            f"- **p-value:** {_fnum(p_float, 4)}\n"
            f"- **-log₂(p):** {log2_p_txt}\n\n"
            f"{sig_txt}"
        )

    top = conn.execute(
        text(
            """
            SELECT hpc_id, expcoef, p
            FROM hpc_survival_analysis
            WHERE p IS NOT NULL
            ORDER BY p ASC
            LIMIT 10
        """
        )
    ).fetchall()

    if top:
        lines = ["### 🧬 Top 10 HPCs associated with survival"]
        for row in top:
            try:
                hr_txt = f"{float(row.expcoef):.3f}" if row.expcoef is not None else "NA"
            except Exception:
                hr_txt = "NA"
            try:
                p_txt = f"{float(row.p):.4f}" if row.p is not None else "NA"
            except Exception:
                p_txt = "NA"
            lines.append(f"- **HPC {row.hpc_id}** → HR={hr_txt}, p={p_txt}")
        return "\n".join(lines)

    return "No survival analysis data available."


def fetch_answer_from_db(
    query: str, engine: Engine, detected: dict, polarity: str | None
) -> tuple[str, list[dict]]:
    """Server-side twin of hpc_chat_handlers_v23.fetch_answer_from_db.

    `detected` is the caller's own detect_entity_patterns(query) result (that
    function is pure — no st.* calls — so tile_server_v2_.py imports it
    directly from app/hpc_chat_handlers_v23 rather than duplicating it here
    too); `polarity` is classify_malignancy_polarity(query), computed once by
    the caller since it is also needed to build the plan.

    Returns (markdown, tile_images) instead of a bare string — tile_images
    replaces the inline st.image() calls the Streamlit version made.
    """
    q = (query or "").lower()

    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))

        if any(kw in q for kw in ["how many", "count", "total hpcs", "number of hpcs"]):
            return handle_analytics(conn, query, "count_hpcs"), []

        if any(kw in q for kw in ["portion", "coverage", "covered by malignant", "percent malignant"]):
            return handle_analytics(conn, query, "slide_malignant_coverage"), []

        if any(kw in q for kw in ["survival", "cox", "hazard ratio", "p-value", "significant", "regression"]):
            return handle_survival(conn, query), []

        is_negative = polarity == "non"
        intents = {
            "malignant": polarity in ("malignant", "non"),
            "negative": is_negative,
            "tile": detected.get("tile"),
            "slide": detected.get("slide"),
            "sample": detected.get("sample"),
            "hpc": detected.get("hpc"),
        }

        output_parts = []
        tile_images: list[dict] = []

        if intents["tile"]:
            part, images = handle_tile(conn, intents["tile"], intents)
            if part:
                output_parts.append(part)
            tile_images.extend(images)

        if intents["slide"]:
            part = handle_slide(conn, intents["slide"], intents)
            if part:
                output_parts.append(part)

        if intents["sample"]:
            part = handle_sample(conn, intents["sample"], intents)
            if part:
                output_parts.append(part)

        if intents["hpc"]:
            part, images = handle_hpc(conn, intents["hpc"], intents, engine)
            if part:
                output_parts.append(part)
            tile_images.extend(images)

        if not output_parts:
            return "Please specify a valid tile, slide, sample, or HPC ID.", []

        return "\n\n---\n\n".join(output_parts), tile_images
