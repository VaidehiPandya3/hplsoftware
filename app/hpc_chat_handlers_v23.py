"""
Full DB chat handlers from app_v21, adapted for app_v23:
- Tile previews use TileServerClient.get_tile_image(slide_tile) instead of local H5.
- Survival / analytics still use PostgreSQL via the passed connection.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
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


def handle_tile(conn, match, intents, ts_client) -> str:
    tile_name = match if isinstance(match, str) else match.group(1)
    tile_name = (tile_name or "").strip()

    if not tile_name:
        return "Please specify a tile name."

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
        return f"No tile found matching '{tile_name}'."
    info = dict(row._mapping)
    out = [f"### Tile `{tile_name}` Summary"]
    out += [f"- **{k}**: {v}" for k, v in info.items()]

    idx = info.get("image_index")
    try:
        idx = int(idx) if idx is not None else None
    except (TypeError, ValueError):
        idx = None

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
            malignant_flag = conn.execute(
                text("SELECT malignant FROM hpc_dictionary WHERE hpc_id = :hpc_id LIMIT 1"),
                {"hpc_id": hpc_id_int},
            ).scalar()
            if isinstance(malignant_flag, (int, np.integer)):
                malignant_flag = bool(malignant_flag)
            elif isinstance(malignant_flag, str):
                s = malignant_flag.strip().lower()
                if s in ["true", "t", "1", "yes", "y"]:
                    malignant_flag = True
                elif s in ["false", "f", "0", "no", "n"]:
                    malignant_flag = False
                else:
                    malignant_flag = None

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

    slide_tile_key = info.get("slide_tile") or provided_slide_tile
    if slide_tile_key:
        slide_tile_key = str(slide_tile_key).strip().upper()
        try:
            img = ts_client.get_tile_image(slide_tile_key)
            st.image(img, caption=f"{tile_name} ({slide_tile_key})", use_container_width=True)
        except Exception as e:
            st.error(f"Could not load tile image from server: {e}")
    elif idx is not None:
        st.warning("No slide_tile key for this row; cannot fetch tile image from tile server.")

    return "\n".join(out)


def handle_slide(conn, match, intents) -> str:
    slide_id = match if isinstance(match, str) else match.group(0)
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

        st.session_state.active_slide = slide_id
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

    st.session_state.active_slide = slide_id
    return "\n".join(out) + f"\n\nThe slide viewer is now updated for **{slide_id}**."


def handle_sample(conn, match, intents) -> str:
    sample_id = match if isinstance(match, str) else match.group(0)
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


def handle_hpc(conn, ids, intents, ts_client, engine: Engine) -> str:
    insp = inspect(engine)
    out = []

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
            malignant_flag = d.get("malignant")
            if isinstance(malignant_flag, (int, np.integer)):
                malignant_flag = bool(malignant_flag)
            elif isinstance(malignant_flag, str):
                s = malignant_flag.strip().lower()
                if s in ["true", "t", "1", "yes", "y"]:
                    malignant_flag = True
                elif s in ["false", "f", "0", "no", "n"]:
                    malignant_flag = False
                else:
                    malignant_flag = None

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
            if table in ["hpc_dictionary", "h_latent_vectors"]:
                continue
            # Stage 6's scratch tables (kb_stage.py) exist only while a load is
            # running and hold that CSV's rows, not Knowledge Bank state. They
            # are named so this loop cannot pick them up — no hpc_id or
            # dominant_hpc column — and skipped by prefix as well, so a future
            # scratch column named less carefully does not start answering
            # questions here. This loop is the reader a grep does not find.
            if table.startswith("hpl_stage_"):
                continue

            cols = [c["name"] for c in insp.get_columns(table)]
            id_col = "hpc_id" if "hpc_id" in cols else ("dominant_hpc" if "dominant_hpc" in cols else None)
            if not id_col:
                continue

            if id_col in ("hpc_id", "dominant_hpc"):
                where_sql = f"{id_col} = :hpc_id"
                params: dict[str, Any] = {"hpc_id": hpc_id_int}
            else:
                where_sql = f"TRIM({id_col}::text) ILIKE TRIM(:h)"
                params = {"h": f"%{str(hpc_id_int)}%"}

            rows = conn.execute(
                text(f"SELECT * FROM {table} WHERE {where_sql} LIMIT 5"),
                params,
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

        if tile_rows:
            block.append("")
            st.subheader(f"Example Tiles for HPC {hpc_id_int}")
            cols_ui = st.columns(3)
            for i, row in enumerate(tile_rows):
                st_key = str(row.slide_tile or "").strip().upper()
                if not st_key:
                    continue
                try:
                    img = ts_client.get_tile_image(st_key)
                    cols_ui[i % 3].image(img, caption=str(row.tiles), use_container_width=True)
                except Exception:
                    continue

        out.append("\n".join(block))

    return "\n\n---\n\n".join(out)


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

    return "Sorry, I couldn’t identify what kind of analytics you want."


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


def detect_entity_patterns(query: str) -> dict:
    q = (query or "").strip().lower()

    def after(keyword):
        # (?![a-z]) blocks the keyword's own plural "s" glued on with no
        # separator: "hpcs"/"slides"/"tiles"/"samples" — all extremely
        # natural phrasings ("which slides have...", "how many tiles...") —
        # used to match the bare keyword and then capture their own trailing
        # "s" as if it were the identifier (slide_match=="s", etc.), which
        # silently skipped the real, digit/pattern-anchored fallback regexes
        # below since those only run `if not <keyword>_match`. Still allows
        # "hpc40"/"hpc:40"/"hpc-40" — a digit or punctuation right after the
        # keyword, never a letter.
        m = re.search(rf"{keyword}(?![a-z])\s*[:=]?\s*([^\s,;]+)", q, re.I)
        return m.group(1).strip() if m else None

    tile_match = after("tile")
    slide_match = after("slide")
    sample_match = after("sample")
    hpc_match = after("hpc")
    if hpc_match and not re.search(r"\d", hpc_match):
        # "hpc" used generically with nothing numeric nearby ("how many
        # tiles does this hpc have" captured hpc_match=="have") — not a real
        # ID. Drop it so the digit-anchored fallback just below gets a
        # chance to correctly find nothing, rather than keeping a captured
        # English word that int(hpc_id) will only fail on downstream.
        hpc_match = None
    if slide_match is None:
        m = re.search(r"\b(TCGA-[A-Z0-9\-]+-?DX\d+)\b", query or "", re.I)
        if m:
            slide_match = m.group(1)
    if not tile_match:
        m = re.search(
            r"\b(tile[_\-]?\d+|[A-Za-z0-9_\-]+\.jpe?g|[A-Za-z0-9_\-]+\.png|[A-Za-z0-9_\-]+\.tif)\b",
            q,
        )
        tile_match = m.group(0) if m else None
    if not hpc_match:
        m = re.search(r"\bhpc[-_\s]*([0-9]+)\b", q, re.I)
        hpc_match = m.group(1) if m else None
    return {
        "tile": tile_match,
        "slide": slide_match,
        "sample": sample_match,
        "hpc": [hpc_match] if hpc_match else None,
    }


from sqlalchemy import text
from sqlalchemy.engine import Engine
import streamlit as st

def fetch_answer_from_db(query: str, engine: Engine, ts_client) -> str:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

            q = (query or "").lower()
            polarity = classify_malignancy_polarity(q)
            is_negative = polarity == "non"

            if any(kw in q for kw in ["how many", "count", "total hpcs", "number of hpcs"]):
                return handle_analytics(conn, query, "count_hpcs")

            if any(kw in q for kw in ["portion", "coverage", "covered by malignant", "percent malignant"]):
                return handle_analytics(conn, query, "slide_malignant_coverage")

            if any(
                kw in q
                for kw in ["survival", "cox", "hazard ratio", "p-value", "significant", "regression"]
            ):
                return handle_survival(conn, query)

            detected = detect_entity_patterns(query)

            intents = {
                "malignant": polarity in ("malignant", "non"),
                "negative": is_negative,
                "tile": detected.get("tile"),
                "slide": detected.get("slide"),
                "sample": detected.get("sample"),
                "hpc": detected.get("hpc"),
            }

            handlers = {
                "tile": lambda c, m, i: handle_tile(c, m, i, ts_client),
                "slide": handle_slide,
                "sample": handle_sample,
                "hpc": lambda c, m, i: handle_hpc(c, m, i, ts_client, engine),
            }

            output_parts = []
            for entity, func in handlers.items():
                match = intents.get(entity)
                if match:
                    part = func(conn, match, intents)
                    if part:
                        output_parts.append(part)

            if not output_parts:
                return "Please specify a valid tile, slide, sample, or HPC ID."

            return "\n\n---\n\n".join(output_parts)

    except Exception as e:
        st.error(f"DB error inside fetch_answer_from_db: {e}")
        return f"Database query failed: {e}"