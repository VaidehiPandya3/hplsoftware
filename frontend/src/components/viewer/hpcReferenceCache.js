// Replaces app_v28.py's load_hpc_titles() / load_hpc_title_map() /
// load_survival_coefficients() — those hit Postgres directly and have no
// bulk REST equivalent (see CLAUDE.md-adjacent task notes). Instead we fetch
// per-HPC via the existing api.getHpcInfo / api.getHpcSurvival endpoints,
// only for the HPC ids actually present on the current slide, and cache
// every result in a module-level Map so the same id is never re-fetched
// during the session (HPC dictionary/survival rows are static reference
// data — safe to cache for as long as the Knowledge Bank does not change).
//
// That caveat is load-bearing: the cache is keyed by hpc id alone, and an hpc
// id only means one thing within a single Knowledge Bank. hpl_kb and
// hpl_kb_test have their own hpc_dictionary and hpc_survival_analysis rows, so
// switching target has to empty this — see clearHpcReferenceCache below, and
// app_v28.py's st.cache_data.clear() on the same switch, for the same reason.

import { api } from "../../api.js";

const infoCache = new Map(); // hpcId -> info object from /hpc/{id}/info
const infoPending = new Map(); // hpcId -> in-flight promise

const NO_SURVIVAL = Symbol("no-survival-data");
const survivalCache = new Map(); // hpcId -> row object | NO_SURVIVAL
const survivalPending = new Map();

/** Fetch (and cache) a single HPC's info row. Rejects on network/API error — not cached on failure so a transient outage can be retried. */
export async function getHpcInfoCached(hpcId) {
  const id = Number(hpcId);
  if (infoCache.has(id)) return infoCache.get(id);
  if (infoPending.has(id)) return infoPending.get(id);

  const promise = api
    .getHpcInfo(id)
    .then((info) => {
      infoCache.set(id, info);
      infoPending.delete(id);
      return info;
    })
    .catch((err) => {
      infoPending.delete(id);
      throw err;
    });
  infoPending.set(id, promise);
  return promise;
}

/**
 * Build a hpc_id -> hpc_title Map for exactly the ids given (typically the
 * unique hpc_id values present on the current slide's tiles — a handful,
 * never more than the ~71 clusters that exist). Mirrors load_hpc_title_map().
 */
export async function getHpcTitleMapFor(hpcIds) {
  const unique = [...new Set(hpcIds.filter((h) => h !== null && h !== undefined).map(Number))];
  const entries = await Promise.all(
    unique.map(async (id) => {
      try {
        const info = await getHpcInfoCached(id);
        return [id, String(info?.hpc_title || "").trim()];
      } catch {
        return [id, ""];
      }
    })
  );
  const map = new Map();
  for (const [id, title] of entries) {
    if (title) map.set(id, title);
  }
  return map;
}

async function getSurvivalRowCached(hpcId) {
  const id = Number(hpcId);
  if (survivalCache.has(id)) return survivalCache.get(id);
  if (survivalPending.has(id)) return survivalPending.get(id);

  const promise = api
    .getHpcSurvival(id)
    .then((row) => {
      survivalCache.set(id, row || NO_SURVIVAL);
      survivalPending.delete(id);
      return row || NO_SURVIVAL;
    })
    .catch(() => {
      // No survival row for this HPC (e.g. 404) — cache the negative result
      // too, since "this HPC has no survival data" is itself static and
      // worth not re-fetching every render.
      survivalCache.set(id, NO_SURVIVAL);
      survivalPending.delete(id);
      return NO_SURVIVAL;
    });
  survivalPending.set(id, promise);
  return promise;
}

/**
 * Build a hpc_id -> {coef, expcoef, p, ciLow, ciHigh} Map for the given ids,
 * keeping only rows with p < pThreshold — mirrors load_survival_coefficients's
 * `p_threshold=0.05` default and its `df[df["p"] < p_threshold]` filter
 * exactly (strict less-than).
 */
export async function getHpcSurvivalMapFor(hpcIds, pThreshold = 0.05) {
  const unique = [...new Set(hpcIds.filter((h) => h !== null && h !== undefined).map(Number))];
  const rows = await Promise.all(unique.map((id) => getSurvivalRowCached(id).then((row) => [id, row])));

  const map = new Map();
  for (const [id, row] of rows) {
    if (row === NO_SURVIVAL) continue;
    const coef = Number(row.coef);
    if (!Number.isFinite(coef)) continue;
    const p = Number(row.p);
    if (pThreshold !== null && pThreshold !== undefined) {
      if (!(p < pThreshold)) continue;
    }
    map.set(id, {
      coef,
      expcoef: Number(row.expcoef),
      p,
      ciLow: Number(row.expcoef_lower_95),
      ciHigh: Number(row.expcoef_upper_95),
    });
  }
  return map;
}


/**
 * Empty every cache in this module.
 *
 * Called when the Knowledge Bank target changes. Without it, the viewer keeps
 * showing the previous database's cluster titles and survival coefficients for
 * any hpc id present in both — which is exactly the "one view reading one
 * database while another reads the other" failure the KB selector exists to
 * prevent, one level down. In-flight promises are dropped too: they were
 * issued against the old target, so their results are answers to a question
 * nobody is asking any more.
 */
export function clearHpcReferenceCache() {
  infoCache.clear();
  infoPending.clear();
  survivalCache.clear();
  survivalPending.clear();
}
