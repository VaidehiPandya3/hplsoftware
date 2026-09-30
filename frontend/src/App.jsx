import { useEffect, useState } from "react";
import { api, KB_PRODUCTION, KB_TEST } from "./api";
import PipelinePanel from "./components/pipeline";
import SlideViewer from "./components/viewer";
import ChatPanel from "./components/chat/ChatPanel.jsx";
import { clearHpcReferenceCache } from "./components/viewer/hpcReferenceCache.js";
import "./App.css";

const KB_ORDER = [KB_PRODUCTION, KB_TEST];

/**
 * Which Knowledge Bank everything downstream reads and writes.
 *
 * Port of app_v28.py:57-83. It is first in the sidebar there for a stated
 * reason: without it "the viewer would read one database while the chatbot read
 * the other, with nothing on screen to say so". This UI had no selector at all
 * and no kb_target on any request, so it silently read and wrote production
 * only — the same failure one level up.
 *
 * Labels come from the server's /health, which reports the resolved database
 * name per target, rather than being hardcoded: the names are env-configurable
 * (HPL_DB_NAME / HPL_DB_NAME_TEST) and a browser cannot read those.
 */
function KbTargetPicker({ target, onChange }) {
  const [names, setNames] = useState({});

  useEffect(() => {
    let live = true;
    Promise.all(
      KB_ORDER.map((t) =>
        // Asked per target because /health reports the one it was asked about.
        // A deployment with no hpl_kb_test is normal — _get_engine builds
        // engines lazily precisely so that is not a startup failure — so a
        // rejection here means "unlabelled", not "broken".
        fetch(`${api.baseUrl}/health?kb_target=${t}`)
          .then((r) => (r.ok ? r.json() : null))
          .then((d) => [t, d && d.database])
          .catch(() => [t, null])
      )
    ).then((pairs) => {
      if (live) setNames(Object.fromEntries(pairs));
    });
    return () => {
      live = false;
    };
  }, []);

  const label = (t) => {
    const pretty = t === KB_PRODUCTION ? "Production" : "Test";
    return names[t] ? `${pretty} — ${names[t]}` : pretty;
  };

  return (
    <div className="app-kb-picker">
      <span className="app-kb-label">Knowledge Bank</span>
      {KB_ORDER.map((t) => (
        <label key={t} className="app-kb-option">
          <input
            type="radio"
            name="kb-target"
            checked={target === t}
            onChange={() => onChange(t)}
          />
          {label(t)}
        </label>
      ))}
      <span className="app-kb-help">
        Registration and the cluster-assignment load write here, and the slide viewer and HPC
        panels read from here. Pipeline execution and run history always stay in production — a run
        is one run regardless of which Knowledge Bank it filled.
      </span>
      {target === KB_TEST && (
        <span className="app-kb-warning">
          Reading and writing {names[KB_TEST] || "the test Knowledge Bank"}. Production is
          untouched.
        </span>
      )}
    </div>
  );
}

function SlidePicker({ slideId, onChange }) {
  const [slides, setSlides] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    api
      .listSlides()
      .then(setSlides)
      .catch((e) => setError(e.message));
  }, []);

  return (
    <div className="slide-picker">
      <label htmlFor="slide-select">Slide</label>
      <select
        id="slide-select"
        value={slideId}
        onChange={(e) => onChange(e.target.value)}
      >
        <option value="">Select a slide…</option>
        {slides.map((id) => (
          <option key={id} value={id}>
            {id}
          </option>
        ))}
      </select>
      {error && <span className="slide-picker-error">Could not load slide list: {error}</span>}
    </div>
  );
}

function App() {
  const [slideId, setSlideId] = useState("");
  const [kbTarget, setKbTarget] = useState(api.getKbTarget());

  function switchKb(target) {
    if (target === kbTarget) return;
    api.setKbTarget(target);
    // The module-level HPC cache is keyed by hpc id alone, so it would keep
    // serving the previous database's titles and survival rows.
    clearHpcReferenceCache();
    // A slide id only means one thing within a single Knowledge Bank, so the
    // selected slide does not carry across.
    setSlideId("");
    setKbTarget(target);
  }

  return (
    <div className="app-shell">
      <header className="app-topbar">
        <h1>HPC Tile Explorer</h1>
        <span className="app-topbar-note">
          JS UI (preview) — talks to the same backend as app_v28.py, which keeps running unchanged.
        </span>
      </header>

      <div className="app-body">
        <aside className="app-sidebar">
          <KbTargetPicker target={kbTarget} onChange={switchKb} />
          {/* Keyed on the target so a switch remounts the whole subtree rather
              than leaving components holding rows fetched from the other
              database. This is the React equivalent of app_v28.py's
              st.cache_data.clear() on the same switch, and it is done once here
              rather than by adding a kb_target argument to every reader — which
              would have to be remembered for every reader added later. */}
          <PipelinePanel key={`pipeline-${kbTarget}`} />
        </aside>

        <main className="app-main">
          {/* Chat first, matching app_v28.py's layout (chat above the slide
              picker/viewer). Naming a slide in chat calls setSlideId directly
              — in this UI there is no separate open/close toggle for the
              viewer, so setting the active slide already is "opening" it. */}
          <ChatPanel key={`chat-${kbTarget}`} slideId={slideId} onOpenSlide={setSlideId} />

          {/* Distinct key prefixes: these two are siblings, and React requires
              keys to be unique among siblings — a bare kbTarget on both is a
              duplicate-key error, which it reports as a warning and then
              silently drops or duplicates one of them. */}
          <SlidePicker key={`picker-${kbTarget}`} slideId={slideId} onChange={setSlideId} />
          <SlideViewer key={`viewer-${kbTarget}`} slideId={slideId} />
        </main>
      </div>
    </div>
  );
}

export default App;
