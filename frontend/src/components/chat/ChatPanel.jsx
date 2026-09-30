// Port of app_v28.py's chat section (~lines 5880-5954): message history,
// input box, the "which slide would you like me to look at?" branch, and the
// debug query-plan expander — all driven by the server's /query pipeline
// (tile_server_v2_.py, completing the "Phase 2" its docstring described).
//
// What's intentionally NOT wired up yet: chat setting the WSI viewer's
// highlight mode / selected HPC / heatmap / adjacency pair (the deeper half
// of app/ui_actions_v25.py's apply_plan_to_session). That state currently
// lives privately inside SlideViewer (components/viewer/index.jsx) — hooking
// chat into it needs that state lifted up first, which is a separate,
// sizeable change from "the chatbot works". What IS wired: naming a slide in
// chat opens/switches to it, exactly like handle_slide's "The slide viewer is
// now updated for X" — the single most common chat behaviour.
import { useEffect, useRef, useState } from "react";
import { api } from "../../api";
import { planNeedsSlide, resolveActiveSlide } from "./applyPlan";
import "./chat.css";

const GREETING =
  "Hello! I can look up HPCs, slides, tiles, and samples, run survival and count " +
  "analytics, and open the WSI viewer.";

function TileImages({ images }) {
  if (!images || !images.length) return null;
  return (
    <div className="chat-tile-images">
      {images.map((img, i) => (
        <figure key={`${img.slide_tile}-${i}`} className="chat-tile-image">
          <img src={api.tileImageUrl(img.slide_tile)} alt={img.caption || img.slide_tile} />
          <figcaption>{img.caption || img.slide_tile}</figcaption>
        </figure>
      ))}
    </div>
  );
}

function Message({ msg }) {
  return (
    <div className={`chat-message chat-message-${msg.role}`}>
      <div className="chat-message-bubble">
        {/* Plain text rather than a markdown renderer — no new dependency for
            a UI whose answers already read fine without bold/heading markup
            rendered. */}
        <div className="chat-message-text">{msg.content}</div>
        <TileImages images={msg.tileImages} />
      </div>
    </div>
  );
}

export default function ChatPanel({ slideId, onOpenSlide }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [lastPlan, setLastPlan] = useState(null);
  const [slideList, setSlideList] = useState([]);
  const historyRef = useRef(null);

  useEffect(() => {
    api
      .listSlides()
      .then(setSlideList)
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (historyRef.current) historyRef.current.scrollTop = historyRef.current.scrollHeight;
  }, [messages, sending]);

  async function send() {
    const prompt = input.trim();
    if (!prompt || sending) return;
    setInput("");
    const withUser = [...messages, { role: "user", content: prompt }];
    setMessages(withUser);
    setSending(true);

    try {
      const result = await api.query(prompt, slideId || null, {
        history: withUser.map(({ role, content }) => ({ role, content })),
        sessionContext: { active_slide: slideId || null },
      });
      const plan = result.plan || {};
      setLastPlan(plan);

      // Port of app_v28.py: if nothing is active yet and the plan implies a
      // slide-scoped action, ask which slide instead of guessing — same
      // wording, same three-example cap.
      if (!slideId && planNeedsSlide(plan)) {
        const examples = slideList.length ? slideList.slice(0, 3).join(", ") : "TCGA-XX-XXXX";
        const ask =
          `Which slide would you like me to look at? Name a slide (e.g. ${examples}) ` +
          "or pick one from the dropdown below, then ask again.";
        setMessages((m) => [...m, { role: "assistant", content: ask }]);
        return;
      }

      const resolved = resolveActiveSlide(plan, slideId);
      if (resolved && resolved !== slideId) {
        onOpenSlide(resolved);
      }

      const finalAnswer = result.final_answer || "I could not find an answer for that question.";
      const tileImages = (result.evidence && result.evidence.tile_images) || [];
      setMessages((m) => [...m, { role: "assistant", content: finalAnswer, tileImages }]);
    } catch (e) {
      const message = e instanceof api.ApiError ? e.message : String((e && e.message) || e);
      setMessages((m) => [...m, { role: "assistant", content: `Query failed: ${message}` }]);
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="chat-panel">
      <div className="chat-history" ref={historyRef}>
        {messages.length === 0 && <div className="chat-empty">{GREETING}</div>}
        {messages.map((msg, i) => (
          <Message key={i} msg={msg} />
        ))}
        {sending && <div className="chat-typing">Thinking…</div>}
      </div>

      <form
        className="chat-input-row"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        <input
          type="text"
          value={input}
          placeholder="Ask something about HPCs..."
          onChange={(e) => setInput(e.target.value)}
          disabled={sending}
        />
        <button type="submit" disabled={sending || !input.trim()}>
          Send
        </button>
      </form>

      {lastPlan && (
        <details className="chat-debug">
          <summary>Query Plan (debug)</summary>
          <pre>{JSON.stringify(lastPlan, null, 2)}</pre>
          <div className="chat-debug-caption">planner: {lastPlan.planner || "?"}</div>
        </details>
      )}
    </div>
  );
}
