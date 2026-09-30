# Chatbot: making answers sound natural — what we need and why

**Status: almost there.** The chatbot already works end-to-end — it understands
a question, looks up the right facts in the Knowledge Bank, and answers
correctly. The one thing left is making the *wording* of the answer sound
like a person wrote it instead of a lab printout. To fix that properly, we
need one thing: **access to a proper LLM API**. This note explains which one,
how we'd keep it safe, and why.

## Where we are today

- The chatbot's fact-finding (looking up an HPC, a slide, a survival stat) is
  100% our own database, on our own infrastructure. That part is done and
  doesn't change with anything below.
- Turning those facts into a sentence currently uses a very small, free AI
  model (called Ollama) that runs locally on our own server. It's small on
  purpose — it runs on a shared HPC login machine, which isn't meant to run
  a big AI model all day.
- Because that model is so small, the wording comes out stiff and
  robotic — technically correct, but not natural to read.

## The one gap: a proper LLM for the wording step only

We want to swap that one small step — "turn these facts into a friendly
sentence" — for a real, high-quality LLM. Nothing else about the chatbot
changes: it still looks up facts from our own database first, always. The
LLM's only job is to phrase the answer nicely.

### Which API would we use?

**The university's own enterprise Copilot / Azure OpenAI access** — the one
IT can issue us a key for, not a public consumer account (not a personal
ChatGPT or API key from OpenAI directly). In plain terms: it's the same
Microsoft AI service, but running under the university's own contract with
Microsoft, on the university's own terms, not the open public version.

### Why that one, and not something else?

- **A bigger local model** doesn't solve the problem — we don't have anywhere
  appropriate to run one. It would need to live on a shared university
  machine that isn't meant for that kind of ongoing heavy use.
- **A personal/public AI API key** (paying for our own OpenAI or similar
  account) would work technically, but then our data would be governed by
  that company's standard consumer terms, not by any agreement the
  university has already negotiated.
- **The university's own Copilot/Azure access** gives us the quality jump we
  need, while staying inside an agreement IT and the university have already
  reviewed for exactly this kind of use — we're not introducing a new vendor
  relationship, just using one that already exists.

### How do we make sure data doesn't just start leaving the HPC?

Three separate safeguards, not just a policy on paper:

1. **Off by default, one switch.** The connection to the outside API is
   controlled by a single setting. Until someone deliberately turns it on,
   the chatbot keeps using the small on-site model exactly as it does today.
   Nothing changes automatically.
2. **Only the minimum leaves, and only for one step.** The only thing ever
   sent out is the short text of the question and the already-looked-up
   answer (e.g. "HPC 40 — associated with X, seen in slide Y") — never raw
   images, never the database itself, never login credentials. The database
   lookup itself always happens on our own servers first; the outside API
   only ever sees the small result of that lookup, to phrase it nicely.
3. **Today's data is already public research data.** Right now this runs on
   TCGA — a public, already-de-identified cancer research dataset, not real
   patient records. So even the "minimum" data above isn't sensitive today.
   The moment we ever bring in a private/clinical cohort, we treat that as
   its own decision — the switch above means we can simply leave it off for
   that data, or revisit this whole question then, rather than it being an
   afterthought.

### What we're asking approval for

Permission to wire up that one switch and point it at the university's
Copilot/Azure OpenAI access, once IT can give us an endpoint and key for it.
The code change itself is small and already scoped out — we're not asking
for time to build something big, just the green light and the credentials to
turn the last step on properly.
