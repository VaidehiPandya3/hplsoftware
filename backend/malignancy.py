"""What "malignant" means, in one place.

`hpc_dictionary.malignant` is loosely typed. The rows already in the live
database carry the flag as text — "True"/"False" — but the column's own type is
one of the four things `backend/kb_live_schema_2026-08-26.txt` never captured
(`\\d hpc_dictionary` has never been run), so nothing in this repository knows
whether it is boolean, text, or something else. A driver may hand it back as
bool, as int, or as any of the spellings a human has typed into it.

Two callers already normalised it independently — the tile viewer's malignancy
filter and the per-slide malignancy breakdown, both in `app/app_v28.py` — and
`select_tumour_slides.py` is a third, deciding which slides ANORAK runs on at
all. Three copies of a rule about which tissue is cancer is two too many, so the
rule lives here and they import it.

Two functions, because the two callers want opposite things from an
unrecognised value:

  parse_malignant()     refuses. Used where the answer decides whether a slide
                        is processed. Defaulting an unknown spelling to
                        non-malignant would drop tumour slides from a cohort and
                        report a smaller cohort as a success.

  describe_malignant()  returns "missing". Used where the answer is being shown
                        to someone. A viewer that raises rather than rendering
                        an unlabelled cluster is worse than a viewer with a grey
                        band in it.

The vocabulary is deliberately the union of both call sites' existing rules
rather than a tightening of them: this file changes where the rule lives, not
what it says.
"""

from __future__ import annotations

#: Recognised spellings, lower-cased and stripped. Taken verbatim from the two
#: normalisations in app/app_v28.py so that lifting the rule out cannot silently
#: reclassify a cluster that was already being read correctly.
_TRUE = frozenset({"true", "t", "1", "yes", "y", "malignant"})
_FALSE = frozenset({"false", "f", "0", "no", "n", "non-malignant", "non malignant"})


class UnrecognisedMalignancy(ValueError):
    """A `malignant` value outside the recognised vocabulary.

    Carries the value so a caller can name it. Raised rather than resolved
    because there is no safe default: read as non-malignant it drops a slide
    from the cohort, read as malignant it puts non-tumour tissue through an
    inference pass, and both look like an ordinary result.
    """

    def __init__(self, value):
        self.value = value
        super().__init__(
            f"unrecognised hpc_dictionary.malignant value {value!r}. "
            f"Recognised: {sorted(_TRUE)} for malignant, {sorted(_FALSE)} for "
            f"non-malignant."
        )


def parse_malignant(value) -> bool:
    """True or False, or raise UnrecognisedMalignancy.

    bool is checked before int deliberately — `isinstance(True, int)` is true,
    and numpy's bool_ is neither — so a driver returning a real boolean is not
    routed through the numeric branch. An integer other than 0 or 1 is *not*
    accepted as truthy: `malignant = 2` is not a flag with a value, it is a
    column holding something this does not understand.

    None and the empty string raise as well. A cluster whose malignancy was
    never recorded is not a non-malignant cluster, and it is the reference data
    that needs fixing rather than this call.
    """
    if isinstance(value, bool):
        return value
    # numpy scalars, without importing numpy for a type check.
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return parse_malignant(value.item())
        except UnrecognisedMalignancy:
            raise UnrecognisedMalignancy(value) from None
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise UnrecognisedMalignancy(value)
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("utf-8", "replace")
    if value is None:
        raise UnrecognisedMalignancy(value)
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise UnrecognisedMalignancy(value)


def describe_malignant(value) -> str:
    """"malignant", "non-malignant", or "missing" — never raises.

    The display counterpart of parse_malignant(), and the exact behaviour the
    two blocks in app/app_v28.py had before this module existed: a NULL or an
    unrecognised spelling is shown as its own category rather than being folded
    into non-malignant, so a dictionary row that needs attention is visible in
    the UI instead of being quietly counted as benign.
    """
    try:
        return "malignant" if parse_malignant(value) else "non-malignant"
    except UnrecognisedMalignancy:
        return "missing"


def malignant_flag(value):
    """True / False / None — describe_malignant()'s answer as a tristate.

    For the viewer's colour lookup, which wants a nullable boolean rather than a
    label.
    """
    try:
        return parse_malignant(value)
    except UnrecognisedMalignancy:
        return None
