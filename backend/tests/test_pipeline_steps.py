"""The pipeline stepper: does the sidebar say what actually has to happen next?

Registration is a new step between cluster classification and the Knowledge
Bank load, and inserting a step into this list is exactly the kind of change
that breaks quietly. Two ways, specifically:

  * a step key with no entry in the renderer dict is a KeyError at render time,
    on a screen nobody opens until a run reaches that stage;
  * a gate that reads the wrong flag shows "ready to load" for a cohort with no
    identity rows, and Stage 6 then refuses at a 0% match rate — a number that
    reads like a slide-naming bug and is in fact a missing step.

app_v28.py imports streamlit and much else at module level, so _pipeline_steps
is pulled out by source and exec'd against stubs, the same technique
test_cohort_shift.py uses on _http_detail.
"""

import re
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
APP = BACKEND.parent / "app" / "app_v28.py"

# Standalone mode is how this suite runs on the cluster, and there nothing else
# has already put backend/ on the path — the seam tests below import
# tile_server_v2_ directly.
sys.path.insert(0, str(BACKEND))

_SOURCE = APP.read_text()


def _pipeline_steps():
    """_pipeline_steps, exec'd with stubs for the three names it closes over."""
    start = _SOURCE.index("def _pipeline_steps")
    end = _SOURCE.index("\ndef ", start)
    namespace = {
        "_SLURM_IN_FLIGHT": {"PENDING", "RUNNING", "CONFIGURING"},
        "_test_packaging_note": lambda status: "",
        "_human_bytes": lambda n: f"{n} B",
    }
    exec(compile(_SOURCE[start:end], "probe", "exec"), namespace)
    return namespace["_pipeline_steps"]


def _renderer_keys():
    """The keys the renderer dict actually handles."""
    start = _SOURCE.index("    renderers = {")
    end = _SOURCE.index("    }", start)
    return set(re.findall(r'"([a-z_]+)": lambda', _SOURCE[start:end]))


def _finished_through_assignment(**overrides):
    """A run whose Stages 1-4 are all done, so only the KB steps are in play."""
    status = {
        "status": "completed",
        "total_slides": 10, "succeeded": 10, "tiling_complete": True,
        "h5_ready": True,
        "extraction_ready": True,
        "assignment_ready": True,
        "registration_ready": True,
    }
    status.update(overrides)
    return status


def _by_key(steps):
    return {s["key"]: s for s in steps}


# --- the wiring ----------------------------------------------------------

def test_every_step_has_a_renderer(_tmp=None):
    """A step key with no renderer is a KeyError on the pipeline screen."""
    steps = _pipeline_steps()(_finished_through_assignment())
    missing = [s["key"] for s in steps if s["key"] not in _renderer_keys()]
    assert not missing, f"steps with no renderer: {missing}"


def test_the_renderer_check_can_fail(_tmp=None):
    """Proves the test above is load-bearing rather than trivially true."""
    assert "a_step_that_does_not_exist" not in _renderer_keys()
    assert _renderer_keys(), "no renderers were parsed at all — the regex is wrong"


def test_the_steps_are_numbered_in_the_order_they_are_listed(_tmp=None):
    """Inserting a step means renumbering the ones after it. A list reading
    1, 2, 3, 4, 5, 5 is the visible half of having forgotten to."""
    steps = _pipeline_steps()(_finished_through_assignment())
    numbers = [int(re.match(r"(\d+)\.", s["title"]).group(1)) for s in steps]
    assert numbers == list(range(1, len(steps) + 1)), numbers


def test_registration_comes_before_the_kb_load(_tmp=None):
    keys = [s["key"] for s in _pipeline_steps()(_finished_through_assignment())]
    assert keys.index("registration") < keys.index("kb_load")


# --- the gates -----------------------------------------------------------

def test_registration_is_gated_on_packaging_not_on_the_assignment(_tmp=None):
    """It reads tile identity out of the .h5 and needs no cluster labels, so
    making it wait for Stage 4 would keep Stage 6 blocked behind a step that
    could have finished hours earlier."""
    steps = _by_key(_pipeline_steps()(_finished_through_assignment(
        assignment_ready=False, extraction_ready=False)))
    assert steps["registration"]["state"] == "action", steps["registration"]


def test_registration_is_blocked_before_packaging_finishes(_tmp=None):
    steps = _by_key(_pipeline_steps()(_finished_through_assignment(
        h5_ready=False, registration_ready=False)))
    assert steps["registration"]["state"] == "blocked"
    assert "packaging" in steps["registration"]["summary"]


def test_the_kb_load_waits_for_registration(_tmp=None):
    """The whole reason the step exists. Without identity rows Stage 6's match
    rate is 0% by construction, and before this the refusal was the first sign
    anything was wrong."""
    steps = _by_key(_pipeline_steps()(_finished_through_assignment(
        registration_done=False)))
    assert steps["kb_load"]["state"] == "blocked", steps["kb_load"]
    assert "registration" in steps["kb_load"]["summary"]


def test_the_kb_load_opens_once_registration_is_done(_tmp=None):
    """The companion — the gate above must not be permanently closed."""
    steps = _by_key(_pipeline_steps()(_finished_through_assignment(
        registration_done=True)))
    assert steps["kb_load"]["state"] == "action", steps["kb_load"]


def test_a_missing_assignment_still_blocks_the_load_even_once_registered(_tmp=None):
    """Registration is an additional gate, not a replacement for the old one."""
    steps = _by_key(_pipeline_steps()(_finished_through_assignment(
        registration_done=True, assignment_ready=False)))
    assert steps["kb_load"]["state"] == "blocked"
    assert "classification" in steps["kb_load"]["summary"]


def test_a_run_that_predates_registration_shows_it_as_outstanding(_tmp=None):
    """A run finished before this step existed reports no registration_done —
    which is correct, not a display bug. Those runs did not register; their
    tiles reached the KB by hand or not at all. The one thing it must not do is
    claim the step is finished."""
    steps = _by_key(_pipeline_steps()(_finished_through_assignment()))
    assert steps["registration"]["state"] == "action"


def test_a_finished_registration_reports_what_it_wrote(_tmp=None):
    steps = _by_key(_pipeline_steps()(_finished_through_assignment(
        registration_done=True,
        registration_rows={"wsi_registry": 10, "tile_registry": 8_192},
    )))
    assert steps["registration"]["state"] == "done"
    assert "8,192" in steps["registration"]["summary"]


def test_a_registration_with_no_recorded_rows_still_reads_as_done(_tmp=None):
    """registration_rows is JSONB and .get()s to None on a deployment that has
    not applied the migration. The step must degrade to 'registered', not crash
    formatting a None."""
    steps = _by_key(_pipeline_steps()(_finished_through_assignment(
        registration_done=True, registration_rows=None)))
    assert steps["registration"]["state"] == "done"
    assert steps["registration"]["summary"] == "registered"


# --- the client/server seam ----------------------------------------------
#
# api_client posts a JSON body straight into a pydantic model. A field the
# client sends and the model does not declare is silently dropped by pydantic,
# so "Also read slide headers" would tick in the UI and do nothing, and the
# only symptom would be an empty wsi_metadata nobody was watching.


def _client_payload_keys(method_name):
    """The keys api_client's method actually posts."""
    source = (BACKEND.parent / "app" / "api_client.py").read_text()
    start = source.index(f"def {method_name}(")
    end = source.index("\n    def ", start)
    body = source[start:end]
    return set(re.findall(r'"([a-z_]+)":', body))


def test_the_registration_payload_matches_the_endpoints_model(_tmp=None):
    import tile_server_v2_ as srv

    declared = set(srv.RegistrationRequest.model_fields)
    for method in ("preview_registration", "commit_registration"):
        sent = _client_payload_keys(method)
        assert sent, f"parsed no payload keys out of {method}"
        extra = sent - declared
        assert not extra, (
            f"api_client.{method} sends {sorted(extra)}, which "
            f"RegistrationRequest does not declare — pydantic drops them "
            f"silently, so the control would appear to work and do nothing"
        )


def test_the_payload_check_can_fail(_tmp=None):
    """Proves the comparison is real: a field the model has never heard of must
    be reported."""
    import tile_server_v2_ as srv
    declared = set(srv.RegistrationRequest.model_fields)
    assert "a_field_nobody_declared" not in declared
    assert {"a_field_nobody_declared"} - declared == {"a_field_nobody_declared"}


def test_both_registration_routes_are_mounted(_tmp=None):
    import tile_server_v2_ as srv
    paths = {r.path for r in srv.app.routes}
    for path in ("/dataset-jobs/{submission_id}/register-preview",
                 "/dataset-jobs/{submission_id}/register"):
        assert path in paths, f"{path} is not mounted"


def test_the_status_payload_carries_what_the_stepper_gates_on(_tmp=None):
    """_pipeline_steps reads these off /status. A flag the server never sets
    reads as False, which shows registration as permanently outstanding."""
    server = (BACKEND / "tile_server_v2_.py").read_text()
    for flag in ("registration_done", "registration_ready", "registration_rows",
                 "registration_dataset_id", "registration_at"):
        assert f'base["{flag}"]' in server, f"/status never sets {flag}"


# --- a KB write running on Slurm ------------------------------------------
#
# Stages 5 and 6 can now be handed to Slurm so the write outlives the server.
# That gives each of them a state they never had: submitted, not yet committed.
# Reading it as "ready to register" is how a cohort gets registered twice.


def test_a_queued_registration_reads_as_running_not_as_ready(_tmp=None):
    steps = _by_key(_pipeline_steps()(_finished_through_assignment(
        registration_job_id="987654", registration_slurm_state="PENDING")))

    assert steps["registration"]["state"] == "running"
    assert "PENDING" in steps["registration"]["summary"]


def test_a_queued_kb_load_reads_as_running_not_as_ready(_tmp=None):
    steps = _by_key(_pipeline_steps()(_finished_through_assignment(
        registration_done=True,
        kb_load_job_id="987655", kb_load_slurm_state="RUNNING")))

    assert steps["kb_load"]["state"] == "running"


def test_a_job_that_ended_without_committing_asks_for_attention(_tmp=None):
    """Not "done" — the job set neither flag — and not "ready", which would hide
    a failed write behind a button that looks like it was never pressed."""
    steps = _by_key(_pipeline_steps()(_finished_through_assignment(
        registration_job_id="987654", registration_slurm_state="FAILED")))

    assert steps["registration"]["state"] == "attention"


def test_a_committed_job_still_reads_as_done(_tmp=None):
    """The job sets registration_done itself, so "done" keeps meaning committed
    however the write ran."""
    steps = _by_key(_pipeline_steps()(_finished_through_assignment(
        registration_done=True, registration_rows={"tile_registry": 38892},
        registration_job_id="987654", registration_slurm_state="COMPLETED")))

    assert steps["registration"]["state"] == "done"
    assert "38,892" in steps["registration"]["summary"]


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_steps_test_"))
        try:
            fn(tmp_path)
            print(f"PASS  {name}")
        except Exception as e:
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
