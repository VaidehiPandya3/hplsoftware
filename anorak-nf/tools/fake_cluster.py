#!/usr/bin/env python3
"""A fake Beatson cluster for running preflight.sh and bootstrap_gpu_extras.sh
for real, on a laptop.

Stand-ins for sinfo, sacctmgr, scontrol, squeue, sbatch, srun, singularity,
nextflow, java and df, driven by one JSON state file, plus a fake inference
image (tensorflow, numpy, pip) and a fake openslide for the native env.

The stand-ins are only as useful as they are strict, so each does what the
real tool does to the thing preflight is checking, and no more:

  - singularity wipes the environment under --cleanenv, maps SINGULARITYENV_X
    to X, lets the image's own PYTHONPATH win over the host's, and makes a path
    that no --bind covers not exist inside the container (every absolute
    argument outside a bind is rewritten to a path that does not exist, which
    is what a container does to it);
  - GPUs are visible only with --nv, on a node srun gave a --gres=gpu, and then
    as many as CUDA_VISIBLE_DEVICES names — or all of the node's if nothing
    passed it through;
  - srun refuses an unknown partition the way Slurm does, and can be broken;
  - pip installs only from a local index, --no-deps, at exactly the pinned
    version, and freeze lists what is in --path.

Used by test_preflight.py and test_bootstrap_gpu_extras.py.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
from pathlib import Path

REAL_PY = sys.executable

# --- Slurm ---------------------------------------------------------------

_COMMON = r'''#!@PY@
import json, os, sys
STATE = @STATE@
def cluster():
    with open(os.path.join(STATE, "cluster.json")) as h:
        return json.load(h)
def log(name):
    with open(os.path.join(STATE, "calls.log"), "a") as h:
        h.write(name + " " + json.dumps(sys.argv[1:]) + "\n")
'''

SINFO = _COMMON + r'''
log("sinfo")
c = cluster(); args = sys.argv[1:]
part = node = None; fmt = "%P %a %l %D %t %N"
i = 0
while i < len(args):
    a = args[i]
    if a in ("-p", "--partition"): part = args[i + 1]; i += 2; continue
    if a in ("-n", "--nodes"): node = args[i + 1]; i += 2; continue
    if a in ("-o", "--format"): fmt = args[i + 1]; i += 2; continue
    i += 1
for name, p in c["partitions"].items():
    if part and name != part:
        continue
    for n in p["nodes"]:
        if node and n["name"] != node:
            continue
        line = fmt
        for token, value in (("%P", name + ("*" if p.get("default") else "")),
                             ("%l", p["maxtime"]), ("%m", str(n["mem"])),
                             ("%c", str(n["cpus"])), ("%G", n.get("gres", "(null)")),
                             ("%N", n["name"]), ("%t", "idle"), ("%D", "1"), ("%a", "up")):
            line = line.replace(token, value)
        print(line)
'''

SACCTMGR = _COMMON + r'''
log("sacctmgr")
c = cluster(); args = [a for a in sys.argv[1:] if a not in ("-n", "-P", "--noheader", "--parsable2")]
if c.get("sacctmgr_broken"):
    print("sacctmgr: error: Problem talking to the database: Connection refused", file=sys.stderr); sys.exit(1)
entity = args[args.index("show") + 1]
fields = next((a.split("=", 1)[1].split(",") for a in args if a.lower().startswith("format=")), [])
names = [a for a in args[args.index("show") + 2:] if "=" not in a and a != "where"]
if entity == "user":
    rows = [{"DefaultAccount": c.get("default_account", "")}]
elif entity == "assoc":
    rows = c.get("assoc", [])
elif entity == "qos":
    rows = [dict(v, Name=k) for k, v in c.get("qos", {}).items() if not names or k in names]
else:
    sys.exit(1)
for row in rows:
    low = {k.lower(): v for k, v in row.items()}
    print("|".join(str(low.get(f.lower(), "")) for f in fields))
'''

SCONTROL = _COMMON + r'''
log("scontrol")
c = cluster(); args = sys.argv[1:]
if args[:2] == ["show", "config"]:
    cfg = c.get("config", {})
    print("Configuration data as of 2026-09-25T10:00:00")
    for key in ("ProctrackType", "TaskPlugin", "SLURM_CONF"):
        if key in cfg:
            print("%-23s = %s" % (key, cfg[key]))
    if "ConstrainDevices" in cfg:
        print("\nCgroup Support Configuration:")
        print("%-23s = %s" % ("ConstrainDevices", cfg["ConstrainDevices"]))
elif args[:2] == ["show", "partition"]:
    p = c["partitions"].get(args[2])
    if p is None:
        print("Partition %s not found" % args[2], file=sys.stderr); sys.exit(1)
    print("PartitionName=%s Default=%s QoS=%s MaxTime=%s" % (
        args[2], "YES" if p.get("default") else "NO", p.get("qos", "N/A"), p["maxtime"]))
'''

SQUEUE = _COMMON + r'''
log("squeue")
if "--version" in sys.argv:
    print("slurm 23.02.7"); sys.exit(0)
for i in range(cluster().get("queued_jobs", 0)):
    print(1000 + i)
'''

SBATCH = _COMMON + r'''
if "--version" in sys.argv:
    print("slurm 23.02.7")
'''

SRUN = _COMMON + r'''
log("srun")
c = cluster(); args = sys.argv[1:]
takes_value = {"-p", "--partition", "-t", "--time", "-n", "-N", "-w", "-c", "--mem", "-J", "--gres", "--jobid"}
part = None; gres = None; i = 0
while i < len(args) and args[i].startswith("-"):
    a = args[i]
    if "=" in a:
        key, value = a.split("=", 1)
        if key in ("--partition", "-p"): part = value
        if key == "--gres": gres = value
        i += 1; continue
    if a in takes_value:
        if a in ("-p", "--partition"): part = args[i + 1]
        if a == "--gres": gres = args[i + 1]
        i += 2; continue
    i += 1
command = args[i:]
if c.get("srun_broken"):
    print("srun: error: Unable to allocate resources: Requested node configuration is not available", file=sys.stderr)
    sys.exit(1)
if part is None:
    part = next(n for n, p in c["partitions"].items() if p.get("default"))
if part not in c["partitions"]:
    print("srun: error: Unable to allocate resources: Invalid partition name specified", file=sys.stderr)
    sys.exit(1)
node = c["partitions"][part]["nodes"][0]
import re
found = re.search(r"gpu(?::[^:(,]+)?:(\d+)", node.get("gres", ""))
gpus = int(found.group(1)) if found else 0
if gres and gres.startswith("gpu") and not gpus:
    print("srun: error: Unable to allocate resources: Requested node configuration is not available", file=sys.stderr)
    sys.exit(1)
env = dict(os.environ)
env["SLURMD_NODENAME"] = node["name"]
tmp = os.path.join(STATE, "node_tmp"); os.makedirs(tmp, exist_ok=True)
env["TMPDIR"] = tmp
env.pop("CUDA_VISIBLE_DEVICES", None)
if gres and gres.startswith("gpu"):
    env["CUDA_VISIBLE_DEVICES"] = "2"
if not c.get("node_has_sbatch", True):
    env["PATH"] = os.pathsep.join(p for p in env["PATH"].split(os.pathsep) if not p.endswith("slurmcli"))
with open(os.path.join(STATE, "current_node.json"), "w") as h:
    json.dump({"node": node["name"], "gpus": gpus if (gres and gres.startswith("gpu")) else 0}, h)
os.execvpe(command[0], command, env)
'''

DF = _COMMON + r'''
d = sys.argv[-1]
print("Filesystem     1024-blocks      Used Available Capacity Mounted on")
print("/dev/fake       9999999999        10 %d       1%% %s" % (cluster().get("tmp_avail_kb", 10 ** 9), d))
'''

JAVA = r'''#!/bin/sh
echo 'openjdk version "17.0.8" 2023-07-18' >&2
'''

# `nextflow`: `config -flat` prints the state's flat config; `config` without
# -flat prints each setting under its last name, which is all the nested
# form's readers (the pre-rewrite preflight) ever matched on; `run -preview`
# exits as told.
NEXTFLOW = _COMMON + r'''
log("nextflow")
args = sys.argv[1:]
if "-v" in args or "-version" in args:
    print("nextflow version 26.04.6.12646"); sys.exit(0)
if "config" in args:
    flat = open(os.path.join(STATE, "config.flat")).read()
    if "-flat" in args:
        sys.stdout.write(flat); sys.exit(0)
    import re
    for line in flat.splitlines():
        m = re.match(r"^((?:[^\s=']|'[^']*')+) = (.*)$", line)
        if m:
            print("   %s = %s" % (re.split(r"\.(?=(?:[^']*'[^']*')*[^']*$)", m.group(1))[-1], m.group(2)))
    sys.exit(0)
if "run" in args:
    c = cluster()
    if c.get("preview_error"):
        print("ERROR ~ " + c["preview_error"], file=sys.stderr); sys.exit(1)
    print("ANORAK: preview ok"); sys.exit(0)
sys.exit(1)
'''

# --- the container runtime -------------------------------------------------

SINGULARITY = _COMMON + r'''
log("singularity")
args = sys.argv[1:]
if not args or args[0] != "exec":
    print("fake singularity: only exec is modelled", file=sys.stderr); sys.exit(255)
i = 1; cleanenv = nv = False; binds = []
while i < len(args) and args[i].startswith("-"):
    a = args[i]
    if a in ("--cleanenv", "-e"): cleanenv = True
    elif a == "--nv": nv = True
    elif a in ("--bind", "-B"): binds += args[i + 1].split(","); i += 1
    elif a.startswith("--bind="): binds += a.split("=", 1)[1].split(",")
    elif a == "--pwd": i += 1
    i += 1
image, command = args[i], args[i + 1:]
if not os.path.isfile(image):
    print("FATAL:   could not open image %s: failed to retrieve path" % image, file=sys.stderr); sys.exit(255)
meta = {}
if os.path.exists(image + ".json"):
    meta = json.load(open(image + ".json"))
# Inside, only what a bind covers exists (plus the image's own filesystem).
roots = [b.split(":")[0] for b in binds] + ["/bin", "/usr", "/dev", "/etc", "/private/etc", "/System", "/Library", STATE]
def visible(p):
    p = os.path.normpath(p)
    return any(p == r or p.startswith(r.rstrip("/") + "/") for r in roots)
command = [a if not a.startswith("/") or visible(a) else "/__not_in_container__" + a for a in command]
env = {} if cleanenv else {k: v for k, v in os.environ.items() if not k.startswith("SINGULARITYENV_")}
env["PATH"] = os.path.join(STATE, "imagebin") + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin")
if "env_pythonpath" in meta:   # the image's own %environment wins over the host's
    env["PYTHONPATH"] = meta["env_pythonpath"]
for k, v in os.environ.items():
    if k.startswith("SINGULARITYENV_"):
        env[k[len("SINGULARITYENV_"):]] = v
env["FAKE_INTERNAL_SITE"] = meta.get("site", "")
node = {}
if os.path.exists(os.path.join(STATE, "current_node.json")):
    node = json.load(open(os.path.join(STATE, "current_node.json")))
env["FAKE_INTERNAL_NODE_GPUS"] = str(node.get("gpus", 0) if nv else 0)
os.execvpe(command[0], command, env)
'''

# python3 inside the fake image: the real interpreter with no site-packages,
# and the image's own site appended after PYTHONPATH, where site-packages is.
IMAGE_PYTHON = r'''#!/bin/sh
PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$FAKE_INTERNAL_SITE" exec "@PYRAW@" -S "$@"
'''

FAKE_TENSORFLOW = r'''
import os
__version__ = "2.11.0"
class _Device:
    def __init__(self, i): self.name = "/physical_device:GPU:%d" % i
class config:
    @staticmethod
    def list_physical_devices(kind="GPU"):
        n = int(os.environ.get("FAKE_INTERNAL_NODE_GPUS", "0") or 0)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if n and visible is not None:
            n = len([v for v in visible.split(",") if v.strip()])
        return [_Device(i) for i in range(n)] if kind == "GPU" else []
    class experimental:
        @staticmethod
        def get_device_details(device):
            return {"device_name": "NVIDIA H200 (fake)", "compute_capability": (9, 0)}
class sysconfig:
    @staticmethod
    def get_build_info():
        return {"cuda_version": "12.1", "cudnn_version": "8.9"}
'''
FAKE_KERAS_MODELS = r'''
class _Prediction:
    def __init__(self, shape): self.shape = shape
class _Model:
    input_shape = (None, 384, 384, 3)
    output_shape = (None, 384, 384, 7)
    def predict(self, patch, verbose=0):
        return _Prediction((1,) + tuple(patch.shape[1:3]) + (7,))
def load_model(path, custom_objects=None, compile=True):
    import os
    if os.path.isfile(path) and b"CORRUPT" in open(path, "rb").read():
        raise ValueError("Unknown layer: TFOpLambda (fake)")
    return _Model()
'''
FAKE_NUMPY = r'''
__version__ = "1.22.2"
class _Array:
    def __init__(self, shape): self.shape = tuple(shape)
    def astype(self, dtype): return self
class random:
    @staticmethod
    def rand(*shape): return _Array(shape)
'''
FAKE_PIP_MAIN = r'''
import json, os, shutil, sys
STATE = @STATE@
args = sys.argv[1:]
def dists(path):
    for entry in sorted(os.listdir(path)) if os.path.isdir(path) else []:
        if entry.endswith(".dist-info"):
            meta = dict(line.split(": ", 1) for line in open(os.path.join(path, entry, "METADATA")).read().splitlines() if ": " in line)
            yield entry, meta["Name"], meta["Version"]
if args[0] == "freeze":
    for _, name, version in dists(args[args.index("--path") + 1]):
        print("%s==%s" % (name, version))
    sys.exit(0)
if args[0] != "install":
    sys.exit(2)
with open(os.path.join(STATE, "calls.log"), "a") as h:
    h.write("pip " + json.dumps(args) + "\n")
if json.load(open(os.path.join(STATE, "cluster.json"))).get("pip_offline"):
    print("ERROR: Could not find a version that satisfies the requirement (offline)", file=sys.stderr); sys.exit(1)
target = args[args.index("--target") + 1]
reqs, i = [], 1
while i < len(args):
    if args[i] == "--target": i += 2; continue
    if args[i] == "-r":
        reqs += [l.strip() for l in open(args[i + 1]) if l.strip() and not l.startswith("#")]; i += 2; continue
    if not args[i].startswith("-"): reqs.append(args[i])
    i += 1
index = os.path.join(STATE, "pypi")
for req in reqs:
    name, _, version = req.partition("==")
    versions = sorted(os.listdir(os.path.join(index, name))) if os.path.isdir(os.path.join(index, name)) else []
    version = version or (versions[-1] if versions else "")
    source = os.path.join(index, name, version)
    if not os.path.isdir(source):
        print("ERROR: No matching distribution found for %s" % req, file=sys.stderr); sys.exit(1)
    os.makedirs(target, exist_ok=True)
    for entry, dist_name, _ in list(dists(target)):
        if dist_name.lower() == name.lower():
            shutil.rmtree(os.path.join(target, entry))
    for item in os.listdir(source):
        dest = os.path.join(target, item)
        if os.path.isdir(dest): shutil.rmtree(dest)
        (shutil.copytree if os.path.isdir(os.path.join(source, item)) else shutil.copy)(os.path.join(source, item), dest)
    info = os.path.join(target, "%s-%s.dist-info" % (name.replace("-", "_"), version))
    os.makedirs(info, exist_ok=True)
    open(os.path.join(info, "METADATA"), "w").write("Name: %s\nVersion: %s\n" % (name, version))
'''
FAKE_OPENSLIDE = r'''
import json
PROPERTY_NAME_OBJECTIVE_POWER = "openslide.objective-power"
PROPERTY_NAME_MPP_X = "openslide.mpp-x"
class OpenSlideUnsupportedFormatError(Exception):
    pass
class OpenSlide:
    def __init__(self, path):
        try:
            with open(path) as handle:
                data = json.load(handle)
        except Exception:
            raise OpenSlideUnsupportedFormatError("Unsupported or missing image file")
        self.properties = data.get("props", {})
        self.level_dimensions = [tuple(data.get("dims", [40000, 30000]))]
    def __enter__(self): return self
    def __exit__(self, *exc): return False
'''

# --- building one ------------------------------------------------------------

PIPELINE = Path(__file__).resolve().parent.parent
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"

#: The effective config, flat, as `nextflow config -profile beatson -flat`
#: prints it for conf/beatson.config — trimmed to what preflight reads.
FLAT_CONFIG = r"""params.slide_column = 'slide_id'
params.sample_column = 'samples'
params.tumour_column = 'is_tumour'
params.output_mpp = 0.22
params.gpu_extras = '@ROOT@/extras'
params.max_cpu_memory = '72.GB'
params.max_cpu_time = '2d'
params.max_gpu_memory = '96.GB'
params.max_gpu_time = '36h'
params.gpu_partition = 'gpu'
params.cpu_partition = 'compute'
params.gpu_type = null
params.tiling_container = null
params.native_env_bin = '@ROOT@/native/bin'
params.gpu_container = '@ROOT@/images/tf.sif'
process.errorStrategy = { task.exitStatus == 65 ? 'finish'
                      : task.attempt <= 2 ? 'retry' : 'finish' }
process.maxRetries = 2
process.'withLabel:process_tiling'.cpus = 1
process.'withLabel:process_tiling'.memory = { def r = 16.GB * task.attempt; params.max_cpu_memory ? [r, MemoryUnit.of(params.max_cpu_memory.toString())].min() : r }
process.'withLabel:process_tiling'.time = { def r = 8.h * task.attempt; params.max_cpu_time ? [r, Duration.of(params.max_cpu_time.toString())].min() : r }
process.'withLabel:process_tiling'.queue = 'compute'
process.'withLabel:process_gpu'.cpus = 4
process.'withLabel:process_gpu'.memory = { def r = 32.GB * task.attempt; params.max_gpu_memory ? [r, MemoryUnit.of(params.max_gpu_memory.toString())].min() : r }
process.'withLabel:process_gpu'.time = { def r = 12.h * task.attempt; params.max_gpu_time ? [r, Duration.of(params.max_gpu_time.toString())].min() : r }
process.'withLabel:process_gpu'.queue = 'gpu'
process.'withLabel:process_gpu'.clusterOptions = { params.gpu_type ? "--gres=gpu:${params.gpu_type}:1" : '--gres=gpu:1' }
process.'withLabel:process_gpu'.container = '@ROOT@/images/tf.sif'
process.'withLabel:process_stitch'.cpus = 1
process.'withLabel:process_stitch'.memory = { def r = 24.GB * task.attempt; params.max_cpu_memory ? [r, MemoryUnit.of(params.max_cpu_memory.toString())].min() : r }
process.'withLabel:process_stitch'.time = { def r = 4.h * task.attempt; params.max_cpu_time ? [r, Duration.of(params.max_cpu_time.toString())].min() : r }
process.'withLabel:process_stitch'.queue = 'compute'
process.'withLabel:process_light'.cpus = 1
process.'withLabel:process_light'.memory = { def r = 8.GB * task.attempt; params.max_cpu_memory ? [r, MemoryUnit.of(params.max_cpu_memory.toString())].min() : r }
process.'withLabel:process_light'.time = { def r = 1.h * task.attempt; params.max_cpu_time ? [r, Duration.of(params.max_cpu_time.toString())].min() : r }
process.'withLabel:process_light'.queue = 'compute'
process.executor = 'slurm'
process.scratch = true
executor.queueSize = 200
singularity.enabled = true
singularity.autoMounts = true
singularity.runOptions = '--nv --bind @ROOT@:@ROOT@'
singularity.envWhitelist = 'CUDA_VISIBLE_DEVICES'
"""

#: A cluster preflight should pass on.
GOOD_CLUSTER = {
    "partitions": {
        "compute": {"default": True, "maxtime": "2-00:00:00", "qos": "N/A",
                    "nodes": [{"name": "c01", "cpus": 64, "mem": 512000, "gres": "(null)"}]},
        "gpu": {"default": False, "maxtime": "infinite", "qos": "N/A",
                "nodes": [{"name": "g01", "cpus": 64, "mem": 1031000,
                           "gres": "gpu:nvidia_h200:4(S:0-1),gpu:nvidia_h200:4"}]},
    },
    "default_account": "lab",
    "assoc": [{"Account": "lab", "Partition": "", "MaxSubmitJobs": "", "MaxJobs": "",
               "QOS": "normal", "DefaultQOS": "normal"}],
    "qos": {"normal": {"MaxSubmitPU": "500", "MaxJobsPU": ""}},
    "config": {"ProctrackType": "proctrack/cgroup", "TaskPlugin": "task/affinity,task/cgroup",
               "ConstrainDevices": "yes"},
    "queued_jobs": 0,
    "tmp_avail_kb": 900 * 1024 * 1024,
}

#: A slide of 40000x30000 at 40x, 0.25 um/px: 108 tiles at output_mpp 0.22.
GOOD_SLIDE = {"props": {"openslide.objective-power": "40", "openslide.mpp-x": "0.25"},
              "dims": [40000, 30000]}
SLIDE_TILES = 108


def _write(path: Path, text: str, executable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _install_dist(target: Path, name: str, version: str, module: str, body: str = "") -> None:
    _write(target / module / "__init__.py", body or f"__version__ = {version!r}\n")
    _write(target / f"{name.replace('-', '_')}-{version}.dist-info" / "METADATA",
           f"Name: {name}\nVersion: {version}\n")


class FakeCluster:
    """Everything preflight touches, under one root."""

    def __init__(self, root: Path, preflight: Path | None = None):
        # The root itself has no space: it is a --bind in runOptions, which
        # Nextflow splits on whitespace, so one there breaks the run as well
        # as this. Everything below it has spaces, quotes or brackets.
        self.root = Path(root) / "cluster"
        self.state = self.root / "state"
        self.bin = self.root / "bin"
        self.slurmcli = self.root / "slurmcli"
        self.pipeline = self.root / "pipe line"
        self.image = self.root / "images" / "tf.sif"
        self.extras = self.root / "extras"
        self.anorak = self.root / "AIgrading clone"
        self.raw = self.root / "raw dir's (x)"
        self.slides_csv = self.root / "lists" / "cohort list.csv"
        self.state.mkdir(parents=True)
        self.cluster = json.loads(json.dumps(GOOD_CLUSTER))
        self.save()
        self.flat = FLAT_CONFIG.replace("@ROOT@", str(self.root))
        self.save_config()

        subs = {"@PY@": REAL_PY, "@PYRAW@": REAL_PY, "@STATE@": repr(str(self.state))}
        def render(text):
            for key, value in subs.items():
                text = text.replace(key, value)
            return text
        for name, text in (("sinfo", SINFO), ("sacctmgr", SACCTMGR), ("scontrol", SCONTROL),
                           ("srun", SRUN), ("singularity", SINGULARITY),
                           ("nextflow", NEXTFLOW), ("java", JAVA), ("df", DF)):
            _write(self.bin / name, render(text), executable=True)
        for name, text in (("sbatch", SBATCH), ("squeue", SQUEUE)):
            _write(self.slurmcli / name, render(text), executable=True)
        os.symlink(REAL_PY, self.bin / "python3")
        _write(self.state / "imagebin" / "python3", render(IMAGE_PYTHON), executable=True)

        # The inference image: TensorFlow, numpy and pip, and no cv2.
        site = self.root / "imgsite" / "tf"
        _write(site / "tensorflow" / "__init__.py", FAKE_TENSORFLOW)
        _write(site / "tensorflow" / "keras" / "__init__.py", "")
        _write(site / "tensorflow" / "keras" / "models.py", FAKE_KERAS_MODELS)
        _write(site / "numpy" / "__init__.py", FAKE_NUMPY)
        _write(site / "pip" / "__init__.py", "")
        _write(site / "pip" / "__main__.py", render(FAKE_PIP_MAIN))
        self.image.parent.mkdir(parents=True)
        self.image.write_bytes(b"SIF")
        self.image_meta = {"site": str(site)}
        self.save_image()
        # A local index for the fake pip, with a newer cv2 than the pin.
        pypi = self.state / "pypi"
        for version in ("4.8.1.78", "4.10.0.84"):
            _write(pypi / "opencv-python-headless" / version / "cv2" / "__init__.py", f"__version__ = {version!r}\n")
        _write(pypi / "pillow" / "10.4.0" / "PIL" / "__init__.py", "")

        # gpu_extras as bootstrap_gpu_extras.sh leaves it, with its lock.
        _install_dist(self.extras, "opencv-python-headless", "4.8.1.78", "cv2")
        self.write_lock(["opencv-python-headless==4.8.1.78"])

        # The AIgrading clone and its checkpoint.
        _write(self.anorak / "generating_tile" / "save_cws.py", "")
        _write(self.anorak / "inference_slide" / "predict_gp.py", "import cv2\nimport numpy\nimport tensorflow\n")
        (self.anorak / "models").mkdir(parents=True)
        (self.anorak / "models" / "AIgrading_anorak.h5").write_bytes(HDF5_SIGNATURE + b"\0" * 256)

        # The native conda env the CPU steps run in: python with openslide.
        _write(self.root / "native" / "site" / "openslide" / "__init__.py", FAKE_OPENSLIDE)
        _write(self.root / "native" / "bin" / "python3",
               f'#!/bin/sh\nPYTHONPATH="{self.root}/native/site${{PYTHONPATH:+:$PYTHONPATH}}" exec "{REAL_PY}" -S "$@"\n',
               executable=True)

        # Slides, under a directory whose name carries a quote and brackets.
        self.add_slide("S1.ndpi", GOOD_SLIDE)
        self.add_slide("sub/S2.SVS", GOOD_SLIDE)   # upper-case extension
        self.write_list([("S1", "T1", "true"), ("S2", "T2", "true")])

        # The pipeline, copied so a test can edit main.nf, running the
        # preflight under test from inside it.
        self.pipeline.mkdir()
        for item in ("main.nf", "bin"):
            source = PIPELINE / item
            if source.is_dir():
                shutil.copytree(source, self.pipeline / item, ignore=shutil.ignore_patterns("__pycache__"))
            else:
                shutil.copy(source, self.pipeline / item)
        (self.pipeline / "tools").mkdir()
        for name in ("scan_slide_headers.py", "check_anorak_model.py"):
            shutil.copy(PIPELINE / "tools" / name, self.pipeline / "tools" / name)
        shutil.copy(preflight or (PIPELINE / "tools" / "preflight.sh"), self.pipeline / "tools" / "preflight.sh")

    # state ------------------------------------------------------------------
    def save(self):
        (self.state / "cluster.json").write_text(json.dumps(self.cluster, indent=1))

    def save_config(self):
        (self.state / "config.flat").write_text(self.flat)

    def save_image(self):
        (self.image.parent / (self.image.name + ".json")).write_text(json.dumps(self.image_meta))

    def set_config(self, key: str, raw: str | None):
        """Replace one flat setting (raw Groovy literal), append it, or drop it (None)."""
        lines = [l for l in self.flat.splitlines() if not l.startswith(key + " = ")]
        if raw is not None:
            lines.append(f"{key} = {raw}")
        self.flat = "\n".join(lines) + "\n"
        self.save_config()

    def write_lock(self, pins, image=None):
        _write(self.extras / "anorak_extras.lock",
               "# anorak gpu_extras lock — written by tools/bootstrap_gpu_extras.sh\n"
               f"# image: {image or self.image}\n# written: 2026-09-25T00:00:00Z by tester\n"
               + "".join(p + "\n" for p in pins))

    def add_slide(self, relative: str, content: dict | None):
        path = self.raw / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content) if content is not None else "not a slide")
        return path

    def write_list(self, rows, header=("slide_id", "samples", "is_tumour")):
        import csv
        self.slides_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(self.slides_csv, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    def calls(self, tool: str) -> list[list[str]]:
        log = self.state / "calls.log"
        if not log.exists():
            return []
        return [json.loads(line.split(" ", 1)[1]) for line in log.read_text().splitlines()
                if line.split(" ", 1)[0] == tool]

    # running ----------------------------------------------------------------
    def env(self, **extra) -> dict:
        env = {"PATH": os.pathsep.join([str(self.bin), str(self.slurmcli), "/usr/bin", "/bin", "/usr/sbin", "/sbin"]),
               "HOME": str(self.root), "USER": "tester", "LANG": "C", "TMPDIR": str(self.state),
               "ANORAK_REPO_DIR": str(self.anorak)}
        env.update(extra)
        return env

    def preflight(self, *args, default_inputs=True, **extra_env):
        argv = ["bash", str(self.pipeline / "tools" / "preflight.sh")]
        if default_inputs:
            argv += ["--slides-csv", str(self.slides_csv), "--raw-dir", str(self.raw),
                     "--outdir", str(self.root / "results")]
        argv += list(args)
        result = subprocess.run(argv, env=self.env(**extra_env), capture_output=True, text=True, timeout=300)
        import re
        result.out = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout + result.stderr)
        return result


import subprocess  # noqa: E402  (used by FakeCluster.preflight)
