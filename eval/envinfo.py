"""Environment block written into every results file."""
import datetime
import os
import platform
import subprocess

# distribution names (as pip knows them)
LIBS = [
    "torch", "sentence-transformers", "transformers", "chromadb", "rank-bm25", "langchain-text-splitters",
    "langchain-core", "numpy", "httpx",
]


def _lib_versions():
    from importlib import metadata

    out = {}
    for name in LIBS:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _cpu_model():
    # EVAL_HOST_CPU is passed in from the host (Docker's VM hides the real CPU model).
    host = os.environ.get("EVAL_HOST_CPU")
    model = None
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith(("model name", "hardware", "cpu model")):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"host_cpu": host, "container_cpuinfo_model": model, "machine": platform.machine()}


def _cgroup_cpus():
    """CPU limit applied to this container (cgroup v2 cpu.max), or None when unlimited/unknown."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            return round(int(quota) / float(period), 2)
    except (OSError, ValueError):
        pass
    return None


def _vcs():
    """git HEAD and a dirty flag when this is a git checkout, else the build-copy note."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(os.path.join(root, ".git")):
        return "none: run from a build copy (not a git checkout)"
    base = ["git", "-c", "safe.directory=*", "-C", root]
    try:
        head = subprocess.run(base + ["rev-parse", "HEAD"], capture_output=True, text=True, timeout=20, check=True).stdout.strip()
        dirty = subprocess.run(base + ["status", "--porcelain"], capture_output=True, text=True, timeout=20, check=True).stdout.strip()
        return "git %s (%s)" % (head, "dirty working tree" if dirty else "clean")
    except (OSError, subprocess.SubprocessError):
        return "git checkout, HEAD not recorded (git unavailable)"


def environment(models, extra=None):
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    try:
        import torch
        torch_threads = torch.get_num_threads()
    except Exception:
        torch_threads = None
    env = {
        "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "vcs": _vcs(),
        "python": platform.python_version(),
        "libraries": _lib_versions(),
        "models": models,
        "os": platform.platform(),
        "cpu": _cpu_model(),
        "docker": {
            "visible_cpus": os.cpu_count(),
            "affinity_cpus": affinity,
            "cgroup_cpu_limit": _cgroup_cpus(),
            "torch_threads": torch_threads,
            "host_note": os.environ.get("EVAL_HOST_NOTE"),
        },
        "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "EVAL_MODEL_ROOT", "HF_HUB_OFFLINE")},
    }
    if extra:
        env.update(extra)
    return env
