#!/usr/bin/env python3
"""Concurrent, trigger-driven launcher for the HQSFlow ablation programme.

The launcher intentionally does not embed experiment semantics. A canonical matrix
YAML defines the ablations; a small trigger YAML selects which jobs to execute and
sets the GPU/concurrency policy. Each job is launched as an isolated subprocess with
its own stdout/stderr log and machine-readable status manifest.

Trigger lifecycle in --watch mode:
    *.trigger.yaml -> *.running.yaml -> *.done.yaml | *.failed.yaml

A safe default of one process per GPU is enforced unless ``allow_gpu_oversubscription``
is explicitly true in the trigger.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from omegaconf import OmegaConf


def load_yaml(path: Path) -> Dict[str, Any]:
    cfg = OmegaConf.load(path)
    value = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping in {path}")
    return value


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def normalise_tags(value: Any) -> Set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    return {str(v) for v in value}


def selected_job_ids(matrix: Mapping[str, Any], trigger: Mapping[str, Any]) -> List[str]:
    jobs = matrix.get("jobs", {})
    if not isinstance(jobs, Mapping):
        raise TypeError("matrix.jobs must be a mapping keyed by ablation ID")

    selection = trigger.get("select", {}) or {}
    explicit = {str(x) for x in selection.get("ids", [])}
    groups = {str(x) for x in selection.get("groups", [])}
    tags = {str(x) for x in selection.get("tags", [])}
    exclude = {str(x) for x in selection.get("exclude_ids", [])}

    selected: List[str] = []
    for job_id, raw in jobs.items():
        job = dict(raw or {})
        if not bool(job.get("enabled", True)):
            continue
        if str(job.get("implementation_status", "ready")).lower() != "ready":
            continue

        include = not (explicit or groups or tags)
        if explicit and str(job_id) in explicit:
            include = True
        if groups and str(job.get("group", "")) in groups:
            include = True
        if tags and (normalise_tags(job.get("tags")) & tags):
            include = True
        if str(job_id) in exclude:
            include = False
        if include:
            selected.append(str(job_id))

    unknown = explicit - set(jobs.keys())
    if unknown:
        raise KeyError(f"Trigger references unknown ablation IDs: {sorted(unknown)}")
    return selected


def topological_order(job_ids: Sequence[str], jobs: Mapping[str, Any]) -> List[str]:
    selected = set(job_ids)
    temporary: Set[str] = set()
    permanent: Set[str] = set()
    order: List[str] = []

    def visit(job_id: str) -> None:
        if job_id in permanent:
            return
        if job_id in temporary:
            raise ValueError(f"Cycle in ablation dependencies at {job_id}")
        temporary.add(job_id)
        job = dict(jobs[job_id] or {})
        for dependency in job.get("depends_on", []) or []:
            dependency = str(dependency)
            if dependency in selected:
                visit(dependency)
        temporary.remove(job_id)
        permanent.add(job_id)
        order.append(job_id)

    for job_id in job_ids:
        visit(job_id)
    return order


def build_train_command(
    *,
    repo_root: Path,
    defaults: Mapping[str, Any],
    job_id: str,
    job: Mapping[str, Any],
) -> List[str]:
    """Build one train_curriculum.py command from matrix fields."""
    python_exe = str(job.get("python", defaults.get("python", sys.executable)))
    entrypoint = repo_root / str(job.get("entrypoint", defaults.get("entrypoint", "train_curriculum.py")))
    base_config = repo_root / str(job.get("config", defaults.get("config", "configs/default.yaml")))
    override = job.get("override", defaults.get("override", None))
    curriculum = job.get("curriculum", defaults.get("curriculum", None))
    if curriculum is None:
        raise ValueError(f"{job_id}: curriculum is required")

    command = [
        python_exe,
        str(entrypoint),
        "--config",
        str(base_config),
    ]
    if override:
        command += ["--override", str(repo_root / str(override))]
    command += ["--curriculum", str(repo_root / str(curriculum))]

    base_run_name = job.get("base_run_name", None)
    if base_run_name:
        command += ["--base-run-name", str(base_run_name)]
    if job.get("start_stage", None):
        command += ["--start-stage", str(job["start_stage"])]
    if job.get("stop_after_stage", None):
        command += ["--stop-after-stage", str(job["stop_after_stage"])]
    if bool(job.get("run_optional", False)):
        command += ["--run-optional"]

    cli_overrides = job.get("cli_overrides", []) or []
    command.extend(str(x) for x in cli_overrides)
    return command


def resolve_command(
    *,
    repo_root: Path,
    defaults: Mapping[str, Any],
    job_id: str,
    job: Mapping[str, Any],
) -> List[str]:
    if job.get("command", None):
        command = job["command"]
        if isinstance(command, str):
            return shlex.split(command)
        return [str(x) for x in command]
    return build_train_command(
        repo_root=repo_root,
        defaults=defaults,
        job_id=job_id,
        job=job,
    )


@dataclass
class RunningJob:
    job_id: str
    gpu: str
    process: subprocess.Popen
    log_handle: Any
    status_path: Path
    started: float
    command: List[str]


def _gpu_slots(resources: Mapping[str, Any]) -> List[str]:
    gpus = resources.get("gpus", None)
    if gpus is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible.strip():
            gpus = [x.strip() for x in visible.split(",") if x.strip()]
        else:
            gpus = ["0"]
    slots = [str(x) for x in gpus]
    if not slots:
        raise ValueError("Trigger resource policy contains no GPUs")

    per_gpu = int(resources.get("max_parallel_per_gpu", 1))
    if per_gpu <= 0:
        raise ValueError("max_parallel_per_gpu must be positive")
    if per_gpu > 1 and not bool(resources.get("allow_gpu_oversubscription", False)):
        raise ValueError(
            "max_parallel_per_gpu > 1 requires allow_gpu_oversubscription: true. "
            "Concurrent full-resolution optical-flow training on one GPU can OOM."
        )
    return [gpu for gpu in slots for _ in range(per_gpu)]


def _status_payload(
    *,
    job_id: str,
    group: str,
    state: str,
    gpu: Optional[str],
    command: Sequence[str],
    started_at: Optional[str] = None,
    ended_at: Optional[str] = None,
    returncode: Optional[int] = None,
    pid: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "group": group,
        "state": state,
        "gpu": gpu,
        "command": list(command),
        "started_at": started_at,
        "ended_at": ended_at,
        "returncode": returncode,
        "pid": pid,
    }


def run_trigger(matrix_path: Path, trigger_path: Path, *, dry_run: bool = False) -> int:
    matrix = load_yaml(matrix_path)
    trigger = load_yaml(trigger_path)
    jobs: Mapping[str, Any] = matrix.get("jobs", {})
    defaults: Mapping[str, Any] = matrix.get("defaults", {}) or {}

    repo_root = Path(str(trigger.get("repo_root", matrix.get("repo_root", ".")))).expanduser().resolve()
    output_root = Path(
        str(trigger.get("output_root", matrix.get("output_root", "ablation_runs")))
    ).expanduser()
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    trigger_id = str(trigger.get("trigger_id", trigger_path.stem.replace(".trigger", "")))
    run_root = output_root / trigger_id
    run_root.mkdir(parents=True, exist_ok=True)

    selected = selected_job_ids(matrix, trigger)
    ordered = topological_order(selected, jobs)
    if not ordered:
        print("No ready/enabled ablation jobs selected.")
        return 0

    resources = trigger.get("resources", {}) or {}
    slots = _gpu_slots(resources)
    max_parallel = int(resources.get("max_parallel", len(slots)))
    max_parallel = max(1, min(max_parallel, len(slots)))
    slots = slots[:max_parallel]

    resume_existing = bool(trigger.get("resume_existing", True))
    fail_fast = bool(trigger.get("fail_fast", False))
    poll_seconds = float(trigger.get("poll_seconds", 2.0))

    print(f"Trigger {trigger_id}: {len(ordered)} jobs; GPUs={slots}; max_parallel={max_parallel}")

    pending = list(ordered)
    running: Dict[str, RunningJob] = {}
    completed: Set[str] = set()
    failed: Set[str] = set()
    free_slots: List[str] = list(slots)

    # Rehydrate completed states when requested.
    if resume_existing:
        for job_id in list(pending):
            status_path = run_root / job_id / "status.json"
            if status_path.is_file():
                try:
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                except Exception:
                    status = {}
                if status.get("state") == "completed" and int(status.get("returncode", 1)) == 0:
                    completed.add(job_id)
                    pending.remove(job_id)
                    print(f"[{job_id}] already completed; skipping")

    def dependencies_satisfied(job_id: str) -> bool:
        dependencies = [str(x) for x in (jobs[job_id].get("depends_on", []) or [])]
        for dependency in dependencies:
            if dependency in ordered and dependency not in completed:
                return False
        return True

    stop_requested = False

    def handle_signal(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print("Shutdown requested; terminating active ablation jobs...", file=sys.stderr)

    old_sigint = signal.signal(signal.SIGINT, handle_signal)
    old_sigterm = signal.signal(signal.SIGTERM, handle_signal)
    try:
        while pending or running:
            # Retire finished processes.
            for job_id, active in list(running.items()):
                returncode = active.process.poll()
                if returncode is None:
                    continue
                active.log_handle.flush()
                active.log_handle.close()
                job = dict(jobs[job_id] or {})
                state = "completed" if returncode == 0 else "failed"
                atomic_write_json(
                    active.status_path,
                    _status_payload(
                        job_id=job_id,
                        group=str(job.get("group", "")),
                        state=state,
                        gpu=active.gpu,
                        command=active.command,
                        started_at=dt.datetime.fromtimestamp(active.started).astimezone().isoformat(timespec="seconds"),
                        ended_at=now_iso(),
                        returncode=returncode,
                        pid=active.process.pid,
                    ),
                )
                free_slots.append(active.gpu)
                del running[job_id]
                if returncode == 0:
                    completed.add(job_id)
                else:
                    failed.add(job_id)
                print(f"[{job_id}] {state} (rc={returncode})")
                if returncode != 0 and fail_fast:
                    stop_requested = True

            if stop_requested:
                for active in running.values():
                    active.process.terminate()
                deadline = time.time() + 20.0
                while running and time.time() < deadline:
                    for job_id, active in list(running.items()):
                        if active.process.poll() is not None:
                            active.log_handle.close()
                            del running[job_id]
                    time.sleep(0.2)
                for active in running.values():
                    active.process.kill()
                    active.log_handle.close()
                return 130 if not failed else 1

            # Jobs depending on a failed selected job cannot run.
            for job_id in list(pending):
                dependencies = [str(x) for x in (jobs[job_id].get("depends_on", []) or [])]
                if any(dep in failed for dep in dependencies):
                    pending.remove(job_id)
                    failed.add(job_id)
                    job_dir = run_root / job_id
                    atomic_write_json(
                        job_dir / "status.json",
                        _status_payload(
                            job_id=job_id,
                            group=str(jobs[job_id].get("group", "")),
                            state="blocked_dependency_failed",
                            gpu=None,
                            command=[],
                            ended_at=now_iso(),
                            returncode=None,
                        ),
                    )
                    print(f"[{job_id}] blocked: dependency failed")

            # Fill free GPU slots with dependency-ready jobs.
            launched_any = False
            while free_slots and len(running) < max_parallel:
                ready_id = next((jid for jid in pending if dependencies_satisfied(jid)), None)
                if ready_id is None:
                    break
                pending.remove(ready_id)
                job = dict(jobs[ready_id] or {})
                command = resolve_command(
                    repo_root=repo_root,
                    defaults=defaults,
                    job_id=ready_id,
                    job=job,
                )
                gpu = free_slots.pop(0)
                job_dir = run_root / ready_id
                job_dir.mkdir(parents=True, exist_ok=True)
                log_path = job_dir / "process.log"
                status_path = job_dir / "status.json"

                print(f"[{ready_id}] GPU {gpu}: {' '.join(shlex.quote(x) for x in command)}")
                if dry_run:
                    atomic_write_json(
                        status_path,
                        _status_payload(
                            job_id=ready_id,
                            group=str(job.get("group", "")),
                            state="dry_run",
                            gpu=gpu,
                            command=command,
                            started_at=now_iso(),
                            ended_at=now_iso(),
                            returncode=0,
                        ),
                    )
                    completed.add(ready_id)
                    free_slots.append(gpu)
                    launched_any = True
                    continue

                env = os.environ.copy()
                env.update({str(k): str(v) for k, v in (job.get("env", {}) or {}).items()})
                env["CUDA_VISIBLE_DEVICES"] = gpu
                env["HQS_ABLATION_ID"] = ready_id
                env["HQS_ABLATION_GROUP"] = str(job.get("group", ""))

                log_handle = log_path.open("a", encoding="utf-8", buffering=1)
                started = time.time()
                process = subprocess.Popen(
                    command,
                    cwd=str(repo_root),
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                atomic_write_json(
                    status_path,
                    _status_payload(
                        job_id=ready_id,
                        group=str(job.get("group", "")),
                        state="running",
                        gpu=gpu,
                        command=command,
                        started_at=now_iso(),
                        pid=process.pid,
                    ),
                )
                running[ready_id] = RunningJob(
                    job_id=ready_id,
                    gpu=gpu,
                    process=process,
                    log_handle=log_handle,
                    status_path=status_path,
                    started=started,
                    command=command,
                )
                launched_any = True

            if not pending and not running:
                break
            if not launched_any:
                time.sleep(poll_seconds)

    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

    summary = {
        "trigger_id": trigger_id,
        "matrix": str(matrix_path.resolve()),
        "trigger": str(trigger_path.resolve()),
        "completed": sorted(completed),
        "failed": sorted(failed),
        "finished_at": now_iso(),
    }
    atomic_write_json(run_root / "summary.json", summary)
    return 0 if not failed else 1


def process_trigger_file(matrix_path: Path, trigger_path: Path, dry_run: bool) -> int:
    if trigger_path.name.endswith(".trigger.yaml"):
        running_path = trigger_path.with_name(trigger_path.name.replace(".trigger.yaml", ".running.yaml"))
    else:
        running_path = trigger_path.with_suffix(".running.yaml")
    trigger_path.replace(running_path)
    try:
        rc = run_trigger(matrix_path, running_path, dry_run=dry_run)
    except Exception:
        failed_path = running_path.with_name(running_path.name.replace(".running.yaml", ".failed.yaml"))
        running_path.replace(failed_path)
        raise
    final_suffix = ".done.yaml" if rc == 0 else ".failed.yaml"
    final_path = running_path.with_name(running_path.name.replace(".running.yaml", final_suffix))
    running_path.replace(final_path)
    return rc


def watch_trigger_directory(
    matrix_path: Path,
    trigger_dir: Path,
    *,
    dry_run: bool,
    poll_seconds: float,
) -> int:
    trigger_dir.mkdir(parents=True, exist_ok=True)
    print(f"Watching {trigger_dir} for *.trigger.yaml")
    while True:
        candidates = sorted(trigger_dir.glob("*.trigger.yaml"))
        if not candidates:
            time.sleep(poll_seconds)
            continue
        for trigger_path in candidates:
            print(f"Consuming trigger: {trigger_path.name}")
            try:
                process_trigger_file(matrix_path, trigger_path, dry_run)
            except Exception as exc:
                print(f"Trigger failed: {trigger_path}: {exc}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trigger", type=Path)
    group.add_argument("--watch", type=Path, metavar="TRIGGER_DIR")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--watch-poll-seconds", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.trigger is not None:
        raise SystemExit(run_trigger(args.matrix, args.trigger, dry_run=args.dry_run))
    watch_trigger_directory(
        args.matrix,
        args.watch,
        dry_run=args.dry_run,
        poll_seconds=args.watch_poll_seconds,
    )


if __name__ == "__main__":
    main()
