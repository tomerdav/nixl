#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sweep NIXL EP channels, proxy workers, and experts per rank."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
EP_DIR = REPO_ROOT / "examples" / "device" / "ep"
ELASTIC_SCRIPT = EP_DIR / "tests" / "elastic" / "elastic.py"
PLAN_PATH = EP_DIR / "tests" / "elastic" / "no_expansion.json"

CUDA_VISIBLE_DEVICES = "0,1,4,5"
GPU_NIC_MAP = "mlx5_0,mlx5_1,mlx5_2,mlx5_4,mlx5_5,mlx5_6"
DEFAULT_BUILD_DIR = REPO_ROOT / "build-proxy-release"
EXPECTED_RANKS = frozenset(range(4))
DEFAULT_TIMEOUT_SECONDS = 40.0
TERMINATION_GRACE_SECONDS = 1.0
SWEEP_ENVIRONMENT_VARIABLES = (
    "CUDA_VISIBLE_DEVICES",
    "NIXL_GPU_NIC_MAP",
    "NIXL_EP_PROXY_CHANNELS",
    "NIXL_EP_PROXY_WORKER_COUNT",
    "NIXL_EP_NUM_CHANNELS",
    "NIXL_PLUGIN_DIR",
    "NIXL_LOG_LEVEL",
    "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE",
    "UCX_NET_DEVICES",
    "UCX_TLS",
)
EXPECTED_NIXL_EP_BACKENDS = {
    "proxy": "proxy",
    "direct": "ucx",
}

DISPATCH_BW_RE = re.compile(
    r"^\[rank (?P<rank>\d+)\] Dispatch bandwidth: "
    r"(?P<bandwidth>\d+(?:\.\d+)?) GB/s, "
    r"avg_t=\d+(?:\.\d+)? us, min_t=\d+(?:\.\d+)? us, "
    r"max_t=\d+(?:\.\d+)? us$"
)
NIXLPUT_ISSUE_RE = re.compile(
    r"^\[rank (?P<rank>\d+)\] nixlPut issue: "
    r"n=(?P<n>\d+)(?: bp=(?P<bp>\d+))? "
    r"avg=(?P<avg>\d+(?:\.\d+)?) us "
    r"p50=(?P<p50>\d+(?:\.\d+)?) us "
    r"p99=(?P<p99>\d+(?:\.\d+)?) us$"
)


@dataclass
class NixlPutIssueStats:
    n: int
    bp: int | None
    avg_us: float
    p50_us: float
    p99_us: float

    @property
    def bp_rate(self) -> float | None:
        if self.bp is None or self.n <= 0:
            return None
        return self.bp / self.n


@dataclass
class RunResult:
    backend: str
    channels: int
    proxy_workers: int | None
    experts_per_rank: int
    repeat_index: int
    repeat_count: int
    status: str
    exit_code: int | None
    timed_out: bool
    elapsed_seconds: float
    rank_bandwidths: dict[int, float]
    bandwidth_mean: float | None
    bandwidth_min: float | None
    bandwidth_max: float | None
    rank_nixlput: dict[int, NixlPutIssueStats]
    nixlput_avg_us_mean: float | None
    nixlput_avg_us_min: float | None
    nixlput_avg_us_max: float | None
    nixlput_bp_rate_mean: float | None
    log_path: str
    error: str | None = None


@dataclass
class PointResult:
    backend: str
    channels: int
    proxy_workers: int | None
    experts_per_rank: int
    requested_repeats: int
    successful_repeats: int
    status: str
    repeat_bandwidths: list[float]
    bandwidth_mean: float | None
    bandwidth_variance: float | None
    bandwidth_stddev: float | None
    repeat_nixlput_avg_us: list[float]
    nixlput_avg_us_mean: float | None
    nixlput_avg_us_variance: float | None
    nixlput_avg_us_stddev: float | None
    repeat_nixlput_bp_rate: list[float]
    nixlput_bp_rate_mean: float | None
    nixlput_bp_rate_variance: float | None
    nixlput_bp_rate_stddev: float | None


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def create_unique_directory(parent: Path, name: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    candidate = parent / name
    suffix = 1
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            candidate = parent / f"{name}_{suffix}"
            suffix += 1


def validate_unique(values: Sequence[int], argument: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{argument} contains duplicate values")


def build_parameter_points(
    backend: str,
    channel_counts: Sequence[int],
    proxy_worker_counts: Sequence[int] | None,
    experts_per_rank: Sequence[int],
) -> list[tuple[int, int | None, int]]:
    if backend == "proxy":
        if proxy_worker_counts is None:
            return [
                (channels, channels, experts)
                for channels, experts in product(
                    channel_counts, experts_per_rank
                )
            ]
        return [
            (channels, proxy_workers, experts)
            for channels, proxy_workers, experts in product(
                channel_counts, proxy_worker_counts, experts_per_rank
            )
            if proxy_workers <= channels
        ]

    if backend == "direct":
        if proxy_worker_counts is not None:
            raise ValueError(
                "--proxy-worker-counts is only valid with --backend proxy"
            )
        return [
            (channels, None, experts)
            for channels, experts in product(
                channel_counts, experts_per_rank
            )
        ]

    raise ValueError(f"unsupported backend: {backend}")


def build_command(experts_per_rank: int, kineto: bool) -> list[str]:
    command = [
        sys.executable,
        str(ELASTIC_SCRIPT),
        "--plan",
        str(PLAN_PATH),
        "--num-processes",
        "4",
        "--num-tokens",
        "128",
        "--num-experts-per-rank",
        str(experts_per_rank),
        "--num-topk",
        "8",
        "--dispatch-only",
        "--disable-ll-nvlink",
    ]
    if kineto:
        command.append("--kineto")
    return command


def build_environment(
    backend: str,
    channels: int,
    proxy_workers: int | None,
    build_dir: Path,
) -> dict[str, str]:
    environment = os.environ.copy()
    for name in SWEEP_ENVIRONMENT_VARIABLES:
        environment.pop(name, None)

    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                (
                    str(REPO_ROOT / "src" / "bindings" / "python" / "nixl-meta"),
                    str(build_dir / "examples" / "device" / "ep"),
                )
            ),
            "NIXL_PLUGIN_DIR": str(build_dir / "src" / "plugins" / "ucx"),
            "NIXL_LOG_LEVEL": "WARN",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
            "UCX_NET_DEVICES": "all",
            "UCX_TLS": "^cuda_ipc",
        }
    )
    if backend == "proxy":
        if proxy_workers is None:
            raise ValueError("proxy backend requires a proxy worker count")
        environment.update(
            {
                "NIXL_GPU_NIC_MAP": GPU_NIC_MAP,
                "NIXL_EP_PROXY_CHANNELS": str(channels),
                "NIXL_EP_PROXY_WORKER_COUNT": str(proxy_workers),
            }
        )
    elif backend == "direct":
        if proxy_workers is not None:
            raise ValueError("direct backend does not use proxy workers")
        environment["NIXL_EP_NUM_CHANNELS"] = str(channels)
    else:
        raise ValueError(f"unsupported backend: {backend}")
    return environment


def verify_nixl_ep_backend(backend: str, build_dir: Path) -> str:
    expected_backend = EXPECTED_NIXL_EP_BACKENDS[backend]
    compile_commands_path = build_dir / "compile_commands.json"
    try:
        compile_commands = json.loads(
            compile_commands_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"could not read Meson compile metadata {compile_commands_path}: "
            f"{error}"
        ) from error

    backend_defines = {
        "proxy": "-DNIXL_GPU_DEVICE_BACKEND_PROXY",
        "ucx": "-DNIXL_GPU_DEVICE_BACKEND_UCX",
    }
    resolved_backends: set[str] = set()
    for entry in compile_commands:
        source_path = str(entry.get("file", "")).replace("\\", "/")
        if not source_path.endswith("examples/device/ep/csrc/nixl_ep.cpp"):
            continue
        command = entry.get("command")
        if command is None:
            command = " ".join(entry.get("arguments", ()))
        for candidate, define in backend_defines.items():
            if define in command:
                resolved_backends.add(candidate)

    if len(resolved_backends) != 1:
        detected = ", ".join(sorted(resolved_backends)) or "none"
        raise RuntimeError(
            "could not resolve exactly one NIXL EP backend from "
            f"{compile_commands_path}; detected: {detected}"
        )

    resolved_backend = resolved_backends.pop()
    if resolved_backend != expected_backend:
        raise RuntimeError(
            f"--backend {backend} requires nixl_ep backend '{expected_backend}', "
            f"but {build_dir} was compiled for '{resolved_backend}'"
        )
    return resolved_backend


def terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    termination_deadline = time.monotonic() + TERMINATION_GRACE_SECONDS
    try:
        process.wait(timeout=TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass

    remaining_grace = termination_deadline - time.monotonic()
    if remaining_grace > 0:
        time.sleep(remaining_grace)

    # Kill the process group even if its leader has exited. A spawned worker may
    # have ignored SIGTERM and would otherwise leak into the next experiment.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    if process.poll() is None:
        process.wait()


def run_process(
    command: Sequence[str],
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    timeout_seconds: float,
) -> tuple[int | None, bool, float]:
    start_time = time.monotonic()
    timed_out = False
    exit_code: int | None = None

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"# started_utc={datetime.now(timezone.utc).isoformat()}\n")
        log_file.write(f"# cwd={cwd}\n")
        for name in SWEEP_ENVIRONMENT_VARIABLES:
            log_file.write(f"# env {name}={environment.get(name, '<unset>')}\n")
        log_file.write(f"# command={shlex.join(command)}\n\n")
        log_file.flush()

        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as error:
            elapsed_seconds = time.monotonic() - start_time
            log_file.write(f"# launch_error={error}\n")
            log_file.write("# exit_code=None\n")
            log_file.write(f"# elapsed_seconds={elapsed_seconds:.3f}\n")
            return None, False, elapsed_seconds

        try:
            exit_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            log_file.write(
                f"\n# TIMEOUT after {timeout_seconds:g} seconds; "
                "terminating process group\n"
            )
            log_file.flush()
            terminate_process_group(process)
            exit_code = process.returncode

        elapsed_seconds = time.monotonic() - start_time
        log_file.write(f"\n# exit_code={exit_code}\n")
        log_file.write(f"# elapsed_seconds={elapsed_seconds:.3f}\n")

    return exit_code, timed_out, elapsed_seconds


def parse_dispatch_bandwidth(log_text: str) -> dict[int, float]:
    rank_bandwidths: dict[int, float] = {}
    duplicate_ranks: set[int] = set()

    for line in log_text.splitlines():
        match = DISPATCH_BW_RE.match(line.strip())
        if match is None:
            continue
        rank = int(match.group("rank"))
        if rank in rank_bandwidths:
            duplicate_ranks.add(rank)
        rank_bandwidths[rank] = float(match.group("bandwidth"))

    if duplicate_ranks:
        duplicates = ", ".join(str(rank) for rank in sorted(duplicate_ranks))
        raise ValueError(
            f"duplicate dispatch bandwidth results for ranks: {duplicates}"
        )
    return rank_bandwidths


def parse_nixlput_issue(log_text: str) -> dict[int, NixlPutIssueStats]:
    rank_stats: dict[int, NixlPutIssueStats] = {}
    duplicate_ranks: set[int] = set()

    for line in log_text.splitlines():
        match = NIXLPUT_ISSUE_RE.match(line.strip())
        if match is None:
            continue
        rank = int(match.group("rank"))
        if rank in rank_stats:
            duplicate_ranks.add(rank)
        bp_text = match.group("bp")
        rank_stats[rank] = NixlPutIssueStats(
            n=int(match.group("n")),
            bp=int(bp_text) if bp_text is not None else None,
            avg_us=float(match.group("avg")),
            p50_us=float(match.group("p50")),
            p99_us=float(match.group("p99")),
        )

    if duplicate_ranks:
        duplicates = ", ".join(str(rank) for rank in sorted(duplicate_ranks))
        raise ValueError(
            f"duplicate nixlPut issue results for ranks: {duplicates}"
        )
    return rank_stats


def classify_result(
    exit_code: int | None,
    timed_out: bool,
    rank_bandwidths: dict[int, float],
) -> tuple[str, str | None]:
    if timed_out:
        return "TIMEOUT", "experiment exceeded its wall-clock timeout"
    if exit_code is None:
        return "FAILED", "failed to launch elastic.py"
    if exit_code != 0:
        return "FAILED", f"elastic.py exited with code {exit_code}"

    observed_ranks = set(rank_bandwidths)
    if observed_ranks != EXPECTED_RANKS:
        missing = sorted(EXPECTED_RANKS - observed_ranks)
        unexpected = sorted(observed_ranks - EXPECTED_RANKS)
        details = []
        if missing:
            details.append(f"missing ranks {missing}")
        if unexpected:
            details.append(f"unexpected ranks {unexpected}")
        return "INCOMPLETE", ", ".join(details)
    return "SUCCESS", None


def run_experiment(
    sweep_dir: Path,
    run_index: int,
    backend: str,
    channels: int,
    proxy_workers: int | None,
    experts_per_rank: int,
    repeat_index: int,
    repeat_count: int,
    timeout_seconds: float,
    build_dir: Path,
    kineto: bool,
) -> RunResult:
    channel_label = (
        f"channels-{channels}_workers-{proxy_workers}"
        if backend == "proxy"
        else f"channels-{channels}"
    )
    run_name = (
        f"{run_index:03d}_{utc_timestamp()}_backend-{backend}_"
        f"{channel_label}_ep-{experts_per_rank}_"
        f"repeat-{repeat_index}-of-{repeat_count}"
    )
    run_dir = create_unique_directory(sweep_dir, run_name)
    log_path = run_dir / "run.log"
    command = build_command(experts_per_rank, kineto)
    environment = build_environment(backend, channels, proxy_workers, build_dir)

    exit_code, timed_out, elapsed_seconds = run_process(
        command, EP_DIR, environment, log_path, timeout_seconds
    )

    rank_bandwidths: dict[int, float] = {}
    rank_nixlput: dict[int, NixlPutIssueStats] = {}
    parse_error = None
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    try:
        rank_bandwidths = parse_dispatch_bandwidth(log_text)
    except ValueError as error:
        parse_error = str(error)

    try:
        rank_nixlput = parse_nixlput_issue(log_text)
    except ValueError as error:
        parse_error = "; ".join(
            part for part in (parse_error, str(error)) if part
        )

    status, error = classify_result(exit_code, timed_out, rank_bandwidths)
    if parse_error is not None:
        if status == "SUCCESS":
            status = "INCOMPLETE"
        error = "; ".join(part for part in (error, parse_error) if part)

    bandwidth_values = list(rank_bandwidths.values())
    nixlput_avg_values = [stats.avg_us for stats in rank_nixlput.values()]
    nixlput_bp_rates = [
        rate
        for stats in rank_nixlput.values()
        if (rate := stats.bp_rate) is not None
    ]
    result = RunResult(
        backend=backend,
        channels=channels,
        proxy_workers=proxy_workers,
        experts_per_rank=experts_per_rank,
        repeat_index=repeat_index,
        repeat_count=repeat_count,
        status=status,
        exit_code=exit_code,
        timed_out=timed_out,
        elapsed_seconds=elapsed_seconds,
        rank_bandwidths=rank_bandwidths,
        bandwidth_mean=(
            statistics.fmean(bandwidth_values) if bandwidth_values else None
        ),
        bandwidth_min=min(bandwidth_values) if bandwidth_values else None,
        bandwidth_max=max(bandwidth_values) if bandwidth_values else None,
        rank_nixlput=rank_nixlput,
        nixlput_avg_us_mean=(
            statistics.fmean(nixlput_avg_values) if nixlput_avg_values else None
        ),
        nixlput_avg_us_min=(
            min(nixlput_avg_values) if nixlput_avg_values else None
        ),
        nixlput_avg_us_max=(
            max(nixlput_avg_values) if nixlput_avg_values else None
        ),
        nixlput_bp_rate_mean=(
            statistics.fmean(nixlput_bp_rates) if nixlput_bp_rates else None
        ),
        log_path=str(log_path),
        error=error,
    )

    metadata = asdict(result)
    metadata["command"] = command
    metadata["cwd"] = str(EP_DIR)
    metadata["environment"] = {
        name: environment.get(name) for name in SWEEP_ENVIRONMENT_VARIABLES
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def aggregate_results(results: Sequence[RunResult]) -> list[PointResult]:
    grouped_results: dict[
        tuple[str, int, int | None, int], list[RunResult]
    ] = {}
    for result in results:
        key = (
            result.backend,
            result.channels,
            result.proxy_workers,
            result.experts_per_rank,
        )
        grouped_results.setdefault(key, []).append(result)

    points: list[PointResult] = []
    for group in grouped_results.values():
        first = group[0]
        successful = [
            result for result in group if result.status == "SUCCESS"
        ]
        repeat_bandwidths = [
            result.bandwidth_mean
            for result in successful
            if result.bandwidth_mean is not None
        ]
        repeat_nixlput_avg_us = [
            result.nixlput_avg_us_mean
            for result in successful
            if result.nixlput_avg_us_mean is not None
        ]
        repeat_nixlput_bp_rate = [
            result.nixlput_bp_rate_mean
            for result in successful
            if result.nixlput_bp_rate_mean is not None
        ]
        successful_repeats = len(repeat_bandwidths)
        point_complete = successful_repeats == first.repeat_count
        points.append(
            PointResult(
                backend=first.backend,
                channels=first.channels,
                proxy_workers=first.proxy_workers,
                experts_per_rank=first.experts_per_rank,
                requested_repeats=first.repeat_count,
                successful_repeats=successful_repeats,
                status="SUCCESS" if point_complete else "INCOMPLETE",
                repeat_bandwidths=repeat_bandwidths,
                bandwidth_mean=(
                    statistics.fmean(repeat_bandwidths)
                    if repeat_bandwidths
                    else None
                ),
                bandwidth_variance=(
                    statistics.pvariance(repeat_bandwidths)
                    if repeat_bandwidths
                    else None
                ),
                bandwidth_stddev=(
                    statistics.pstdev(repeat_bandwidths)
                    if repeat_bandwidths
                    else None
                ),
                repeat_nixlput_avg_us=repeat_nixlput_avg_us,
                nixlput_avg_us_mean=(
                    statistics.fmean(repeat_nixlput_avg_us)
                    if repeat_nixlput_avg_us
                    else None
                ),
                nixlput_avg_us_variance=(
                    statistics.pvariance(repeat_nixlput_avg_us)
                    if repeat_nixlput_avg_us
                    else None
                ),
                nixlput_avg_us_stddev=(
                    statistics.pstdev(repeat_nixlput_avg_us)
                    if repeat_nixlput_avg_us
                    else None
                ),
                repeat_nixlput_bp_rate=repeat_nixlput_bp_rate,
                nixlput_bp_rate_mean=(
                    statistics.fmean(repeat_nixlput_bp_rate)
                    if repeat_nixlput_bp_rate
                    else None
                ),
                nixlput_bp_rate_variance=(
                    statistics.pvariance(repeat_nixlput_bp_rate)
                    if repeat_nixlput_bp_rate
                    else None
                ),
                nixlput_bp_rate_stddev=(
                    statistics.pstdev(repeat_nixlput_bp_rate)
                    if repeat_nixlput_bp_rate
                    else None
                ),
            )
        )
    return points


def write_summary(results: Sequence[RunResult], summary_path: Path) -> None:
    fieldnames = [
        "backend",
        "channels",
        "proxy_workers",
        "experts_per_rank",
        "repeat_index",
        "repeat_count",
        "status",
        "exit_code",
        "timed_out",
        "elapsed_seconds",
        "dispatch_bw_mean_gbps",
        "dispatch_bw_min_gbps",
        "dispatch_bw_max_gbps",
        "rank_0_dispatch_bw_gbps",
        "rank_1_dispatch_bw_gbps",
        "rank_2_dispatch_bw_gbps",
        "rank_3_dispatch_bw_gbps",
        "nixlput_avg_us_mean",
        "nixlput_avg_us_min",
        "nixlput_avg_us_max",
        "nixlput_bp_rate_mean",
        "rank_0_nixlput_avg_us",
        "rank_1_nixlput_avg_us",
        "rank_2_nixlput_avg_us",
        "rank_3_nixlput_avg_us",
        "rank_0_nixlput_bp_rate",
        "rank_1_nixlput_bp_rate",
        "rank_2_nixlput_bp_rate",
        "rank_3_nixlput_bp_rate",
        "log_path",
        "error",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "backend": result.backend,
                    "channels": result.channels,
                    "proxy_workers": result.proxy_workers,
                    "experts_per_rank": result.experts_per_rank,
                    "repeat_index": result.repeat_index,
                    "repeat_count": result.repeat_count,
                    "status": result.status,
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                    "elapsed_seconds": f"{result.elapsed_seconds:.3f}",
                    "dispatch_bw_mean_gbps": _format_optional_float(
                        result.bandwidth_mean
                    ),
                    "dispatch_bw_min_gbps": _format_optional_float(
                        result.bandwidth_min
                    ),
                    "dispatch_bw_max_gbps": _format_optional_float(
                        result.bandwidth_max
                    ),
                    **{
                        f"rank_{rank}_dispatch_bw_gbps": _format_optional_float(
                            result.rank_bandwidths.get(rank)
                        )
                        for rank in EXPECTED_RANKS
                    },
                    "nixlput_avg_us_mean": _format_optional_float(
                        result.nixlput_avg_us_mean
                    ),
                    "nixlput_avg_us_min": _format_optional_float(
                        result.nixlput_avg_us_min
                    ),
                    "nixlput_avg_us_max": _format_optional_float(
                        result.nixlput_avg_us_max
                    ),
                    "nixlput_bp_rate_mean": _format_optional_float(
                        result.nixlput_bp_rate_mean
                    ),
                    **{
                        f"rank_{rank}_nixlput_avg_us": _format_optional_float(
                            (
                                stats.avg_us
                                if (stats := result.rank_nixlput.get(rank))
                                is not None
                                else None
                            )
                        )
                        for rank in EXPECTED_RANKS
                    },
                    **{
                        f"rank_{rank}_nixlput_bp_rate": _format_optional_float(
                            (
                                stats.bp_rate
                                if (stats := result.rank_nixlput.get(rank))
                                is not None
                                else None
                            )
                        )
                        for rank in EXPECTED_RANKS
                    },
                    "log_path": result.log_path,
                    "error": result.error or "",
                }
            )


def write_aggregate_summary(
    points: Sequence[PointResult], summary_path: Path
) -> None:
    fieldnames = [
        "backend",
        "channels",
        "proxy_workers",
        "experts_per_rank",
        "status",
        "requested_repeats",
        "successful_repeats",
        "dispatch_bw_mean_gbps",
        "dispatch_bw_variance_gbps_squared",
        "dispatch_bw_stddev_gbps",
        "repeat_dispatch_bw_gbps",
        "nixlput_avg_us_mean",
        "nixlput_avg_us_variance",
        "nixlput_avg_us_stddev",
        "repeat_nixlput_avg_us",
        "nixlput_bp_rate_mean",
        "nixlput_bp_rate_variance",
        "nixlput_bp_rate_stddev",
        "repeat_nixlput_bp_rate",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "backend": point.backend,
                    "channels": point.channels,
                    "proxy_workers": point.proxy_workers,
                    "experts_per_rank": point.experts_per_rank,
                    "status": point.status,
                    "requested_repeats": point.requested_repeats,
                    "successful_repeats": point.successful_repeats,
                    "dispatch_bw_mean_gbps": _format_optional_float(
                        point.bandwidth_mean
                    ),
                    "dispatch_bw_variance_gbps_squared": _format_optional_float(
                        point.bandwidth_variance
                    ),
                    "dispatch_bw_stddev_gbps": _format_optional_float(
                        point.bandwidth_stddev
                    ),
                    "repeat_dispatch_bw_gbps": ";".join(
                        f"{bandwidth:.3f}"
                        for bandwidth in point.repeat_bandwidths
                    ),
                    "nixlput_avg_us_mean": _format_optional_float(
                        point.nixlput_avg_us_mean
                    ),
                    "nixlput_avg_us_variance": _format_optional_float(
                        point.nixlput_avg_us_variance
                    ),
                    "nixlput_avg_us_stddev": _format_optional_float(
                        point.nixlput_avg_us_stddev
                    ),
                    "repeat_nixlput_avg_us": ";".join(
                        f"{value:.3f}" for value in point.repeat_nixlput_avg_us
                    ),
                    "nixlput_bp_rate_mean": _format_optional_float(
                        point.nixlput_bp_rate_mean
                    ),
                    "nixlput_bp_rate_variance": _format_optional_float(
                        point.nixlput_bp_rate_variance
                    ),
                    "nixlput_bp_rate_stddev": _format_optional_float(
                        point.nixlput_bp_rate_stddev
                    ),
                    "repeat_nixlput_bp_rate": ";".join(
                        f"{value:.3f}" for value in point.repeat_nixlput_bp_rate
                    ),
                }
            )


def _format_optional_float(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def write_plots(
    points: Sequence[PointResult],
    backend: str,
    channel_counts: Sequence[int],
    experts_per_rank: Sequence[int],
    proxy_worker_counts: Sequence[int] | None,
    plot_path: Path,
) -> list[Path]:
    try:
        import matplotlib
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is required to generate dispatch bandwidth plots; "
            "install it with 'python3 -m pip install matplotlib'"
        ) from error

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    decoupled_proxy_workers = (
        backend == "proxy" and proxy_worker_counts is not None
    )
    successful_points = {
        (
            point.experts_per_rank,
            point.proxy_workers if decoupled_proxy_workers else None,
            point.channels,
        ): point
        for point in points
        if point.status == "SUCCESS"
    }
    ordered_channels = sorted(channel_counts)
    channel_positions = {
        channels: position for position, channels in enumerate(ordered_channels)
    }

    if decoupled_proxy_workers:
        assert proxy_worker_counts is not None
        plot_specs: Sequence[tuple[int | None, Path]] = [
            (
                proxy_workers,
                plot_path.with_name(
                    f"{plot_path.stem}_workers_{proxy_workers}"
                    f"{plot_path.suffix}"
                ),
            )
            for proxy_workers in sorted(proxy_worker_counts)
        ]
    else:
        plot_specs = [(None, plot_path)]

    written_paths: list[Path] = []
    for proxy_workers, current_plot_path in plot_specs:
        figure, axis = plt.subplots(figsize=(9, 6))
        plotted_series = False
        for experts in experts_per_rank:
            series = [
                successful_points[(experts, proxy_workers, channels)]
                for channels in ordered_channels
                if (experts, proxy_workers, channels) in successful_points
            ]
            if not series:
                continue

            x_values = [channel_positions[point.channels] for point in series]
            y_values = [point.bandwidth_mean for point in series]
            standard_deviations = [
                point.bandwidth_stddev for point in series
            ]
            axis.errorbar(
                x_values,
                y_values,
                yerr=standard_deviations,
                marker="o",
                capsize=4,
                linewidth=2,
                label=f"EP{experts}",
            )
            plotted_series = True

        if plotted_series:
            axis.legend(title="Experts per Rank")
        else:
            axis.text(
                0.5,
                0.5,
                "No successful experiments",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axis.transAxes,
            )

        worker_title = (
            f" — {proxy_workers} Proxy Workers"
            if decoupled_proxy_workers
            else ""
        )
        axis.set_title(
            f"DFW {backend.capitalize()} Dispatch Bandwidth{worker_title}\n"
            "Mean across repeats; error bars show ±1 standard deviation"
        )
        axis.set_xlabel("Channel Count")
        axis.set_ylabel("Dispatch BW (GB/s)")
        axis.set_xticks(range(len(ordered_channels)), labels=ordered_channels)
        axis.set_ylim(0, 50)
        axis.set_yticks(range(0, 51, 10))
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(current_plot_path, dpi=160)
        plt.close(figure)
        written_paths.append(current_plot_path)

    return written_paths


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep NIXL EP channel counts, proxy worker counts, and "
            "experts per rank."
        )
    )
    parser.add_argument(
        "--backend",
        choices=("proxy", "direct"),
        required=True,
        help="NIXL EP backend environment to configure",
    )
    parser.add_argument(
        "--channel-counts",
        nargs="+",
        type=positive_int,
        help="positive integer channel counts for direct or decoupled mode",
    )
    proxy_worker_mode = parser.add_mutually_exclusive_group()
    proxy_worker_mode.add_argument(
        "--coupled-proxy-channel-worker",
        nargs="+",
        type=positive_int,
        help="positive integer counts used for both proxy channels and workers",
    )
    proxy_worker_mode.add_argument(
        "--proxy-worker-counts",
        nargs="+",
        type=positive_int,
        help="positive integer counts for an independent proxy worker sweep",
    )
    parser.add_argument(
        "--experts-per-rank",
        nargs="+",
        type=positive_int,
        required=True,
        help="positive integer experts-per-rank values",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="wall-clock timeout in seconds for each experiment (default: 40)",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help=(
            "configured NIXL build directory containing the EP Python module "
            "and UCX plugin (default: build-proxy-release)"
        ),
    )
    parser.add_argument(
        "--kineto",
        action="store_true",
        help="enable Kineto profiling in elastic.py",
    )
    parser.add_argument(
        "--repeats",
        type=positive_int,
        default=1,
        help="number of repeated runs for each parameter combination (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "automations" / "results",
        help="parent directory for timestamped sweep results",
    )
    parsed = parser.parse_args(arguments)

    try:
        if parsed.channel_counts is not None:
            validate_unique(parsed.channel_counts, "--channel-counts")
        if parsed.coupled_proxy_channel_worker is not None:
            validate_unique(
                parsed.coupled_proxy_channel_worker,
                "--coupled-proxy-channel-worker",
            )
        if parsed.proxy_worker_counts is not None:
            validate_unique(
                parsed.proxy_worker_counts, "--proxy-worker-counts"
            )
        validate_unique(parsed.experts_per_rank, "--experts-per-rank")
    except ValueError as error:
        parser.error(str(error))
    if parsed.backend == "proxy":
        if parsed.coupled_proxy_channel_worker is not None:
            if parsed.channel_counts is not None:
                parser.error(
                    "--channel-counts cannot be used with "
                    "--coupled-proxy-channel-worker"
                )
            parsed.channel_counts = parsed.coupled_proxy_channel_worker
        elif parsed.proxy_worker_counts is None:
            parser.error(
                "--backend proxy requires either "
                "--coupled-proxy-channel-worker "
                "or --proxy-worker-counts"
            )
        elif parsed.channel_counts is None:
            parser.error(
                "--proxy-worker-counts requires --channel-counts"
            )
    else:
        if (
            parsed.coupled_proxy_channel_worker is not None
            or parsed.proxy_worker_counts is not None
        ):
            parser.error(
                "proxy worker mode options are only valid with "
                "--backend proxy"
            )
        if parsed.channel_counts is None:
            parser.error("--backend direct requires --channel-counts")
    parsed.build_dir = parsed.build_dir.resolve()
    return parsed


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    parameter_points = build_parameter_points(
        args.backend,
        args.channel_counts,
        args.proxy_worker_counts,
        args.experts_per_rank,
    )
    required_files = (ELASTIC_SCRIPT, PLAN_PATH)
    required_directories = (
        args.build_dir / "examples" / "device" / "ep",
        args.build_dir / "src" / "plugins" / "ucx",
    )
    for required_path in required_files:
        if not required_path.is_file():
            print(
                f"error: required file does not exist: {required_path}",
                file=sys.stderr,
            )
            return 2
    for required_path in required_directories:
        if not required_path.is_dir():
            print(
                f"error: required build directory does not exist: {required_path}",
                file=sys.stderr,
            )
            return 2

    try:
        loaded_backend = verify_nixl_ep_backend(
            args.backend,
            args.build_dir,
        )
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    selected_mode = args.backend
    if args.backend == "proxy":
        selected_mode += (
            ", workers coupled to channels"
            if args.coupled_proxy_channel_worker is not None
            else ", independent worker sweep"
        )
    if args.kineto:
        selected_mode += ", Kineto enabled"
    print(
        f"Verified nixl_ep backend: {loaded_backend} "
        f"(selected mode: {selected_mode})",
        flush=True,
    )

    sweep_dir = create_unique_directory(
        args.output_dir, f"{utc_timestamp()}_dfw_{args.backend}_sweep"
    )
    runs = [
        (channels, proxy_workers, experts, repeat_index)
        for channels, proxy_workers, experts in parameter_points
        for repeat_index in range(1, args.repeats + 1)
    ]
    print(f"Sweep output: {sweep_dir}", flush=True)
    print(
        f"Running {len(parameter_points)} {args.backend} combinations, "
        f"{args.repeats} repeats each ({len(runs)} total runs), with "
        f"{args.timeout:g}s timeout per run",
        flush=True,
    )

    results: list[RunResult] = []
    for run_index, (
        channels,
        proxy_workers,
        experts,
        repeat_index,
    ) in enumerate(runs, start=1):
        if args.backend == "proxy":
            channel_description = (
                f"channels=workers={channels}"
                if args.coupled_proxy_channel_worker is not None
                else f"channels={channels}, workers={proxy_workers}"
            )
        else:
            channel_description = f"channels={channels}"
        print(
            f"[{run_index}/{len(runs)}] {channel_description}, "
            f"experts_per_rank={experts}, "
            f"repeat={repeat_index}/{args.repeats}",
            flush=True,
        )
        result = run_experiment(
            sweep_dir,
            run_index,
            args.backend,
            channels,
            proxy_workers,
            experts,
            repeat_index,
            args.repeats,
            args.timeout,
            args.build_dir,
            args.kineto,
        )
        results.append(result)
        result_detail = (
            f", mean={result.bandwidth_mean:.2f} GB/s"
            if result.bandwidth_mean is not None
            else ""
        )
        print(
            f"  {result.status} in {result.elapsed_seconds:.2f}s{result_detail}",
            flush=True,
        )

    points = aggregate_results(results)
    summary_path = sweep_dir / "summary.csv"
    aggregate_summary_path = sweep_dir / "aggregate_summary.csv"
    plot_path = sweep_dir / "dispatch_bw.png"
    write_summary(results, summary_path)
    write_aggregate_summary(points, aggregate_summary_path)
    try:
        plot_paths = write_plots(
            points,
            args.backend,
            args.channel_counts,
            args.experts_per_rank,
            args.proxy_worker_counts,
            plot_path,
        )
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        print(f"Per-run summary: {summary_path}", flush=True)
        print(f"Aggregate summary: {aggregate_summary_path}", flush=True)
        return 2

    print(f"Per-run summary: {summary_path}", flush=True)
    print(f"Aggregate summary: {aggregate_summary_path}", flush=True)
    for written_plot_path in plot_paths:
        print(f"Plot: {written_plot_path}", flush=True)
    failed_results = [result for result in results if result.status != "SUCCESS"]
    if failed_results:
        print(
            f"{len(failed_results)} of {len(results)} experiments did not succeed",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
