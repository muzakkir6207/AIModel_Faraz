#!/usr/bin/env python3
"""Direct GPU workload generator for the ResNet50 ONNX model.

This pod does not call the current ort-server HTTP service. It loads the same
model directly with onnxruntime-gpu and continuously runs inference on the
shared GPU slot exposed by the NVIDIA device plugin time-slicing config.
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import onnxruntime as ort


@dataclass
class Stats:
    lock: threading.Lock = field(default_factory=threading.Lock)
    requests: int = 0
    images: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    latencies_window: list[float] = field(default_factory=list)

    def record_success(self, batch_size: int, latency_ms: float) -> None:
        with self.lock:
            self.requests += 1
            self.images += batch_size
            self.total_latency_ms += latency_ms
            self.max_latency_ms = max(self.max_latency_ms, latency_ms)
            self.latencies_window.append(latency_ms)

    def record_error(self) -> None:
        with self.lock:
            self.errors += 1

    def snapshot_and_reset_window(self) -> tuple[int, int, int, float, float, list[float]]:
        with self.lock:
            values = (self.requests, self.images, self.errors, self.total_latency_ms, self.max_latency_ms, self.latencies_window[:])
            self.latencies_window.clear()
            return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/models/resnet50/1/model.onnx")
    parser.add_argument("--input-name", default="data")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--stats-interval", type=float, default=5.0)
    parser.add_argument("--duration-seconds", type=float, default=0.0, help="0 means run forever")
    parser.add_argument("--sleep-ms", type=float, default=0.0, help="Optional delay after each inference per worker")
    parser.add_argument("--gpu-mem-limit-mb", type=int, default=2048)
    parser.add_argument("--random-input", action="store_true", help="Regenerate random input each iteration")
    return parser.parse_args()


def create_session(model_path: str, gpu_mem_limit_mb: int) -> ort.InferenceSession:
    provider_options = {
        "device_id": 0,
        "gpu_mem_limit": str(gpu_mem_limit_mb * 1024 * 1024),
        "arena_extend_strategy": "kNextPowerOfTwo",
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": True,
    }
    session = ort.InferenceSession(
        model_path,
        providers=[("CUDAExecutionProvider", provider_options), "CPUExecutionProvider"],
    )
    providers = session.get_providers()
    print(f"session providers={providers}", flush=True)
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(f"CUDAExecutionProvider not active. providers={providers}")
    return session


def worker_main(
    worker_id: int,
    args: argparse.Namespace,
    stop_event: threading.Event,
    stats: Stats,
) -> None:
    session = create_session(args.model_path, args.gpu_mem_limit_mb)
    output_names = [out.name for out in session.get_outputs()]
    shape = (args.batch_size, 3, args.height, args.width)
    input_tensor = np.random.rand(*shape).astype(np.float32)

    for _ in range(args.warmup):
        session.run(output_names, {args.input_name: input_tensor})

    while not stop_event.is_set():
        if args.random_input:
            input_tensor = np.random.rand(*shape).astype(np.float32)
        start = time.perf_counter()
        try:
            session.run(output_names, {args.input_name: input_tensor})
        except Exception as exc:
            stats.record_error()
            print(f"worker={worker_id} error={exc}", flush=True)
            time.sleep(0.5)
            continue

        latency_ms = (time.perf_counter() - start) * 1000.0
        stats.record_success(args.batch_size, latency_ms)

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)


def main() -> None:
    args = parse_args()
    print(f"torch.cuda.is_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}", flush=True)
    print(f"ort available providers={ort.get_available_providers()}", flush=True)
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()

    stop_event = threading.Event()
    stats = Stats()
    threads = [
        threading.Thread(target=worker_main, args=(i, args, stop_event, stats), daemon=True)
        for i in range(args.workers)
    ]
    for thread in threads:
        thread.start()

    start_time = time.time()
    prev_requests = 0
    prev_images = 0
    prev_errors = 0

    try:
        while True:
            time.sleep(args.stats_interval)
            requests, images, errors, total_latency_ms, max_latency_ms, window = stats.snapshot_and_reset_window()
            interval_reqs = requests - prev_requests
            interval_imgs = images - prev_images
            interval_errors = errors - prev_errors
            prev_requests = requests
            prev_images = images
            prev_errors = errors

            avg_latency_ms = (sum(window) / len(window)) if window else float("nan")
            p95_latency_ms = statistics.quantiles(window, n=20)[18] if len(window) >= 20 else (max(window) if window else float("nan"))
            print(
                "elapsed={elapsed:.1f}s total_req={requests} total_img={images} err={errors} "
                "interval_req_s={reqps:.2f} interval_img_s={imgps:.2f} avg_ms={avg:.2f} p95_ms={p95:.2f} max_ms={maxv:.2f}".format(
                    elapsed=time.time() - start_time,
                    requests=requests,
                    images=images,
                    errors=errors,
                    reqps=interval_reqs / args.stats_interval,
                    imgps=interval_imgs / args.stats_interval,
                    avg=avg_latency_ms,
                    p95=p95_latency_ms,
                    maxv=max_latency_ms,
                ),
                flush=True,
            )
            if interval_errors:
                print(f"interval_errors={interval_errors}", flush=True)

            if args.duration_seconds > 0 and (time.time() - start_time) >= args.duration_seconds:
                break
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=3.0)


if __name__ == "__main__":
    main()
