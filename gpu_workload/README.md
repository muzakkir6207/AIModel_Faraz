# GPU Workload for `aimodel`

This directory contains the GPU load-generation bundle for `aimodel`.
It is separate from `ort-server`.

The important distinction is:

- `ort-server` is not generating the GPU load.
- `resnet50-gpu-loadgen` is generating the GPU load.

For the controlled testcase matrix, validated workflow, and recommended
execution order, see
[`../load_testing/README.md`](/home/user/AIModel_Faraz/load_testing/README.md).

## Recommended Way To Run Load

Use the launcher script and pass the runtime at execution time.
That is the intended interface now.

```bash
cd ~/AIModel_Faraz
./export_model.sh
./gpu_workload/run_load_job.sh <duration-seconds> [parallelism]
./gpu_workload/run_load_job.sh <duration-seconds> --percent-load <1-100> [--instance-count N] [--max-workers-per-pod N]
./gpu_workload/run_load_job.sh <duration-seconds> --percent-loads <P1,P2,...> [--max-workers-per-pod N]
```

Examples:

```bash
./export_model.sh
./gpu_workload/run_load_job.sh 600
./gpu_workload/run_load_job.sh 300 1
./gpu_workload/run_load_job.sh 900 2
./gpu_workload/run_load_job.sh 600 --percent-load 10 --instance-count 1 --max-workers-per-pod 10
./gpu_workload/run_load_job.sh 600 --percent-load 20 --instance-count 3 --max-workers-per-pod 10
./gpu_workload/run_load_job.sh 600 --percent-loads 10,20,30 --max-workers-per-pod 10
```

Primary controls:

- `duration-seconds`: how long each load pod should generate GPU load
- `parallelism` or `--instance-count`: how many pods to run in parallel
- `--percent-load`: requested uniform load per pod
- `--percent-loads`: requested varied load per pod
- `--max-workers-per-pod`: reference worker budget for `100%` requested load

Percent mapping:

- `10%` with `--max-workers-per-pod 10` maps to `1` worker per pod
- `20%` with `--max-workers-per-pod 10` maps to `2` workers per pod
- `40%` with `--max-workers-per-pod 10` maps to `4` workers per pod
- `100%` with `--max-workers-per-pod 10` maps to `10` workers per pod

The script prints the actual worker mapping for every percent-based run.
These percentages are requested control levels, not guaranteed GPU utilization
percentages.

Optional per-run environment overrides:

- `JOB_NAME_PREFIX`
- `INPUT_NAME`
- `BATCH_SIZE`
- `WORKERS`
- `WARMUP`
- `STATS_INTERVAL`
- `SLEEP_MS`
- `GPU_MEM_LIMIT_MB`
- `RANDOM_INPUT_FLAG`

Example:

```bash
WORKERS=6 BATCH_SIZE=16 JOB_NAME_PREFIX=tc20 ./gpu_workload/run_load_job.sh 600 2
```

Equivalent percent-based example:

```bash
./gpu_workload/run_load_job.sh 600 --percent-load 60 --instance-count 2 --max-workers-per-pod 10
```

The launcher expects the exported model file at:

- host path: `~/AIModel_Faraz/model/resnet50.onnx`
- container path: `/models/resnet50.onnx`
- default input name: `input`

If the model directory is missing, the launcher now exits locally with an
actionable error instead of creating a stuck Kubernetes Job.

## What The Launcher Does

`gpu_workload/run_load_job.sh` creates a one-shot Kubernetes Job at runtime.
The Job name is generated dynamically, so each run is separate.

For each run, it creates pods that:

- request `nvidia.com/gpu: 1`
- use `runtimeClassName: nvidia`
- install `onnxruntime-gpu==1.20.1`
- run `gpu_workload/loadgen.py`
- pass your requested duration through `DURATION_SECONDS`
- stop automatically when that duration is reached

The Job also has:

- `restartPolicy: Never`
- `backoffLimit: 0`
- `ttlSecondsAfterFinished: 120`

So it does not keep consuming GPU indefinitely.

## How The GPU Load Is Generated

1. You run `./gpu_workload/run_load_job.sh <duration-seconds> [parallelism]`.
2. The script creates a Kubernetes Job in namespace `aimodel`.
3. Each Job pod gets a shared GPU slot through the NVIDIA time-slicing setup.
4. The container starts `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`.
5. The container installs `onnxruntime-gpu==1.20.1`.
6. The container runs `python /workspace/gpu_workload/loadgen.py`.
7. `loadgen.py` loads `/models/resnet50.onnx`.
8. `loadgen.py` creates one ONNX Runtime CUDA session per worker thread.
9. Each worker performs warmup iterations.
10. After warmup, each worker loops: generate input, run `session.run(...)` on GPU, record latency, and increment counters.
11. Once `DURATION_SECONDS` is reached, the process exits and the Job completes.

## Current Default Runtime Settings

The generated Job uses these default pod settings:

- `BATCH_SIZE=16`
- `WORKERS=4`
- `WARMUP=10`
- `STATS_INTERVAL=5`
- `GPU_MEM_LIMIT_MB=2048`
- `RANDOM_INPUT_FLAG=--random-input`

With `parallelism=2`, the effective shape is:

- 2 pods
- 1 container per pod
- 1 Python process per container
- 4 CUDA-backed ORT workers per process
- total 8 concurrent CUDA inference workers on the shared H100

## Watch A Run

After starting a run:

```bash
kubectl get jobs -n aimodel
kubectl get pods -n aimodel -l app=resnet50-gpu-loadgen -w
kubectl logs -n aimodel -l app=resnet50-gpu-loadgen --all-containers=true -f
```

The launcher also prints the exact `job-name` selector to use.

## Verify That GPU Load Is Real

You should see pod log lines like:

```text
torch.cuda.is_available=True device_count=1
session providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
elapsed=5.0s total_req=... total_img=... interval_img_s=... avg_ms=... p95_ms=...
```

Check node-level GPU activity:

```bash
nvidia-smi
nvidia-smi dmon -s u
```

## What To Monitor Per Pod

If you want per-pod GPU metrics later, monitor these identities:

- `namespace=aimodel`
- `pod=resnet50-gpu-loadgen-*`
- `container=loadgen`

Do not attribute this GPU load to `ort-server`.

Useful commands:

```bash
kubectl get pods -n aimodel -l app=resnet50-gpu-loadgen -o wide
kubectl logs -n aimodel -l app=resnet50-gpu-loadgen --all-containers=true -f
kubectl exec -n aimodel <pod-name> -- ps -ef
```

Inside each pod you should see a process similar to:

```bash
python /workspace/gpu_workload/loadgen.py --model-path /models/resnet50.onnx --input-name input --batch-size 16 --workers 4 ...
```

## Continuous Deployment Mode

A continuous Deployment manifest still exists at:

- `gpu_workload/k8s/loadgen-deployment.yaml`

But it is intentionally safe now:

- `replicas: 0`
- it will not consume GPU unless you explicitly scale it up

Manual start:

```bash
kubectl apply -f gpu_workload/k8s/loadgen-deployment.yaml
kubectl scale deployment/resnet50-gpu-loadgen -n aimodel --replicas=2
```

Manual stop:

```bash
kubectl scale deployment/resnet50-gpu-loadgen -n aimodel --replicas=0
```

## Why This Helps Per-Pod GPU Monitoring

This design isolates the GPU load in dedicated pods.
That gives you a clean Kubernetes target for GPU attribution later.

The correct attribution target is:

- pod: `resnet50-gpu-loadgen-*`
- container: `loadgen`

## Important Note

Node-level `nvidia-smi` shows total GPU usage for the whole node, not clean per-pod attribution.
Use the `resnet50-gpu-loadgen-*` pod identity in your monitoring stack when you move to per-pod GPU metrics.
