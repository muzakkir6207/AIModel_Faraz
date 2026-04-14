# GPU Workload for `aimodel`

This directory contains the GPU load-generation bundle for `aimodel`.
It is separate from `ort-server`.

The important distinction is:

- `ort-server` is not generating the GPU load.
- `resnet50-gpu-loadgen` is generating the GPU load.

## Recommended Way To Run Load

Use the launcher script and pass the runtime at execution time.
That is the intended interface now.

```bash
cd ~/AIModel_Faraz
./gpu_workload/run_load_job.sh <duration-seconds> [parallelism]
```

Examples:

```bash
./gpu_workload/run_load_job.sh 600
./gpu_workload/run_load_job.sh 300 1
./gpu_workload/run_load_job.sh 900 2
```

Arguments:

- `duration-seconds`: how long each load pod should generate GPU load
- `parallelism`: how many pods to run in parallel, default `2`

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
7. `loadgen.py` loads `/models/resnet50/1/model.onnx`.
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
python /workspace/gpu_workload/loadgen.py --model-path /models/resnet50/1/model.onnx --input-name data --batch-size 16 --workers 4 ...
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
