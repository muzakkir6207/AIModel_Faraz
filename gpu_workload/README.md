# GPU Workload for `aimodel`

This directory contains a standalone GPU load generator for the `aimodel` namespace.
It is separate from `ort-server`.

The important distinction is:

- `ort-server` is not generating the GPU load.
- `resnet50-gpu-loadgen` is generating the GPU load.

## Files

- `gpu_workload/loadgen.py`: Python load generator that uses `onnxruntime-gpu`
- `gpu_workload/k8s/loadgen-deployment.yaml`: Kubernetes deployment for the load generator
- `gpu_workload/README.md`: this document

## What This Workload Does

- Loads the same ResNet50 ONNX model from `/models/resnet50/1/model.onnx`
- Uses `CUDAExecutionProvider`
- Runs continuous inference on the shared H100 GPU
- Creates load in separate pods so you can later attribute GPU metrics per pod

## How The GPU Load Is Generated

1. Apply `gpu_workload/k8s/loadgen-deployment.yaml`.
2. Kubernetes starts the deployment `resnet50-gpu-loadgen` in namespace `aimodel`.
3. Each pod requests `nvidia.com/gpu: 1` and uses `runtimeClassName: nvidia`, so it gets a shared GPU slot from time-slicing.
4. Each pod starts the image `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`.
5. The container startup command installs `onnxruntime-gpu==1.20.1`.
6. The same startup command then runs `python /workspace/gpu_workload/loadgen.py`.
7. `LD_LIBRARY_PATH` is set so ONNX Runtime can find cuDNN from PyTorch and activate `CUDAExecutionProvider`.
8. `loadgen.py` opens `/models/resnet50/1/model.onnx`.
9. `loadgen.py` creates one ONNX Runtime CUDA session per worker thread.
10. Each worker warms up the model for `WARMUP` iterations.
11. After warmup, each worker loops forever:
    - generate input
    - run `session.run(...)` on GPU
    - record latency and counters
12. Every `STATS_INTERVAL` seconds, the process prints throughput and latency for that pod.

## Current Manifest Defaults

The current deployment manifest is configured with:

- `replicas: 2`
- `BATCH_SIZE=16`
- `WORKERS=4`
- `WARMUP=10`
- `STATS_INTERVAL=5`
- `RANDOM_INPUT_FLAG=--random-input`
- `GPU_MEM_LIMIT_MB=2048`

That means the current setup is effectively:

- 2 pods
- 1 container per pod
- 1 Python process per container
- 4 CUDA-backed ORT workers per process
- total 8 concurrent CUDA inference workers on the shared H100

## Deploy

```bash
kubectl apply -f gpu_workload/k8s/loadgen-deployment.yaml
kubectl rollout status deployment/resnet50-gpu-loadgen -n aimodel --timeout=300s
kubectl get pods -n aimodel -l app=resnet50-gpu-loadgen -o wide
```

## Verify That GPU Load Is Real

Check the pod logs:

```bash
kubectl logs -n aimodel deployment/resnet50-gpu-loadgen -f --all-pods=true
```

You should see lines like:

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
kubectl logs -n aimodel deployment/resnet50-gpu-loadgen -f --all-pods=true
kubectl exec -n aimodel <pod-name> -- ps -ef
```

Inside each pod you should see a process similar to:

```bash
python /workspace/gpu_workload/loadgen.py --model-path /models/resnet50/1/model.onnx --input-name data --batch-size 16 --workers 4 ...
```

## Why This Helps Per-Pod GPU Monitoring

This design is useful because the GPU load is isolated in dedicated pods.
That gives you a clean Kubernetes target for later GPU attribution.

For example, if your monitoring stack can attribute GPU usage by pod or container, the correct target is:

- pod: `resnet50-gpu-loadgen-*`
- container: `loadgen`

## Tune The Load

To push more or less load, update the environment variables in `gpu_workload/k8s/loadgen-deployment.yaml`.
The main knobs are:

- `BATCH_SIZE`
- `WORKERS`
- `replicas`
- `GPU_MEM_LIMIT_MB`
- `SLEEP_MS`
- `DURATION_SECONDS`

Examples:

```bash
kubectl scale deployment/resnet50-gpu-loadgen -n aimodel --replicas=1
kubectl scale deployment/resnet50-gpu-loadgen -n aimodel --replicas=2
```

After changing the manifest:

```bash
kubectl apply -f gpu_workload/k8s/loadgen-deployment.yaml
kubectl rollout status deployment/resnet50-gpu-loadgen -n aimodel --timeout=300s
```

## Stop Or Remove The Load

Scale down without deleting:

```bash
kubectl scale deployment/resnet50-gpu-loadgen -n aimodel --replicas=0
```

Delete completely:

```bash
kubectl delete -f gpu_workload/k8s/loadgen-deployment.yaml
```

## Important Note

Node-level `nvidia-smi` shows total GPU usage for the whole node, not clean per-pod attribution.
Use the `resnet50-gpu-loadgen-*` pod identity in your monitoring stack when you move to per-pod GPU metrics.
