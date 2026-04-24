# AIModel_Faraz Load Testing Guide

This folder is the working reference for future load testing in
`/home/user/AIModel_Faraz`.

It captures the load-testing flow that was validated on the current cluster for
namespace `aimodel`.

## Scope

This guide is for GPU load generation using:

- `gpu_workload/run_load_job.sh`
- `gpu_workload/loadgen.py`
- one-shot `resnet50-gpu-loadgen-*` Jobs

It is not the same as request-level load testing against `ort-server`.

Current model:

- host file: `~/AIModel_Faraz/model/resnet50.onnx`
- container path: `/models/resnet50.onnx`
- default model input name: `input`

## Validated Status

The current workflow is working.

Validated baseline from this repo and cluster:

- command shape: `--percent-load 40 --instance-count 1 --max-workers-per-pod 10`
- actual mapping: `1` pod x `4` workers
- observed GPU: about `40% sm`, `12% mem`
- observed throughput: about `107 req/s`
- observed image rate: about `1710 img/s`
- observed latency: about `4.8 ms avg`, `5.0 ms p95`
- observed errors: `0`

Example healthy log lines:

```text
elapsed=10.0s total_req=561 total_img=8976 err=0 interval_req_s=106.60 interval_img_s=1705.60 avg_ms=4.91 p95_ms=5.58 max_ms=25.45
elapsed=15.0s total_req=1098 total_img=17568 err=0 interval_req_s=107.40 interval_img_s=1718.40 avg_ms=4.76 p95_ms=4.98 max_ms=25.45
```

## Important Behavior

- `ort-server` is not generating this GPU load
- `resnet50-gpu-loadgen` is generating this GPU load
- percentages are requested control levels, not guaranteed GPU utilization percentages
- requested percent is converted into workers per pod
- the script prints the actual worker mapping for every percent-based run

## One-Time Setup

### 1. Move to the repo

```bash
cd ~/AIModel_Faraz
```

### 2. Export the ONNX model

Do this at least once on a fresh clone.

```bash
./export_model.sh
ls -l ~/AIModel_Faraz/model/resnet50.onnx
```

### 3. Confirm cluster state

```bash
kubectl get ns aimodel
kubectl get pods -n aimodel
kubectl get jobs -n aimodel
```

## Main Interface

The primary launcher is:

```bash
./gpu_workload/run_load_job.sh <duration-seconds> [parallelism]
./gpu_workload/run_load_job.sh <duration-seconds> --percent-load <1-100> [--instance-count N] [--max-workers-per-pod N]
./gpu_workload/run_load_job.sh <duration-seconds> --percent-loads <P1,P2,...> [--max-workers-per-pod N] [--job-name-prefix PREFIX]
```

Recommended standard reference:

- duration: `600`
- max workers per pod: `10`
- batch size: `16`
- warmup: `10`
- stats interval: `5`
- sleep: `0`

## Percent Mapping

With `--max-workers-per-pod 10`:

- `10%` -> `1` worker per pod
- `20%` -> `2` workers per pod
- `30%` -> `3` workers per pod
- `40%` -> `4` workers per pod
- `50%` -> `5` workers per pod
- `60%` -> `6` workers per pod
- `70%` -> `7` workers per pod
- `80%` -> `8` workers per pod
- `90%` -> `9` workers per pod
- `100%` -> `10` workers per pod

The mapping is rounded to whole workers. Record both:

- requested load percent
- actual mapping printed by the script

## Common Commands

### Idle baseline

```bash
kubectl delete job -n aimodel -l app=resnet50-gpu-loadgen
nvidia-smi dmon -s u -d 1 -c 60
```

### One pod at `10%`

```bash
./gpu_workload/run_load_job.sh 600 \
  --percent-load 10 \
  --instance-count 1 \
  --max-workers-per-pod 10
```

### One pod at `20%`

```bash
./gpu_workload/run_load_job.sh 600 \
  --percent-load 20 \
  --instance-count 1 \
  --max-workers-per-pod 10
```

### One pod at `40%`

```bash
./gpu_workload/run_load_job.sh 600 \
  --percent-load 40 \
  --instance-count 1 \
  --max-workers-per-pod 10
```

### Two pods at `20%` each

```bash
./gpu_workload/run_load_job.sh 600 \
  --percent-load 20 \
  --instance-count 2 \
  --max-workers-per-pod 10
```

### Three pods at `20%` each

```bash
./gpu_workload/run_load_job.sh 600 \
  --percent-load 20 \
  --instance-count 3 \
  --max-workers-per-pod 10
```

### Varied load across three pods: `10%,20%,30%`

```bash
./gpu_workload/run_load_job.sh 600 \
  --percent-loads 10,20,30 \
  --max-workers-per-pod 10 \
  --job-name-prefix tc70
```

That launches three separate one-pod Jobs so each pod can carry a different
requested load.

## Recommended Test Matrix

### TC-00: Idle baseline

- no loadgen Jobs running
- record idle GPU `sm` and `mem`

### TC-10: Single-pod percent sweep

Run these one at a time:

- `10%`
- `20%`
- `30%`
- `40%`
- `50%`
- `60%`
- `70%`
- `80%`
- `90%`
- `100%`

Command pattern:

```bash
./gpu_workload/run_load_job.sh 600 \
  --percent-load 40 \
  --instance-count 1 \
  --max-workers-per-pod 10
```

### TC-20: Two-pod uniform percent sweep

Command pattern:

```bash
./gpu_workload/run_load_job.sh 600 \
  --percent-load 20 \
  --instance-count 2 \
  --max-workers-per-pod 10
```

### TC-30: Three-pod uniform percent sweep

Command pattern:

```bash
./gpu_workload/run_load_job.sh 600 \
  --percent-load 20 \
  --instance-count 3 \
  --max-workers-per-pod 10
```

### TC-40: Four-pod uniform percent sweep

Command pattern:

```bash
./gpu_workload/run_load_job.sh 600 \
  --percent-load 20 \
  --instance-count 4 \
  --max-workers-per-pod 10
```

### TC-50: Five-pod uniform percent sweep

Command pattern:

```bash
./gpu_workload/run_load_job.sh 600 \
  --percent-load 20 \
  --instance-count 5 \
  --max-workers-per-pod 10
```

### TC-60: Two-pod varied percent load

Suggested pairs:

- `10,20`
- `25,50`
- `50,100`

Command pattern:

```bash
./gpu_workload/run_load_job.sh 600 \
  --percent-loads 10,20 \
  --max-workers-per-pod 10 \
  --job-name-prefix tc60
```

### TC-70: Three-pod varied percent load

Suggested sets:

- `10,20,30`
- `25,50,75`
- `50,75,100`

Command pattern:

```bash
./gpu_workload/run_load_job.sh 600 \
  --percent-loads 10,20,30 \
  --max-workers-per-pod 10 \
  --job-name-prefix tc70
```

## Monitoring During A Run

Watch pods:

```bash
kubectl get pods -n aimodel -l app=resnet50-gpu-loadgen -w
```

Watch logs:

```bash
kubectl logs -n aimodel -l app=resnet50-gpu-loadgen --all-containers=true -f
```

Watch GPU:

```bash
nvidia-smi dmon -s u -d 1
```

Healthy signs:

- `torch.cuda.is_available=True`
- `session providers=['CUDAExecutionProvider', 'CPUExecutionProvider']`
- `total_req` increasing
- `total_img` increasing
- `err=0`
- non-zero GPU `sm` and `mem`

## Stopping Load

Normal stop:

```bash
kubectl delete job -n aimodel -l app=resnet50-gpu-loadgen
```

Verify:

```bash
kubectl get jobs -n aimodel
kubectl get pods -n aimodel
```

If pods linger in `Terminating`, force-remove them:

```bash
kubectl delete pod -n aimodel -l app=resnet50-gpu-loadgen --force --grace-period=0
```

Expected final state after stopping load:

- `ort-server` still running if you want it
- no `resnet50-gpu-loadgen-*` pods left

## Recording Results

For each run, record:

- test case ID
- Job name
- pod name or pod names
- requested load percent or percents
- actual worker mapping
- batch size
- sleep
- average GPU `sm`
- average GPU `mem`
- total requests
- total images
- errors
- average image rate
- average latency
- p95 latency
- notes

Suggested table:

| Test Case | Job Name | Pods | Requested Load | Actual Mapping | Batch | Sleep | Avg GPU SM | Avg GPU MEM | Total Req | Total Img | Errors | Avg Img/s | Avg ms | P95 ms | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Troubleshooting

### Model directory missing

Symptom:

- launcher fails before Job creation

Fix:

```bash
./export_model.sh
ls -l ~/AIModel_Faraz/model/resnet50.onnx
```

### Pod stuck in `Pending` or `ContainerCreating`

Check:

```bash
kubectl describe pod -n aimodel <pod-name>
kubectl get nodes -o wide
```

Common causes:

- no schedulable GPU slice
- missing model directory
- runtime class or device plugin issue

### Pod runs but GPU stays at `0%`

Check pod logs first:

```bash
kubectl logs -n aimodel -l app=resnet50-gpu-loadgen --all-containers=true -f
```

The fixed workflow should use:

- model path `/models/resnet50.onnx`
- input name `input`

If `total_req` and `total_img` are not increasing, the pod is not driving real
inference yet.

### Load is too weak or too strong

Adjust one or more of:

- requested percent load
- `--max-workers-per-pod`
- `--instance-count`
- `SLEEP_MS`

### Old pods stay around after deleting Jobs

Use:

```bash
kubectl delete pod -n aimodel -l app=resnet50-gpu-loadgen --force --grace-period=0
```

## Related Files

- [gpu_workload/run_load_job.sh](/home/user/AIModel_Faraz/gpu_workload/run_load_job.sh)
- [gpu_workload/loadgen.py](/home/user/AIModel_Faraz/gpu_workload/loadgen.py)
- [gpu_workload/README.md](/home/user/AIModel_Faraz/gpu_workload/README.md)
- [README.md](/home/user/AIModel_Faraz/README.md)
