#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
  cat >&2 <<'EOF'
usage:
  run_load_job.sh <duration-seconds> [parallelism]
  run_load_job.sh <duration-seconds> --percent-load <1-100> [--instance-count N] [--max-workers-per-pod N]
  run_load_job.sh <duration-seconds> --percent-loads <P1,P2,...> [--max-workers-per-pod N]

examples:
  ./gpu_workload/run_load_job.sh 600 1
  ./gpu_workload/run_load_job.sh 600 3
  ./gpu_workload/run_load_job.sh 600 --percent-load 10 --instance-count 1 --max-workers-per-pod 10
  ./gpu_workload/run_load_job.sh 600 --percent-load 20 --instance-count 3 --max-workers-per-pod 10
  ./gpu_workload/run_load_job.sh 600 --percent-loads 10,20,30 --max-workers-per-pod 10

notes:
  - plain mode: create one Job with the requested parallelism and explicit WORKERS
  - --percent-load: map a requested percent to workers per pod using max-workers-per-pod
  - --percent-loads: create one one-pod Job per requested percent so each pod can have different load
  - percentages are requested control levels, not guaranteed GPU utilization percentages
EOF
}

require_positive_int() {
  local name="$1"
  local value="$2"

  if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
    echo "${name} must be a positive integer" >&2
    exit 1
  fi
}

require_nonnegative_int() {
  local name="$1"
  local value="$2"

  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "${name} must be a non-negative integer" >&2
    exit 1
  fi
}

require_positive_number() {
  local name="$1"
  local value="$2"

  if [[ ! "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "${name} must be a positive number" >&2
    exit 1
  fi

  awk -v value="${value}" 'BEGIN { exit !(value > 0) }' || {
    echo "${name} must be > 0" >&2
    exit 1
  }
}

require_percent() {
  local name="$1"
  local value="$2"

  require_positive_int "${name}" "${value}"
  if (( value < 1 || value > 100 )); then
    echo "${name} must be in the range 1..100" >&2
    exit 1
  fi
}

ensure_paths() {
  if [[ ! -d "${WORKLOAD_HOST_PATH}" ]]; then
    echo "error: workload directory not found: ${WORKLOAD_HOST_PATH}" >&2
    exit 1
  fi

  if [[ ! -d "${MODEL_HOST_PATH}" ]]; then
    cat >&2 <<EOF
error: model directory not found: ${MODEL_HOST_PATH}

Generate the ONNX model first:
  cd ${REPO_DIR}
  ./export_model.sh

Expected file after export:
  ${MODEL_HOST_PATH}/${MODEL_RELATIVE_PATH}
EOF
    exit 1
  fi

  if [[ ! -f "${MODEL_HOST_PATH}/${MODEL_RELATIVE_PATH}" ]]; then
    cat >&2 <<EOF
error: model file not found: ${MODEL_HOST_PATH}/${MODEL_RELATIVE_PATH}

Either:
  1. export the default model with ./export_model.sh
  2. or override MODEL_RELATIVE_PATH / MODEL_CONTAINER_PATH for a different layout
EOF
    exit 1
  fi
}

workers_for_percent() {
  local percent="$1"
  local max_workers="$2"
  local workers=$(( (percent * max_workers + 50) / 100 ))
  if (( workers < 1 )); then
    workers=1
  fi
  printf '%s\n' "${workers}"
}

make_job_name() {
  local prefix="$1"
  printf '%s-%s-%s\n' "${prefix}" "$(date +%s)" "${RANDOM}"
}

create_job() {
  local job_name_prefix="$1"
  local parallelism="$2"
  local workers="$3"
  local job_name

  job_name="$(make_job_name "${job_name_prefix}")"

  echo "creating job: ${job_name}"
  echo "  duration_seconds=${DURATION_SECONDS}"
  echo "  parallelism=${parallelism}"
  echo "  workload_host_path=${WORKLOAD_HOST_PATH}"
  echo "  model_host_path=${MODEL_HOST_PATH}"
  echo "  model_relative_path=${MODEL_RELATIVE_PATH}"
  echo "  model_container_path=${MODEL_CONTAINER_PATH}"
  echo "  input_name=${INPUT_NAME}"
  echo "  batch_size=${BATCH_SIZE}"
  echo "  workers=${workers}"
  echo "  warmup=${WARMUP}"
  echo "  stats_interval=${STATS_INTERVAL}"
  echo "  sleep_ms=${SLEEP_MS}"
  echo "  gpu_mem_limit_mb=${GPU_MEM_LIMIT_MB}"
  echo "  random_input_flag=${RANDOM_INPUT_FLAG}"

  kubectl create -f - <<YAML
apiVersion: batch/v1
kind: Job
metadata:
  name: ${job_name}
  namespace: aimodel
  labels:
    app: resnet50-gpu-loadgen
    workload-mode: one-shot
spec:
  completions: ${parallelism}
  parallelism: ${parallelism}
  backoffLimit: 0
  ttlSecondsAfterFinished: 120
  activeDeadlineSeconds: $((DURATION_SECONDS + 300))
  template:
    metadata:
      labels:
        app: resnet50-gpu-loadgen
        workload-mode: one-shot
    spec:
      runtimeClassName: nvidia
      restartPolicy: Never
      containers:
      - name: loadgen
        image: pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
        imagePullPolicy: IfNotPresent
        command:
        - /bin/bash
        - -lc
        - |
          set -euo pipefail
          python -m pip install --no-cache-dir --root-user-action=ignore onnxruntime-gpu==1.20.1
          exec python /workspace/gpu_workload/loadgen.py \
            --model-path "${MODEL_CONTAINER_PATH}" \
            --input-name "\${INPUT_NAME}" \
            --batch-size "\${BATCH_SIZE}" \
            --workers "\${WORKERS}" \
            --warmup "\${WARMUP}" \
            --stats-interval "\${STATS_INTERVAL}" \
            --duration-seconds "\${DURATION_SECONDS}" \
            --sleep-ms "\${SLEEP_MS}" \
            --gpu-mem-limit-mb "\${GPU_MEM_LIMIT_MB}" \
            \${RANDOM_INPUT_FLAG}
        env:
        - name: LD_LIBRARY_PATH
          value: /opt/conda/lib/python3.11/site-packages/torch/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
        - name: INPUT_NAME
          value: "${INPUT_NAME}"
        - name: BATCH_SIZE
          value: "${BATCH_SIZE}"
        - name: WORKERS
          value: "${workers}"
        - name: WARMUP
          value: "${WARMUP}"
        - name: STATS_INTERVAL
          value: "${STATS_INTERVAL}"
        - name: DURATION_SECONDS
          value: "${DURATION_SECONDS}"
        - name: SLEEP_MS
          value: "${SLEEP_MS}"
        - name: GPU_MEM_LIMIT_MB
          value: "${GPU_MEM_LIMIT_MB}"
        - name: RANDOM_INPUT_FLAG
          value: "${RANDOM_INPUT_FLAG}"
        resources:
          requests:
            cpu: "2"
            memory: "6Gi"
            nvidia.com/gpu: "1"
          limits:
            cpu: "4"
            memory: "12Gi"
            nvidia.com/gpu: "1"
        volumeMounts:
        - name: workload-src
          mountPath: /workspace/gpu_workload
          readOnly: true
        - name: model
          mountPath: /models
          readOnly: true
      volumes:
      - name: workload-src
        hostPath:
          path: ${WORKLOAD_HOST_PATH}
          type: Directory
      - name: model
        hostPath:
          path: ${MODEL_HOST_PATH}
          type: Directory
YAML

  echo "created job: ${job_name}"
  echo "watch: kubectl get pods -n aimodel -l job-name=${job_name} -w"
  echo "logs : kubectl logs -n aimodel -l job-name=${job_name} --all-containers=true -f"
}

parse_percent_loads() {
  local csv="$1"
  local -n out_array_ref="$2"
  local item=""

  IFS=',' read -r -a out_array_ref <<< "${csv}"
  if [[ "${#out_array_ref[@]}" -eq 0 ]]; then
    echo "percent-loads must contain at least one entry" >&2
    exit 1
  fi

  for item in "${out_array_ref[@]}"; do
    require_percent "percent-loads entry" "${item}"
  done
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  usage
  exit 0
fi

DURATION_SECONDS="$1"
shift
require_positive_int "duration-seconds" "${DURATION_SECONDS}"

PARALLELISM="${PARALLELISM:-2}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-resnet50-gpu-loadgen}"
WORKLOAD_HOST_PATH="${WORKLOAD_HOST_PATH:-${REPO_DIR}/gpu_workload}"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-${REPO_DIR}/model}"
MODEL_RELATIVE_PATH="${MODEL_RELATIVE_PATH:-resnet50.onnx}"
MODEL_CONTAINER_PATH="${MODEL_CONTAINER_PATH:-/models/${MODEL_RELATIVE_PATH}}"
INPUT_NAME="${INPUT_NAME:-input}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-4}"
WARMUP="${WARMUP:-10}"
STATS_INTERVAL="${STATS_INTERVAL:-5}"
SLEEP_MS="${SLEEP_MS:-0}"
GPU_MEM_LIMIT_MB="${GPU_MEM_LIMIT_MB:-2048}"
RANDOM_INPUT_FLAG="${RANDOM_INPUT_FLAG:---random-input}"
PERCENT_LOAD=""
PERCENT_LOADS=""
INSTANCE_COUNT=""
MAX_WORKERS_PER_POD="${MAX_WORKERS_PER_POD:-10}"

LEGACY_ARGS=()
while [[ $# -gt 0 && "$1" != --* ]]; do
  LEGACY_ARGS+=("$1")
  shift
done

case "${#LEGACY_ARGS[@]}" in
  0)
    ;;
  1)
    PARALLELISM="${LEGACY_ARGS[0]}"
    ;;
  *)
    usage
    exit 1
    ;;
esac

while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallelism|--instance-count)
      INSTANCE_COUNT="${2:-}"
      shift 2
      ;;
    --percent-load)
      PERCENT_LOAD="${2:-}"
      shift 2
      ;;
    --percent-loads)
      PERCENT_LOADS="${2:-}"
      shift 2
      ;;
    --max-workers-per-pod)
      MAX_WORKERS_PER_POD="${2:-}"
      shift 2
      ;;
    --job-name-prefix)
      JOB_NAME_PREFIX="${2:-}"
      shift 2
      ;;
    --workers)
      WORKERS="${2:-}"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="${2:-}"
      shift 2
      ;;
    --input-name)
      INPUT_NAME="${2:-}"
      shift 2
      ;;
    --warmup)
      WARMUP="${2:-}"
      shift 2
      ;;
    --stats-interval)
      STATS_INTERVAL="${2:-}"
      shift 2
      ;;
    --worker-delay-ms|--sleep-ms)
      SLEEP_MS="${2:-}"
      shift 2
      ;;
    --gpu-mem-limit-mb)
      GPU_MEM_LIMIT_MB="${2:-}"
      shift 2
      ;;
    --model-relative-path)
      MODEL_RELATIVE_PATH="${2:-}"
      MODEL_CONTAINER_PATH="/models/${MODEL_RELATIVE_PATH}"
      shift 2
      ;;
    --fixed-input)
      RANDOM_INPUT_FLAG=""
      shift
      ;;
    --random-input)
      RANDOM_INPUT_FLAG="--random-input"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -n "${INSTANCE_COUNT}" ]]; then
  PARALLELISM="${INSTANCE_COUNT}"
fi

require_positive_int "parallelism" "${PARALLELISM}"
require_positive_int "workers" "${WORKERS}"
require_positive_int "batch-size" "${BATCH_SIZE}"
require_nonnegative_int "warmup" "${WARMUP}"
require_positive_int "max-workers-per-pod" "${MAX_WORKERS_PER_POD}"
require_positive_number "stats-interval" "${STATS_INTERVAL}"
require_nonnegative_int "worker-delay-ms" "${SLEEP_MS}"
require_positive_int "gpu-mem-limit-mb" "${GPU_MEM_LIMIT_MB}"

if [[ -n "${PERCENT_LOAD}" && -n "${PERCENT_LOADS}" ]]; then
  echo "--percent-load cannot be combined with --percent-loads" >&2
  exit 1
fi

ensure_paths

if [[ -n "${PERCENT_LOADS}" ]]; then
  LOAD_PERCENTS=()
  parse_percent_loads "${PERCENT_LOADS}" LOAD_PERCENTS

  if [[ -n "${INSTANCE_COUNT}" ]] && (( INSTANCE_COUNT != ${#LOAD_PERCENTS[@]} )); then
    echo "instance-count must match the number of percent-loads entries" >&2
    exit 1
  fi

  if [[ "${#LEGACY_ARGS[@]}" -gt 0 ]]; then
    echo "positional parallelism cannot be combined with --percent-loads" >&2
    exit 1
  fi

  echo "running percent-load series"
  echo "mode                        : varied per-pod requested load"
  echo "pod count                    : ${#LOAD_PERCENTS[@]}"
  echo "reference max workers/pod    : ${MAX_WORKERS_PER_POD}"
  echo "duration                     : ${DURATION_SECONDS}s"
  echo "batch size                   : ${BATCH_SIZE}"
  echo "worker delay                 : ${SLEEP_MS}ms"

  for i in "${!LOAD_PERCENTS[@]}"; do
    workers_for_target="$(workers_for_percent "${LOAD_PERCENTS[i]}" "${MAX_WORKERS_PER_POD}")"
    echo "requested load for pod $((i + 1)) : ${LOAD_PERCENTS[i]}%"
    echo "actual mapping for pod $((i + 1)) : 1 pod x ${workers_for_target} workers"
    create_job "${JOB_NAME_PREFIX}-p$((i + 1))" "1" "${workers_for_target}"
  done
  exit 0
fi

if [[ -n "${PERCENT_LOAD}" ]]; then
  require_percent "percent-load" "${PERCENT_LOAD}"

  if [[ -n "${INSTANCE_COUNT}" ]]; then
    require_positive_int "instance-count" "${INSTANCE_COUNT}"
    PARALLELISM="${INSTANCE_COUNT}"
  fi

  workers_for_uniform="$(workers_for_percent "${PERCENT_LOAD}" "${MAX_WORKERS_PER_POD}")"
  echo "running percent-load job"
  echo "mode                        : uniform per-pod requested load"
  echo "instance count              : ${PARALLELISM}"
  echo "requested load per pod      : ${PERCENT_LOAD}%"
  echo "reference max workers/pod   : ${MAX_WORKERS_PER_POD}"
  echo "actual mapping              : ${PARALLELISM} pod(s) x ${workers_for_uniform} workers"
  create_job "${JOB_NAME_PREFIX}" "${PARALLELISM}" "${workers_for_uniform}"
  exit 0
fi

create_job "${JOB_NAME_PREFIX}" "${PARALLELISM}" "${WORKERS}"
