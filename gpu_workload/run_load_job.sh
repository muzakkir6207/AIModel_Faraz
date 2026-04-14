#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <duration-seconds> [parallelism]" >&2
  exit 1
fi

DURATION_SECONDS="$1"
PARALLELISM="${2:-2}"
JOB_NAME="resnet50-gpu-loadgen-$(date +%s)"

kubectl create -f - <<YAML
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: aimodel
  labels:
    app: resnet50-gpu-loadgen
    workload-mode: one-shot
spec:
  completions: ${PARALLELISM}
  parallelism: ${PARALLELISM}
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
            --model-path /models/resnet50/1/model.onnx \
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
          value: data
        - name: BATCH_SIZE
          value: "16"
        - name: WORKERS
          value: "4"
        - name: WARMUP
          value: "10"
        - name: STATS_INTERVAL
          value: "5"
        - name: DURATION_SECONDS
          value: "${DURATION_SECONDS}"
        - name: SLEEP_MS
          value: "0"
        - name: GPU_MEM_LIMIT_MB
          value: "2048"
        - name: RANDOM_INPUT_FLAG
          value: --random-input
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
          path: /home/user/AIModel_Faraz/gpu_workload
          type: Directory
      - name: model
        hostPath:
          path: /home/user/AIModel_Faraz/model
          type: Directory
YAML

echo "created job: ${JOB_NAME}"
echo "watch: kubectl get pods -n aimodel -l job-name=${JOB_NAME} -w"
echo "logs : kubectl logs -n aimodel -l job-name=${JOB_NAME} --all-containers=true -f"
