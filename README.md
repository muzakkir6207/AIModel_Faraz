# Native Image Classification (CPU) - ORT gRPC + Metrics

This repository exports a ResNet50 model to ONNX, serves it with ONNX Runtime
Server, and provides scripts for local benchmarking and metrics collection.

For the controlled GPU load-test workflow used in Kubernetes, see
[`load_testing/README.md`](/home/user/AIModel_Faraz/load_testing/README.md).

## Local quickstart

Install Docker first, then run:

```bash
sudo apt-get install -y python3-venv
bash ./export_model.sh
bash ./simple_run.sh
```

## Deploying to Kubernetes

The Kubernetes manifests for this repo are under [`k8s/`](/home/user/AIModel_Faraz/k8s).
They deploy the public ONNX Runtime Server image and mount the exported
`resnet50.onnx` model from the Kubernetes node filesystem.

### 1. Export the ONNX model

Generate the model file locally:

```bash
bash ./export_model.sh
```

This creates `model/resnet50.onnx`.

### 2. Update the Deployment host path

Open [`k8s/deployment.yaml`](/home/user/AIModel_Faraz/k8s/deployment.yaml) and
update the `hostPath.path` value so it points to the absolute path of the
directory that contains `resnet50.onnx` on your Kubernetes node.

Example:

```yaml
volumes:
  - name: model
    hostPath:
      path: /absolute/path/to/AIModel_Faraz/model
      type: Directory
```

The container expects the model at `/models/resnet50.onnx`, so the mounted host
directory must contain that file.

### 3. Create the namespace

```bash
kubectl apply -f k8s/namespace.yaml
```

### 4. Apply the Deployment

```bash
kubectl apply -f k8s/deployment.yaml
```

This creates the `ort-server` Deployment in the `aimodel` namespace.

### 5. Expose the application with a Service

Choose one of the service manifests:

#### Option A: ClusterIP service

Use this if the application only needs to be reachable inside the cluster, or
if you plan to use `kubectl port-forward`.

```bash
kubectl apply -n aimodel -f k8s/service-clusterIP.yaml
```

Port-forward it locally:

```bash
kubectl -n aimodel port-forward svc/ort-server 8000:8000 8001:8001
```

#### Option B: NodePort service

Use this if you want to reach the service from outside the cluster through the
node IP.

```bash
kubectl apply -f k8s/service-NodePort.yaml
```

This exposes:

- HTTP on `30080`
- gRPC on `30081`

Example endpoints:

```text
http://<node-ip>:30080
<node-ip>:30081
```

### 6. Verify the deployment

```bash
kubectl -n aimodel get pods
kubectl -n aimodel get svc
kubectl -n aimodel describe deployment ort-server
```

You should see:

- a running pod labeled `app=ort-server`
- a service named `ort-server`
- container ports `8000` and `8001`

## Kubernetes manifest summary

- [`k8s/namespace.yaml`](/home/user/AIModel_Faraz/k8s/namespace.yaml) creates
  the `aimodel` namespace.
- [`k8s/deployment.yaml`](/home/user/AIModel_Faraz/k8s/deployment.yaml)
  creates the ONNX Runtime Server Deployment and mounts the ONNX model from the
  host.
- [`k8s/service-clusterIP.yaml`](/home/user/AIModel_Faraz/k8s/service-clusterIP.yaml)
  exposes the pod internally in the cluster.
- [`k8s/service-NodePort.yaml`](/home/user/AIModel_Faraz/k8s/service-NodePort.yaml)
  exposes the same ports through Kubernetes NodePort.
