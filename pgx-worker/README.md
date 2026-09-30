# PGX02–04 Alpamayo inference Worker

This package runs **one model-only HTTP Worker**. It provides `GET /health` and
`POST /infer` on port 8092. PGX01 remains the main node responsible for camera
and Pose intake, scheduling, post-processing, and vehicle UDP. Do not run
`run_backend_real.sh` on a Worker.

## Directory layout on each Worker

Place this folder next to the already transferred source, weights, and assets:

```text
701/
├── alpamayo1.5/
├── model_weights/                 # config.json and checkpoint shards
├── model_assets/
│   └── Qwen3-VL-2B-Instruct/
└── pgx-worker/
    ├── run_worker.sh
    └── inference/
        ├── __init__.py
        ├── http_worker.py
        └── alpamayo.py
```

The ARM64 `alpamayo-pgx:1.0` Docker image must also be imported on the target.
The image is **not** included in this small code package. Before starting,
check that the target PGX has compatible NVIDIA driver/Docker GPU support and
that its paths above exist.

## Start and check

On PGX02, PGX03, or PGX04, from the `701` directory:

```bash
sudo bash pgx-worker/run_worker.sh start
sudo bash pgx-worker/run_worker.sh check
```

`check` exits unsuccessfully while the model is loading; wait and rerun it.
Once loaded, it prints `{"ready": true, "state": "ready"}`. To inspect or
stop the exact Worker container:

```bash
sudo bash pgx-worker/run_worker.sh status
sudo bash pgx-worker/run_worker.sh logs
sudo bash pgx-worker/run_worker.sh stop
```

The script discovers the single `192.168.50.x` address and publishes port 8092
only on that interface. If the target uses another LAN or has multiple matching
addresses, set `WORKER_BIND_IP` explicitly, for example:

```bash
sudo env WORKER_BIND_IP=192.168.50.4 bash pgx-worker/run_worker.sh start
```

If the files are not placed beside `pgx-worker`, set `DATA_ROOT` to the directory
containing `alpamayo1.5`, `model_weights`, and `model_assets`. `IMAGE_NAME`,
`WORKER_PORT`, and `CONTAINER_NAME` can also be overridden. Paths *inside* the
container stay consistent with PGX01 to accommodate checkpoint configuration.

From PGX01, verify LAN reachability with the target's actual IP:

```bash
curl -fsS http://192.168.50.3:8092/health
```

`ready: true` proves model loading and HTTP access, not an actual inference.
An end-to-end check additionally requires a valid `POST /infer` request from
PGX01 and observing an accepted result there. The Worker endpoint has no
authentication: expose port 8092 only on a trusted/private LAN.

## Updating dependencies

The Worker Python files and model data are separate from the Docker image.
When dependencies change, build and validate a new ARM64 image on PGX01,
export/import that image, and set `IMAGE_NAME` accordingly; no need to re-copy
unchanged model weights.
