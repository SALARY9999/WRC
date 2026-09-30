# PGX01 容器启动与验证

实车启动脚本是 `run_backend_real.sh`。它使用 `alpamayo-pgx:1.0` 镜像，以及同级的
`alpamayo1.5/`、`model_weights/`、`model_assets/` 和 `wrc-backend/`。镜像还没导入时，
脚本从 `offline/alpamayo-pgx-1.0.tar.zst` 导入。

在 PGX01 的 `/home/xz-01/data/user_data/wanghaohao/701` 下执行：

```bash
bash wrc-backend/run_backend_real.sh test
```

`test` 只绑定 `127.0.0.1` 的测试端口，使用真实后端和本机 GPU Worker，向它发送
合成 CIMG、Pose 数据；航点也仅发到本机回环地址。出现 `"status": "ok"` 且
`backend_inference_count >= 1` 才是一次后端推理通过。测试成功后容器自动停止并移除。
测试配置 `config/pgx01-test.json` 的结果时限为 10 秒，仅用于诊断，不是实车配置。

`config/pgx01.json` 是独立的 PGX01 运行配置，不覆盖现有 `config/config.json`。
本机 GPU `[0]` 和 PGX02–04 的 Worker URL 已预填。修改其中的 `mode` 即可选择：

- `inline`：PGX01 单机同步推理；预填的 `network_workers` 不参与调度，模型在首次推理时加载。
- `multi_process`：PGX01 本机 GPU Worker 启动时加载模型，同时启用已预填的网络 Worker；远端服务必须另行启动。默认 `minimum_ready_workers=1` 仅代表至少一个 Worker 可用，不代表四台全部就绪。

`config/pgx01.json` 仍写着 `result_max_age_ms=2500` 和 `max_capture_lag_ms=2500`。
本机推理约 2.8–5.5 秒，按这个门槛结果会在发出去之前被丢掉。`run_backend_real.sh`
只在进程环境里把这两项放到 10000 毫秒，不改 JSON。10 秒窗口让推理结果能够下发；
它不证明车辆会执行一条已经滞后数秒的轨迹。接管位仍由规划航线偏差决定。

另外三台先各自启动 Worker，只监听 `0.0.0.0:8092`，不接收无人车、也不下发航点。目录不在同一路径时设置 `DATA_ROOT`。

```bash
bash wrc-backend/run_backend_real.sh worker
```

三台的 `GET /health` 出现 `"ready": true` 之后，在 PGX01 启动调度端。它按 `config/pgx01.json` 连接 `192.168.50.3:8092`、`192.168.50.4:8092`、`192.168.50.5:8092`。远端单次推理超时在进程环境里是 10 秒。

```bash
bash wrc-backend/run_backend_real.sh
bash wrc-backend/run_backend_real.sh status
bash wrc-backend/run_backend_real.sh logs
bash wrc-backend/run_backend_real.sh stop
```

Worker 容器用 `status worker`、`logs worker`、`stop worker`。

`run_backend_real.sh` 不修改模型、驱动或 `config/pgx01.json`，也不会停掉其他服务。
容器起来之后，还要用实车的 CIMG、位姿、规划航线和车辆 UDP 接收把整条链路核对一遍。
