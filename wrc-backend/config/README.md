# `config.json` 字段说明

`config.json` 使用标准 JSON 格式，不能写 `//`、`#` 或 `/* ... */` 注释。配置项说明统一维护在本文件中。

## 顶层字段

- `version`：配置文件版本，便于以后做格式兼容。当前值为 `1`。
- `navigation_text`：发送给模型的导航文本；空字符串表示不提供额外导航指令。

## `vehicle_heading`

- `azimuth_offset_deg`：车身安装偏角，单位为度，加在协议换算之后。车辆 `azimuth` 以正西为 0°、顺时针增大；后端罗盘航向为 `(azimuth - 90 + 偏移量) % 360`（0° 为正北）。没有额外安装偏角时设为 `0`。

## `camera_freshness`

- `camera_stale_ms`：最新相机帧相对当前时间允许的最大延迟，超过后本轮不推理。
- `camera_pose_delta_ms`：相机时间与位姿时间允许的最大差值，超过后本轮不推理。
- `window_max_age_ms`：四帧相机窗口中最老帧允许的最大年龄，用于防止窗口包含过旧画面。
- `frame_timeout_ms`：等待相机帧的超时时间；超时后认为当前相机输入不可用。

## `time_sync`

相机和位姿都使用车辆 Unix 时间，换成微秒后对齐。这一节不再有可调字段。

- 相机时间固定为 CIMG 包里的 `capture_time_ns`（纳秒），整除 1000 得到微秒。
- 位姿时间固定为车辆 timestamp（秒），乘 1_000_000 得到微秒。
- `enabled`、`pose_timestamp_mode`、`arrival_fallback`、`pose_unix_unit`、`max_future_offset_ms`、`camera_timestamp_mode`、`camera_unix_unit` 已删除。写进配置时进程启动失败。

收到包的时刻仍用来判断数据到了多久，不参与相机和位姿对齐。

时间同步只做单位换算。两路能够对应，是因为它们本来就在同一条车辆 Unix 时间轴上。

## `inference_pool`

- `mode`：推理运行模式。
  - `inline`：主流程内同步推理，保持原有单模型行为。
  - `multi_process`：启用多进程推理池，每张配置的 GPU 启动一个独立模型进程。
- `gpu_devices`：多进程模式使用的物理 GPU 编号，即 `nvidia-smi` 显示的编号。数组长度就是推理 worker 数量；`inline` 模式下不用于创建 worker。
- `latest_only`：所有 worker 忙碌时是否只保留最新的一项待推理任务。设为 `true` 可避免积压旧画面。
- `pending_capacity`：等待队列容量。`latest_only=true` 时必须为 `1`。
- `result_max_age_ms`：推理结果从输入冻结开始计算允许的最大年龄；超时结果会被丢弃，不再发送给车辆。
- `max_capture_lag_ms`：任务相对最新相机采集时间允许的最大落后量，同时用于识别相机时间戳倒退。向前间隔再大也不记为异常，该采集时间成为新的比较基准。
- `scheduler_poll_interval_ms`：多进程调度器检查 worker 结果和分配新任务的间隔，单位毫秒。
- `worker_start_timeout_s`：worker 加载模型允许的最长时间，单位秒。
- `minimum_ready_workers`：至少需要多少个 worker 完成启动，服务才进入 ready 状态。上限是本机 GPU 数量加上 `network_workers` 的数量。保持为 `1` 时，一台 worker 可用系统即可继续运行。
- `metrics_window_s`：推理池吞吐量、丢弃量等统计指标的时间窗口，单位秒。
- `worker_warmup`：worker 启动后是否执行模型预热；设为 `true` 可把首次推理开销提前到启动阶段。
- `dispatch_interval_ms`：两次派发之间的最小间隔，单位毫秒。`submit()` 直接派发和空闲补位都遵守它。`0` 表示不限速。环境变量 `ALPAMAYO_DISPATCH_INTERVAL_MS` 优先于本字段。
- `network_workers`：PGX-2、PGX-3、PGX-4 的推理服务根地址，例如 `http://10.0.0.2:8092`。每台一个槽位，只接收 JPEG 任务。环境变量 `ALPAMAYO_NETWORK_WORKERS` 用逗号分隔，优先于本字段。
- `network_connect_timeout_s`：连接网络 worker 的超时，单位秒。环境变量 `ALPAMAYO_NETWORK_CONNECT_TIMEOUT_S`。
- `network_infer_timeout_s`：单次推理的超时，单位秒。超时后该槽位变为 `unreachable`，不会立刻再次派发。环境变量 `ALPAMAYO_NETWORK_INFER_TIMEOUT_S`。
- `network_health_interval_ms`：对仍在加载或已不可达的网络 worker 调用 `GET /health` 的间隔，单位毫秒。只有响应明确为 ready 才重新加入调度。环境变量 `ALPAMAYO_NETWORK_HEALTH_INTERVAL_MS`。

### 常用模式

单模型同步推理：

```json
"mode": "inline",
"gpu_devices": []
```

四卡多进程推理示例：

```json
"mode": "multi_process",
"gpu_devices": [1, 3, 4, 5],
"minimum_ready_workers": 4
```

## `network`

PGX 收发的地址和端口只在本节配置。`app.py`、`pipeline.py` 和
`run_backend_real.sh` 都通过 `config.loader` 读取，不再各自硬编码。

监听地址可以是 `0.0.0.0` 或 PGX 的局域网地址。无人车配置的是 PGX 的实际
局域网 IP，不能是 `0.0.0.0`。`waypoint` 是相反方向：PGX 把航点发到无人车。

每个条目都是 `{"host": ..., "port": ...}`：

| 键 | 用途 | 默认值 | 协议 | 方向 | 绑定方 | 连接发起方 |
| --- | --- | --- | --- | --- | --- | --- |
| `cimg` | CIMG 相机帧 | `0.0.0.0:30106` | TCP | 车辆 → PGX | PGX 监听 | 车辆连接 |
| `pose_udp` | 车辆位姿 | `0.0.0.0:7001` | UDP | 车辆 → PGX | PGX 监听 | 车辆发送 |
| `route_tcp` | 规划航线 | `0.0.0.0:30006` | TCP | 车辆 → PGX | PGX 监听 | 车辆连接 |
| `waypoint` | 航点下发 | `168.130.182.10:7010` | UDP | PGX → 车辆 | 车辆监听 7010 | PGX 发送 |
| `http` | `/health`、`/ready`、`/status` | `0.0.0.0:8090` | TCP | 运维 → PGX | PGX 监听 | 运维连接 |

`network.relay`、`PC_IP`、`RELAY_HOST`、`RELAY_PORT` 以及 `127.0.0.1:7003`
是已删除的 PC 中转配置。写进文件或导出到环境变量时，进程启动失败，不会退回旧地址。

`route_tcp` 额外有 `enabled`：设为 `false` 时不启动规划路径 TCP 接收。

相机只有 `cimg` 这一路：默认监听 `0.0.0.0:30106`。Alpamayo 相机顺序由
`ALPAMAYO_CIMG_CAMERA_INDICES` 决定，当前是 `1,2,0,4`（前、右、左、后）。
读取超时、包大小、JPEG 大小、队列和组率不在本文件里，用
`ALPAMAYO_CIMG_READ_TIMEOUT_S`、`ALPAMAYO_CIMG_MAX_PAYLOAD_BYTES`、
`ALPAMAYO_CIMG_MAX_JPEG_BYTES`、`ALPAMAYO_CIMG_QUEUE_FRAMES`、
`ALPAMAYO_CIMG_MAX_BACKLOG_GROUPS`、`ALPAMAYO_CIMG_GROUP_FPS` 覆盖，缺省分别为
15 秒、64MiB、16MiB、60、2、10。

`network.camera_sources` 以及 `ALPAMAYO_CAMERA_INPUT`、`ALPAMAYO_CAMERA_SOURCES`、
`ALPAMAYO_CAMERA_INDICES`、`ALPAMAYO_FFMPEG`、`ALPAMAYO_MPEGTS_LOG_DIR` 是已删除的
MPEG-TS 输入。写进文件或导出到环境变量时，进程启动失败。

`http_client` 是客户端的出站地址：

- `hf_endpoint`：模型下载的 HuggingFace 镜像地址，默认 `https://hf-mirror.com`。
- `package_test_url`：离线测试 API，默认 `http://127.0.0.1:8091/infer`；
  `tools/request_model_test.py` 的 `--url` 默认值取自这里，PC 侧
  `run_package_test.py` 传给它的也是同一个值。

### 取值顺序

命令行参数 > 环境变量 > 本文件 > 代码内置默认值。内置默认值与本节当前写下的值
完全一致，因此删掉整节或整个文件，启动结果与加本节之前逐字节相同。环境变量为
空字符串视为未设置。

对应的环境变量：`ALPAMAYO_CIMG_HOST/PORT`、`ALPAMAYO_POSE_UDP_HOST/PORT`、
`ALPAMAYO_WAYPOINT_HOST/PORT`、`ALPAMAYO_ROUTE_TCP_HOST/PORT`、
`ALPAMAYO_ROUTE_TCP_ENABLED`、`ALPAMAYO_HOST/PORT`、
`HF_ENDPOINT`、`ALPAMAYO_PACKAGE_TEST_URL`。
`ALPAMAYO_RELAY_*`、`PC_IP`、`RELAY_HOST`、`RELAY_PORT` 若被设置，加载直接失败。
`ALPAMAYO_CAMERA_INPUT`、`ALPAMAYO_CAMERA_SOURCES`、`ALPAMAYO_CAMERA_INDICES`、
`ALPAMAYO_FFMPEG`、`ALPAMAYO_MPEGTS_LOG_DIR` 同样直接失败。
`ALPAMAYO_CONFIG` 指定要读取的配置文件名（默认本文件同目录的 `config.json`）。

### 约束

- `port` 必须是 1..65535 的整数（`true` 之类的布尔值会被拒绝）。
- 监听端点的 `host` 可以是 `0.0.0.0`；发送目标不行，必须写实际接收方的地址。
- 每个端点的协议与方向由使用它的组件决定，写进配置也不生效（写了会直接报错）。
- 同一进程内的两个监听端点，协议相同、端口相同且地址重叠（含通配地址）时会被
  拒绝：例如把 `http` 改到 `0.0.0.0:30106` 会与 `cimg` 冲突。不同协议（TCP 与
  UDP）同端口不算冲突。`waypoint` 是发送目标，不参与监听冲突检查。
- 以上校验在加载配置时完成（`load_network_config`，在 `AppConfig.from_env()`
  内），因此非法端口是启动报错，而不是运行一段时间后的 bind 失败。

### 生效方式

监听地址/端口没有热加载：改完需要重启 `run_backend_real.sh` 才会生效。

离线模型测试 URL 默认 `http://127.0.0.1:8091/infer`，与生产 HTTP `8090` 分开。
PC 仓库里的 `config/network.json` 只给本机诊断和离线客户端用，不是 PGX 的监听配置。
