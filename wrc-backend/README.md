# Alpamayo 1.5 实时避障后端

## TCP 30006 原规划航线接收

`vehicle/planned_route.py` 解析小端帧：`<BBIIB`（0xF1、0x10、总长度、航点数、reached），
后接 `num` 个 `<ff`（经度、纬度，单位度）。总长度必须为 `11 + 8*num`。
完整报文原子替换航线；`reached=1` 或零航点清除缓存。TCP 拆包、粘包均支持。
非法长度、坐标 NaN/Inf 或范围异常拒绝该连接，不写入缓存；断流允许重新连接。

后端默认启用监听；绑定地址默认 `0.0.0.0`、端口默认 `30006`
（可用 `ALPAMAYO_ROUTE_TCP_HOST/PORT` 修改）。
启用前必须确认端口没有被其他服务占用。接收日志关键词为 `planned route`，
`/status` 的 `planned_route_cache.receiver_listening/receiver_error` 表示接收状态。

已接入 planned route 几何转换和偏差接管，无 history pose 回退。
双方经纬度 datum 已确认一致；float32 量化误差仍需在实车调参时评估。

离线测试：`python -m unittest discover -s tests -p test_planned_route_tcp.py -v`。

实时链路（无人车与 PGX 在同一局域网，不经过 PC）：

```text
车辆 CIMG TCP 推 JPEG → PGX 0.0.0.0:30106
车辆 UDP vehicle_state → PGX 0.0.0.0:7001
车辆 TCP planned_route → PGX 0.0.0.0:30006
        ↓
4路相机×最近4帧 + 16步 ego 历史(10Hz)
        ↓
Alpamayo 推理 → CoT 文本 + 64 点未来轨迹
        ↓
车体系 x/y → WGS84 经纬度 → takeover
        ↓
UDP control_flag + 航点 → 车辆 168.130.182.10:7010
```

PGX 可以监听 `0.0.0.0` 或指定局域网地址。无人车必须填写 PGX 的实际局域网 IP，不能填写 `0.0.0.0`。

相机输入只有 CIMG：车辆把 4 路 JPEG 打成一条 TCP 流推到 `network.cimg`（默认 `0.0.0.0:30106`）。

模型使用官方提示词输出 CoT 与轨迹，不请求或解析模型控制位。
后端每个有效推理周期向车发送一个 UDP 包，control 字节由
`TakeoverDecisionEngine` 根据真实航线平均横向偏差和连续确认生成。
旧手工覆盖不再影响下发，航点协议未改变。

## 启动

生产由 `run_backend_real.sh` 在 PGX01 上启动 `alpamayo-pgx:1.0` 容器，容器内运行
`python app.py`。不需要在 PC 上启动中转、隧道或代理。

当前生产环境变量以 `run_backend_real.sh` 为准：`ALPAMAYO_POSE_MODE=udp`，
`ALPAMAYO_MOCK_MODE=0`，`ALPAMAYO_CIMG_CAMERA_INDICES=1,2,0,4`，
`ALPAMAYO_RESULT_MAX_AGE_MS=10000`，`ALPAMAYO_MAX_CAPTURE_LAG_MS=10000`。
相机接收端直接监听 `network.cimg`。推理用本机 GPU 0。mock 位姿是
`ALPAMAYO_POSE_MODE=mock`，没有本地视频回退。

## 常用配置

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `ALPAMAYO_CIMG_HOST` / `ALPAMAYO_CIMG_PORT` | `0.0.0.0:30106` | CIMG 监听地址，对应 `network.cimg` |
| `ALPAMAYO_CIMG_CAMERA_INDICES` | `1,2,0,4` | CIMG `camera_id` 1..4 对应的 Alpamayo index，顺序为前、右、左、后 |
| `ALPAMAYO_CIMG_READ_TIMEOUT_S` | `15` | 收不到完整包时断开，让车辆重连 |
| `ALPAMAYO_CIMG_MAX_PAYLOAD_BYTES` | 64MiB | 单个 CIMG 包上限 |
| `ALPAMAYO_CIMG_MAX_JPEG_BYTES` | 16MiB | 单张 JPEG 上限 |
| `ALPAMAYO_CIMG_QUEUE_FRAMES` | `60` | 每路相机缓冲帧数 |
| `ALPAMAYO_CIMG_MAX_BACKLOG_GROUPS` | `2` | 一次读取里允许积压的完整组数，超出只保留最新 |
| `ALPAMAYO_CIMG_GROUP_FPS` | `10` | 车辆推流组率，用来对齐采样网格 |
| `ALPAMAYO_FRAMES_PER_CAMERA` | `4` | 每路时间帧数 |
| `ALPAMAYO_TARGET_FPS` | `10` | 采样帧率 |
| `ALPAMAYO_POSE_MODE` | mock | `udp`/`mock` |
| `ALPAMAYO_POSE_UDP_PORT` | `7001` | 车辆状态入站端口（pose + chassis 同一 JSON，车辆直连 PGX） |
| `ALPAMAYO_CONTROL_FLAG` | — | 旧接口保留兼容，已不影响实际下发 |
| `ALPAMAYO_CONTROL_MODE` | — | 不读取。控制模式固定为真实航线偏差 |
| `ALPAMAYO_CONFIG` | `config/config.json` | 统一后端 JSON 配置文件路径 |
| `ALPAMAYO_TRAJECTORY_DT_S` | `0.1` | 模型轨迹相邻点时间间隔（秒） |
| `ALPAMAYO_TAKEOVER_THRESHOLD_M` | `0.8` | 接管横向平均偏差阈值（米，实车调参初始值） |
| `ALPAMAYO_RELEASE_THRESHOLD_M` | `0.4` | 释放横向平均偏差阈值（米，实车调参初始值，必须小于接管阈值） |
| `ALPAMAYO_TAKEOVER_CONFIRMATIONS` | `2` | 连续多少次超过阈值后接管 |
| `ALPAMAYO_RELEASE_CONFIRMATIONS` | `2` | 连续多少次低于阈值后释放 |
| `ALPAMAYO_TAKEOVER_MIN_REMAINING_POINTS` | `10` | 时间补偿后允许判断的最少有效轨迹点数 |
| `ALPAMAYO_ROUTE_TCP_PORT` | `30006` | 真实航线 TCP 接收端口 |
| `ALPAMAYO_ROUTE_MAX_AGE_S` | `30` | 航线缓存最大接收年龄（秒，初始值） |
| `ALPAMAYO_ROUTE_MAX_MATCH_DISTANCE_M` | `10` | 当前车辆到航线最大匹配距离（米，初始值） |
| `ALPAMAYO_WAYPOINT_HOST/PORT` | `168.130.182.10:7010` | 航点 UDP 目标，即无人车地址。不能是 `0.0.0.0`，也不能是已删除的中转 `127.0.0.1:7003` |
| `ALPAMAYO_WAYPOINT_MAGIC` | `0` | 正式 magic 到位前不发送（mock 除外） |
| `ALPAMAYO_VEHICLE_ID` | `0` | 正式 ID 到位前不发送（mock 除外） |
| `ALPAMAYO_CHECKPOINT` | 701 model_weights | Alpamayo 权重路径 |

轨迹偏差接管阈值配置示例（初始值需实车调参）。不要导出
`ALPAMAYO_CONTROL_MODE`，加载器不读它：

```bash
export ALPAMAYO_TAKEOVER_THRESHOLD_M=0.8
export ALPAMAYO_RELEASE_THRESHOLD_M=0.4
```

相机新鲜度（`camera_stale_ms`、`window_max_age_ms`、`frame_timeout_ms`）只来自
`config/config.json` 的 `camera_freshness`。`ALPAMAYO_CAMERA_STALE_MS`、
`ALPAMAYO_WINDOW_MAX_AGE_MS`、`ALPAMAYO_CAMERA_FRAME_TIMEOUT_MS` 不生效。

快照和推理包各保留最新 600 个目录（约 1 Hz 下 10 分钟），由 `config.json` 的
`retention` 配置。`ALPAMAYO_SNAPSHOT_KEEP=0` 或 `ALPAMAYO_INFER_PACKAGE_KEEP=0`
关闭对应轮转。

每轮判断会输出 `takeover decision:` 日志，并写入推理包的
`takeover_decision` 字段；`/status` 可查看 `last_takeover_decision`。

### 原规划航线接入

已按真实小端 TCP 协议接收航线。`trajectory/takeover.py` 提供协议无关的
`PlannedRoute`、`PlannedRouteAdapter`、线程安全 `PlannedRouteCache` 和
`PlannedRouteReferenceProvider.update_route()`。当前调用链：

```text
wire payload -> 协议专属 PlannedRouteAdapter -> PlannedRoute
             -> PipelineRunner.update_planned_route() -> PlannedRouteCache
             -> PlannedRouteReferenceProvider
```

参考来源固定为 planned_route，不读取旧 history pose 参考配置，不执行回退。
Provider 将经纬度转换到模型局部坐标，匹配当前车辆位置并按当前速度采样。
航线不足、缺失或无效不新增接管。完整配置、边界和测试见 [docs/PLANNED_ROUTE.md](docs/PLANNED_ROUTE.md)。

## 文档

本 README 只留运行入口；细节在 [docs/](docs/)：

| 文档 | 内容 |
|---|---|
| [docs/真实场景.md](docs/真实场景.md) | 实车端到端链路、数据格式、已确认项与已知断点 |
| [docs/PLANNED_ROUTE.md](docs/PLANNED_ROUTE.md) | 真实航线接管：协议、坐标转换、阈值与验证 |
| [docs/MODEL_TEST.md](docs/MODEL_TEST.md) | 离线模型测试入口 `test_backend:app` |
| [docs/无人车UDP位姿数据协议说明.md](docs/无人车UDP位姿数据协议说明.md) | 车辆端 `vehicle_state` 报文格式 |

文档里的 `tests/`、`tools/`、`config/` 路径都相对**仓库根目录**，命令请在根目录执行。

## 说明

- 入站 `vehicle_state` 已确认为 JSON；`chassis` 嵌在 `pose` 内
  （`pose.chassis`），后端从 7001 报文中一并解析。
- 模型 CoT 仅保留用于调试，不参与控制位判断；旧模型控制位模式已移除。
- 本目录文件为实时后端实现，不修改 `alpamayo1.5/src` 与 `alpasim` 上游。

## API

- `GET /health`：进程/管道状态
- `GET /status`：最近位姿/推理/标志位
