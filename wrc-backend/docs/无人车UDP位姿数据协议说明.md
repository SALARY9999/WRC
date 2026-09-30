# 无人车位姿及底盘数据

> **更新（2026-09-10）**
>
> - 实车抓包确认：报文中的 `chassis` 实际嵌在 `pose` 内部（`pose.chassis`），
>   不是与 `pose` 同级；解析端应同时兼容 `chassis` 和 `pose.chassis` 两种布局。
> - `azimuth` 已按实车核对：**0° = 正西，90° = 正北，180° = 正东，
>   270° = 正南，角度顺时针增大**。后端罗盘航向是
>   `(azimuth - 90 + azimuth_offset_deg) % 360`（0° = 正北，顺时针）。
>   不要使用 `90 - azimuth`：`azimuth = 0` 时那会得到正东，而实车是正西。
> - 顶层 `timestamp` 为 Unix 秒，用于与相机时间对齐；
>   `local_time` 为车辆本地时间，不要用于跨设备对齐。

## 1. 数据格式

无人车通过 UDP 协议发送车辆状态数据，数据格式为 JSON。

数据示例：

```json
{
  "type": "vehicle_state",
  "timestamp": 1788426445.204,
  "pose": {
    "latitude": 0.0,
    "longitude": 0.0,
    "height": 0.0,
    "azimuth": 89.956,
    "vNorth": 0.0,
    "vEast": 0.0,
    "local_time": 17,
    "chassis": {
      "cur_spd": 0.0,
      "yaw_rate": 0.0,
      "local_time": 0.0
    }
  }
}
```

（早期文档中曾把 `chassis` 写成 `pose` 的同级字段；实际车辆抓包为
`pose.chassis`，解析代码需要两种都兼容。）

## 2. 数据字段说明

### 2.1 顶层字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| type | string | 数据类型 |
| timestamp | double | 数据时间戳 |

## 3. 位姿数据（pose）

`pose` 包含车辆当前位置信息、航向信息以及速度信息。

| 字段 | 类型 | 单位 | 说明 |
| --- | --- | --- | --- |
| latitude | double | ° | 车辆当前纬度 |
| longitude | double | ° | 车辆当前经度 |
| height | double | m | 车辆当前高度 |
| azimuth | double | ° | 车辆当前航向角，**0°=正西，90°=正北，180°=正东，270°=正南，顺时针增大**。后端再换成罗盘航向（0°=正北，90°=正东） |
| vNorth | double | m/s | 北向速度 |
| vEast | double | m/s | 东向速度 |
| local_time | int | - | 本地时间 |

## 4. 底盘数据（chassis）

`chassis` 包含车辆底盘运动状态信息。

| 字段 | 类型 | 单位 | 说明 |
| --- | --- | --- | --- |
| cur_spd | double | m/s | 当前车辆速度 |
| yaw_rate | double | °/s | 横摆角速度（后端按角度制转换成弧度使用） |
| local_time | double | - | 本地时间 |

> 实际报文中 `chassis` 位于 `pose` 内部，即 `pose.chassis`；
> 解析端应优先查找顶层 `chassis`，不存在时再读取 `pose.chassis`。

## 5. 数据结构

```text
vehicle_state
│
├── type
├── timestamp
│
└── pose
    ├── latitude
    ├── longitude
    ├── height
    ├── azimuth
    ├── vNorth
    ├── vEast
    ├── local_time
    └── chassis          ← 实际抓包中的位置
        ├── cur_spd
        ├── yaw_rate
        └── local_time

（兼容旧文档格式：chassis 也可能出现在顶层 vehicle_state.chassis）
```

## 6. 时间戳与传输链路

| 项 | 说明 |
|---|---|
| `timestamp` | Unix 秒（double），车辆时钟，用于与相机 PTS 做绝对时间对齐 |
| `pose.local_time` | 车辆本地时间，非 Unix，不用于跨设备对齐 |
| `pose.chassis.local_time` | 底盘本地时间，非 Unix |
| 上行端口 | 车辆 → PGX UDP 7001（位姿，含 chassis） |
| 端口 | 只使用 UDP **7001**；旧的 7002 独立 chassis 通道已移除 |
| 接收 | PGX `UdpPoseListener` 直接监听 7001；chassis 从 `pose.chassis` 一并解析。不再经过 PC 中转 |

实际抓包示例（2026-09-03，源 `168.130.182.10:7000`）：

```json
{
  "type": "vehicle_state",
  "timestamp": 1788426435.404,
  "pose": {
    "latitude": 0.0,
    "longitude": 0.0,
    "height": 0.0,
    "azimuth": 89.956,
    "vNorth": 0.0,
    "vEast": 0.0,
    "local_time": 17,
    "chassis": {
      "cur_spd": 0.0,
      "yaw_rate": 0.0,
      "local_time": 0.0
    }
  }
}
```

## 7. 解析注意事项

1. 位姿解析只看 `pose.latitude/longitude/height/azimuth/vNorth/vEast`；
2. 底盘解析兼容 `pose.chassis`（当前实车格式）与顶层 `chassis`（旧文档格式），后端在 7001 报文中一并解析；
3. `azimuth` 先换成罗盘航向 `(azimuth - 90) % 360`（0=正北，90=正东，顺时针），不要做 `90 - azimuth`；
4. `vNorth/vEast` 仅表示运动速度，不用于替代车身航向；安装偏角由
   `config/config.json` 的 `vehicle_heading.azimuth_offset_deg` 统一配置；
5. 四路视频 PTS 与位姿 `timestamp` 的绝对对齐需要一个
   `Unix ↔ PTS` 锚点（车辆端后续提供正确时间后接入）。
6. 后端会把 7001 报文中的 `cur_spd / yaw_rate` 解析成 `ChassisInfo`
   （`yaw_rate` 从 °/s 转成 rad/s），并在 `/status` 中输出
   `chassis_speed_mps / chassis_yaw_rate_rad_s / chassis_packets`。
