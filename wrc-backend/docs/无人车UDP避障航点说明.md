# 无人车避障 UDP 接口字段说明

## 1. UDP 数据结构

| 偏移 | 字段 | 类型 | 长度 | 说明 |
|---:|---|---|---:|---|
| 0 | magic | uint32 | 4 Byte | 固定数据包标识：`0x514D3031` |
| 4 | version | uint8 | 1 Byte | 协议版本号 |
| 5 | control_flag | uint8 | 1 Byte | 控制权标志位，`0`：不接管，`1`：接管 |
| 6 | seq | uint32 | 4 Byte | 数据包序号 |
| 10 | vehicle_id | uint16 | 2 Byte | 固定无人车编号：`1` |
| 12 | num_waypoints | uint16 | 2 Byte | 航点数量 |
| 14 | points[] | Waypoint[] | 16 × N Byte | 航点数据 |

多字节字段都是小端（little-endian）。报头是 `<IBBIHH`（magic、version、control_flag、seq、vehicle_id、num_waypoints），每个航点是 `<dd`（latitude、longitude）。

## 2. 航点数据

每个航点包含经纬度：

```c
struct Waypoint {
    double latitude;    // 纬度
    double longitude;   // 经度
};
```

| 字段 | 类型 | 长度 | 说明 |
|---|---|---:|---|
| latitude | double | 8 Byte | 纬度 |
| longitude | double | 8 Byte | 经度 |

单个航点共 **16 Byte**。

航点按照数组顺序排列：

```text
points[0] → 第 1 个航点
points[1] → 第 2 个航点
...
points[N-1] → 第 N 个航点
```

## 3. control_flag

| 值 | 含义 |
|---:|---|
| `0` | 不接管，无人车按照指挥所原航线行驶 |
| `1` | 接管，无人车按照我方下发的避障航点行驶 |

避障完成后，将 `control_flag` 设置为 `0`，表示释放控制权。

## 4. 字段说明

### magic

- 类型：`uint32`
- 长度：4 Byte
- 用途：用于识别当前数据包是否属于本协议。
- 约定值：`0x514D3031`
- 发送端必须填写该固定值，接收端应校验该值，不匹配的数据包不应按本协议解析。

### version

- 类型：`uint8`
- 长度：1 Byte
- 用途：协议版本号。
- 当前版本：`1`

### control_flag

- 类型：`uint8`
- 长度：1 Byte
- 用途：表示当前是否由我方系统接管无人车控制权。
- `0`：不接管
- `1`：接管

### seq

- 类型：`uint32`
- 长度：4 Byte
- 用途：数据包序号，每发送一个数据包递增。

### vehicle_id

- 类型：`uint16`
- 长度：2 Byte
- 用途：标识目标无人车。
- 约定值：`1`
- 当前接口下发的避障航点均面向车辆 `1`。

### num_waypoints

- 类型：`uint16`
- 长度：2 Byte
- 用途：表示当前数据包包含的航点数量。

### points[]

- 类型：`Waypoint[]`
- 长度：`16 × N Byte`
- 用途：存放无人车需要执行的航点。
- 每个航点包含纬度和经度。
