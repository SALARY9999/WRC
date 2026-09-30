# 真实航线接管第一版

运行链路：车辆 TCP 30006 → VehiclePlannedRouteAdapter（小端，11 字节帧头，
每点 float32 经度/纬度共 8 字节）→ PlannedRouteCache →
PlannedRouteReferenceProvider → TrajectoryDeviationCalculator →
TakeoverDecisionEngine → 原有 WaypointSender。每帧替换整条航线；reached=1 清除缓存。

仅使用 planned_route，不再读取 ALPAMAYO_REFERENCE_SOURCE 或
ALPAMAYO_REFERENCE_HISTORY_POINTS；没有 history pose 回退。
模型输入历史 Pose 仍保留。旧手工 control flag/运行时覆盖入口不再控制实际下发。
实际下发始终来自 decision.control_flag。

坐标转换使用与 app.py 一致的短距离 ENU 近似和罗盘航向旋转，
以模型输入 Pose 为原点，x 向前、y 向左，单位米。双方经纬度基准已确认一致。
匹配当前车辆位置到有序航线的最近线段，然后按当前 GlobalPose 速度
沿航线累计弧长采样到补偿后的时间点。没有假设报文包含速度或航点时间。
重复点移除；不向终点之外外推。静止、退化、过短、过远、非有限值、
航线过期或缺失均拒绝新增接管。

延迟使用当前 Pose 时间轴加其单调时钟接收年龄，减模型输入 t0。
索引通过整数微秒除以 dt 微秒计算。默认 0.1 秒，每次至少剩余 10 点。

## 配置

配置在 app.py Settings/from_env 中，以下都是实车调参初始值：

- ALPAMAYO_TAKEOVER_THRESHOLD_M=0.8：平均横向偏差严格大于阈值。
- ALPAMAYO_RELEASE_THRESHOLD_M=0.4：平均横向偏差严格小于阈值。
- ALPAMAYO_TAKEOVER_CONFIRMATIONS=2 / ALPAMAYO_RELEASE_CONFIRMATIONS=2。
- ALPAMAYO_TRAJECTORY_DT_S=0.1 / ALPAMAYO_TAKEOVER_MIN_REMAINING_POINTS=10。
- ALPAMAYO_ROUTE_MAX_AGE_S=30：从服务器收到完整航线起计时；
  若未来只发送一次完整任务航线，应依据任务时长调整，而非假设它持续更新。
- ALPAMAYO_ROUTE_MAX_MATCH_DISTANCE_M=10：车辆离航线过远拒绝匹配。
- ALPAMAYO_ROUTE_TCP_HOST=0.0.0.0 / ALPAMAYO_ROUTE_TCP_PORT=30006。

run_backend_real.sh 启用航线接收。启动前确认独立 receive_route_test.py 已退出，
否则会占用同一端口；绑定失败显示在 /status，缺失航线不新增接管。

## 判断与验证

阈值之间是滞回区，接管和释放各要求连续两次有效推理。
无效数据重置确认计数，保持已有接管状态，不新增接管；并不自动释放
已经接管的车辆。发送端协议不变，仍发送原轨迹，模型预测中已经过期的点
仅在偏差计算时裁剪，原有航点下发格式/流程没有更改。

日志 INFO 搜索 `takeover decision:`，包含 control_flag、mean/max_lat_error、
确认计数、reason、delay、start_index、剩余点数和 reference_source。
无有效参考时 reference_fallback_reason 字段记录失败原因，但不执行回退。
输入在推理前被 freshness gate 拒绝时看 freshness gate 日志。

在仓库根目录（本文件上一级的 backend 目录）运行，使用已安装 numpy 的 Python。
`-s tests` 相对根目录解析，在 `docs/` 里执行会找不到测试模块：

```sh
cd /data/users_data/wanghaohao/701/alpamayo_infer_backend    # 或本机 F:\wrc\wrc-backend
python -m unittest discover -s tests -p 'test_planned_route*.py' -v
python -m unittest discover -s tests -p test_trajectory_takeover.py -v
```

涵盖 TCP 分包/粘包/重连/清除、坐标旋转、直行不接管、单次偏离不接管、
连续两次偏离接管、两次恢复释放、stale Pose/Camera、缺失航线、过期模型、NaN。
backend 控制方法通过 AST 原样加载测试，不需要加载 GPU/摄像头或给实车发控制。

尚需实车/回放确认：GPS float32 量化（可能米级）、Pose 误差、速度单位、
弯道恒速近似、自交或往返航线的最近段匹配歧义。第一版不处理复杂任务匹配。
此前真实两点样例仅约 0.84 米，行驶时一般不足以覆盖未来预测，不应强行外推。
通过离线测试不代表已验证实车安全；上线前先记录轨迹和偏差进行回放调参。
