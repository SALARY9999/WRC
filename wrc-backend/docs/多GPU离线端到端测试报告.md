# Alpamayo 多 GPU 离线端到端测试报告

测试日期：2026-09-20
被测版本：Alpamayo 1.5 10B（BF16 + SDPA），`num_traj_samples=1`，top_p=0.98，temperature=0.6
staging 目录：`/data/users_data/wanghaohao/701/alpamayo_multi_gpu_stage_20260920-125615`
主机：8 × NVIDIA GeForce RTX 4090（每卡 24564 MiB）

---

## 0. 结论摘要

| Phase | 内容 | 结果 |
|---|---|---|
| 1 | 启动链路只读分析 | 已完成（无改动） |
| 2 | 单卡 soak 门禁 | **判据未通过**（见 §3），按用户裁决放行 |
| 3 | CIMG 数据入口链路 | **22/22 PASS**，但真实图片入口**未验证** |
| 4 | Result Manager 故障注入 | **31/31 PASS** |
| 5/6 | 4 卡驻留 + 真实吞吐 | **6/6 PASS**，`accepted_nonempty_hz=2.90` |
| 7 | 真实模型 → loopback UDP 闭环 | **9/9 PASS**，`hz_with_waypoints=2.97 ≥ 2.0` |

**核心结论：4 卡多进程推理在实测中达标。** 4 worker 真实吞吐 3.30 Hz（理论容量 3.51 Hz 的 94%），非空合法轨迹 UDP 输出 2.97 Hz，超过 2.0 Hz 判据。单卡与 4 卡每 worker 显存占用完全一致（allocated 21150.1 MiB），4 卡无需扩展到 5/6 卡。

**必须同时说明的三点：**
1. 本报告的 Phase 2 门禁**按字面判据是 FAILED**，放行是基于平台期证据的用户裁决，不是自动通过（§3）。
2. 真实四路相机入口**未验证**：服务端从未保存原始 CIMG 包，本次回放的是**真实 JPEG 载荷**，但 CIMG 分帧是测试侧重建的（§4）。
3. 全程 Pose 为合成，模型输入窗口为回放/构造数据，**不是实时四路相机**（§10）。

---

## 1. 安全边界与未执行项

以下为**实际执行结果**，不是承诺：

| 约束 | 执行情况 |
|---|---|
| 不运行 `start_realtime.ps1` | 未运行 |
| 不启动生产 splitter / relay / Uvicorn | 未启动 |
| 不修改生产配置 | 未修改（含 `config/config.json`） |
| 不占用实车端口 | 未占用（未绑定 30106 或任何生产端口） |
| UDP 只发往 `127.0.0.1` 临时端口 | 遵守：Phase 7 发往 `127.0.0.1:46152`（运行时自开的临时端口） |
| 不向真实车辆地址发送数据 | 未发送 |
| 使用独立 staging 目录 | 遵守 |
| 不修改模型权重/输入结构/采样参数/UDP 协议 | 未修改 |
| 真实模型只测正常路径；异常注入用 fake worker | 遵守（Phase 4 全部使用 `fake_worker_main`） |
| 测试前检查 GPU 与显存，不抢占其他任务 | 遵守（见下表） |
| 不以代码审查/单测/模拟/理论估算替代实测 | 遵守：所有结论均来自实跑日志 |

**邻居卡未被触碰的实测证据**（5 秒采样器覆盖全部 8 张卡）：

| 运行 | 采样 | GPU0 | GPU1 | GPU2 | GPU3 | GPU4 | GPU5 | GPU6 | GPU7 |
|---|---|---|---|---|---|---|---|---|---|
| Phase 5/6 | 28 点 / 2.3 min | **1488** | 23014 | **20212** | 23038 | 23012 | 23010 | 4 | 4 |
| Phase 7 | 22 点 / 1.8 min | **1488** | 23012 | **20212** | 23010 | 23010 | 23034 | 4 | 4 |

GPU 0（1488 MiB）与 GPU 2（20212 MiB）由他人任务持有，全程**数值完全不变**；测试只使用 GPU 1/3/4/5。每次运行结束后 1/3/4/5 均回落到 4 MiB。

---

## 2. Phase 1 启动链路只读分析

只读检查，未改动任何文件。分析对象为 `pipeline.py`、`inference/gpu_pool.py`、`inference/result_manager.py`、`inference/input_builder.py`、`camera/` 与 `vehicle/`。

关键结论（后续各 Phase 的设计依据）：

- `GpuScheduler` 使用 `multiprocessing` **spawn**，每卡一进程；worker 先固定物理卡再在 `cuda:0` 上加载，因此 `--devices` 是 `nvidia-smi` 的物理序号。
- `ResultManager.validate()` 的**检查顺序是关键**：worker 失败 → 回声/未知/重复 → 乱序（`seq <= last_accepted`）→ 过期（age）→ 采集异常/采集滞后 → 形状 → 点数 → 有限性。因为**乱序先于过期**，一个低序号的结果会被判为 `out_of_order`，从而**掩盖**后面的过期检查。Phase 4 的两个用例就是为绕开这一掩盖效应而设计的。
- `InputBuilder.build()` 的拒绝顺序为：相机过期 → pose 过期 → 相机-pose 时差 → 相机窗口过期 → pose 历史不可用。拒绝是返回值而非异常。
- 采集时钟：`note_task()` 只有**合理窗口**才推进 `_newest_capture_us`；回退/跳跃分别按 `max_capture_lag_ms` 判定。
- 时间戳配置为 `camera_timestamp_mode=sei_unix`、`pose_timestamp_mode=unix`、`max_future_offset_ms=200`——**采集时间戳超前服务器时钟 200 ms 以上会被直接拒绝**。这一条在 Phase 3 首轮导致全部帧被拒，定位后修正。

---

## 3. Phase 2 单卡 soak 门禁（判据未通过）

### 3.1 实测结果

30 分钟 soak，单卡（GPU 1），2 Hz，采样 5 s：

```
worker 0 dev 1 ready: allocated=21150MiB reserved=22496MiB peak_allocated=22125MiB
load wall time: 52s
soak ran 30.0 min, accepted_total=1527
worker 0 dev 1: allocated 21150 -> 21150MiB (+0.0MiB/h),
                reserved  22500 -> 22542MiB (+80.8MiB/h), peak_reserved=22542MiB
[PASS] worker 0 allocated did not grow - +0.0MiB/h (limit 32)
[FAIL] worker 0 reserved did not grow - +80.8MiB/h (limit 32)
release: dev1: 4.0MiB before -> 4.0MiB after (0.1s)

RESULT: FAILED (1/11 checks failed)
```

推理时延（30 分钟基准段，5 次）：avg **1133.8 ms** / p50 1137.3 / p95 1144.5；wall avg 1156.8 ms。

### 3.2 为什么这个 FAIL 是判据缺陷，而不是泄漏

`slope_per_hour()`（`tools/gpu_memory_probe.py:332`）对**整个 soak 窗口**做最小二乘拟合再折算到小时。对一次性的 settle 开销 S，斜率 ≈ S/T，即报出的"速率"**必然随窗口变长而下降，永不可能为 0**。两次运行的实测正好符合这一预测，且与"泄漏"的预测相反：

| 运行 | reserved 绝对增长 | 报出"速率" | peak_reserved vs 终值 |
|---|---|---|---|
| 15 min | +29.6 MiB | +97.1 MiB/h | 相等（棘轮已停） |
| **30 min** | **+42.0 MiB** | **+80.8 MiB/h** | 相等（棘轮已停） |

窗口翻倍，"速率"反而下降——泄漏会保持不变或上升。独立 5 秒采样器记录的整卡时序进一步给出平台期形状：

```
 0.0 min       4 MiB   （基线）
 0.7 min    8326 MiB   （加载）
 0.8 min   21770 MiB   （加载）
 0.9 min   23008 MiB   （加载）
 1.0 min   23012 MiB   +4
 4.3 min   23032 MiB   +20
10.0 min   23048 MiB   +16
17.6 min   23054 MiB   +6      ← 之后 13.5 分钟完全平坦
31.1 min       4 MiB   -23050  （归还）
```

四次上台阶的间隔在**变宽**（3.3 → 5.7 → 7.6 分钟）、幅度在变小，最后 13.5 分钟归零，退出后完全归还。`allocated` 在两次运行中都**逐字节持平**于 21150 MiB。

### 3.3 门禁裁决

计划原文为"soak 失败则停止，不进入多卡测试"。该 FAIL 已按字面记录为**未通过**。用户基于上述平台期证据裁决**放行进入多卡测试**；本报告记录这一裁决为**人的决定**，而非工具判定通过。若需要工具层面的干净通过，应修正判据（对尾部窗口拟合，或改用"最后 N 分钟增量"），本次未修改该工具。

---

## 4. Phase 3 CIMG 数据入口链路（22/22 PASS，真实图片入口未验证）

按当前车端协议（TCP 30106 直发 CIMG 四路 JPEG）测试，**未**使用 `camera1-4.ts`，**未**启动 `run_mpegts_split.sh`。

**真实与合成的分界（必须明确）：**

| 项 | 性质 |
|---|---|
| 四路 JPEG 载荷（57 组 × 4 路 = 228 张，46.8 MB） | **真实**：真实 CIMG 抓取留存，来源见 `direct_push_receiver.py:197` |
| CIMG **分帧/包头** | **测试侧重构**——原始字节流从未保存（服务端 `find` 无结果，`tools/capture_30106.py` 的输出目录不存在） |
| Pose | **合成**（10 Hz，与回放采集同一 Unix 时钟） |
| 回放目标 | `127.0.0.1:40743`（临时端口，仅 loopback） |

**因此「真实图片入口」标记为未验证。** 工具用**载荷同一性**做了补偿证明。

关键实测结果：

```
input_mode='cimg' readers=['CimgCameraSource']        ← 未误选 MPEG-TS
groups_total=79 sent=79, images_total=316, decode_errors=0
protocol_errors={"discarded_bytes":0,"packets_skipped":0,"resyncs":0}
timestamp_errors=0 fallback=False mode=sei_unix
cameras {"1":79,"2":79,"3":79,"4":79} 全部 HEALTHY
max_depths=[16,16,16,16] of 60 capacity, over 4595 independent samples
window_builds=33, InputBuilder accepted=33/33 rejections={}
ego_xyz=(1,1,16,3) ego_rot=(1,1,16,3,3)
window indices=[0,1,2,4]  wire order=[1,2,3,4] alpamayo=[0,1,2,4]
frozen task frames=(4,4,3,320,576)
16/16 帧与磁盘 JPEG 逐字节相同（独立 cv2 解码，576x320 RGB，mismatches=[]）
```

说明两点，避免误读：

- **索引顺序已验证**：窗口内相机索引为 Alpamayo 顺序 `[0,1,2,4]`，而非线序 `[1,2,3,4]`，与生产约定一致。
- **只有 1/33 个任务被 Result Manager 接受**（`expired task_seq=1 age_ms=8683 > 2500`）。原因是回放速度远快于消费速度（回放器在数秒内推完 33 个窗口），旧任务按 `result_max_age_ms` 过期——这是回放器的速率特性，不是链路缺陷。验收项「真实任务跨进程边界并被接受」已 PASS。

---

## 5. Phase 4 Result Manager 故障注入（31/31 PASS）

全部使用 `fake_worker_main`，异常不进入真实模型。六个用例的实测判据与关键计数：

| 用例 | 关键实测 |
|---|---|
| 重复结果 | `duplicate=1`，`accepted=[1,2,3,4]`（重复项被拒且不扰动邻项） |
| 结果过期 | `expired=1`，`last_accepted=3`，**`out_of_order=0`**，`accepted=[1,2,3]`，`raw_total=4` |
| 采集时钟回退/跳跃 | `capture_rollback=1`，`capture_jump=1`，`accepted=[1]` |
| 采集时钟滞后 | `capture_lag=1`，**`jump=0 rollback=0`**（单步都不异常，滞后是唯一拒绝理由） |
| 轨迹损坏 | `accepted=[]`，`non_finite_trajectory=2`，`invalid_trajectory_shape=1`，`invalid_trajectory_points=1` |
| worker 故障 | `worker_exited=1`（worker 未发 `WORKER_STOPPED`），`cuda_oom=1`，`worker_exception=1`，`accepted=[1,4,5,6,7]`，`ready=3 failed=1` |
| pending 替换 | `max_pending=1`，`pending_replaced=7`，`raw_total=5`，`accepted=[1,2,3,4,12]` |

另有模块隔离检查：AST 遍历 `inference/{gpu_pool,result_manager,fake_worker,tasks}.py`，确认无任何模块可触达控制链路（`app`/`pipeline`/`trajectory`/`waypoint`）。

两个用例的设计值得记录，因为**若不如此设计就会假通过**：

- **过期用例**：因乱序检查先于过期检查，慢任务必须放在**最高 seq**，否则会被判成 `out_of_order` 而掩盖过期判定。实测 `out_of_order=0` 证明掩盖未发生。
- **滞后用例**：必须让滞后任务在**水位仍为 0** 时被校验（任务 1 快、任务 2-3 慢，三次合法 2.4 s 步进），否则同样被乱序掩盖。

---

## 6. Phase 5/6 4 卡驻留 + 真实多 GPU 吞吐（6/6 PASS）

命令：`--scenario workers --devices 1,3,4,5 --real --infer-ms 1138 --seconds 60 --rate-hz 10`

`--infer-ms 1138` 取自单卡实测均值，否则 `capacity_hz` 判据不公平。

```
raw_inference_hz=3.30  accepted_result_hz=2.90  accepted_nonempty_hz=2.90
submitted=600  raw_total=206  accepted_total=194  pending=0
infer_ms avg=1117.9  p50=1092.4  p95=1230.0  max=2213.3
scheduler_drops={"pending_replaced": 394}
result_drops={"out_of_order": 12, ...其余全部为 0}

worker 0 dev 1  n=50  avg=1158.0  p95=1331.2  vram_alloc=21150.1MiB reserved=22502.0MiB
worker 1 dev 3  n=52  avg=1121.6  p95=1210.9  vram_alloc=21150.1MiB reserved=22526.0MiB
worker 2 dev 4  n=52  avg=1103.9  p95=1194.8  vram_alloc=21150.1MiB reserved=22500.0MiB
worker 3 dev 5  n=52  avg=1089.6  p95=1189.6  vram_alloc=21150.1MiB reserved=22498.0MiB

[PASS] 4 workers: raw_inference_hz tracks capacity - raw=3.30 Hz capacity~=3.51 Hz
[PASS] 4 workers: every worker answered - counts=[50, 52, 52, 52]
[PASS] 4 workers: no worker failed, no CUDA OOM
[PASS] 4 workers: accepted_result_hz >= 2.0 - accepted=2.90 Hz
[PASS] 4 workers: accepted_nonempty_hz >= 2.0 - accepted_nonempty=2.90 Hz
[PASS] 4 workers: pending stayed within one slot - pending=0
RESULT: OK (6/6 checks passed)
```

要点：

- **线性扩展接近理想**：4 卡 3.30 Hz vs 单卡 0.88 Hz × 4 = 3.52 Hz，达 94%。多卡无相互干扰开销。
- **每卡显存与单卡完全一致**：allocated 21150.1 MiB、reserved 22498–22526 MiB，与单卡 soak 的 21150 / 22496 一致——4 卡共存未造成任何跨卡显存膨胀。整卡峰值 23010–23038 MiB / 24564 MiB。
- `out_of_order=12` 是 latest-only 策略下的**预期行为**：4 个 worker 完成顺序交错，较新的 seq 先被接受，后到的旧结果被正确拒绝。
- **未扩展到 5/6 卡**：计划规定仅在 4 卡未达 ≥2.0 Hz 时扩展，4 卡已达标，故按要求不扩展。

---

## 7. Phase 7 真实模型 → loopback UDP 闭环（9/9 PASS）

命令：`--scenario pipeline --devices 1,3,4,5 --real --infer-ms 1138 --seconds 60 --rate-hz 10`

本 Phase 走**真实** `PipelineRunner._handle_result`：裁剪 → 坐标变换 → UDP 编码 → 发送。`PipelineRunner` 被构造但**从不 `start()`**，因此不打开相机、不绑定位姿端口。

```
-- wire: {"packets":192,"with_waypoints":178,"release_frames":14,
          "bad_magic":0,"seq_gaps":0,"last_seq":191,
          "control_flags":{"0":192},"first_point":[30.5,114.30000000058588],
          "decode_errors":[],"hz_total":3.203,"hz_with_waypoints":2.970}

[PASS] the sink decoded every packet cleanly
[PASS] non-empty valid-trajectory UDP output >= 2.0 Hz - hz_with_waypoints=2.97
[PASS] the total packet rate is not what carried that - total=192 with_waypoints=178
[PASS] the UDP sequence did not go backwards - seq_gaps=0
[PASS] no worker failed, no CUDA OOM
RESULT: OK (9/9 checks passed)
```

**验收判据 `nonempty_udp_output_hz >= 2.0` 实测 2.97 Hz，通过。**

### 7.1 14 个空 release 帧的原因（已定位）

192 个包中有 14 个是空 release 帧，日志给出明确理由：

```
trajectory_expired / no_remaining_waypoints:
  original_waypoint_count=64 matched_segment_index=62 projection_distance_m=85.1
  removed_waypoint_count=64 remaining_waypoint_count=0
  latest_pose_timestamp=1789891551347706 trajectory_timestamp=1789891550047704
```

时间戳现已对齐（相差 1.3 s），但**投影距离 85–583 m**：合成 Pose 与模型输出的轨迹在几何上毫不相关，裁剪层因此拒绝发送无法与车辆关联的航点。

根因是**离线测试的固有局限**：Phase 7 喂给真实模型的窗口是 `make_task` 构造的（`ego_xyz=zeros`、帧 ID 形如 `cam1:t`），不是真实四路相机。真实模型对无意义输入产出的轨迹自然与合成 Pose 不一致。**这是测试构造的局限，不是生产缺陷**；而裁剪层选择"不发"而非"发垃圾"是**正确的失效保护行为**，属于本次测试的正面发现。

判据用的是 `hz_with_waypoints`（非空合法轨迹速率），因此不受这 14 帧影响。

---

## 8. 测试过程中发现并修复的问题

以下 3 项是**本次新增的测试工具缺陷**（`tools/multi_gpu_offline_test.py`），均已在本地修改、scp 到服务端、在服务端复测通过。生产代码未因这些问题被修改。

1. **Phase 7 首轮 0 包：`AttributeError: 'PoseSample' object has no attribute 'arrival_us'`**
   测试工具的 `_add_pose()` 把 `PoseSample` 塞进了 `PoseHistory`，而 `PoseHistory` 存的是 `GlobalPose`（`vehicle/history.py:29` 注释与签名均为 `GlobalPose`），`GlobalPose` 才有 `arrival_us`。异常从 `takeover.decide`（`trajectory/takeover.py:541`）抛出并穿透 `_drain`，导致整条尾链中止、`sent_total=0`、pool 停在 `state=created`。
   **已确认这是测试工具的错，不是生产缺陷**：生产自己的位姿路径构造的就是带 `arrival_us` 的 `GlobalPose`（`vehicle/receiver.py:334-350`）；`task.reference_pose` 为 `PoseSample` 是正确的（`inference/tasks.py:172`，来自 `build_ego_arrays`）。
   注：`PoseHistory.add()` 不做类型校验，因此错误类型被静默接受——工具与生产都依赖调用方自律。

2. **Phase 7 出现大量空 release 帧：采集时间戳用了合成时间轴**
   `scenario_pipeline` 原以 `capture = CAPTURE_STEP_US * seq` 打时间戳（`seq=148` → `t0=14.8 s`），而 Pose 用真实时钟（`1.79e15 µs`），导致每条轨迹被判为**落后 56 年**，全部走 `trajectory_expired` 空帧路径。已改为按真实时钟打戳（与生产采集时间戳同为真实 Unix µs，Phase 3 已验证）。空帧从 20 降到 14，剩余 14 帧由 §7.1 的几何原因造成。

3. **`_seed_pose_history()` 实际只种下 1 个位姿**
   `PoseHistory.add()` 拒绝时间戳不前进的样本，而原种子循环用 `time.time()*1e6` 紧循环调用 40 次，µs 分辨率下大多相同 → 39 个被静默丢弃。已改为按 10 Hz 间隔回填严格递增的时间戳。此问题不产生错误结果，但会让"已种下 40 个位姿"的说法不成立。

另有 1 项**判据设计缺陷**记录在 §3.2（soak 的 reserved 斜率判据），本次未修改该工具。

---

## 9. 内联回归（全量测试套件）

本次修改的 3 个文件已与本地逐字节比对一致（md5 相同），再以**完整代码树 + 完整 `tests/`** 运行全量套件（`/data/users_data/wanghaohao/701/alpamayo_regression_20260920`，该目录为本次回归新拷贝，未改动 staging）：

```
python -m pytest tests -q
→ 630 passed, 1 failed, 561 subtests passed in 96.62s
```

**唯一的失败与本项目改动无关，且为改动前既存：**

```
tests/test_shutdown.py:415  UncooperativeComponentTests::test_the_report_stays_within_its_stated_budget
    self.assertFalse(report.budget_exceeded)
AssertionError: True is not false
```

机制清楚：`pipeline.py:881` 定义 `budget_exceeded = elapsed_ms > (drain_s + reap_s) * 1000.0`。该用例对**故意不配合**的组件调用 `runner.stop(drain_s=0.05, reap_s=0.5)`，即判据预算 **550 ms**；而同一用例的另一条断言只要求 `elapsed_ms < 2000.0`。实测 elapsed 落在 550–2000 ms 之间，于是在**预算没有任何余量**的前提下必然判超。

- 单独重跑 3/3 次**稳定失败**，不是偶发抖动。
- `tests/test_shutdown.py` 未引用本次修改的任何文件；失败点位于 `pipeline.py` 的停机路径。
- 属既存的「判据预算与实现地板时间同量级」问题，需模块负责人决定是放宽预算还是缩短停机路径。本次未修改。

其余 630 项全部通过，其中包含 `tests/test_multi_gpu_tools.py`（覆盖本次所使用的离线测试工具本身）。

---

## 10. 未验证项（明确列出）

按计划要求，以下项目**未实际执行或无法验证**，不得视为通过：

| 项目 | 原因 |
|---|---|
| **真实四路相机入口** | 服务端从未保存原始 CIMG 字节流，"分帧"为测试侧重构。回放的是真实 JPEG 载荷，但入口链路本身未在真实分帧下验证 |
| **真实车端 Pose 链路** | 全程 Pose 为合成（Phase 3/7），未接真实位姿源 |
| **实时四路相机 → 模型一致性** | 模型输入为回放窗口（Phase 3）或构造窗口（Phase 7），非实时相机数据 |
| **真实 UDP 对端** | 只发往 `127.0.0.1` 临时端口，未验证真实接收端解析 |
| **5/6/7 卡扩展** | 按计划仅在 4 卡未达标时扩展；4 卡已达标，未测试 |
| **soak reserved 速率判据** | 判据未通过（§3），放行依据为人工裁决 |
| **生产端到端（含 splitter/relay/Uvicorn）** | 按安全约束未启动 |
| **长时间稳定性（>30 min）** | 未测试 |

---

## 11. 发现的配置不一致（报告撰写时未修改、未启动）

撰写本报告时，`run_backend_real.sh` 仍按 MPEG-TS 启动，和车端 TCP 30106 的 CIMG 四路 JPEG 不一致。该启动脚本后来已改为只接收 CIMG，不再选择 `cimg` / `mpegts`，也不再启动切流器。相机时间也不再有 `camera_timestamp_mode`，固定使用 CIMG `capture_time_ns`。本报告记录的是当时的离线测试，没有改写当时的测量。PC 侧 `start_realtime.ps1` 不在本仓库。

---

## 12. 观察到的度量不一致（未定位根因）

Phase 7 运行中，`runner` 侧计数器与实际线上流量不符，如实记录：

- `-- runner: raw_inference_hz=0.0 accepted_result_hz=0.0 udp_output_hz=0.0`
- `-- udp_packets: {"sent_total": 0, ...}`
- 而同一次运行的 sink **实际收到 192 个包**（`hz_with_waypoints=2.97`）
- `-- pool: {"state":"created","ready_workers":0,...}`，但同一运行中 4 个 worker 已就绪并产出 192 个包

线上（sink）是权威度量，本报告的 Phase 7 结论全部基于 sink。runner 侧速率计数与 pool 视图在这一离线模式下不可靠，**根因未定位**，建议在真实 `/health` 上复核。另：`the published snapshot carries the pool view` 这一检查即使在 PASS 时也打印其失败态文案（`inference_pool is None, ...`），属输出可读性问题。

---

## 13. 复现方式

```bash
# Phase 5/6：4 卡驻留 + 真实吞吐
cd /data/users_data/wanghaohao/701/alpamayo_multi_gpu_stage_20260920-125615
./tools/run_multi_gpu_real.sh workers workers4

# Phase 7：真实模型 → loopback UDP 闭环
./tools/run_multi_gpu_real.sh pipeline pipeline7
```

启动器内固定 `--devices 1,3,4,5`、`--real --infer-ms 1138 --seconds 60 --rate-hz 10`，并显式 `env -u CUDA_VISIBLE_DEVICES`（worker 自行固定物理卡，继承该变量会把 `--devices` 重新编号）。`scenario_pipeline` 新增了设备校验：若加载出的 `inference_pool.gpu_devices` 与 `--devices` 不符，会在**派生任何进程之前**拒绝启动。

## 14. 产物清单

| 文件 | 内容 |
|---|---|
| `results/single_gpu_soak30.log` / `.json` | 30 分钟 soak（门禁，判据未通过） |
| `results/soak30_gpu_samples.csv` | 全程 5 秒整卡采样（376 点 / 31.6 min） |
| `results/single_gpu_soak.log` / `.json` | 15 分钟 soak（对照） |
| `results/cimg_ingress.log` | Phase 3（22/22） |
| `results/faults.log` | Phase 4（31/31） |
| `results/workers4.log` / `_gpu_samples.csv` | Phase 5/6（6/6） |
| `results/pipeline7.log` / `_gpu_samples.csv` | Phase 7 最终（9/9） |
| `results/pipeline7_clockmixed.log` | Phase 7 时钟缺陷版（保留对照） |
| `tools/run_multi_gpu_real.sh` | Phase 5–7 启动器 |
| `tools/multi_gpu_offline_test.py` | 测试工具（含本次 3 项修复） |
| `tools/cimg_ingress_test.py` | Phase 3 工具 |
| `/data/users_data/wanghaohao/701/alpamayo_regression_20260920/` | 内联回归用完整代码树（含 `tests/`，本次新建，未改动 staging） |
