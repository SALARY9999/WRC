# Alpamayo 1.5 多 GPU 多实例负载均衡推理改造说明

本文是本次改造的交付说明。**本地代码、单元测试与离线测试已完成；未部署实车、未启动实车
UDP 控制链路、未上传服务器。** 所有需要在 GPU 机器上执行的测量，本文明确标注为"待执行"，
并给出可直接复制的命令与需要记录的字段。

改造前的实时链路（相机/Pose 接收与同步 → InputBuilder → 单实例推理 → 后处理/接管状态机/
坐标转换 → UDP）保持不变；本次只把中间"一次只能跑一个推理"这一段换成可横向扩展的池。

---

## 1. 修改与新增文件

### 新增（推理池本体）

| 文件 | 作用 |
| --- | --- |
| `inference/tasks.py` | `InferenceTask`（不可变、完整输入快照）、`InferenceResult`、`WorkerStatus`、`GpuMemory` |
| `inference/gpu_pool.py` | `GpuScheduler`（worker 生命周期、pending 槽、结果回收）、`worker_main`、GPU 探测与启动校验 |
| `inference/result_manager.py` | 结果校验与水印，`if result.task_seq <= last_accepted_task_seq: drop_as_out_of_order()` |
| `inference/metrics.py` | 三个频率、UDP 计数、丢弃/故障计数、延迟统计 |
| `inference/fake_worker.py` | 不加载模型、不碰 CUDA 的同签名 worker，供无 GPU 环境确定性测试 |
| `tools/multi_gpu_offline_test.py` | 离线负载均衡测试（三种场景；`scheduler` 模式不需要 GPU） |
| `tools/gpu_memory_probe.py` | 分阶段 VRAM 测量与单卡验收：load / warmup / bench / soak / release（见下） |
| `tests/test_tasks.py`、`tests/test_result_manager.py`、`tests/test_inference_metrics.py`、`tests/test_inference_scheduler.py`、`tests/test_health_pool_view.py`、`tests/test_multi_gpu_tools.py`、`tests/_fake_pool.py` | 对应的单元测试与测试替身 |

### 修改

| 文件 | 改动 |
| --- | --- |
| `pipeline.py` | `run()` 拆成 `_run_inline()` / `_run_multi_process()`；抽出 `_acquire_window()` 与 `_handle_result()`（单写入者的后处理尾部）；池的启动/停止、快照发布 |
| `health.py` | 新增 `InferencePoolView` / `InferenceWorkerView`；`model_loaded` 判断改为按模式；`/health`、`/status` 末尾追加 `inference_pool`、`inference_metrics` |
| `config/loader.py` | `InferencePoolConfig`（JSON + 环境变量 + 默认值，默认 `inline`） |
| `config/config.json` | 新增 `inference_pool` 段，默认 `mode=inline`、`gpu_devices=[]` |
| `inference/alpamayo.py` | `AlpamayoRunner(settings, device=...)`，去掉硬编码 `"cuda"` |
| `app.py` | 启动日志新增 `[inference_pool]` 段与 `min_cycle_s=... INERT(...)` 标记 |
| `tests/_source_tree.py`、`tests/test_waypoint_trimming.py`、`tests/test_replay_tool.py`、`tests/test_model_control_removed.py`、`tests/test_log_context.py`、`tests/test_health_snapshot.py`、`tests/test_local_full_chain.py`、`tests/test_inference_scheduler.py` | 源码检查类测试从 `run` 改指向 `_handle_result`；字段集合与新增日志行同步 |

> 工作区中还包含其他**非本次改造**的未提交改动（CIMG/cv2 相关文件等）。改造开始前已执行
> `git status` 确认，过程中没有 `reset`、`checkout` 或覆盖任何既有改动。

### E 阶段（单卡实测）追加的改动

| 文件 | 改动 |
| --- | --- |
| `tools/gpu_memory_probe.py` | 新增 `bench` 阶段（串行提交 N 次、逐次 `infer_ms`/`wall_ms` 与 `LatencyStats`、输出 shape 与有限值校验）与 `release` 阶段（`wait_for_release`，逐卡比对基线与归还）；新增 `nvidia_smi_memory()` 作为 NVML/nvidia-smi 读数的来源；新增 `check_device_pinning()`，在 `CUDA_VISIBLE_DEVICES` 被继承时**拒绝启动**（`exit 2`）；`stop()` 与 JSON 落盘移入 `finally`；新增 CLI `--inferences` / `--release-tolerance-mib` / `--release-timeout-s` |
| `inference/gpu_pool.py` | **修掉一个真实缺陷**：`probe_gpu_inventory()` 的 torch 回退分支不再调用 `torch.cuda.mem_get_info()`——该调用需要活动 context，会在**机器上每一张卡**各建一个约 390MiB 的 context，包括其他用户的在用卡，而该函数每次池启动都会走一遍（证据与修复见第 8 节） |
| `tests/test_multi_gpu_tools.py` | 新增 `NvidiaSmiReadingTests`、`DevicePinningTests`、`ReleaseCheckTests`（共 36 项） |
| `tests/test_inference_scheduler.py` | 新增 `InventoryIsReadOnlyTests`：以 AST 遍历断言该模块**永不调用** `mem_get_info`，并断言 torch 回退仍报出 `total_memory`（共 41 项） |
| `docs/MULTI_GPU_POOL.md` | 第 8 节改写为实测数据；第 9 节改为基于实测的推算；第 10、12 节同步（本条） |

---

## 2. 架构

```
相机接收 + Pose 接收 → camera/sync 同步  →  InputBuilder
                                              │ 只在这一刻分配 task_seq
                                              ▼
                                   InferenceTask（frozen，完整输入）
                                              │
                                       GpuScheduler（单线程）
                        ┌──────────────┬──────┴───────┬──────────────┐
                        ▼              ▼              ▼              ▼
                   worker 0        worker 1       worker 2       worker 3
                (spawn 进程)    (spawn 进程)    (spawn 进程)   (spawn 进程)
                整模型 cuda:0    整模型 cuda:0   整模型 cuda:0   整模型 cuda:0
                每个进程独享一张物理卡（CUDA_VISIBLE_DEVICES）
                        └──────────────┴──────┬───────┴──────────────┘
                                              ▼ 结果按完成顺序返回，可能乱序
                                        ResultManager
                          全部校验通过才交给下面这一条单写入者链路
                                              ▼
                      接管状态机 → 延迟裁剪/补偿 → 坐标转换 → WaypointSender(UDP)
                                              └→ 推理包落盘
```

要点：

* worker 是 `multiprocessing` **spawn** 进程，不是线程；每个进程加载一个完整 Alpamayo 1.5
  实例，不做模型切分、张量并行或流水线并行。进程内 `CUDA_VISIBLE_DEVICES` 只指向自己那张卡。
* worker 里**没有**接管决策、裁剪、坐标转换、UDP、Pose 历史或相机缓冲，也不读任何线上状态。
  它拿到的 `InferenceTask` 是自包含的，因此结构上不可能误做控制决策。
* 接管状态机、`WaypointSender`、UDP 序号与最近发送状态仍然只有 pipeline 线程一个写入者。
* 结果可以乱序返回；`task_seq` 是唯一的身份与排序键。

---

## 3. `task_seq` 与 Result Manager 的精确规则

### task_seq

* 由 pipeline 线程在 `InputBuilder` **接受**一个完整输入的那一刻分配，只分配一次；
  被拒绝的周期没有 task，也就没有 task_seq。
* 进程生命周期内严格递增、唯一。
* 是**唯一**的排序键。`capture_timestamp_us` 只用于新鲜度、端到端数据年龄和相机时钟异常
  检测——它来自远端时钟，可能回退、跳变或换域，永远不参与"谁更新"的判断。

### 核心规则

```python
if result.task_seq <= last_accepted_task_seq:
    drop_as_out_of_order()
```

其中 `==` 单独计为 `duplicate`，`<` 计为 `out_of_order`；两者都不会写入状态机。

### 校验顺序（`ResultManager.validate`，逐条计数，任何一条不过即丢弃）

1. worker 报告成功（失败也是一种结果，走同一条路径，以便释放 worker 槽位并计数）；
2. 回显的 `task_seq` 与该 worker 实际被派发的任务一致；
3. `task_seq` 严格大于上一个被接受的水印；
4. 自任务冻结起未超过 `result_max_age_ms`；
5. 该任务没有触发相机时钟异常，且未落后最新采集窗口超过 `max_capture_lag_ms`；
6. 轨迹形状可用（`(T, ≥2)`）；
7. 至少有一个点；
8. 所有坐标有限。

`validate()` 永不抛异常：畸形结果属于正常控制流，校验路径里的一次异常会带走唯一被允许发送的
那个线程。

**最新优先（latest-only）**：所有 worker 都忙时只保留最新的一个任务（`pending_capacity` 必须
为 1，配置不一致直接拒绝加载）。被顶掉的任务记 `pending_replaced`——它不是错误，是策略的代价。
不使用普通 FIFO 积压。

---

## 4. `min_cycle_s` 的行为

| 模式 | 行为 |
| --- | --- |
| `inline`（默认） | **完全不变**：`推理 → 后处理 → UDP → sleep(min_cycle_s)`。唯一被保留的细节是：轨迹非法那条分支不 sleep，与改造前 `continue` 的语义一致（这是刻意保留，不是遗漏）。 |
| `multi_process` | **不使用、也不受 `min_cycle_s` 节流**。循环由 `scheduler_poll_interval_ms`（默认 20ms）驱动。当 `min_cycle_s > 0` 时：启动日志出现 `min_cycle_s=<v> INERT(mode=multi_process ignores it)`，循环起点另打一条 WARNING（含实际轮询频率与 worker 数），提示 `min_cycle_s=0` 可消除该提示。 |

---

## 5. 配置样例

`config/config.json`（**已提交的默认值就是回滚态**）：

```json
{
  "inference_pool": {
    "mode": "inline",
    "gpu_devices": [],
    "latest_only": true,
    "pending_capacity": 1,
    "result_max_age_ms": 2500,
    "max_capture_lag_ms": 2500,
    "scheduler_poll_interval_ms": 20,
    "worker_start_timeout_s": 180,
    "minimum_ready_workers": 1,
    "metrics_window_s": 60,
    "worker_warmup": true
  }
}
```

启用 4 worker（`gpu_devices` 的长度**就是** worker 数，1/4/5/6 都可以；代码中没有任何地方
假定 4 张卡）：

```json
{ "inference_pool": {
    "mode": "multi_process",
    "gpu_devices": [1, 3, 4, 5],
    "latest_only": true,
    "pending_capacity": 1,
    "result_max_age_ms": 2500,
    "max_capture_lag_ms": 2500,
    "scheduler_poll_interval_ms": 20,
    "worker_start_timeout_s": 180,
    "minimum_ready_workers": 4,
    "metrics_window_s": 60,
    "worker_warmup": true
} }
```

环境变量覆盖（优先级：环境变量 → JSON → 默认值）：`ALPAMAYO_INFERENCE_MODE`、
`ALPAMAYO_GPU_DEVICES`、`ALPAMAYO_LATEST_ONLY`、`ALPAMAYO_PENDING_CAPACITY`、
`ALPAMAYO_RESULT_MAX_AGE_MS`、`ALPAMAYO_MAX_CAPTURE_LAG_MS`、
`ALPAMAYO_SCHEDULER_POLL_INTERVAL_MS`、`ALPAMAYO_WORKER_START_TIMEOUT_S`、
`ALPAMAYO_MINIMUM_READY_WORKERS`、`ALPAMAYO_METRICS_WINDOW_S`、`ALPAMAYO_WORKER_WARMUP`。

启动时会先做设备校验：重复/越界设备号、显卡可见性、基线显存不足都在加载模型之前快速失败并给出
可读错误。`minimum_ready_workers` 决定 `/ready` 何时转正——worker 没就绪时 `/ready` 返回 503，
而不是让一个缺卡的池悄悄半速运行。

---

## 6. 三个频率的定义

三者刻意分开计数，因为原来一个 `inference_count` 无法回答"模型跑到 4Hz、但只有 1Hz 的结果被
允许发车"这个唯一要紧的问题。每个频率都带 10s 滑窗、60s 滑窗和自启动平均。

| 频率 | 定义 |
| --- | --- |
| `raw_inference_hz` | 模型**跑完一次**。worker 结果到达即计数，在 ResultManager 看它之前——因此包含随后被乱序/过期/内容非法丢弃的结果；不含 worker 报错、CUDA OOM，也不含被 latest-only 顶掉而根本没跑的任务。 |
| `accepted_result_hz` | 结果通过了全部校验，被允许进入轨迹/接管/UDP 链路。 |
| `udp_output_hz` | `WaypointSender.send()` 返回。 |

`udp_output_hz` **单独也不够**：零航点的释放帧（`count=0, control=0`）是合法发送，不含任何
轨迹。所以另有 UDP 包分类计数：

* `sent_total` / `sent_with_waypoints` / `sent_release_frames` / `send_failures`
* `last_waypoint_count` / `last_control_flag` / `last_seq`

**验收标准是"非空有效轨迹的 UDP 输出 ≥ 2.0Hz"，不是 UDP 总包率。** 总包率可以是释放帧撑起
来的。

其余计数器：`drops`（`pending_replaced`、`out_of_order`、`duplicate`、`expired`、
`capture_lag`、`capture_timestamp_anomaly`、`capture_rollback`、`capture_jump`、
`invalid_trajectory_shape`、`invalid_trajectory_points`、`non_finite_trajectory`、
`worker_error`、`cuda_oom`、`worker_exception`、`unknown_task`、`task_seq_mismatch`）；
`faults`（`worker_failed`、`worker_exited`、`worker_load_timeout`、`worker_dispatch_failed`、
`message_handling_error`、`unknown_message`、`accepted_without_task`）；延迟
`queue_wait` / `infer`（纯模型时间）/ `validate` / `end_to_end` / `task_to_send`，各带
count/last/avg/p50/p95/min/max。每 GPU 另有：推理次数、最近/平均/P50/P95 推理时间、
busy/loading/ready/failed 状态、当前 `task_seq`、`allocated`/`reserved`/峰值显存。

`/health`、`/status` 的全部数据来自一个单写入者不可变快照。

---

## 7. 单元测试与离线测试结果

### 单元测试（本机）

```
python -m unittest discover -s tests
Ran 608 tests in 96.5s
FAILED (errors=4, skipped=2)
```

* 4 个 error 全部是**环境缺失 `cv2` 导致的历史遗留**，与本次改造无关：
  `test_camera_sei.CamtTests` ×3、`test_model_test_backend`（import 阶段失败）。
* 0 failures，2 skipped。改造前基线为 583 tests / 同样 4 errors / 2 skipped；
  新增 25 个测试后为 608。

### 离线测试 `tools/multi_gpu_offline_test.py`

以下为**本机（无 GPU）**真实 `multiprocessing` spawn + 真实 `GpuScheduler` /
`ResultManager` / 指标 + 替身 worker 的结果。替身只替换模型调用，因此这些是**调度数字**，
不是模型数字；真实模型吞吐需要在 GPU 机器上以 `--real` 重跑。

**乱序完成（强制完成顺序 1,3,5,2,4）** — 4/4 PASS：

```
accepted, in the order the watermark allowed them: [1, 3, 5]
[PASS] only the increasing task_seqs were accepted
[PASS] the two losers were dropped as out_of_order - out_of_order=2
[PASS] no result overwrote a newer one - last_accepted=5
[PASS] the inferred-but-lost results were still counted as inferences - raw=5, accepted=3
```

**积压测试（10Hz 输入 vs 每个 1500ms 的池）** — 5/5 PASS：

```
submitted=80  raw_total=21  accepted_total=21  max pending observed=1
scheduler_drops={"pending_replaced": 59}   faults={}
raw_inference_hz=2.66   (容量 4/1.5=2.67 Hz，输入 10 Hz)
result_drops 全 0
```

即：输入 10Hz、池容量 2.67Hz 时，pending 始终 ≤ 1，59 次提交被"替换"而不是排队，
只有 21 次真正推理——积压被丢弃，不是被推后。

**worker 数 1/2/4/5/6（每个推理固定 1500ms，6s 窗口）** — 26/26 PASS：

| workers | raw_inference_hz | accepted_result_hz | accepted_nonempty_hz |
| --- | --- | --- | --- |
| 1 | 0.80 | 0.80 | 0.80 |
| 2 | 1.47 | 1.47 | 1.47 |
| 4 | 2.78 | 2.78 | 2.78 |
| 5 | 3.43 | 3.43 | 3.43 |
| 6 | 4.02 | 4.02 | 4.02 |

`accepted_nonempty_hz` 是**调度侧的替身指标**（通过校验且带非空轨迹的结果数），
`udp_output_hz` 在这一模式下恒为 0，因为不发送。真实 UDP 线速率只有 `--scenario pipeline`
能测，工具自身也这样标注，避免被误读成线速率。

> 注：窗口短于运行时，`Frequency` 从自己第一个事件开始计时，把排空尾巴算进分子而没算等待首次
> 推理的时间，速率会偏高一点。请读数比例而非第三位小数；要引用容量数字请用 `--seconds 30`。

### `tools/gpu_memory_probe.py`

本机无 CUDA，只能验证失败路径（这一点本身很重要——探测工具必须在缺依赖时**快速可诊断地
失败**，而不是挂住）：

```
[FAIL] every worker loaded and became ready - ready=0/1 failed=1 loading=0 after 1s
worker 0 (device 0): load failed: ModuleNotFoundError: No module named 'alpamayo1_5'
RESULT: FAILED (1/1 checks failed)
```

1 秒内退出、错误原因明确、无残留进程。

---

## 8. 单 GPU 实测：显存与推理时间

**已实测。** 2026-09-20 13:07–13:08，机器 `xz-gpu02`，驱动 `580.126.09`，物理 GPU 1
（RTX 4090 24564MiB，测试前 4MiB，无人占用），
1 个 worker，真实完整 Alpamayo 1.5 10B，BF16 + SDPA，四路相机 `[1,4,0,2]` 每路 4 帧、
320×576、16 步 Ego History、`num_traj_samples=1 / top_p=0.98 / temperature=0.6`，
一次 warm-up + 5 次正式推理。全部产物在 staging 目录
`/data/users_data/wanghaohao/701/alpamayo_multi_gpu_stage_20260920-125615/`：
`probe_single_gpu.log`（完整日志）、`probe_single_gpu.json`（机器可读）、
`gpu_samples.csv` + `gpu_peaks.txt`（逐卡采样，见下）、`run_single_gpu_probe.sh`（执行脚本）。
**未使用 `CUDA_VISIBLE_DEVICES`**（它会让 `--devices` 的物理含义失效，工具会拒绝启动）；
worker 自己钉卡后在 `cuda:0` 上加载。**未跑 soak、未启动任何服务、未发送任何 UDP。**

### 加载

| 项 | 值 |
| --- | --- |
| 加载墙钟时间 | **44s**（5 个 safetensors 分片） |
| 进程内 `allocated` | 21150 MiB |
| 进程内 `reserved` | 22496 MiB |
| 进程内 `peak_allocated` | 22125 MiB |
| 进程内 `peak_reserved` | 22496 MiB |
| 设备 whole-device used | 4 → **23008 MiB** |
| 设备剩余 free | **1324 MiB**（共 24564 MiB） |

### warm-up 与正式推理

warm-up 1 次：**1118.6ms**（不计入统计）。

5 次正式推理（`task_seq` 2–6，`task_seq` 1 是 warm-up），单次一条、串行提交，总墙钟 5.786s：

| task_seq | infer_ms（worker 内，仅模型调用） | wall_ms（提交→被接受） | 输出 shape | 有限值 |
| --- | --- | --- | --- | --- |
| 2 | 1126.7 | 1149.0 | (64, 3) | 是 |
| 3 | 1127.3 | 1147.6 | (64, 3) | 是 |
| 4 | 1155.3 | 1169.2 | (64, 3) | 是 |
| 5 | 1152.4 | 1168.7 | (64, 3) | 是 |
| 6 | 1127.3 | 1148.9 | (64, 3) | 是 |

| 统计 | infer_ms | wall_ms |
| --- | --- | --- |
| mean | 1137.8 | 1156.7 |
| P50 | 1127.3 | 1149.0 |
| P95 | 1154.7 | 1169.1 |
| min | 1126.7 | 1147.6 |
| max | 1155.3 | 1169.2 |

5 次落成**两个档位**：1127 / 1127 / 1155 / 1152 / 1127——低、低、高、高、低，差额 28ms
（2.5%）。不是单调漂移（否则第三次之后不会回落），形态上更像固定的调度/同步行为而非热积累；
但 5 个样本不足以定论，真正要看的是第 9 节并发下的分布。

* 输出 **shape (64, 3)**，64 个航点；**无 NaN / Inf**（逐元素 `np.isfinite`）。
* **无 CUDA OOM**，无 worker 异常退出，无失败结果；15 个丢弃计数器全为 0。
* `wall - infer ≈ 19ms`，即队列等待 + 派发 + 校验的固定开销，对 1138ms 的模型时间可忽略。
* **单卡吞吐 ≈ 0.88 Hz**（1 / 1.138s）。

### 关闭与显存释放

`stop()` 返回 `terminated: []`、`killed: []`，789ms 内优雅退出。释放检查：
dev1 `4.0MiB → 4.0MiB`，**0.1s 内归还**，9/9 项检查通过。

运行期间对**全部 8 张卡**每 5s 采样一次（11 个采样点），只有 GPU 1 增长，
**GPU 0、2、3、4、5、6、7 全程 +0MiB**——包括他人正在使用的 GPU 2：

| dev | start | peak | end | growth |
| --- | --- | --- | --- | --- |
| 0 | 1488 | 1488 | 1488 | +0 |
| 1 | 4 | 23008 | 23008 | **+23004** |
| 2 | 20212 | 20212 | 20212 | +0（他人的任务） |
| 3 | 4 | 4 | 4 | +0 |
| 4 | 4 | 4 | 4 | +0 |
| 5 | 4 | 4 | 4 | +0 |
| 6 | 4 | 4 | 4 | +0 |
| 7 | 4 | 4 | 4 | +0 |

> 记录说明：探测**本身**的完整日志（load / warmup / bench / release 四段与 9/9 PASS）在
> `probe_single_gpu.log`；上面这张逐卡表由同一批次 `gpu_samples.csv`（11 行原始 `nvidia-smi`
> 采样）重新汇总到 `gpu_peaks.txt`——首轮运行里这张表因 stdout 缓冲没落进日志，原始 CSV 是
> 完整的，数字没有变。

### 从这次实测修掉的一个真实缺陷

`probe_gpu_inventory()` 的 torch 回退分支此前会为每张卡调用
`torch.cuda.mem_get_info(index)`。实测：该调用需要**活动 context**，因此会为**机器上每一张卡**
各创建一个 CUDA context，**每张约 390MiB**——包括其他用户正在跑任务的卡（GPU 2 实测
20212 → 20601MiB）。而 `GpuScheduler.start()` 每次启动都会调用它，也就是说**生产链路每次启动
都会在全部 8 张卡上各占 390MiB**。

已修复：torch 回退只保留 `get_device_name()` 与 `get_device_properties().total_memory`
（实测两者均无副作用、不建 context），不再调用 `mem_get_info`，`free_bytes` 留 `None`。
所有使用方（`gpu_pool` 启动日志、工具的诊断输出）本就已处理 `None`。NVML 分支与 nvidia-smi
仍能给出 free，且都不碰设备。修复后重测：其他 7 张卡全程 +0MiB，且本卡空闲显存从 934MiB
回升到 **1324MiB**（少掉的就是父进程那个 context）。

### 这次实测**没有**覆盖的

* **soak 未跑**（`--soak-minutes 0`）：显存是否随时间增长仍无数据。这是 rollout 前必须补的
  一项，见第 12 节。
* **单卡单 worker**：无法体现 4–6 个 worker 同时加载时的相互影响（PCIe 带宽、host 内存带宽、
  CPU 解码）。真实的每 worker 推理时间在多 worker 下**通常高于**这里的 1138ms。
* **1324MiB 剩余显存是本方案最紧的地方**。当前足够，但没有为"torch/CUDA 版本升级带来的
  额外开销"或更长的上下文留出余量。

---

## 9. 4 / 5 / 6 worker 吞吐对比 —— 待实测（下为基于实测的推算）

第 7 节的替身数字建立在"每个推理固定 1500ms"的假设上；**实测单次为 1138ms**，比假设更快，
所以那里的预测是**保守**的。按 1138ms / 卡重新推算（均匀耗时时完成顺序与派发顺序一致，
因此 `accepted ≈ raw`）：

| worker 数 | 推算 raw/accepted Hz | 对 2.0Hz 目标的余量 |
| --- | --- | --- |
| 1 | 0.88 | 不足 |
| 4 | ~3.5 | ~75% |
| 5 | ~4.4 | ~120% |
| 6 | ~5.3 | ~165% |

**注意这是推算，不是实测**：它假设每张卡的推理时间与单卡时相同。多 worker 并发时通常会更慢
（共享 PCIe 与 host 带宽），所以真实值会低于该表。即便如此，**4 卡对 2.0Hz 的余量看起来是
充足的**——这与第 7 节"1500ms 假设下 4 卡只剩 33% 余量"的判断相比乐观得多，但**必须实测确认**，
因为它直接决定"非空有效轨迹 UDP ≥ 2.0Hz"能否成立。

**并发时还要重新看显存**：每张被点名的卡仍是一个完整模型（约 23GiB）、只剩 1324MiB 余量
（风险 6），父进程那 390MiB 的开销虽已消除，但余量本身没有变大。

真实对比命令：

```bash
python tools/multi_gpu_offline_test.py --scenario workers --real \
    --device-sets "1,3,4,5;1,3,4,5,6;1,3,4,5,6,7" --seconds 30
# 纯调度侧对照（无 GPU 也能跑）：
python tools/multi_gpu_offline_test.py --scenario workers \
    --device-sets "1;1,3;1,3,4;1,3,4,5;1,3,4,5,6;1,3,4,5,6,7"
```

判据（工具会自动断言）：

* 每个 worker 都有推理次数（没有空闲 worker）；
* `raw_inference_hz >= 0.6 × N/T`（N 卡、单次 T 秒）；
* 无 worker 异常、无 CUDA OOM；
* 4 卡及以上：`accepted_result_hz >= 2.0` 且 `accepted_nonempty_hz >= 2.0`；
* pending 始终 ≤ 1。

真实的 UDP 线速率与包分类（唯一算数的口径）：

```bash
python tools/multi_gpu_offline_test.py --scenario pipeline --devices 1,3,4,5 --real \
    --seconds 60
```

该模式构造真实 `PipelineRunner`（不调用 `start()`，不开相机、不绑 Pose 端口、不接实车），
任务直接投给 runner 自己的 scheduler，由真实的 `_handle_result`（单写入者）完成裁剪、坐标转换
与发送，UDP 指向 127.0.0.1 的本地 sink，由独立实现的解码器统计总包数、非空轨迹包数、释放帧数、
序号连续性。需要 `cv2` 与真实 checkpoint，只能在 GPU 机器上跑。

---

## 10. 已知风险与限制

1. **多 worker 并发仍未验证**：单卡单 worker 已实测（第 8 节，全部通过），但 **4/5/6 个 worker
   同时加载、同时推理**未测，所以**每卡显存是否够、并发下的推理时间与吞吐**仍然是未知项。
   第 9 节的 3.5 / 4.4 / 5.3 Hz 是由单卡实测值推算的，不是实测值。这是当前最大的未知项。
2. **`--scenario scheduler` 的 `accepted_nonempty_hz` 是替身**：真实线速率只有 `pipeline`
   模式能测。不要用替身数字对外汇报 2.0Hz 达标。
3. **`result_max_age_ms=2500` 可以放回默认值**：实测单次 `infer_ms` 的 P95 为 **1154.7ms**，
   最坏一次 wall 为 1169.2ms。2500ms 允许一个任务在最坏情况下排到一个完整周期（~1.17s）之后
   仍有 ~2.1 倍的自身耗时余量，原先"若实测到 2s 就吃紧"的担心不成立。**但这条余量是按单 worker
   测的**：并发下每 worker 变慢，且 `min_cycle_s` 的节流会叠加等待，所以第 9 节并发实测时应把
   这个值一起复核。
4. **latest-only 会丢输入**：积压测试里 80 次提交只有 21 次被推理，59 次被替换。如果部署预期是
   "每个输入都要出一个包"，latest-only 就是错误的策略——这是设计选择，不是缺陷，但必须确认它
   符合实车预期。
5. **相机时钟检查是相对的**。相邻提交若倒退超过 `max_capture_lag_ms`，记 `capture_rollback`
   并丢掉该任务，比较基准不前进。向前间隔再大也不记 `capture_jump`，该采集时间成为新基准。
   结果相对最新采集窗口落后超过同一阈值时记 `capture_lag`。上线前应观察 `capture_rollback`
   和 `capture_lag`。
6. **10B 模型 × N 实例**：显存是 N 倍，且是**每卡一份、互不共享**。实测单卡加载后 whole-device
   used 为 23008MiB、**只剩 1324MiB**。这是本方案最紧的一项，两件事必须在并发实测里确认：
   每张被点名的卡都独立容纳一个完整模型（而不是挤在一张上），以及 1324MiB 的余量在并发下不被
   吃穿（`loading 期间峰值` 22125MiB 高于稳态 allocated 21150MiB，多 worker 同时加载时叠加的是
   这部分）。**不要为了塞进更多 worker 而用降低精度或 CPU offload 绕开**——那会改变模型输出，
   违反本项目"不改变精度"的约束。
7. **推理包落盘是磁盘成本**：每周期 16 张 JPEG + JSON。生产由 `infer_package_dir` 控制，并只保留最新 `infer_package_keep` 个目录（默认 600）。离线工具
   默认**不写**（`--package-dir` 才写），避免把磁盘写满。
8. **P50/P95 是每 worker 有界样本**（512 条滑窗，约 8 分钟 @1Hz/worker）。样本数会一并输出，
   避免把 4 个样本的 P95 当成 400 个样本的 P95。
9. **`minimum_ready_workers` 设为 worker 数**会在缺卡时让 `/ready` 一直 503。这是刻意的
   （缺卡的池应当拒绝服务），但要知道它意味着"一卡挂了就不接流量"；若希望降级运行，把它调小，
   此时 `/health` 的 `inference_workers_failed` 是告警位置。

---

## 11. 回滚到 inline

**回滚就是"不启用"**：`config/config.json` 里已提交的默认值就是 `mode=inline`、
`gpu_devices=[]`，即完全不改变现有单卡行为。

```bash
# 方式一：改配置（推荐，与已提交默认值一致）
#   config/config.json → inference_pool.mode = "inline"
# 方式二：环境变量覆盖，不改文件
ALPAMAYO_INFERENCE_MODE=inline
```

回滚后：

* pipeline 线程自己加载一个完整模型、自己阻塞在推理上，行为与改造前逐行一致；
* `gpu_devices` 被忽略，不启动任何 worker 进程；
* `min_cycle_s` 恢复其节流作用（`推理 → 后处理 → UDP → sleep(min_cycle_s)`）；
* `/health`、`/status` 的**字段顺序与数量不变**，只在末尾多出 `inference_pool`、
  `inference_metrics` 两个键；`/ready` 仍然是原来那 8 项检查（池存在时不会多出一项，
  有测试锁死这一点）。

inline 路径的行为等价性由测试保证：源检查类测试现在指向 `_handle_result`（两条路径共用的
唯一后处理尾部），因此 inline 与 multi_process 不可能各自漂移。

---

## 12. 尚未执行的实车部署步骤

以下是**除第 2 步外**均未执行；第 2 步已完成到"单卡不接车"这一档（见第 8 节），但其中
soak 一项仍未做。按顺序在人工确认后逐级放行：

1. **只读确认 checkpoint**：确认 `ALPAMAYO_CHECKPOINT` 指向的路径可读、符号链接目标未被修改。
   若目标不可读或不稳定，**报告并停止**，不复制、不移动、不修改其他用户目录。
   —— 已在单卡实测中隐式确认（模型从该路径加载成功，且全程只读）。
2. ~~单卡探测~~ **已完成，尚欠 soak**：单次推理、显存峰值、优雅关闭与显存归还均已实测并全部
   通过（第 8 节）。**未做的是 `--soak-minutes 15/30`**，即"显存是否随时间增长"没有数据。
   补测命令：`python tools/gpu_memory_probe.py --devices 1 --soak-minutes 15`。
3. **多卡探测**：`--devices 1,3,4,5`，确认 N 个完整实例能同时驻留、每卡都长出自己的模型。
   注意每卡只剩 1324MiB 余量（风险 6）。
4. **离线吞吐对比**：第 9 节的 `--scenario workers --real --device-sets`，
   得到 4/5/6 卡的真实对比，据此冻结 `gpu_devices` 与 `minimum_ready_workers`。
5. **复核 `result_max_age_ms`**（见风险 3）：单 worker 下 2500ms 已有充足余量，但并发下每
   worker 会变慢，需用第 4 步的实测值再确认一次。随后用 `--scenario pipeline --real`
   在本地 loopback sink 上验证"非空有效轨迹 UDP ≥ 2.0Hz"，同时确认没有 `seq_gaps`。
6. **`--soak-minutes 30`**：确认 `allocated` 与 `reserved` 双双不增长（与第 2 步同一项）。
7. **接入真实相机与 Pose，但不接控制链路**：观察 `/health` 的
   `inference_pool.workers[].state`、`inference_metrics.drops`、相机时钟异常计数、
   `queue_wait`。此时 UDP 仍指向本地 sink 或保持 `allow_unconfigured_udp` 关闭。
8. **接实车 UDP**：确认 `magic` / `vehicle_id` / 端口与协议一致后再打开；先在静止状态验证
   释放帧与航点包格式，再低速行驶。
9. **实车低速验证**：重点看接管决策、坐标转换结果与 UDP 序列号，确认没有任何旧 `task_seq`
   覆盖新结果（`out_of_order` 可以为非零，但**被接受的**必须严格递增）。

---

## 附：离线工具速查

```bash
# 不需要 GPU：调度、积压、乱序、吞吐（替身 worker）
python tools/multi_gpu_offline_test.py
python tools/multi_gpu_offline_test.py --scenario order
python tools/multi_gpu_offline_test.py --scenario backlog --devices 1,3,4,5 --infer-ms 1500
python tools/multi_gpu_offline_test.py --scenario workers --device-sets "1;1,3;1,3,4;1,3,4,5;1,3,4,5,6;1,3,4,5,6,7"

# 需要 GPU 与真实 checkpoint
python tools/multi_gpu_offline_test.py --scenario workers --real --device-sets "1,3,4,5;1,3,4,5,6;1,3,4,5,6,7"
python tools/multi_gpu_offline_test.py --scenario pipeline --devices 1,3,4,5 --real
python tools/gpu_memory_probe.py --devices 1,3,4,5 --soak-minutes 30 --json soak.json
```

两个工具的退出码为 0 仅当所选场景的每一条断言都成立；失败会逐条列出未通过的检查名。
