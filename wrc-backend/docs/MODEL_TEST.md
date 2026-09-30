# 离线模型测试后端

独立入口 `test_backend:app`，不会启动实时 PipelineRunner、相机、车辆状态监听或航点发送。复用当前 AlpamayoRunner 做一次离线推理。响应里是 `cot`、`positions`、`rotations` 等观察字段，不包含 `control_flag`，也绝不向车辆发送。模型源码、权重及实车 `app.py` 均不修改。下面的 8091 只在 `127.0.0.1` 上给这个离线进程用，不是生产后端的 8090。

## 启动

在有模型依赖和权重的服务器环境执行。单进程、单 worker，模型在第一次有效请求时加载。请选择有足够空闲显存的 GPU，避免与实车推理竞争。不要对生产的 `uvicorn app:app` 做 pkill，也不要占用 8090。

```bash
cd /data/users_data/wanghaohao/701/alpamayo_infer_backend
sudo -n env PYTHONPATH=/data/users_data/wanghaohao/701/alpamayo1.5/src \
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/data/users_data/wanghaohao/701/alpasim/.venv.bak/bin/python \
  -m uvicorn test_backend:app --host 127.0.0.1 --port 8091 --workers 1
```

PC 开 SSH 隧道，然后调用请求脚本：

```powershell
ssh -N -L 8091:127.0.0.1:8091 -p 24204 wanghaohao@117.190.94.226
```

另一个终端执行：

```powershell
python F:\wrc\wrc-backend\tools\request_model_test.py F:\wrc\wrc-backend\examples\model_test_sample.json --output model_result.json
```

## 输入约定

`examples/model_test_sample.json` 已采用你提供的 16 步 ego_history 位姿。图片路径仍为占位路径，需要替换为同一次采集的真实图片。图片路径相对于样例 JSON 所在目录；客户端转换成 Base64 后 POST /infer。不要求服务器访问 PC 文件。

将图片放在 `examples/images/` 下，样例采用以下文件名，t0 到 t3 按时间从旧到新排列：

```text
left_t0.jpg  ～ left_t3.jpg    左相机（0）
front_t0.jpg ～ front_t3.jpg   前相机（1）
right_t0.jpg ～ right_t3.jpg   右相机（2）
back_t0.jpg  ～ back_t3.jpg    后相机（4）
```

文件名通过 image_paths 指定，可自定义；相机归属由 camera_index 指定。

- nav_text：导航指令文本。
- cameras：左 0、前 1、右 2、后 4，每路恰好 4 张图片，按时间从旧到新排列，内部按相机编号排序。图片转为 RGB 并缩放至 576×320，与现有输入流程一致。
- ego_history：直接填写 16 步 timestamp_us、latitude、longitude、altitude、heading_deg，与你提供的数据格式一致。经纬度为度，海拔为米，heading_deg 沿用现有后端语义（北为 0°，顺时针增加）。接口复用实时后端 build_ego_arrays，转换为以最后一步为原点的车体系位置及相对旋转，不需要你手动计算 xyz/yaw_rad。
- x_m、y_m：可选的记录字段，允许直接保留；模型位置以经纬度和海拔重新计算，避免重复旋转已在最后一步车体系下的 x_m/y_m。
- 车体系 +x 向前，+y 向左，+z 向上。模型张量形状为 [1,1,16,3] 和 [1,1,16,3,3]。
- 兼容旧 history 输入（xyz 米制固定坐标、yaw_rad 弧度），但 history 和 ego_history 必须二选一，不能同时提供。
- timestamp_us：同一时间轴的微秒时间。16 步间隔 100000μs，允许误差 1000μs；每路 4 张图片分别对应最后 4 步位姿时间，允许误差 1000μs。该离线接口不会根据当前墙钟拒绝历史数据。

## 输出及错误

GET /health 不触发模型加载。POST /infer 返回 cot（模型生成文本）、positions（通常为 64×3，单位米）、rotations、t0_us、trajectory_dt_s、elapsed_s（第一次包含加载耗时）、prompt_text 和输入 token 指纹，不返回模型控制位。轨迹以输入最后一步车体系为参考，不做经纬度转换或延迟补偿。

字段/图片/同步错误返回 422，模型繁忙返回 409，加载或推理失败返回 500 并在服务日志记录原因。未传入 chassis；沿用当前模型调用实际使用的图像与历史位姿张量。
