# UUV 综合路径规划训练流程与程序结构说明（供后续 Agent 阅读）

本文档面向后续自动化 Agent，概述当前项目在二维网格地图下，结合地形、海流、信标覆盖与能量约束的训练框架。

## 1. 项目定位与入口

- 任务：UUV 在 2D 网格中从起点到终点进行路径规划。
- 影响因素：
  - 地形深度（可航行性）
  - 海流方向与强度（推进助力/逆流能耗）
  - 信标覆盖（可重置 INS 误差）
  - 能量预算与终止惩罚
- 统一入口：`runs.py`
- 配置驱动：Hydra + YAML（默认 `configs/config1.yaml`）

## 2. 目录与职责（按训练主链路）

- `runs.py`
  - 全流程编排：清理旧输出 -> 生成环境 -> 选择算法训练 -> 训练后分析。
- `configs/config1.yaml`
  - 环境尺寸、动作维度、奖励参数、训练步数、DQN 参数、信标位置等。
- `envs/`
  - `env.py`：核心环境、step 奖励计算、终止判定、step 级日志。
  - `generate_terrain.py`：生成地形并保存。
  - `generate_ocean_current.py`：基于地形生成海流场并保存。
  - `generate_beacon_area.py`：生成信标覆盖图并保存。
  - `read_terrain_current.py` / `read_beacon_info.py`：读取环境数据。
  - `async_step_logger.py`：异步写入 `step_rewards.csv`。
  - `robot.py`：机器人状态（位置、能量、INS 误差、轨迹）。
- `trainers/`
  - `train_qlearning.py`：Q-Learning 训练循环。
  - `train_double_dqn.py`：Double DQN 训练循环（CUDA-only）。
- `agents/`
  - `q_learning.py`：Q 表动作选择与 TD 更新。
  - `double_dqn.py`：在线/目标网络、Double DQN 更新、软更新。
- `models/dqn_model.py`
  - 多尺度 Dueling Q 网络（标量 + 局部图块 + 动作特征融合）。
- `buffers/prioritized_replay_gpu.py`
  - GPU 优先经验回放 + n-step return。
- `scripts/`
  - `collect_step_rewards.py`：step 奖励分解统计与图表。
  - `collect_summary.py`：训练日志摘要统计。
  - `beacon_layout_editor.py`：独立信标布局编辑器（交互式增删改半径，保存 JSON）。
  - `ins_threshold_visual_tester.py`：动态 INS 阈值可视化验证工具（点击起终点并显示阈值细节）。
  - `pressure_test_buffer_capacity.py`：GPU 回放池容量压测（按容量阶梯写入并监控显存占用）。

## 3. 端到端训练执行流程

### 3.1 `runs.py` 主流程

1. 通过 Hydra 读取配置，确定本次 `run_dir`（时间戳输出目录）。
2. 清理 `outputs/` 下旧结果（仅保留最近 2 次）。
3. 环境生成顺序：
   - 生成地形
   - 生成海流
   - 生成信标覆盖
4. 按 `cfg.train.algorithm` 分支：
   - `q_learning` -> `train_with_cfg(...)`
   - `double_dqn` -> `train_double_dqn_with_cfg(...)`
5. 训练结束后自动后处理：
   - 解析 `step_rewards.csv` 产出 reward 分析
   - 汇总 `train_log.csv` 统计信息

### 3.2 关键不变量

- 同一轮训练必须共享同一个 `run_dir`（环境输入与日志输出一致）。
- 地形应先读取/归一化，再生成地形奖励图（否则地形奖励可能异常弱化）。
- DQN 分支要求 CUDA 可用，且 device 必须是 `cuda` 或 `cuda:N`。

## 4. 环境机制（`envs/env.py`）

## 4.1 观测空间

环境返回观测为 5 维元组：

- `(x, y, dx, dy, ins_error)`
- 其中 `dx = goal_x - x`, `dy = goal_y - y`

## 4.2 动作空间

- 当前配置默认 4 动作：上下左右（在 `move_robot` 中按编码更新坐标）。

## 4.3 终止条件

每步先执行动作，再检查终止：

- 越界：`out_of_bounds`
- 能量耗尽：`energy_exhausted`
- INS 误差超过阈值：`ins_diverged`
- 到达目标：`goal_reached`

对应会设置终止惩罚或目标奖励，并记录 `last_termination_reason`。

## 4.4 奖励组成（step 粒度）

未终止时，step 总奖励由以下分量相加：

- `step_penalty`：每步基础代价
- `approach_reward`：朝目标前进奖励（远离目标时按惩罚倍率）
- `terrain_reward`：地形奖励（带 clip 和权重）
- `current_reward`：海流奖励（带 clip 和权重）
- `energy_reward`：能耗项（基础能耗 + 逆流附加能耗）
- `beacon_reward`：进入新信标区域奖励
- `revisit_penalty`：重复访问位置惩罚（有上限）

终止步会优先写入终止相关奖励/惩罚。

## 4.5 信标与 INS 机制

- 每步动作后 `INS_error += 1`
- 若当前位置被信标覆盖，INS 误差立即归零
- 通过该机制鼓励轨迹经过信标区域，抑制定位发散
- 环境支持动态 INS 阈值：
  - 开启 `env.dynamic_ins_error_threshold=true` 时，按起终点、信标布局和走廊宽度动态计算本轮阈值。
  - 关键参数：`dynamic_ins_corridor_width`、`dynamic_ins_safety_factor`、`dynamic_ins_min`、`dynamic_ins_max`。
  - 关闭时回退到固定阈值 `env.ins_error_threshold`。

## 4.7 reset 初始化策略（已更新）

- `Env.reset()` 当前为随机起终点：
  - 起点：在地图范围内随机采样
  - 终点：在地图范围内随机采样，且强制不与起点重合
- 每轮会基于 `step_total` 计算：
  - 初始能量（短路径与常规路径使用不同 scale，并受 `init_energy_min` 下限约束）
  - 本轮 `ins_error_threshold`（动态阈值开启时）
- reset 后会执行一次信标同步（若落在覆盖区则 INS 误差归零）。

## 4.6 step 日志

环境通过 `AsyncStepLogger` 异步写 `step_rewards.csv`，关键字段包括：

- 轨迹信息：episode、step、动作、位置、目标、距离变化
- 终止信息：terminated、termination_reason
- 奖励分项：step_penalty、goal_reward、approach_reward、terrain_reward、current_reward、energy_reward、beacon_reward 等
- 能耗/对齐信息：energy_step_cost、current_alignment
- 汇总：step_reward

## 5. Q-Learning 分支

## 5.1 状态与 Q 表

- 状态索引基于 `(x, y, dx, dy, ins_error)`。
- Q 表维度约为：
  - `[width, height, dx_range, dy_range, ins_range, action_dim]`

## 5.2 训练循环

每个 episode：

1. 根据 episode 进度计算阶段调度权重（exploration / transition / optimization）。
2. `env.reset()` 初始化。
3. 按 epsilon-greedy 选动作，执行 step。
4. 用 TD 目标更新 Q 表。
5. 写 `train_log.csv`（episode 级）。
6. 若奖励最优，保存 `q_table_best.npz`。

## 5.3 阶段调度

- `compute_stage_schedule(...)` 动态调整奖励权重：
  - 早期偏探索（靠近目标主导）
  - 中期过渡
  - 后期强调多目标平衡（地形/海流/能耗）

## 6. Double DQN 分支

## 6.1 状态编码（比 Q-Learning 更丰富）

`train_double_dqn.py` 中 `encode_observation(...)` 将环境编码为：

- `scalar`：归一化坐标、目标方向、能量比、INS 比例、当前位置海流/地形/信标等
- `patch_s`：小尺度局部多通道图块（11×11，包含地形、海流、信标、目标先验）
- `patch_l`：大尺度局部多通道图块（21×21，通道同上）
- `action_feat`：每个动作的风险/收益先验特征（越界、距离变化、逆流风险、深度风险、信标可达）
- `patch_meta`（new）：patch 重建元数据，存储 `[x, y, goal_x, goal_y, step_total]`，用于回放池延迟重建

### 6.1.1 Patch 元数据存储与重建机制（2026-04-09 新增）

支持可选的"patch 延迟重建"模式以大幅降低回放池显存占用：

**启用方式**：
- 配置 `dqn.reconstruct_patch_on_sample: true`
- 配置 `dqn.cache_maps_on_gpu: true`（必须配套）

**工作原理**：
- 状态编码后，`patch_meta` 捕获截取坐标与 episode 上下文（目标位置、总步数）
- 回放池存储时只保存 `scalar`、`action_feat`、`patch_meta`，不存 `patch_s/patch_l` 张量
- 采样时从 GPU 缓存地图（terrain、u_norm、v_norm、beacon）和重建的 `goal_prior` 恢复 patch

**GPU 地图缓存**：
- 环境地图在 buffer 初始化时一次性转为设备张量
- 采样时直接在 GPU 上切片，避免 CPU-GPU 往返

**目标先验图缓存**：
- 按 `(goal_x, goal_y, step_total, width, height)` 做 LRU 字典缓存（容量 `goal_prior_cache_size`）
- 若目标点固定，缓存命中率接近 100%，避免重复指数计算

**兼容性**：
- `reconstruct_patch_on_sample=False` 时采用原始 eager 存储
- 配置一键切换，便于性能对比

## 6.2 网络结构

`MultiScaleDuelingQNetwork`：

- 标量 MLP 分支
- 小图块与大图块共享 CNN 编码
- 动作特征 MLP
- 融合后走 Dueling Head（Value + Advantage）输出 Q

## 6.3 训练机制

- 经验回放：`PrioritizedNStepReplayBufferGPU`
  - PER 按优先级采样
  - n-step 回报累计
  - **新增**（2026-04-09）：可选的 patch 延迟重建、GPU 地图缓存、goal_prior LRU 缓存（见 6.1.1）
- Double DQN 目标计算：
  - online 选 `argmax_a Q_online(s',a)`
  - target 网络评估对应动作 Q 值
- 软更新：`tau` 更新 target 参数
- CUDA 强约束：
  - 启动时检查 CUDA
  - 首个 state 和 sample batch 均校验在 GPU 上
  - 若启用 patch 重建，强制 `cache_maps_on_gpu=True` 以确保地图切片在设备上进行

## 6.4 日志与模型保存

- 训练日志：`train_log.csv`（增加 `loss_mean`、`buffer_size`）
- 最优模型：`dqn_model_best.pt`

## 7. 输出物与后处理

每次训练输出到时间戳目录 `outputs/YYYY-MM-DD/HH-MM-SS/`，典型文件：

- `environment_grid.csv`：环境网格（地形/海流/信标融合信息）
- `step_rewards.csv`：逐步奖励分解日志
- `train_log.csv`：episode 级指标
- `q_table_best.npz` 或 `dqn_model_best.pt`：最优策略参数
- `reward_plots/`：奖励分析脚本产物

训练后由 `runs.py` 自动调用：

- `scripts/collect_step_rewards.py`
- `scripts/collect_summary.py`

## 8. 配置要点（`configs/config1.yaml`）

- `train.algorithm`: `q_learning` / `double_dqn`
- `env.*`: 地图尺寸、动作维度、INS 阈值
- `env.dynamic_ins_*`: 动态 INS 阈值参数（走廊宽度/安全系数/上下限）
- `env.init_energy_*`: 初始能量计算参数（常规倍率、短程倍率、短程阈值、最小能量）
- `reward.*`: 所有奖励/惩罚超参数
- `stage.*`: Q-Learning 的阶段调度比例
- `buffer.capacity`: DQN 回放容量
- `dqn.*`: patch 尺寸、PER、n-step、tau、device 等
  - **新增（2026-04-09）**：
    - `reconstruct_patch_on_sample`：启用 patch 延迟重建（true/false，当前配置默认 false）
    - `cache_maps_on_gpu`：环境地图张量化驻留 GPU（true/false，默认 true）
    - `goal_prior_cache_size`：目标先验图 LRU 缓存容量（默认 64）
- `beacon.beacon_config_path`: 外部信标布局文件路径（JSON，优先级高于 `beacon_locations`）
- `beacon.beacon_locations`: 兼容回退字段（当 JSON 文件不存在时启用）

## 8.1 信标布局配置来源（当前实现）

- 环境在构造阶段仅加载一次信标布局，不在每次 reset 重复读取。
- 优先从 `beacon.beacon_config_path` 指向的 JSON 读取信标列表（每个信标含 `x/y/radius`）。
- 当 JSON 路径为空或文件不存在时，回退读取 `beacon.beacon_locations` 与 `beacon_signal_area`。

## 8.2 回放池容量压测脚本（新增）

- 脚本：`scripts/pressure_test_buffer_capacity.py`
- 配置：`configs/buffer_pressure_test.yaml`
- 目标：
  - 按容量阶梯创建/填充 GPU 回放池
  - 打印 `cap / len / writes / used / reserved / peak_allocated`
  - 在达到目标显存占用比例（默认 90%）或 OOM 时停止
  - 给出正式训练建议容量（保留 10%-15% 显存余量）

## 9. 后续 Agent 常见任务入口建议

- 调奖励：优先改 `configs/config1.yaml`，避免硬编码。
- 加奖励分量：
  - `envs/env.py` 的 step 计算 + `STEP_REWARD_COLUMNS` 同步补字段。
- 调 DQN 表达能力：
  - `trainers/train_double_dqn.py` 的编码 + `models/dqn_model.py`。
- 分析训练异常：
  - 先看 `train_log.csv`，再看 `step_rewards.csv` 分量。

## 10. 当前实现中的显式行为备注

- `Env.reset()` 当前使用随机起终点，且保证起终点不重合；相关初始能量与 INS 阈值按本轮起终点动态确定。
- `runs.py` 每次启动会清理旧输出目录（保留最近 2 次），重要实验结果需提前备份。

## 11. 近期已落实的性能与存储优化（2026-04-11）

本节记录最近已经落到代码中的改动，便于后续 Agent 快速判断当前主路径与旧实现的关系。当前推荐的服务器训练路径已经从“逐样本重建 patch”切换为“元数据回放 + 预计算目标先验 + 批量索引重建”。

### 11.1 AMP 覆盖修复

- `agents/double_dqn.py` 的 `train_step()` 已接入 `autocast` 和 `GradScaler`。
- 当前训练路径不再只把 loss 计算放进混合精度，而是把在线网络前向、目标网络前向、Q 值计算和 loss 计算统一纳入 AMP 作用域。
- 反向传播顺序保持为 `scale -> backward -> unscale -> clip -> step -> update`，避免梯度裁剪作用在缩放梯度上。
- `configs/config1.yaml` 中已保留 `dqn.use_amp: true` 作为默认训练开关。

### 11.2 goal_prior 全量预计算

- `envs/env.py` 已加入全量 `goal_prior` 库，按 `(goal_x, goal_y)` 预先缓存目标先验图。
- `Env.reset()` 会优先使用预计算库，其次回退到 episode 级缓存，最后才走即时计算。
- 这样可以避免每次 `encode_observation()` 和 replay 采样时重复重算目标先验图。
- 当前文档里的“按 episode 缓存”只保留为回退逻辑，主路径已经升级为全量预计算。

### 11.3 Replay 的 meta_only 存储模式

- `buffers/prioritized_replay_gpu.py` 已支持 `replay_store_mode=meta_only`。
- 在该模式下，transition 不再长期保存完整 `patch_s` / `patch_l`，而是保存用于重建 patch 的元数据。
- 这些元数据至少包含位置、目标位置和步数上下文，供采样时按需恢复图块。
- `replay_dtype` 也已接入，可在 `float16` 与 `float32` 间切换，当前服务器默认档倾向 `float16`。

### 11.4 批量索引版 patch 重建

- replay 采样已从逐样本 Python 循环重建，切换为 batch 级张量索引重建。
- 现在 `sample()` 会按批次一次性恢复 `state_patch_s`、`state_patch_l`、`next_state_patch_s`、`next_state_patch_l`，减少大量小 kernel 启动和 Python 循环开销。
- 静态图层 `terrain`、`u_norm`、`v_norm`、`beacon` 只保留单份缓存；`goal_prior` 则从预计算库按目标索引读取。
- 这条路径是当前单卡 T4 上的主推荐路径，目的是降低显存增长斜率并提高采样吞吐。

### 11.5 配置默认档与兼容性

- `configs/config1.yaml` 已接入并保留与上述优化相关的开关：`enable_step_logging`、`cache_goal_prior`、`use_precomputed_goal_prior`、`replay_store_mode`、`replay_dtype`、`use_batch_patch_indexing`、`use_amp`。
- 服务器训练默认建议是 `meta_only + precomputed_goal_prior + float16 + batch indexing + AMP`。
- 旧的 `full_patch` / `reconstruct` 路径仍作为回退方案保留，便于排障和对照实验。
- `trainers/train_double_dqn.py` 负责把这些开关透传给环境和 replay buffer，训练主循环本身不需要感知具体存储细节。

### 11.6 DQN epsilon 曲线优化（2026-04-11 新增）

- DQN 的 epsilon 探索策略已从原来的硬编码三段式改为更灵活的配置驱动方式。
- 当前实现在 [trainers/train_double_dqn.py](trainers/train_double_dqn.py#L39) 的 `three_stage_epsilon` 函数中，优先从 `dqn.*` 配置段读取 epsilon 参数，回退到 `agent.*` 段。
- **推荐参数档位**：
  - `dqn.epsilon_start: 1.0`（初始探索率）
  - `dqn.epsilon_end: 0.05`（最终利用率）
  - `dqn.epsilon_phase1_ratio: 0.05`（前 5% episodes 保持高探索）
  - `dqn.epsilon_phase2_ratio: 0.40`（5% 到 40% episodes 平滑线性下降）
  - `40%` 之后保持低随机性，驱动收敛和局部微调
- 这套参数相比原来的 0.2/0.6 比例更激进，因为多约束路径规划任务中盲目尝试效率低，早期尽快进入学习区间更有利。
- 如果需要更保守，可调整：
  - `epsilon_phase1_ratio: 0.10`，`epsilon_phase2_ratio: 0.50`
  - `epsilon_end: 0.03` 或 `0.02`（更低位置利用）

### 11.7 给后续 Agent 的判断规则

- 如果目标是继续压缩显存，优先看 `buffers/prioritized_replay_gpu.py` 的存储模式与采样路径，而不是先调大 capacity。
- 如果目标是提高 GPU 利用率，优先看 batch 索引重建和 AMP 覆盖范围，而不是先加大 `updates_per_step`。
- 如果目标是调试观测输入，优先看 `trainers/train_double_dqn.py` 的 `encode_observation()`，再看 `envs/env.py` 的 `goal_prior` 生成逻辑。
- 如果目标是调整探索与利用的平衡，优先改 `dqn.epsilon_*` 相关参数，而不是改 buffer 容量或 batch size。

---

若后续 Agent 需要在此文档基础上继续扩展，建议保持“入口编排 -> 环境机制 -> 算法分支 -> 输出分析”的叙述顺序，便于快速定位改动影响范围。
