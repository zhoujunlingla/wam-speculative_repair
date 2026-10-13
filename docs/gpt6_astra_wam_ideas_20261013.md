# GPT-6 Astra × World Action Model：问题驱动的 Idea 报告

日期：2026-10-13
写法：先找"有价值且未被解决"的问题，再引入一个借鉴自其他领域、大概率有效的新模块。每个 idea 都按三段式组织：整体 pipeline、模块 A、模块 B。

> 说明：本环境的网络策略屏蔽了 arxiv.org、scholar.google.com 和 openai.com，所以原文没法直接打开。下文的事实来自网页搜索摘要，以及 GitHub 上公开的 README（FastWAM、Faster-WAM、VPP2、LingBot-VA、Awesome 列表）。论文中的数字都是作者自报，**真正立项前请对照原文核实**。你给的 Scholar 主页我打不开，所以写作风格参考的是你给的 Sentinel-VLA（arXiv 2605.01191）："问题 → 一个总体框架 → 两三个各管一段的模块 → 自动化数据/持续学习"。

---

## 0. TL;DR

| # | 名字 | 解决的关键问题 | Astra 的独特作用（为什么换成普通 VLM 不行） | 借鉴来源 |
|---|---|---|---|---|
| 1 | **ProgAnchor：程序锚定想象** | **想象悖论**：WAM 在 OOD 下最需要"想象未来"，偏偏此时想象最不可靠 | 能写出可执行的动态场景程序（4DCodeBench 排名第一），几何最准（Video2World） | 组合扩散 / PoE 能量引导、LLM-grounded 视频扩散、MPC 重对齐 |
| 2 | **ProbeWAM：先试探、再想象** | **物理混叠**：外观相同、隐藏物理不同（质量、摩擦、铰接方式），WAM 会"自信地想错" | 能从少量交互中归纳出紧凑的符号世界模型（ARC-AGI-3，96% 的关卡用的动作数少于中位数人类） | 贝叶斯实验设计（BOED/BALD）、RMA/TTT 快权重元学习 |
| 3 | **FutureTense：将来时推理** | **时间尺度错配**：Astra 每次调用约 40 s，世界不会等它；现有做法要么暂停仿真，要么执行过期计划 | Astra 是最强但最慢的"慢脑"，推理越强延迟越大，这个趋势只会加剧 | Smith 预估器（时滞控制）、网游的客户端预测 + 服务器对账、CPU 分支预测 / 推测执行 |

**推荐**：以 **Idea 1** 作为主线论文（故事最完整，最"fancy"，直接对准泛化）。**Idea 3** 和你仓库里现成的推测验证 / 修复基础设施（draft / teacher / RiskRouter / SVDR）衔接最紧，原型最快。**Idea 2** 问题新、干净，还能顺带贡献一个 benchmark。

---

## 1. 调研结论

### 1.1 GPT-6 Astra 能做什么、不能做什么（与具身相关的部分）

| 维度 | 证据（作者 / 第三方自报） | 对 idea 的含义 |
|---|---|---|
| 基本信息 | 2026-09-03 发布；OpenRouter 定价为输入 $10/M tokens、输出 $50/M tokens，上下文 1.05M | 贵，只能稀疏调用，不能逐帧调用 |
| 符号世界模型归纳 | ARC Prize：Astra 会把陌生环境转成**紧凑的符号世界模型**，96% 的关卡用的动作数少于中位数人类；标准 harness 下 ARC-AGI-3 得分 62.7%，OpenAI 的有状态 harness 约 99.9% | 擅长**主动实验 + 归纳隐藏规律**，对应 Idea 2 |
| 记忆 / 持续学习 | 高分主要来自 harness 在权重之外保存笔记、跨请求保留推理状态 | "权重外记忆"可以存放程序和物理假设 |
| 视觉 / 空间 | 2D 空间推理与人类持平（96.0 vs 96.0）；检测、分割超过专用模型（+10.7 / +4.3）；视频分割仍比专用模型低 7.9 | 感知和"在哪里"非常强 |
| 视频 → 代码 | 4DCodeBench（从视频写代码重建动态场景）：Astra [Max] 在 18 个 agent 中综合第一，动力学领先 31%；Video2World：几何最准，但功能性任务成功率只有 10.7% | **知道"是什么、在哪、大致怎么动"，不懂接触物理**，对应 Idea 1 |
| 直接控制 | Galbot 系统评测（2609.38537）：与 π0.5 混合控制在 RoboDojo 子集上 48%；直接控制在接触密集任务上表现差；步态控制 0/5 | 不能让它直接出低层动作 |
| 延迟 / 成本 | 同一评测：一段 30 s 的步态需要 250 次调用，**平均每次 39.86 s，期间暂停物理仿真**；50 个 RoboDojo 实例消耗 6.2 亿（辅助控制）到 11.3 亿（直接控制）tokens。RoboDojo 官方评测因安全原因中止了真机测试 | 对应 Idea 3 |
| 自进化 harness | PhysEvo：冻结 Astra + 元 agent 改写工具，RoboDojo 62% vs 47.17%，真机 25 次中 84%；RHD 把 Astra 的经验蒸馏成 playbook 交给更便宜的 Luna 使用 | 不改权重的自进化已经有人做了 |

### 1.2 社区已经做过的 Astra × 具身（避免撞车）

按 Awesome-Astra-Embodied-AI（42 个案例）和相关论文归纳，现有工作分成四类：

1. **零样本直接控制**：写控制代码或直接输出位姿（Rubik 魔方、插头插入、键盘打字等），慢且脆。
2. **Agent 调用冻结的 VLA**：Astra 规划，π0.5 / FluxVLA 执行（RPent 在 LIBERO-PRO 上 92.63%，RoboMME 三层架构 79.13%，Harness VLA，RHD）。**接口是语言子任务。**
3. **Real2Sim**：从视频重建可交互仿真（Video2World、4DCodeBench、Real2Gym、厨房重建）。
4. **自动搭建 RL 环境并训练**（转笔、四足、SimEX 自动科研）。

**空白：没有一项工作把 Astra 接进 WAM 的"想象"内部。** 现有组合都停留在"语言子任务 → 策略"的接口层面，没有用 Astra 去塑造、校正或调度 WAM 对未来的预测。

### 1.3 WAM 现状，以及哪些方向已经拥挤

**你点名的三个开源 WAM：**
- **Fast-WAM**（2603.16666）：提出"测试时到底需不需要想象未来"这个问题。新版 Optional-IDM 在同一个模型里支持两种模式；LIBERO 上 IDM 模式 98.55%，first-frame 模式 97.75%，**分布内差距很小**；H20 上推理 210 ms。
- **LingBot-VA 2.0**（2607.08639）：原生视频-动作预训练（不再微调通用视频生成器），语义视觉-动作 tokenizer + 因果 DiT + 稀疏 MoE，异步推理时用学到的前向动力学对最新观测重新对齐（re-grounding），控制频率最高 150 Hz（另有来源称 225 Hz）。
- **VPP2**（2610.10270）：事件级视频预训练 → 单步视觉规划蒸馏 → MoT 动作模块；RoboDojo-Sim 排名第一（39.26 分 / 32.26% SR）。论文动机原话：**现有 WAM 在开放场景中经常预测出错误的运动**。

**直接支撑"想象悖论"的证据：**
- **Faster-WAM**（2608.04404）：未来表征"不只是辅助训练信号，而是分布偏移下鲁棒动作预测所必需的推理时上下文"。
- **Temporal Ratio**（2607.08127，NeurIPS 2026）：动作头在微调后丢失了视频骨干的组合泛化能力；动作头对未来帧的注意力比例能预测泛化好坏。
- **WAM vs VLA 鲁棒性研究**（2603.22078）：LingBot-VA 在 RoboTwin 2.0-Plus 上 74.2%，Cosmos-Policy 在 LIBERO-Plus 上 82.2%。**扰动类型只有视觉和语言，没有物理参数扰动。**

**已经拥挤、建议避开的方向：**
- 失败检测 / 恢复：FAVOR（2610.06280，用 WAM 自己预测的未来锚点做验证，VLM 发纠正指令）、FARE（2609.18016）、FARM（2609.11445）、Foresight（2606.23085）。我原本想做的"惊奇驱动的元认知 WAM"已经被覆盖，所以放弃了。
- 记忆：MemoryWAM、HiMem-WAM。
- 加速：Flash-WAM、Faster-WAM、LingBot-VA 2.0 的异步推理，以及你自己的推测验证。
- 从人类视频做测试时适应：WAM-TTT（2607.06988）、Zero-WAM。
- Agent 友好接口：AR-WAM（2609.23578，用视觉点提示代替语言，**是静态的"在哪里"**）、DSWAM（双系统，System 2 输出子任务语言）。

---

## 2. 值得做的三个问题

1. **P1 想象悖论（泛化）**：OOD 时 WAM 的动作质量取决于想象质量，而 OOD 恰恰是想象最差的时候。WAM 无法自我纠正：更大的 teacher WAM 和 draft 来自同一训练分布，错误是相关的（你 V6 的记录里写过："两个数值上接近的动作块可能导致不同的物体/接触结果"）。需要一个**错误来源正交**的外部世界模型。
2. **P2 物理混叠（物理泛化）**：视觉泛化有网络视频预训练撑腰，但"看起来一样、物理不同"无法靠被动观察解决，必须主动交互。目前没有 WAM 层面的方案，也没有对应的 benchmark。
3. **P3 时间尺度错配（部署）**：最强的推理器是分钟级的，现实世界是 Hz 级的。现有的延迟补偿（AHEAD、DexWorldModel、RTC、LingBot-VA 2.0 异步）都针对**毫秒到亚秒级的策略自身延迟**。几十秒的延迟意味着场景会发生**子任务 / 语义层面**的变化，特征级外推不够用。

---

## 3. Idea 1：ProgAnchor —— 用可执行程序锚定 WAM 的想象

### 3.1 动机（写进 Introduction 的逻辑链）
1. Faster-WAM 和 Temporal Ratio 证明，**OOD 下动作好坏取决于未来想象的好坏**。
2. VPP2 等工作指出，**开放场景下 WAM 会预测出错误的运动**，长时域还会出现物体幻觉和漂移。
3. 现有 "VLM + 策略" 组合用语言传递意图，但语言说不清"物体应该沿什么轨迹、在何时、到达哪里"。AR-WAM 证明空间接口优于语言，但它的点提示是静态的。
4. Astra 的能力恰好互补：能把动态场景写成**可执行程序**（4DCodeBench 第一），几何最准；但不懂接触物理（Video2World 功能成功率 10.7%）。WAM 正好相反：分布内接触 / 外观动力学很好，OOD 下语义和几何会出错。

**核心 insight**：让 Astra 写出 **"应该发生什么"**，也就是一个以物体为中心、可执行的未来程序；让 WAM 补上 **"实际会怎样发生"**，包括接触、形变、机械臂自身外观，并据此解码动作。二者在每个时空位置上**按各自的可靠度加权融合**。

### 3.2 整体 Pipeline

```
          ┌──────────── 慢回路（Astra，事件触发，异步）────────────┐
 obs, 指令 ─►  Astra 感知 + 推理  ─► 程序 P = (场景图 S, 运动脚本, 不变量, 置信度)
          │                         │ 执行 / 渲染（毫秒级，无需物理引擎）
          │                         ▼
          │          代理未来 Π = {物体掩码 M_t, 3D 物体流 F_t, 深度 D_t}
          └─────────────────────────┬──────────────────────────────┘
                                    ▼
 快回路（WAM）：视频分支去噪 ── 模块 A：置信门控的 PoE 引导 ──► 被锚定的想象 z
                                    ▼
                   动作分支（按锚定程度自适应地加大对未来的依赖）──► 动作块
                                    ▼
           执行 → 跟踪物体 → 模块 B：程序-现实重对齐
              ├─ 小偏差：可微地重拟合程序参数（毫秒级，不调用 Astra）
              └─ 不变量被违反 / 残差超过共形阈值：带"差异报告"请 Astra 重写程序
```

程序示例（Astra 输出的是代码，不是语言）：
```python
scene = Scene(objects={"mug": Obj(pts=..., pose=T0), "rack": Obj(...)})
with phase("grasp", conf=0.9):    mug.stays()
with phase("lift",  conf=0.8):    mug.move_to(T0 @ up(0.12), dur=1.2)
with phase("hang",  conf=0.4):    mug.align(handle_axis, rack.peg_axis); mug.move_to(rack.peg_tip, dur=1.5)
invariant(upright(mug), until="hang"); goal(on(mug.handle, rack.peg))
```

### 3.3 模块 A：置信门控的专家乘积（PoE）想象引导

**借鉴**：组合扩散 / Product-of-Experts 能量引导（"Reduce, Reuse, Recycle"）、LLM-grounded Video Diffusion 的布局能量引导、DPS 类后验采样，以及 Temporal Ratio 的自适应引导。

**做法**：把想象写成两个专家的乘积：
$$p(z\mid o,l,P)\ \propto\ p_{\text{WAM}}(z\mid o,l)\cdot p_P(z)^{\lambda(p,t,\tau)}$$

在 flow matching 采样中（LingBot-VA 的视频循环，`wan_va_server.py` 里 `self.scheduler.step` 之前），修正速度场：
$$\tilde v = v_\theta(x_\tau) + \lambda(p,t,\tau)\odot\nabla_{x_\tau} E_P\big(\hat x_0(x_\tau)\big),\qquad \hat x_0 = x_\tau-\tau\,v_\theta$$

- $E_P$：在**潜空间**中计算（避免 VAE 解码）。训练一个轻量探针头 $h_\psi(\hat x_0)\to$（物体掩码，物体流），然后令 $E_P=\sum_t\sum_{obj}\mathrm{BCE}(h_\psi^{mask},\Pi^{mask})+w\lVert h_\psi^{flow}-\Pi^{flow}\rVert_1$。为了提速，可以对 $v_\theta$ 做 stop-grad，只对 $\hat x_0$ 求梯度（"DPS-lite"）。
- **门控权重是这个模块的关键**：
  $$\lambda = \lambda_0\cdot c_P(p,t)\cdot s\big(u_\theta(p,t)\big)\cdot\kappa(\tau)$$
  - $c_P$：Astra 在程序里为每个物体、每个阶段自报的置信度，渲染到 patch 上，再用校准集做 isotonic 校准。自由空间刚体运动置信度高，接触 / 形变阶段低。
  - $u_\theta$：WAM 的认知不确定度，用**不同噪声水平 τ 下终点重建的离散程度**来估计。你仓库里的 tau-verifier（`verify_world_latent_chunk`，τ∈{150, 300}）可以直接复用。
  - $\kappa(\tau)$：高噪声阶段引导强，低噪声阶段引导弱，因为布局在扩散早期就已确定，后期留给 WAM 补细节和接触。
  - 直观含义："**程序有把握、WAM 没把握的地方听程序；接触物理听 WAM**"。
- **可学习版本**：加一个零初始化的 cross-attention "程序适配器"，把 Π 编码成 token 注入视频流。训练所需的程序**从演示数据中自动抽取**：用现成跟踪器（如 SAM 2 / CoTracker）得到物体轨迹，再转成关键帧程序。训练时对程序做 dropout（p≈0.3）和扰动，让 WAM 学会"利用程序，但不盲从"。免训练引导版作为第一版和消融基线。
- **动作侧**：被程序确认的未来 token，在动作分支中获得更大的注意力权重。这是 Temporal Ratio 结论的直接利用：被锚定的未来更值得依赖。

### 3.4 模块 B：程序-现实重对齐（惊奇驱动的程序修正）

**借鉴**：MPC 滚动时域、卡尔曼式状态校正、代码 agent 的执行反馈自调试、EMPIRIC 的"失败驱动修正"，以及 Astra 在 ARC-AGI-3 里"记笔记、改世界模型"的行为。

两个层级：
1. **连续层（毫秒级，不调用 Astra）**：程序参数 $\theta_P$（关键帧位姿 $\xi_k\in SE(3)$、时间扭曲 $w(t)$）可微。用跟踪得到的实际物体状态做几步 Gauss-Newton：$\min_\theta\sum_t\lVert T^{obs}_t - T_P(t;\theta)\rVert$。物体滑了 2 cm、动作慢了半拍时，程序会"弯过来"贴合现实。
2. **离散层（秒级，调用 Astra）**：在跟踪状态上检查不变量谓词 $\phi_j(S_t)$。若某个谓词被违反，或残差 $r>q_{1-\alpha}$（$q_{1-\alpha}$ 是在校准 rollout 上得到的**共形分位数**），就把结构化的**差异报告**（想象关键帧 vs 实际关键帧、哪个谓词失败、跟踪数据）发给 Astra，让它重写程序结构（例如插入一次重新抓取），同时更新它的笔记。
3. 共形阈值让"误触发率 ≤ α"有理论保证，从而**可控地约束 Astra 的调用预算**。

**可选模块 C（降成本）**：把 Astra 生成的 (obs, 指令, P) 蒸馏进 WAM 内的一个小"程序头"，只在真正新的任务上调用 Astra。思路同 RHD，但蒸馏的是结构化 token。

### 3.5 为什么大概率有效
- LLM 布局引导视频扩散已被证明能显著提升组合生成；Faster-WAM / Temporal Ratio 证明更好的未来带来更好的动作。两段因果链都有先例。
- Astra 只负责它最强的部分（几何、语义、运动的"骨架"），接触物理留给 WAM，分工与两者各自的能力边界一致。
- 门控 λ 是安全阀：程序错了但 WAM 有把握时，WAM 不会被带偏；有 dropout 训练兜底。

### 3.6 与最接近工作的区别
| 工作 | 区别 |
|---|---|
| RPent / Harness VLA / RoboMME / DSWAM | 接口是语言或技能调用；本方案在**想象内部**用可执行的动力学程序干预 |
| AR-WAM | 静态视觉点提示（只说"在哪"）；本方案是随时间演化的程序（"怎么动"），加上门控引导和重对齐 |
| LVD / Code World Model（游戏）/ PROWBench | 只做视频生成，没有动作，也没有闭环重对齐 |
| FAVOR | 锚点来自 WAM 自己的想象，而 OOD 下出错的正是它；本方案的锚点来自**独立的符号世界模型**，错误来源正交 |
| Dream2Flow / FlowWAM / Skel-WAM | 方向相反（视频 → 流 → 机器人），或条件来自人手骨架；没有 LLM 程序 |

### 3.7 实验设计
- **基座**：LingBot-VA（本仓库，直接改视频循环）、FastWAM Optional-IDM（IDM 模式）、VPP2（Stage-2 / LIBERO）。用三个基座说明方法与架构无关。
- **Benchmark**：LIBERO-Plus、LIBERO-PRO、RoboTwin 2.0-Plus（OOD 扰动）、RoboDojo（已有 Astra 系列和 VPP2 的数字可以对比），外加真机"新物体 × 新指令"组合任务。
- **指标**：成功率；**想象保真度**（想象 vs 实际未来的物体掩码 IoU 和轨迹误差，分 ID / OOD 报告）；每个 episode 的 Astra 调用次数；延迟；tokens。
- **关键图**："想象误差 vs 成功率"散点图。基线在 OOD 下是一条强相关的下坡线，ProgAnchor 把这条线**拉平**。
- **消融**：语言子任务 / 视觉点提示 / 程序（接口之争）；免训练引导 vs 适配器；常数 λ vs 门控 λ；去掉重对齐；**用启发式程序代替 Astra**；更换推理器（Qwen-VL / GPT-6 Luna / Astra），画"推理器越强，收益越大"的 scaling 曲线，直接回答"为什么要用 Astra"。

### 3.8 两周验证门槛（先用 oracle 验证机制）
1. **验证问题存在**：在 RoboTwin 2.0-Plus 上统计 LingBot-VA 的想象误差（用 `last_video_latent` 对比实际编码的潜变量）与失败之间的相关性。不相关就说明想象悖论不成立，应该止损。
2. **验证机制有效**：用仿真中的**真值物体轨迹**生成 oracle 程序做引导。oracle 引导能显著提升成功率，再把 oracle 换成 Astra。
3. 两步都通过后，再做适配器训练和重对齐。

### 3.9 风险
程序本身出错（由门控、重对齐和 dropout 训练兜底）；Astra 的调用延迟（程序异步到达，到达前 WAM 无引导运行；也可以与 Idea 3 结合）；成本（事件触发 + 蒸馏）。

---

## 4. Idea 2：ProbeWAM —— 先试探、再想象，解决"物理混叠"

### 4.1 动机
- **物理混叠**：外观相同而隐藏物理不同。例如按压式抽屉和拉开式抽屉、空箱和装满的箱子（后者需要双臂）、左铰链门和右铰链门、上了锁的盖子、海绵和硬块、空杯和满杯。WAM 的想象以外观为条件，会**自信地**预测训练分布里"最典型"的动力学，然后用错误的力或方向执行。
- **证据**：现有 WAM 鲁棒性评测只扰动视觉和语言（2603.22078）。OneWorld（2609.30946）发现世界模型在同一场景、不同动作下的预测会隐含互相矛盾的质量 / 摩擦。"Physically Viable World Models"（2605.30542）指出质量、摩擦、粘度、隐藏接触几何无法被动推断，**必须主动交互**。DyWA 只做基于历史轨迹的被动适应。
- **Astra 的独特作用**：隐藏物理本质上是**假设检验问题**。Astra 在 ARC-AGI-3 上用少于人类的动作数归纳出环境规律，正是"高效主动实验"的能力。它的常识物理先验（"陶瓷杯约 300 g""抽屉通常是平移关节"）也能给出好的假设空间。
- **核心 insight**：**Astra 提出假设和实验，WAM 充当"实验室模拟器"预测每种假设下实验会看到什么**。选信息量最大的试探动作执行，再用结果在测试时更新 WAM 的物理上下文。整个过程不需要物理引擎。

### 4.2 整体 Pipeline
```
新物体 / 首次接触时残差高 ──► 进入试探模式
  1) Astra 写假设程序 H={h_k}：离散结构（平移 / 转动 + 轴、刚体 / 柔体、锁定 / 自由）
                              + 连续范围（质量、摩擦），并给出先验 π_k
  2) Astra 提出候选试探 Q={q_j}（短、安全、低力：轻推、抬 1 cm、沿轴轻拉）
  3) 模块 A：WAM 在每个 (h_k, q_j) 下想象结果 → 选期望信息增益最大的 q*
  4) 执行 q* → 模块 B：用 WAM 似然更新后验 π' → 测试时训练物理快权重 φ*
  5) 适应后的 WAM 执行任务；残差再次升高时回到第 1 步；识别结果写入 Astra 的物体记忆
```

### 4.3 模块 A：基于想象的贝叶斯实验设计
**借鉴**：贝叶斯最优实验设计（BOED）、主动学习的 BALD、主动系统辨识、"科学家 agent"。

$$\mathrm{EIG}(q)=\mathbb H\Big[\tfrac1K\textstyle\sum_k p_\theta(o\mid q,h_k)\Big]-\tfrac1K\sum_k\mathbb H\big[p_\theta(o\mid q,h_k)\big]$$

- 用 WAM 在每个假设上下文 $\phi(h_k)$ 下采样 M 个未来潜变量来近似：**假设间方差 / 假设内方差**（类似 Fisher 判别比）。
- 打分：$\mathrm{Score}(q)=\mathrm{EIG}(q)-\beta\,\mathrm{Risk}_{Astra}(q)-\gamma\,\mathrm{Time}(q)$。风险项由 Astra 评估（是否会打翻、是否用力过大）。
- 分工：**Astra 负责假设空间、候选试探和安全性，WAM 负责似然 / 模拟**。不依赖物理引擎，所以能直接用到真实世界。

### 4.4 模块 B：元学习的物理上下文快权重（测试时训练）
**借鉴**：RMA（快速运动适应）、TTT 层 / Titans 记忆、MAML、WAM-TTT 的 KV 记忆。

- **训练（仿真）**：固定外观，域随机化隐藏物理（RoboTwin 基于 SAPIEN，物理参数可以直接改）。轨迹形式为"试探前缀 + 任务"。物理 token $\phi$（K 个额外 KV token 注入 WAM 的注意力，或动力学层上的 LoRA）通过内循环梯度获得：在试探片段的预测损失上走几步。外循环优化适应后的视频 + 动作损失。
- **符号-神经对齐**：一个假设编码器 $g(h)\to\phi_0$，在仿真中与真值参数做对比学习对齐，这样 Astra 的后验可以直接初始化快权重：$\phi_0=\sum_k\pi'_k\,g(h_k)$。
- **后验更新**：$\pi'_k\propto\pi_k\exp\!\big(-\lVert\hat x(q,\phi(h_k))-x^{obs}\rVert^2/2\sigma^2\big)$，即以 WAM 作为似然。
- **测试**：在真实试探片段上走 1–3 步内循环，得到 $\phi^*$。
- **一致性正则**（借鉴 OneWorld）：同一个 $\phi$ 必须同时解释试探片段和任务未来，防止快权重过拟合在试探片段上。

### 4.5 附带贡献：PhysAlias-Bench
在 RoboTwin 2.0 / ManiSkill 中构造**外观完全相同、只有隐藏物理不同**的任务对，覆盖五类：质量 / 重心、摩擦（滑动 vs 倾倒）、铰接（方向、轴、锁定）、柔顺性、灌装量。再加少量真机任务。

### 4.6 实验
- **基线**：原版 WAM（LingBot-VA / FastWAM / VPP2）、DyWA 式被动历史适应、无试探的 TTT（只在任务交互上更新）、Astra 直接控制（PhysEvo 式）、随机试探、Astra 选试探但不用 WAM 估计 EIG、oracle 物理上下文（上界）。
- **指标**：成功率、试探次数和耗时、隐藏参数识别准确率、Astra 调用次数。
- **两周验证门槛**：在 RoboTwin 里先做 3 组混叠对，测出 LingBot-VA 的成功率掉了多少（验证问题存在）。再用 oracle 物理上下文微调，看能恢复多少（验证上下文通路有效）。
- **风险**：真机试探的安全性（由 Astra 风险项和低力限制约束）；元训练依赖仿真（先只做离散假设，再加少量真机数据微调）。

---

## 5. Idea 3：FutureTense —— 让慢脑在"将来时"里推理

### 5.1 动机
- Astra 每次调用平均 39.86 s，评测中**要暂停物理仿真**；RoboDojo 的真机测试因安全原因中止。真实世界不会暂停，尤其在动态场景里：有人移动物体、传送带分拣（LingBot-VA 2.0 的 demo 里就有）。
- 现有混合系统（RoboMME 三层架构、Harness VLA）的计划**是为一个已经不存在的状态算的**：计划到达时，场景早就变了。
- 现有的延迟补偿（AHEAD、DexWorldModel 推测预去噪、LingBot-VA 2.0 异步、RTC）处理的都是毫秒到亚秒级的策略自身延迟。几十秒的延迟下，场景会在**子任务 / 语义层面**变化，必须换一种机制。
- 这个问题会随时间加剧：Roboflow 的测试显示，高推理档位会让单张图的成本大约翻倍，推理越强延迟越长。所以这是一个**不会过时**的问题。
- **核心 insight**：控制理论里的 **Smith 预估器**处理纯时滞系统的办法，是让控制器基于"模型预测的输出"而不是延迟的测量值做决策，再用模型与实际的偏差校正。**WAM 天然就是这个过程模型**：它能想象 Astra 的答案到达那一刻世界的样子。

### 5.2 整体 Pipeline
```
t 时刻：估计 Astra 延迟 Δ̂（历史分位数，取决于推理档位）
   WAM 便宜地 rollout 到 t+Δ̂，采 M 个未来 → 模块 A 选出 N 个"决策相关"的分支
   Astra 一次批量调用：基于 {当前观测, N 个分支的想象关键帧} 为每个分支出计划 P^(n)
   与此同时 WAM 继续执行，世界不暂停
t+Δ 到达：模块 B 匹配实际状态 → 选分支 n* → 把计划搬运到实际状态 → 校验前置条件
   ├─ 匹配良好：提交
   ├─ 部分前置条件成立：部分提交
   └─ 所有分支都超过共形阈值：快速回退（小 VLM / WAM 自身）+ 以低推理档位重新查询 Astra
```

### 5.3 模块 A：决策相关的推测分支
**借鉴**：CPU 分支预测与推测执行、你自己的推测解码思路、MCTS 渐进展宽、DPP / particle guidance 的多样性采样。

- **问题**：分支数 N 直接决定 tokens 成本。朴素的多样采样会把分支浪费在光照、微小位姿这类不影响决策的差异上。
- **做法**：从日志中收集（想象未来, Astra 的计划类别）对，训练一个**决策嵌入** $e(\hat z)$，用来预测 Astra 会做出哪一类决策。先从 WAM 廉价采样 M≫N 个未来，然后在 $e$ 空间里用 DPP 或最远点采样选出 N 个，**只保留会改变决策的分歧**。
- 想象的时域也自适应：取延迟分布的分位数 $\hat\Delta$；N 根据预算在 1–4 之间调整。

### 5.4 模块 B：计划搬运与对账
**借鉴**：网游的服务器对账（reconciliation）、点集配准 / 最优传输、Smith 预估器的偏差校正。

- **计划必须可搬运**：要求 Astra 把计划写在**物体坐标系**下（锚定到物体 id 和关键点），而不是像素坐标。
- 到达时：分支选择 $n^*=\arg\min_n d(s_{t+\Delta},\hat s^{(n)})$，$d$ 是物体状态距离。用跟踪 + 特征匈牙利匹配建立"想象物体 → 实际物体"的对应，对锚点施加 SE(3) 变换得到 $\tilde P$，然后逐条检查前置条件谓词，只提交成立的部分。
- 偏差统计量 $d$ 用共形方法校准成阈值 $\delta_\alpha$，**同时控制误提交率和重查率**。偏差还会回传给延迟预测和 WAM 的想象校准。

### 5.5 与你现有代码的对应关系
| FutureTense | 你仓库里已有的部件 |
|---|---|
| 推测分支 | draft 的推测块（`RiskRouterClientPolicy`） |
| 到达对账 / 提交 | action / world-latent verifier 的接受或拒绝 |
| 计划搬运 | SVDR 的视频引导修复（`video_guided_blend`） |
| 共形阈值 | `risk_low` / `risk_high` 阈值体系 |

也就是说，你在**动作块时间尺度**上搭好的"推测-验证-修复"框架，可以上移一个时间尺度，用来调度**慢脑的计划**。故事是连贯的。

### 5.6 实验
- **实时协议**（本身也是一个贡献）：仿真**不暂停**，按真实耗时注入 Astra 延迟（实测或回放），加入脚本化的动态扰动（"人"移动物体、传送带）。
- **基线**：暂停世界（规划质量上界）、执行过期计划（RoboMME / Harness 式）、到达后重新查询、AHEAD 式特征外推、只用 WAM 不用 Astra。
- **指标**：成功率、完成时间、计划过期误差（stale-plan error）、tokens。
- **两周验证门槛**：在 RoboDojo 或 RoboTwin 上比较三种情况：暂停 vs 过期计划 vs 用**仿真真值未来状态**规划（oracle 将来时）。只要"过期计划"明显差、"oracle 将来时"能追回大部分差距，问题和机制就都成立了。

---

## 6. 备选 idea（简述）

- **B1 共识幻觉与不可逆提交**：draft 和 teacher WAM 来自同一分布，验证通过不等于正确（对应你 V6 的观察）。让 Astra 离线标注哪些动作阶段不可逆（松手、倾倒、切割、插入），训练一个小的不可逆性预测器，**只对不可逆的块**加做 world-latent 验证和 Astra 程序校验。类比 CPU 中 store 指令必须先提交。
- **B2 Astra 校验的梦境数据引擎**：WAM 针对失败模式生成反事实视频，Astra 写程序化不变量过滤掉幻觉"梦境"，再用 IDM 标注动作做后训练。即 DreamGen 加上程序化校验。
- **B3 无 Astra 推理**：只做 Idea 1 的模块 C。Astra 只在训练时充当老师，测试时零调用，成本最低。
- **B4 跨本体的"身体程序"**：Astra 用约 30 分钟探索一台新机器人，写出它的运动学 / 控制接口；WAM 的物体中心想象经轨迹优化转成新本体的动作，再自训练动作头。与 Dream2Flow 有部分重叠，优先级低。

---

## 7. 优先级与落地建议

1. **主线：Idea 1（ProgAnchor）**。问题最核心（泛化），证据链最完整（Faster-WAM、Temporal Ratio、VPP2 三篇互相印证），方法最 "fancy"（门控 PoE 引导 + 程序重对齐），与 Astra 的独特能力绑定最紧。
2. **最快出原型：Idea 3（FutureTense）**。可以复用 RiskRouter / SVDR 的工程。如果把实时协议做成标准评测，影响面会很广。
3. **新问题 + benchmark：Idea 2（ProbeWAM）**。适合想开一个新坑；仿真里 SAPIEN 物理参数可改，搭建成本低。
4. 三者可以组合成一个统一系统：**FutureTense 是时间层，ProgAnchor 是接口层，ProbeWAM 是物理层**。但写论文时建议拆开，每篇只讲一个问题。

**成本提醒**：Astra 的输入 $10/M、输出 $50/M。每次调用的 tokens 取决于图像数量和推理档位，务必先实测再估算总预算。建议：事件触发而非周期调用；消融和大规模扫描用 GPT-6 Luna / Sol 或开源 VLM；只在主结果上用 Astra。

---

## 8. 参考来源
- GPT-6 Astra：[Wikipedia](https://en.wikipedia.org/wiki/GPT-6_Astra) · [OpenAI 发布页](https://openai.com/index/gpt-6-astra-next-generation-work/) · [OpenRouter 定价](https://openrouter.ai/openai/gpt-6-astra) · [DataCamp](https://www.datacamp.com/blog/gpt-6-astra) · [ARC Prize：Astra on ARC-AGI-3](https://arcprize.org/blog/astra) · [Hard Vision, Easy Vision (2609.35718)](https://arxiv.org/html/2609.35718) · [Roboflow 视觉评测](https://blog.roboflow.com/gpt-6-astra-vision/)
- Astra × 具身：[Awesome-Astra-Embodied-AI](https://github.com/zjwzcx/Awesome-Astra-Embodied-AI) · [Systematically Exploring GPT-6 Astra as Embodied Policies (2609.38537)](https://arxiv.org/abs/2609.38537) · [Robot Manipulation with GPT-6-Astra (2609.31770)](https://arxiv.org/html/2609.31770v1) · [PhysEvo (2610.08995)](https://huggingface.co/papers/2610.08995) · [World Action Agent (2609.29964)](https://arxiv.org/pdf/2609.29964) · [4DCodeBench (2610.03715)](https://hyper.ai/en/papers/2610.03715) · [VLN-CE 评测 (2609.29861)](https://arxiv.org/pdf/2609.29861)
- WAM：[Fast-WAM (2603.16666)](https://github.com/yuantianyuan01/FastWAM) · [Faster-WAM (2608.04404)](https://github.com/hustvl/FasterWAM) · [LingBot-VA 2.0 (2607.08639)](https://arxiv.org/html/2607.08639v1) · [VPP2 (2610.10270)](https://github.com/roboterax/video-prediction-policy-2) · [Temporal Ratio (2607.08127)](https://arxiv.org/pdf/2607.08127) · [WAM vs VLA 鲁棒性研究 (2603.22078)](https://huggingface.co/papers/2603.22078) · [WAM-TTT (2607.06988)](https://arxiv.org/abs/2607.06988) · [AR-WAM (2609.23578)](https://arxiv.org/pdf/2609.23578) · [DSWAM (2607.04927)](https://arxiv.org/pdf/2607.04927) · [World Action Verifier (2604.01985)](https://arxiv.org/abs/2604.01985) · [awesome-vla-wam](https://github.com/DravenALG/awesome-vla-wam) · [OpenMOSS/Awesome-WAM](https://github.com/OpenMOSS/Awesome-WAM)
- 失败恢复（已拥挤）：[FAVOR (2610.06280)](https://arxiv.org/html/2610.06280) · [FARE (2609.18016)](https://arxiv.org/pdf/2609.18016) · [FARM (2609.11445)](https://arxiv.org/pdf/2609.11445) · [Foresight (2606.23085)](https://arxiv.org/html/2606.23085)
- 物理 / 延迟相关：[OneWorld (2609.30946)](https://arxiv.org/html/2609.30946v1) · [Physically Viable World Models (2605.30542)](https://arxiv.org/html/2605.30542v1) · [DyWA (2503.16806)](https://arxiv.org/html/2503.16806) · [AHEAD (2606.02486)](https://arxiv.org/pdf/2606.02486) · [DexWorldModel (2604.16484)](https://arxiv.org/pdf/2604.16484) · [awesome-code-world-models](https://github.com/yifliu3/awesome-code-world-models)
- 写作风格参考：[Sentinel-VLA (2605.01191)](https://arxiv.org/abs/2605.01191)
