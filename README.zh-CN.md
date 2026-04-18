# 分区贝叶斯错误率 · RLHF 偏好数据

> 📖 Languages · [English](README.md) · **中文**

> **代码与数据补充材料**，对应一份硕士研究计划书（东京大学 / Sugiyama-Ishida 研究室，2026）。本仓库承载为期两年的研究计划中「面向 RLHF 偏好数据的**校准感知贝叶斯错误率估计**」所做的前期实验。研究计划书本身是一份独立文档；本 README 提供完整的实证支撑。

---

## 核心想法

### 我们做什么

我们诊断 **RLHF 偏好数据里的分歧有多少属于不可约的标注员噪声，又有多少属于可约的评判器误差**，具体做法是把
[Ishida et al. ICLR 2023 oral](https://arxiv.org/abs/2304.01247) 提出的
无需实例的贝叶斯错误率估计器
`R̂* = (1/N) Σ min(c_i, 1−c_i)` 从软标签推广到 Bradley–Terry
偏好设定（[BT 1952] / Plackett–Luce /
[DPO; Rafailov 2023](https://arxiv.org/abs/2305.18290)），并在
**6 个开源 7B/8B LLM 评判器 × 2 个偏好数据集**（HelpSteer2、
UltraFeedback）上做压力测试。与此独立地，我们在
HelpSteer2-Preference（9125 对 × 3 标注员）上用无分类器的方式测量
**纯人类标注员**异质性，并构造了一个校准感知估计器
**`R̂*_CA`**（单调耦合情形下的间隔匹配），在一个已知解析 `R*`
的 2-高斯玩具问题上得到验证。

### 我们发现了什么（所有数字都在本 repo 中，见下方 C1–C10）

1. **在我们测试的两个数据集上，插入法估计器被评判器的*容量*混淆，而不是被数据侧噪声混淆。** 跨评判器
   `r(acc, R̂*_iso)` 在 HS2 上为 `−0.97`（Fisher-z 95% 置信区间 [−0.996, −0.719]），
   在 UF 上为 `−0.999`（置信区间 [−0.9999, −0.993]）。在同一数据上，
   更弱的评判器系统性地给出更高的 `R̂*`——估计器读的是分类器，不是标签。
2. **单调校准无法在多评判器情形下移除这种混淆。** Platt / Temperature / Isotonic / Beta
   ([Kull 2017](https://arxiv.org/abs/1707.01535)) 把每个评判器的 ECE
   降低了 10–20 倍，却*保留*了跨评判器的 `r(acc, R̂*) ≤ −0.94`。更糟的是，保序回归
   在 UF 上**退化成 `1 − accuracy`**（Spearman = −0.986）——正是我们校正估计器
   必须避免的、依赖数据集的失败模式。
3. **分区条件信号真实存在且与分类器无关。** HS2-Preference 上逐标注员错误率
   在 `|preference-strength|` bin（0 → 3）之间跨度 **3.3×**，95% 置信区间互不相交，
   计算过程不使用任何 LLM。这独立于「评判器混淆」那条叙事线地验证了
   「分区条件 R*|A_k」这一前提。
4. **在已知 `R*` 的合成数据上，校正方案是可行的。** 在已知解析解
   `R* = Φ(−1) ≈ 0.1587` 的 2-高斯玩具问题上，`R̂*_CA` 在 23× 的插入法波动
   （softmax T ∈ {0.1,…,5.0}）下保持在 0.001 以内。
   **我们暂不声称该校正在真实 LLM 评判器数据上同样有效**——
   那是第 1 年的工作。
5. **这个*问题*——但尚未包括其修复——在真实 LLM 上同样出现。** 在 Llama-3-8B + HS2 上，
   扫描条件 `P(A∨B, T)` 得到 **21× 的插入法跨度**，而准确率平坦（0.586）——即
   容量混淆不是合成实验的产物。（这是交叉验证，不是修复。）
6. **合成 Bradley–Terry 样本复杂度与理论一致。** 已知奖励的 BT 仿真中，
   `|R̂*_pref − R*|` 相对 `N` 的经验对数-对数斜率为 `−0.511`
   （理论 `−0.5`；插入法速率参考
   [Nguyen 2005](https://papers.nips.cc/paper/2005/hash/f8c59ccbf5d05b8d4b6e0d5f3b8c8f95-Abstract.html)、
   [Niu et al. 2013]）。

### 为什么对 RLHF 重要

现有流水线（DPO、PPO）把偏好对当作噪声同质的样本来处理。
我们的发现表明：**噪声在分区之间是异质的；任何用 LLM 评判器对它的测量都会被容量混淆；
而标准修复手段（单调校准）并不够用。** 研究计划书第 1 年的目标是给出形式化定理，
把 `R̂*_CA` 从合成数据上的概念验证推进为 BT 下可证的有限样本偏差界。
第 2 年把该偏差界转化为 RLHF 的组件：分区加权 DPO
（[Lodkaew et al. 2025]）、互补偏好学习
（[Ishida 2017](https://arxiv.org/abs/1711.10151) 脉络 × Yin & Ishida
2026 可扩展监督），以及跨语言 / 在线漂移检测。

*作者背景：硕士毕设做的是类别不平衡下的不规则时序分类（Neural LNSDE + 连续时间
Transformer，在 MACHO / LINEAR / ASAS / PhysioNet 上实验）。可迁移的角度是
在不平衡、不规则观测数据上做噪声诊断的经验；不主张任何形式化的同构关系。*

## 贡献 (preliminary)

**实证诊断**（强、已复现）：
- **C1** 6 个评判器 (Falcon / OLMo / Mistral / Qwen / Granite / Llama-3) ×
  HelpSteer2：`r(acc, R̂*_iso) = −0.97`，Fisher-z 置信区间 [−0.996, −0.719]——容量混淆真实存在（图 4）。
- **C6** UltraFeedback 跨数据集复现：`r = −0.999`，置信区间
  [−0.9999, −0.993]——混淆并非 HS2 特有（图 9）。
- **C3** HelpSteer2-Pref 9125 × 3 标注员：逐标注员错误率按
  `|strength|` bin 跨度 3.3 倍，置信区间互不相交，与分类器无关（图 2）。
- **C5** AB/BA 自一致性从 41%（Granite）→ 95%（Falcon）；
  偶然噪声下界随评判器变弱而放大（图 6）。

**合成理论锚**：
- **C4** 在已知 `R* = 0.159` 的 2-高斯玩具问题上，`R̂*_CA` 在 23× 插入法波动的跨度下
  仍能把真实值钉在 0.001 以内（图 3）。
- **C2** Platt / Temperature / Isotonic / Beta 全都*保留*了混淆
  （`r ≤ −0.94`），尽管 ECE 降低了 10–20 倍——单调校准
  结构上不够用（图 7）。

**观察性发现**（需谨慎解读——明确列出 caveats）：
- **C7** Llama-3 条件偏好温度扫描：插入法 `R̂*` 跨度
  21×，而准确率平坦；`r(signed_bias, gap) = −0.908`。等价于
  对 A∨B 条件分布的全词表温度缩放（配分函数相消）。相对合成实验
  `r = −1.00` 的 ~9% 残差本身是一个 real-world finding，*不是*完美复现（图 10）。
- **C8** 保序回归的混淆移除能力是**依赖数据集**的：在 HS2 上把跨评判器
  R̂* 范围压了 3 倍，但在 UF 上退化为 `1 − accuracy`
  (Spearman = −0.986)。正是这一 limitation 激发了基于间隔匹配的
  `R̂*_CA`（图 11）。
- **C9** HS2 上逐子样本重拟合保序回归得到对数-对数斜率 −0.750 (R² =
  0.968)。这是 empirical 观察，**不是**形式化的速率——早期版本使用固定校准器
  trivially 验证了 CLT (−0.500)，该实验已修正（图 12）。
- **C10** HS2 ↔ UF 6 评判器排名一致性：Spearman 0.77 (N=6)。两个
  数据集在协变量漂移和概念漂移上都有差异（人类多数 vs. GPT-4
  分数差金标），所以这是一项排名稳定性检查，*不是*一次
  干净的 OOD 迁移测试（图 13）。

## 第 1 年 / 第 2 年研究计划

**第 1 年 — 理论 + 估计器**
- **BT 下 `R̂*_CA` 的有限样本偏差界。** 把 [Nguyen 2005]、[Niu et al. 2013] 的插入法界
  与 Ushio et al. (ICLR 2026) 的保序速率扩展为联合分析形式：目标为
  `|R̂*_CA − R*| = O_P(M^{−1/3}) + O_P(N^{−1/2})`，其中 M = 校准器拟合样本量，
  N = 估计器样本量。C9 的 −0.750 是实证线索。
- **免校准偏好 R̂\*。** 完全避开保序回归 (C8 说明了为什么)。候选方案：
  基于 logit 差的 k-NN 密度比估计器，或借用 [Patrini et al. 2017]
  前向校正的转移矩阵表述。
- **Bradley-Terry 原生贝叶斯错误率。** 闭式解 `R̂*_pref = 1/2 −
  (1/2)E[|tanh(Δ/2)|]` 是我们 P3 的锚点；扩展到 Plackett-Luce (`K ≥ 2`
  个响应) 和群体式锦标赛。
- **LLM 评判器失败模式分类学。** 把「低 ECE ≠ 好评判器」
  (Falcon 边界样例，C5 / P6) 形式化为 4 象限诊断。与
  [Zheng 2023] (MT-Bench)、[Dubois 2024] (AlpacaEval LC)、位置偏差
  文献建立联系。

**第 2 年 — 下游应用**
- **基于 `R̂*_CA` 分区权重的 IW-DPO**（脉络：
  [Rafailov 2023] DPO + [Lodkaew et al. 2025] TMLR 重要性加权 DPO）。
  假设：下调高 `R̂*` 分区的权重可以让 AlpacaEval LC 相对均匀 DPO
  提升 ≥1.5pp。
- **互补标签偏好学习** —— 融合 [Ishida 2017] cL 脉络
  与 Yin & Ishida (ICLR 2026) 可扩展监督：把「A *不*
  比 B 差」建模为平局密集分区下的弱监督信号。
- **跨语言偏好漂移。** HelpSteer2 EN + OASST 多语言 +
  日语 RLHF 语料；以语言作为分区，R̂*_CA 作为漂移度量。
- **PPO 更新过程中基于 R̂*_CA 漂移的在线奖励黑客检测**
  （可行性对冲：需要一种容忍不规则更新节奏的平滑估计器；
  候选方案是对分区级 R̂*_CA 做序贯 CUSUM 式检验，成本仅
  相当于每个 checkpoint 做一次几百对样本的重估计）。

## 论断-证据对照表

| # | 论断 | 证据 | 指标（如适用，标注 95% 置信区间） |
|---|-------|----------|-----------------------------------|
| C1 | 跨评判器 `R̂*` 是一个容量代理 | [图 4](experiments/P2-ece-vs-rstar/fig_acc_vs_rstar.png) + [stats_with_ci_6judges.json](experiments/P2-ece-vs-rstar/stats_with_ci_6judges.json) | N=6 评判器；Pearson `r = −0.97`，Fisher-z 置信区间 [−0.996, −0.719]，`p = 0.002` |
| C2 | 没有单调校准器能去除该混淆 | [图 7](experiments/P5-calibration-showdown/fig_calibration_showdown.png) + [summary.json](experiments/P5-calibration-showdown/summary.json) | Platt/Temp/Iso/Beta 之后，`r(acc, R̂*) ∈ [−0.99, −0.94]`；ECE 下降 10-20 倍 |
| C3 | 数据侧异质性与分类器无关 | [图 2](experiments/P1-human-partition/fig_partition_hs2pref.png) + [partition_stats_with_ci.json](experiments/P1-human-partition/partition_stats_with_ci.json) | err_rate：k=0 → 0.331 [0.318, 0.344]；k=3 → 0.099 [0.091, 0.107]；置信区间互不相交 |
| C4 | 单调耦合情形下 `R̂*_CA` 钉住真实 `R*` | [图 3](experiments/P0-synthetic-confounding/fig_synthetic_rstar_v4.png) + [results_v4.json](experiments/P0-synthetic-confounding/results_v4.json) | 10 个 T 的跨度：插入法 0.371，`R̂*_CA` 0.001；均值 0.1573 vs 真实 0.1587 |
| C5 | BT 下 `O_P(N^{-1/2})` 速率成立 | [图 5](experiments/P3-synthetic-bt/fig_synthetic_bt.png) + [results_synthetic_bt.json](experiments/P3-synthetic-bt/results_synthetic_bt.json) | 对数-对数斜率 `−0.511`（理论 `−0.500`），在 9 个 N × 50 个种子上拟合 |
| C6 | 混淆并非 HelpSteer2 所独有 | [图 9](experiments/analysis_uf_6judges/fig_rstar_partition.png) + [stats_with_ci.json](experiments/analysis_uf_6judges/stats_with_ci.json) | UltraFeedback N=6 评判器；`r = −0.999`，Fisher-z 置信区间 [−0.9999, −0.993]，`p < 10⁻⁴` |
| C7 | 在真实 LLM 上做条件偏好温度扫描——插入法 `R̂*` 受校准混淆，保序不受 | [图 10](experiments/P7-llama3-tempscan/fig_tempscan.png) + [stats.json](experiments/P7-llama3-tempscan/stats.json) | Llama-3-8B，10 个 T 值：插入法跨度 0.358（21×），保序跨度 0.0008，acc 稳定在 0.586；`r(signed_bias, gap) = −0.908`，p = 2.9e-4。（全词表温度缩放可精确化简为 `P(A\|A∪B,T) = σ((l_A−l_B)/T)`——配分函数相消。） |
| C8 | 单调校准是依赖数据集的权宜之计——激发 `R̂*_CA` | [图 11](experiments/P8-baselines/fig_baselines.png) + [stats.json](experiments/P8-baselines/stats.json) | HS2：保序范围 0.084（比插入法 0.258 低 3 倍）；UF：保序范围 0.337 ≈ 1−acc 范围 0.367，Spearman(iso, acc) = **−0.986**（保序在 UF 上退化） |
| C9 | 联合（保序重拟合 + 插入法）HS2 样本复杂度的实证斜率为 −0.750 | [图 12](experiments/P9-hs2-sample-complexity/fig_sample_complexity.png) + [stats.json](experiments/P9-hs2-sample-complexity/stats.json) | 无放回子抽样，N ∈ {100…800}，100 个种子 × 6 个评判器。对数-对数斜率 = **−0.750**（R² = 0.968），比 CLT 参考值 −0.500 更陡——校准器拟合以非平凡方式耦合了偏差和方差 |
| C10 | 评判器排名在 HS2 与 UF 之间稳定 | [图 13](experiments/P10-cross-dataset-rank/fig_cross_dataset_rank.png) + [stats.json](experiments/P10-cross-dataset-rank/stats.json) | Spearman ρ = +0.77（p = 0.07）；Pearson r = +0.89，Fisher-z 置信区间 [0.30, 0.99]；两者金标定义不同（HS2 用人类多数，UF 用 GPT-4 分数差），因此这是排名一致性，而非 OOD 迁移 |

---

## TL;DR

- Ishida（ICLR 2023 oral）给出了一种从软标签估计贝叶斯错误率的无需实例估计器 `R̂* = (1/N) Σ min(c_i, 1-c_i)`。
- 在 RLHF 中，"软标签"来自 LLM 评判器，而它们存在**系统性的校准偏差**。直接代入会得到被**分类器容量**主导、而非数据侧噪声主导的估计值。
- 我们在一个 2-高斯玩具问题上实证了这一现象（**固定准确率下 R̂* 跨度达 23×**，图 1），并在 6 个 HelpSteer2 上的开源 7B/8B LLM 评判器上复现（**Pearson `r(accuracy, R̂*_iso) = -0.97`**，Fisher-z 置信区间 [-0.996, -0.719]，图 4）。
- HelpSteer2 的 9125 条人类标注员偏好展现出真实的 **3.3× 数据侧异质性**，分区强度 bin 之间差异显著（图 2）——且与分类器无关。
- 第一版校正估计器（保序 + 插入法）在玩具问题 10 个温度上把 `R̂*` 钉在真实值 ±0.001 以内（图 3），证明在单调耦合情形下该校正是可行的。第 1 年的研究目标是**破秩多模型**情形。
- *整个*单调事后校准家族（Platt / Temperature / Isotonic / Beta）都保留或加剧了容量混淆，而不仅仅是保序回归（图 7）。第 1 年的目标必须跳出这一家族之外。

---

## 八张头条图

### 图 1 — 插入法 R̂* 由校准而非数据噪声驱动

![P0 V3](experiments/P0-synthetic-confounding/fig_synthetic_rstar_v3.png)

在已知 `R* = Φ(-1) ≈ 0.1587` 的 2-高斯问题上训练的单个 MLP。扫描 softmax 温度 `T ∈ {0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0}` 对分类器本身没有任何改变（准确率稳定在 0.836），但使插入法 `R̂*` 的取值从 0.017 变化到 0.387（**23×**）。符号校准偏差完美预测了 `R̂*` 间隔差（Pearson `r = -1.00`，p ≈ 10⁻⁵³）。无符号 ECE 丢掉了符号信息，相关系数仅为 r = +0.44。脚本：[synthetic_demo_v3.py](experiments/P0-synthetic-confounding/synthetic_demo_v3.py)。原始数据：[results_v3.json](experiments/P0-synthetic-confounding/results_v3.json)。

### 图 2 — HelpSteer2 人类标注员分区（不涉及任何分类器）

![P1](experiments/P1-human-partition/fig_partition_hs2pref.png)

HelpSteer2-Preference 共 9125 对，每对由 3 名标注员按 `strength ∈ {0, ±1, ±2, ±3}` 打分。按 `|aggregate strength|` 分区：

| 分区 `|strength|` | n_pairs | 一致率 | 逐标注员错误率（相对多数） |
|---|---|---|---|
| 0 (tie) | 2007 | 0.671 | **0.331** |
| 1 (slight) | 3033 | 0.902 | 0.121 |
| 2 (moderate) | 2562 | 0.876 | 0.154 |
| 3 (strong) | 1523 | 0.917 | **0.099** |

逐标注员错误率跨度 3.3 倍，在不使用任何分类器或 softmax 的情况下得到。脚本：[analyze_preference.py](experiments/P1-human-partition/analyze_preference.py)。原始数据：[partition_stats.json](experiments/P1-human-partition/partition_stats.json)。

### 图 3 — 校正后的估计器有效（玩具问题，单模型情形）

![P0 V4](experiments/P0-synthetic-confounding/fig_synthetic_rstar_v4.png)

在相同的 10 温度网格上运行四种估计器：

| 估计器 | T 上跨度 | 均值 | 与 0.159 的间隔差 |
|---|---|---|---|
| 插入法原始 | **0.371** | 0.165 | 严重依赖 T |
| 1 − 准确率 | 0.000 | 0.161 | 平坦（argmax 不变） |
| **保序校准 R̂\*** | **0.001** | 0.157 | 钉在真实值 |
| 间隔匹配 R̂\*_CA | 0.001 | 0.157 | 钉在真实值 |

保序校准基于秩序，因此对温度缩放不变。在这种单调耦合情形下，校正后的估计器达到了目标一致性。脚本：[synthetic_demo_v4.py](experiments/P0-synthetic-confounding/synthetic_demo_v4.py)。原始数据：[results_v4.json](experiments/P0-synthetic-confounding/results_v4.json)。

### 图 4 — 多模型情形下仅靠保序不够（6 个评判器）

![P2](experiments/P2-ece-vs-rstar/fig_acc_vs_rstar.png)

六个开源 LLM 评判器（Llama-3-8B、Granite-3.0-8B、Qwen-2.5-7B、Mistral-7B-v0.3、OLMo-7B-hf、Falcon-7B）分别对 1000 条 HelpSteer2 对做 AB/BA 消偏打分。逐评判器的保序校准把 ECE 从 0.03–0.21 降到 0.03–0.04，**然而** `R̂*_iso` 仍与评判器准确率强相关：

| 评判器 | 准确率 | R̂*_iso | ECE_iso |
|---|---|---|---|
| Llama-3-8B | 0.587 | **0.386** | 0.037 |
| Granite-3.0-8B | 0.587 | 0.398 | 0.039 |
| Qwen-2.5-7B | 0.583 | 0.414 | 0.043 |
| Mistral-7B-v0.3 | 0.578 | 0.414 | 0.032 |
| OLMo-7B-hf | 0.518 | 0.461 | 0.029 |
| Falcon-7B | 0.519 | **0.471** | 0.030 |

Pearson `r(accuracy, R̂*_iso) = −0.97`，p = 0.002（Fisher-z 95% 置信区间 [−0.996, −0.719]；自助法置信区间 [−0.998, −0.822]，10K 次重采样）。由于底层 `(X, Y)` 分布在评判器之间是一致的，真实 `R*` 只有一个值；0.085 的跨度（Llama-3 低 → Falcon 高）反映的是**破秩**校准差异，单模型保序回归无法调和。脚本：[plot.py](experiments/P2-ece-vs-rstar/plot.py)。原始数据：[cross_partition_stats.json](experiments/analysis_hs2_6judges/cross_partition_stats.json)、[stats_with_ci_6judges.json](experiments/P2-ece-vs-rstar/stats_with_ci_6judges.json)。

### 图 5 — 已知奖励的合成 Bradley-Terry（理论锚点）

![P3](experiments/P3-synthetic-bt/fig_synthetic_bt.png)

在 100K 条响应上采样奖励 `r ~ N(0, 1)`；偏好通过 `P(A ≻ B) = σ(r_A - r_B)` 生成。闭式解 `R*_pref = (1/2) − (1/2) E[|tanh(Δ/2)|] = 0.27455`（与 `E[min(σ(Δ), σ(-Δ))]` 交叉验证至 10⁻⁵）。在 `N ∈ {100, …, 50000}` 上各做 50 个种子，插入法估计器经验上无偏（N = 50000 时平均 `R̂*` = 0.27455 ± 0.0005），且 `|R̂* - R*|` 的衰减对数-对数斜率为 **−0.511**（理论值 −0.500，偏差 0.011）。**从实证上验证了 BT 下 `O_P(N^{-1/2})` 的一致性。** 脚本：[synthetic_bt_demo.py](experiments/P3-synthetic-bt/synthetic_bt_demo.py)。

### 图 6 — AB/BA 自一致性：每个评判器的偶然噪声下界

![P4](experiments/P4-self-consistency/fig_self_consistency.png)

对每个评判器同时跑 AB 和 BA 两种提示顺序，计算两者在 argmax 胜者上分歧的频率。这是一个与分类器无关的偶然不确定性下界：

| 评判器 | 分歧率 [95% 置信区间] | 位置偏差 |
|---|---|---|
| Granite-3.0-8B | 0.412 [0.382, 0.442] | +0.280 |
| Qwen-2.5-7B | 0.434 [0.405, 0.463] | **−0.152**（偏好第 2 位） |
| Mistral-7B-v0.3 | 0.585 [0.556, 0.616] | +0.461 |
| OLMo-7B-hf | 0.660 [0.630, 0.690] | +0.273 |
| **Falcon-7B** | **0.954 [0.940, 0.966]** | +0.549（近乎全翻转） |

分歧率与准确率紧密相关：强评判器给出稳定的跨 pass 决策；弱评判器近乎抛硬币。这是 `R̂*` 捕获容量信号的第三条通道（除校准偏差与数据侧噪声之外）。脚本：[analyze_ab_ba.py](experiments/P4-self-consistency/analyze_ab_ba.py)。

### 图 7 — 校准方法大比拼：所有单调校准器都保留了混淆

![P5](experiments/P5-calibration-showdown/fig_calibration_showdown.png)

四种事后校准方法经 5 折 CV 拟合；跨评判器 Pearson `r(accuracy, R̂*_plug)` 及 Fisher-z 95% 置信区间：

| 方法 | r | Fisher-z 95% 置信区间 |
|---|---|---|
| raw | **−0.84** | [−0.989, +0.172]（置信区间跨越 0） |
| Platt（2 参数） | **−0.99** | [−0.999, −0.868] |
| Temperature（1 参数） | −0.98 | [−0.999, −0.691] |
| **Isotonic（非参数）** | **−0.99** | [−0.999, −0.822] |
| Beta（3 参数，Kull 2017） | −0.94 | [−0.996, −0.306] |

每一种校准器都把 ECE 降低了 10-20 倍。但 `r(accuracy, R̂*)` 反而*增强*了——从 −0.84（raw，不显著）增强到 ≤ −0.94（校准后）。容量混淆挺过了所有单调事后校准器；第 1 年的目标必须跳出这一家族之外。脚本：[calibration_showdown.py](experiments/P5-calibration-showdown/calibration_showdown.py)。

### 图 8 — 逐评判器可靠性图

![P6](experiments/P6-reliability-diagrams/fig_reliability_diagrams.png)

5×2 网格的可靠性图：raw（上行）和 isotonic-CV（下行）。Qwen/Mistral/Granite/OLMo 表现出典型的现代 LLM 过度自信（原始 ECE 0.13–0.22，经保序回归降至 0.04）。**Falcon 是一个富有启发的边界样例**：原始 ECE = 0.02，因为输出集中在 `p = 0.5` 附近（没有自信的决策）；保序回归反而轻微*恶化* ECE 至 0.034，原因是它注入了阶梯状伪像。**低 ECE 并不代表评判器好**——Falcon 的 95% AB/BA 分歧率（图 6）和最低的准确率（0.519）清楚地说明了这一点。脚本：[reliability_diagrams.py](experiments/P6-reliability-diagrams/reliability_diagrams.py)。

---

## 仓库布局

```
partitioned-bayes-rlhf/
├── README.md                                  (this file)
├── requirements.txt
├── LICENSE
├── .gitignore
├── src/
│   ├── build_pairs.py                         HelpSteer2 + UltraFeedback pair construction
│   ├── calibration_utils.py                   isotonic / temperature / ECE helpers
│   ├── bootstrap_ci.py                        percentile bootstrap + Fisher-z CI
│   ├── llm_judge_infer.py                     vLLM pairwise preference inference (AB/BA)
│   ├── analyze_partitioned_rstar.py           cross-judge R*_plug / R*_iso + permutation test
│   └── verify_params.py                       tokenizer / prompt-length sanity check
├── run_pipeline.sh                            orchestrates Steps 1–3
├── run_all_judges.sh                          5 judges × single-GPU parallel launcher
├── download_models.sh                         fetch 5 judges from HuggingFace (via HF-Mirror)
└── experiments/
    ├── P0-synthetic-confounding/              2-Gaussian toy, V1–V4 (Fig. 1, 3)
    ├── P1-human-partition/                    HelpSteer2-Preference strength analysis (Fig. 2)
    ├── P2-ece-vs-rstar/                       5-judge cross-correlation (Fig. 4)
    ├── P3-synthetic-bt/                       BT with known reward + N-scaling (Fig. 5)
    ├── P4-self-consistency/                   AB/BA aleatoric noise per judge (Fig. 6)
    ├── P5-calibration-showdown/               4 calibrators × 5 judges (Fig. 7)
    ├── P6-reliability-diagrams/               per-judge reliability plots (Fig. 8)
    ├── analysis_hs2_5judges/                  raw outputs of 5-judge pipeline
    └── judges_hs2/                            per-judge preference JSON (AB/BA)
```

`data/` 和 `models/` 被有意地加入了 gitignore——分别通过 `download_models.sh` 和 HuggingFace 数据集（HelpSteer2）下载。所有图形与 JSON 指标均进行版本化管理。

---

## 安装配置

### 硬件
在单张 96 GB RTX PRO 6000（Blackwell）上测试 7B BF16 推理。每个评判器大约占用 20 GB + KV cache。5 个评判器并行可从 5 张 GPU 获益；单 GPU 串行也能跑（N = 1000 对约 30 分钟）。

### 软件
```bash
conda create -n pbrhf python=3.12
conda activate pbrhf
pip install -r requirements.txt
```

精确的固定版本见 `requirements.txt`。关键：`vllm==0.19.0`、`torch==2.10.0`。已知问题——vLLM 0.19 + torch 2.10 组合需要在 `torch._inductor` 里打两个 assertion 补丁；详见下方 [Known issues](#known-issues)。

### 数据
HelpSteer2 rating 子集会在首次运行时自动从 HuggingFace 加载。对 P1（preference 子集）：

```bash
export HF_ENDPOINT=https://hf-mirror.com    # or omit if outside China
huggingface-cli download nvidia/HelpSteer2 --repo-type dataset \
    --local-dir data/hs2_extra \
    --include 'preference/preference.jsonl.gz' 'disagreements/disagreements.jsonl.gz'
```

### 模型
```bash
bash download_models.sh     # ~85 GB total across 5 models
```

---

## 复现四张图

### 图 1 (P0 V3) — 温度缩放混淆

```bash
cd experiments/P0-synthetic-confounding
python synthetic_demo_v3.py    # ~30 s on single GPU
```

输出：`fig_synthetic_rstar_v3.png`、`results_v3.json`。随机种子：42。

### 图 2 (P1) — HelpSteer2 人类分区

```bash
cd experiments/P1-human-partition
python analyze_preference.py    # ~10 s, CPU only
```

需要 `data/hs2_extra/preference/preference.jsonl.gz`（见「安装配置」）。

### 图 3 (P0 V4) — 校正估计器的概念验证

```bash
cd experiments/P0-synthetic-confounding
python synthetic_demo_v4.py    # ~30 s on single GPU
```

### 图 4 (P2) — 5 评判器相关性

前置条件：先完成 5 评判器推理流水线。

```bash
# Full pipeline: build pairs → 5-judge inference → cross-judge analysis
bash run_pipeline.sh 1000

# Then the scatter plot:
cd experiments/P2-ece-vs-rstar
python plot.py
```

`run_pipeline.sh` 在 5 张 GPU 上并行约 30 分钟（或单张 GPU 串行约 2.5 小时）。

---

## 完整结果表格

### P0 V3 — 原始数据

| T | acc | avg_conf | signed_bias | ECE | sharpness | R̂* | 与 0.159 的间隔差 |
|---|---|---|---|---|---|---|---|
| 0.10 | 0.836 | 0.983 | +0.148 | 0.111 | 0.484 | 0.017 | −0.142 |
| 0.20 | 0.836 | 0.967 | +0.132 | 0.128 | 0.467 | 0.033 | −0.126 |
| 0.30 | 0.836 | 0.951 | +0.116 | 0.115 | 0.451 | 0.049 | −0.110 |
| 0.50 | 0.836 | 0.918 | +0.083 | 0.083 | 0.418 | 0.082 | −0.077 |
| 0.75 | 0.836 | 0.879 | +0.044 | 0.044 | 0.379 | 0.121 | −0.038 |
| 1.00 | 0.836 | 0.843 | +0.008 | 0.009 | 0.343 | 0.157 | −0.002 |
| 1.50 | 0.836 | 0.783 | −0.052 | 0.052 | 0.283 | 0.217 | +0.058 |
| 2.00 | 0.836 | 0.738 | −0.098 | 0.098 | 0.238 | 0.262 | +0.104 |
| 3.00 | 0.836 | 0.676 | −0.160 | 0.160 | 0.176 | 0.324 | +0.165 |
| 5.00 | 0.836 | 0.613 | −0.222 | 0.222 | 0.113 | 0.387 | +0.228 |

### P0 V4 — 校正估计器的稳定性

| 估计器 | T 上跨度 | T 上均值 |
|---|---|---|
| 插入法 R̂*_raw | 0.371 | 0.165 |
| 1 − 准确率 | 0.000 | 0.161 |
| 保序 R̂*_iso | 0.001 | 0.157 |
| 间隔-CA R̂*_CA | 0.001 | 0.157 |

### P1 — 逐分区统计

| 分区 | n_pairs | 一致率 | 错误率 | Strength 方差 |
|---|---|---|---|---|
| 0 | 2007 | 0.6714 | 0.3312 | 0.038 |
| 1 | 3033 | 0.9017 | 0.1206 | 0.172 |
| 2 | 2562 | 0.8763 | 0.1539 | 0.401 |
| 3 | 1523 | 0.9173 | 0.0993 | 0.217 |

### P2 — 5 评判器全量指标

| 评判器 | n | 相对金标的准确率 | R*_raw | R*_iso | ECE_raw | ECE_iso |
|---|---|---|---|---|---|---|
| Falcon-7B-Instruct | 999 | 0.5185 | 0.4699 | 0.4705 | 0.0279 | 0.0300 |
| Granite-3.0-8B-Instruct | 999 | 0.5866 | 0.2438 | 0.3980 | 0.1800 | 0.0393 |
| Mistral-7B-Instruct-v0.3 | 999 | 0.5776 | 0.2947 | 0.4141 | 0.1368 | 0.0315 |
| OLMo-7B-Instruct-hf | 999 | 0.5175 | 0.3325 | 0.4607 | 0.1540 | 0.0292 |
| Qwen2.5-7B-Instruct | 999 | 0.5826 | 0.2187 | 0.4140 | 0.2120 | 0.0427 |

`R*_iso` 的跨评判器统计：均值 0.4315，标准差 0.0287，方差 8.21e-4，方差的自助法 95% 置信区间 [1.0e-5, 2.7e-4]，置换检验 p 值 0.000。

---

## 已知问题

### vLLM 0.19 + torch 2.10 兼容性

在较新的 torch 版本上，inductor 后端在模型加载时会以两个 `AssertionError` 崩溃。当前变通方案是打一个小的源码级补丁：

```bash
# In-place patch 1: torch/_inductor/select_algorithm.py line ~1695
# Replace: assert name not in self.all_templates, "duplicate template name"
# With:    if name in self.all_templates: return

# In-place patch 2: same file line ~2164
# Replace: assert not hasattr(extern_kernels, name), f"duplicate extern kernel: {name}"
# With:    if hasattr(extern_kernels, name): self.name = name; return
```

另外，`vllm/model_executor/models/falcon.py:171` 里针对 Falcon 还需修一个缺失属性：

```python
# Before:
rope_parameters=config.rope_parameters,
# After:
rope_parameters=getattr(config, "rope_parameters",
                         {"rope_theta": getattr(config, "rope_theta", 10000.0)}),
```

务必给 `LLM(...)` 传 `enforce_eager=True` 以绕过 torch.compile 路径。

### Tokenizer A/B 多 ID 问题

每个评判器都会根据前置空白把 "A"/"B" 编码成多个 token ID。参见 `src/llm_judge_infer.py` 中的 `get_ab_token_ids()`。处理不当会在某些模型上丢掉约 50% 的概率质量。

### OLMo 模型变体

请使用 **`allenai/OLMo-7B-Instruct-hf`**（HuggingFace 原生架构），而非带有 vLLM 不支持的遗留 `OLMoForCausalLM` 架构的 `allenai/OLMo-7B-Instruct`。此外注意 OLMo 的 `max_position_embeddings = 2048`；脚本对 OLMo 和 Falcon 设置了 `max_model_len=2048`。

---

## 引用

如果您基于本代码开展研究，请引用以下三篇基石论文：

```bibtex
@inproceedings{ishida2023good,
  title={Is the Performance of My Deep Network Too Good to Be True? A Direct Approach to Estimating the Bayes Error in Binary Classification},
  author={Ishida, Takashi and Yamane, Ikko and Charoenphakdee, Nontawat and Niu, Gang and Sugiyama, Masashi},
  booktitle={ICLR},
  year={2023}
}

@article{wang2024helpsteer2,
  title={HelpSteer2: Open-source dataset for training top-performing reward models},
  author={Wang, Zhilin and others},
  journal={arXiv:2406.08673},
  year={2024}
}

@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Kwon, Woosuk and others},
  booktitle={SOSP},
  year={2023}
}
```

---

## 许可证

MIT（代码）。图形与数值结果使用 CC-BY-4.0——衍生作品中使用时请注明本仓库出处。
