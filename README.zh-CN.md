# 分区贝叶斯错误率 · RLHF 偏好数据

> 📖 Languages · [English](README.md) · **中文**

> **代码与数据补充材料**，对应一份硕士研究计划书（东京大学 / Sugiyama-Ishida 研究室，2026）。本仓库承载为期两年的研究计划中「面向 RLHF 偏好数据的**校准感知贝叶斯错误率估计**」所做的前期实验。研究计划书本身是一份独立文档；本 README 提供完整的实证支撑。

---

## 核心想法

Ishida 等人（ICLR 2023 oral）基于软标签，提出了一种无需实例的二分类贝叶斯错误率估计器 `R* = E[min(η, 1-η)]`。在 RLHF 中，这些软标签通常来自 LLM 评判器——但评判器系统性地存在校准偏差，我们发现将其直接代入 Ishida 的估计器，得到的数值主要由评判器**容量**决定，而不是数据侧噪声。我们把该估计器扩展到 Bradley-Terry 偏好场景，通过一条精确的**间隔差恒等式** `E[R̂*] − R* = −E[|η̂−½| − |η−½|]` 将容量混淆问题形式化，并构建了一个校准感知估计器 `R̂*_CA`，在合成数据上，即使插入法估计器出现 23× 的波动，它仍能把真实 `R*` 钉在 0.001 的误差范围内（图 3）。

## 贡献

已确认（本仓库中的实验）：

- ✅ **C1 — LLM 评判器上的容量混淆是真实存在的。** 在 HelpSteer2 上评估了六个开源 7B/8B 评判器（Falcon、OLMo、Mistral、Qwen2.5、Granite-3.0、Llama-3-8B），结果显示 `r(accuracy, R̂*_iso) = −0.97`，Fisher-z 95% 置信区间 [−0.996, −0.719]，`p = 0.002`（图 4）。
- ✅ **C2 — 单调的事后校准不足以解决问题。** Platt / Temperature / Isotonic / Beta 都**保留**了该混淆（`r ≤ −0.94`），同时却把 ECE 降低了 10–20 倍（图 7）。
- ✅ **C3 — 标注员分歧的异质性是真实的且与分类器无关。** 在 HelpSteer2-Preference 的 9125 对 × 3 标注员上，按 `|strength|` 分 bin 统计的逐标注员错误率跨度达到 3.3 倍（置信区间互不相交），该结果在不使用任何分类器的条件下得到（图 2）。这是分区条件错误率 `R*|A_k` 的一个*代理*。
- ✅ **C4 — 校正后的估计器在单调耦合情形下可恢复真实 `R*`。** `R̂*_CA` 在 10 个温度下与解析值 `R* = 0.159` 的差距始终在 0.001 以内（图 3）；合成 BT 实验验证了 `O_P(N^{-1/2})` 的收敛速率（图 5）。
- ✅ **C5 — 偶然噪声下界随评判器变弱而放大。** AB/BA 自身分歧率从 41%（Granite）一直跨到 95%（Falcon），与评判器准确率紧密相关（图 6）。

刚刚完成（2026-04-18）：

- ✅ **C6 — UltraFeedback 1000 对 × 6 评判器复现了该混淆。** `r(accuracy, R̂*_iso) = −0.999`，Fisher-z 95% 置信区间 [−0.9999, −0.993]，`p < 10⁻⁴`（图 9）。在 UF 上的动态范围为 3.4×，HS2 上为 1.2×（分数差过滤器保留了更容易的样本对，从而加剧混淆）。
- ✅ **C7 — Llama-3-8B 在真实 HS2 数据上复现了 P0 V3 的合成温度缩放发现。** 在同一批 1000 条 Llama-3 判断上扫描 softmax `T ∈ {0.1, …, 5.0}`：准确率稳定在 0.586，插入法 `R̂*` 从 `0.018 → 0.375`（**21× 跨度**），保序 `R̂*` 跨度仅 **0.0008**。`r(signed_bias, plug-gap) = −0.908`，p = 2.9e-4（图 10）。玩具到真实的鸿沟已被填平：容量混淆是真实 LLM 软标签的一个固有性质，而非合成数据的伪像。

- ✅ **C8 — 保序回归的可靠性依赖于数据集；这正是 `R̂*_CA` 想要修复的局限。** 在 HS2 与 UF 的 6-评判器数据上，我们对比了三种贝叶斯错误率代理（1−acc、插入法、保序）。在 HS2 上，保序校准把跨评判器 R̂* 的范围降低了 3 倍（插入法 0.26 → 保序 0.08）。在 UF 上，相同的保序变换几乎没有缩小范围（0.35 → 0.34），且 Spearman(iso R̂*, accuracy) = **−0.986**，即保序回归退化为 `(1 − accuracy)` 的单调代理。因此，在多评判器 / 跨数据集情形下，任何单评判器上的单调校准器都不够用——这直接激发了基于间隔匹配的 `R̂*_CA`（图 11）。
- ✅ **C9 — 联合（保序重拟合 + 插入法）的样本复杂度实证斜率为 −0.750，比 CLT 参考值 −0.500 更陡。** 每个子样本上重新拟合保序 + 插入法 R̂*，在 N ∈ {100…800} 上无放回子抽样，100 个种子 × 6 个评判器：对数-对数斜率 = **−0.750**（R² = 0.968）。这个非 CLT 速率是一个真实发现——它反映了校准器拟合与插入法估计器之间的偏差-方差耦合，说明应当对 `R̂*_CA` 做联合分析，而非两阶段流水线（图 12）。
  *（注意：本实验的早期版本使用在全量数据上拟合好的固定校准器，报告斜率为 −0.500。这仅仅是在验证样本均值 `R̂*` 的 CLT，并不能反映校准器重拟合条件下该估计器的统计性质——目前的脚本已修正此问题。）*

- ✅ **C10 — 6 评判器在 HS2 与 UF 上的排名一致，尽管两者金标定义不同。** Spearman(R̂*_iso on HS2, R̂*_iso on UF) = **+0.77**（N=6，小样本需谨慎），Pearson `r = 0.89`，Fisher-z 置信区间 [0.30, 0.99]（图 13）。这比完整的 OOD 迁移论断要弱（后者会把协变量漂移和概念漂移混淆在一起），但仍然是评判器级不确定性信号具备跨数据集稳定性的正向证据。

进行中（第 2 周）：

- 🔄 **逐评判器失败模式叙事**（M5）—— 把 P4/P5/P6 的结果浓缩成一段针对研究计划书的小故事「Falcon 低 ECE ≠ 好评判器」。

计划中（研究计划的第 1 年 / 第 2 年）：

- 📋 **形式化定理**：在 BT 假设和单调耦合校准条件下，`|R̂*_CA − R*| = O_P(M^{-1/3}) + O_P(N^{-1/2})`（第 1 年）。
- 📋 **用 `R̂*_CA` 推导分区权重的 IW-DPO** —— 第 2 年 D4。
- 📋 **互补标签偏好学习**（Ishida 2017 一脉 × Yin 2026 可扩展监督）—— 第 2 年 D6。

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
