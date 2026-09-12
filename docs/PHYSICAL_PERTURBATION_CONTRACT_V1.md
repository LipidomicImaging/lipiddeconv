# PHYSICAL_PERTURBATION_CONTRACT_V1

2026-09-12。状态：FROZEN_SUPPORTED_SCOPE。本合同冻结目前有数据依据的扰动子集，不声称完整覆盖真实 MSI 误差。V2 已关闭；没有 V2.1、空间权重改版、新 rho 或新架构。下一项任务名称固定为 **主线 identity robustness pilot**，不再增加前置 CE 审计。

## 1. Shared fragment 的系统性强度变化

直接使用 `results/ce133_joint_variation_audit/observed_joint_changes.json` 中同一 donor identity、同一 CE 对、同一 shared fragment 集合的完整 centered-log-change 向量。源端和目标端必须都为正值、有效且有明确峰归属。不得逐 fragment 抽样、拼接极值、重新拟合 q95 box、采样独立 Gaussian、按结果裁剪倍率或选择 PCA 维数。

本版使用原 v1 已有的十个不同身份样本量参照作为冻结的保守准入政策：exact rule × CE pair × 完整 shared-channel set 至少十个不同身份。它不是置信区间或独立重复数保证，也不因为本轮结果而扩大到其他规则。符合的五个 stratum 及全部观测向量封存在 `joint_patterns.json`。支持范围是 PEO-H 的三/四个 shared fragment，完整匹配现有 component 映射的 69/391 个 production candidates；其余 322 个固定。不得把局部支持推广为全谱库覆盖。

三 fragment：sn2_fragment、sn2_ketene_loss、sn2_loss。四 fragment 另含 sn2_fragment_co2。这里 shared 不等于经过独立验证的 strong；不新增强/弱峰强度阈值。所有支持内正峰均保持正值，不做随机消失；不声称它们的强度恒定或始终高于检测阈值。

允许的变化是**有限的完整实测方向端点**，加 nominal；只用实际 low-CE→high-CE 方向，不反向扩展，不插值或外推，不按任意 severity 系数放大。每个 dataset 先固定一个 donor CE pair；一个 identity 的完整向量一次选择后对全部 pixel 固定。同一 rule、同名 identity、同 channel set 的重复候选共用 donor 向量。跨身份 donor 转移是明确的开发假设；固定 donor CE pair 只是建模约束，不能证明库中所有脂质共享同一个物理响应。

沿用现有 fragment component 和 isotope/profile response 映射，记 D_jc 为候选 j 的物理 fragment component，a_j^fixed = a_j − Σ_c D_jc。对一个完整向量 δ，定义：

```text
t_j(δ) = a_j^fixed + Σ_c exp(δ_c) D_jc
a_j_target(δ) = ||a_j||2 * t_j(δ) / ||t_j(δ)||2
U_j_supported = {a_j} ∪ {a_j_target(δ): δ is an allowed whole donor vector}
```

nominal 分支直接保留原 a_j。同一 fragment 的全部 isotope sticks / profile bins 共用倍率。映射外 fragment、precursor 及其他固定部分在列归一化前不加扰动；统一列归一化会共同缩放它们，不称 precursor 的绝对强度不变。不得产生新 support、m/z shift、负值或任意新强峰。列归一化只固定谱形尺度，不能转为修改 X_true 的 per-identity abundance。沿用已有空间/abundance construction 及一个全局 dataset signal scalar，production A_solver 始终是原固定谱库。

不同 donor 的 CE 强度比转移到预测库，是 **CE-informed controlled systematic mismatch**；不是已测得的 library→当前 MSI 预测误差分布。CE30/35/40 来源已用于 production adapter 训练，不能称独立实测验证。

## 2. Weak-peak censoring

`status = NOT_ESTIMATED`。现有保留表没有同一身份、同 CE 的重复观测，不能把 mask、归属变化或未列出的峰当作漏采率。本 pilot 不增加 dropout、强度截断或 detection threshold 模拟；这表示排除该机制，**不表示真实 censoring 为零**。缺少这一估计不延期受限主线 pilot。

## 3. Pixel-level fluctuation 与测量噪声

`status = NOT_SEPARATELY_IDENTIFIABLE`，范围限定为当前已验证证据：尚不能把真实 MSI 的空间组成差异与测量谱形波动分离；不是一般不可能性结论。不指定“小幅 sigma”，不默认比系统误差小，不把 CE 变化当 pixel noise。首次 pilot 的 A_target 对 pixel 固定，B_p = A_target X_true,p；不添加未经估计的 measurement noise。省略这些项是本轮适用边界，不是估计值为零，也不阻止 pilot。

## 4. 主线 pilot 的唯一后继入口

固定谱库 → 本合同支持的系统性扰动 → 完整 production solver → X_hat → identity confidence。不新增 architecture、rho、K ladder、spatial variant、CE benchmark；不因结果弱而改变本合同。原 production training、early-stop/hard cap、报告 gate、rho_zero 和分子身份定义保持原实现。missing-library 继续是必须面对的挑战，不靠未扰动身份的好结果掩盖支持内 truth 的失败。

pilot 实现阶段一次性封存 case、MODEL/CAL/EVAL donor membership、primary truth population/分母、运行数量和推断实现，随后运行。此项封存属于这一次主线 pilot 的准备，不是另开审计或实验。donor identity 的所有 CE 记录/端点以及共享来源谱文件的记录必须同组；inference uncertainty 不得含 CAL/EVAL target donor 的端点或利用它们选择范围。评估 target 及 donor ID 不传入 confidence 方法。已曝光 V2 case 只可用于程序检查，不做新性能选择；不根据筛选结果换 donor/case。

若有限生成端点本来就在推断字典中，结果只能称“字典含真值的表示检查”，不能作为未见失配鲁棒性 GO。本合同冻结生成侧允许向量全集；pilot 的推断侧 U 只能来自预先指定的 MODEL 部分，不能悄悄使用整个全集。CE 数据对 production 的历史训练暴露须在最终结论中保留，因此第一次仍是开发 pilot。

full/delete 必须共享同一观测与同一推断模型；删除全部同名 candidate 和 associated components。单列 separability margin 仅作可选诊断，不替代观测判据。若为计算使用 cone/convex relaxation，须明确它允许同一 identity 的多个端点混合，**不属于本合同有限物理端点集**：放松问题的 deletion lower bound 可作为保守下界；full-model acceptance 必须提供原物理模型允许的 witness，不能靠放松模型的低残差冒充相容性。优化失败或没有有效证明只允许弃权。

开发目标保持经验 FDP≤1%、TP retention≥40%，60%仅为强成功；正式 FDR 结论需要独立验证和统计不确定性。预先明确支持内受扰动 truth 的 retention，并分列全部 truth、未扰动 truth、missing-library 各 arm、distinct identities 与 abstention；禁止靠 fixed 候选稀释分母取得“失配鲁棒性”结论。solver misses 与 filter-induced misses 分开，未保留 truth 不从原分母移除。

一次 pilot 后按封存合同 GO/NO-GO。成功再作独立正式验证；失败保存结果并停止该版，不改本合同重试已曝光 EVAL。本合同无需等待 weak censoring/pixel 数据；以后得到直接重复测量证据才另行扩展适用范围。

## 封存与执行状态

机器可读合同、固定 donor 向量、candidate/component 映射及内容哈希位于 `results/physical_perturbation_contract_v1/`。构建器只做原记录提取/映射和封存校验，不生成 B、不运行 solver、不读取 pilot outcomes。V2 原结果与文件均不变。本轮完成物理合同并 Git 保存；下一项直接是主线 identity robustness pilot，当前尚未启动 GPU 或新评分。
