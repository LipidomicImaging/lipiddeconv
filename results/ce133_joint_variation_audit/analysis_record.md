# CE 成组变化的有界检查 — 2026-09-12

V2 保持关闭，不做 V2.1，不修改空间权重，不启动 open-set V3。本轮只检查已有 CE 数据及 v1 的构造，确认相关扰动建模有什么依据；不拟合新的 uncertainty model，不运行 solver、训练、rho、删除检验或新模拟。

## 已确认的模型结构

`analysis/build_ce133_uncertainty.py` 的 `main()` 按 rule/channel 导出独立的 fragment 倍率上下界；同一 fragment 的 isotope sticks 和原始 instrumental response 共享倍率。`analysis/run_ce_uncertainty_identity_pilot.py` 的 `model()` / `Problem.__init__()` 对每个 component 设置 `lower*x <= w <= upper*x`。共同 abundance 保留，但不同 fragment 的实测联合变化没有作为约束进入 LP。

因此“丢失了 fragment 间的联合约束”属实。“所有通道都独立变化”不准确；“这个 box 导致了 V2 失败”仍 NOT_VERIFIED。边际区间的笛卡尔积允许未观测的组合，但“未观测”也不等于“物理不可能”。不能因为 V2 失败就直接收窄区间，或把实测有限点集宣称为全部真实误差的覆盖集。

## 现有联合信息

只读取原 CE 导出及 provenance 直接指向的 2.2 MB target 表。原 244 个有效配对涉及 120 个身份、302 条参与配对的来源记录。重新读取原强度后，中心化 log change 与冻结导出的最大差异为 4.44e-16。所有输入小文件哈希及输出哈希保存在 `provenance.json`。

本轮保存了每个配对的**完整共同 fragment 变化向量**、原强度、相对最强 shared fragment 的强度、CE 对、身份及来源文件，不再只保存每个 fragment 的独立上下界。它们是观测记录，不是新合成谱或已冻结的允许误差集合。

按相同 rule、CE 对、完全相同 shared fragment 集合分为 123 组，其中 90 组只有一个身份。采用原 v1 的 n=10 作为描述性参照，达到这个样本量的只有五组，均为 PEO-H，分别有 10–20 个身份。这个分组用于避免把不同 centering support 或 CE 跨度混在一起；它不证明其他组合完全不能通过预先定义的公共 fragment contrasts 分析，也不是新的建模门槛。

| PEO-H fragment 数 | CE 对 | 身份数 | 第一方向的样本方差占比 |
|---|---|---:|---:|
| 3 | 30→35 | 20 | 72.89% |
| 3 | 30→40 | 17 | 83.95% |
| 3 | 35→40 | 19 | 84.40% |
| 4 | 30→35 | 13 | 68.52% |
| 4 | 30→40 | 10 | 87.35% |

这些是固定 CE 对内、跨身份中心化 log change 的描述性 SVD。三 fragment 为 sn2_fragment / sn2_ketene_loss / sn2_loss；四 fragment 另含 sn2_fragment_co2。各组身份可能重叠、来源谱可能共享，不能把五组相加为独立样本。没有用方差解释率挑模型维数。

**不能把上表当成已证明的低维物理模型。** 每行变化之和为零，d 个 fragment 的秩天然不超过 d−1。三个 CE 条件在单身份上的中心化轨迹秩天然不超过 2。62 个具有三组有效配对的身份，在共同 support 上满足 Δ30→40 = Δ30→35 + Δ35→40，最大代数误差 8.88e-16；三组差异不是三次独立技术重复。旧相关表中的 PEO-H sn2_fragment / sn2_fragment_co2 相关系数 0.7745（15 身份）也带有 compositional centering 及 CE/support 混合，不能直接拿来定义预测误差 covariance。

## 强峰、弱峰与 pixel：哪些尚未测到

- 这里的“最强 shared fragment”只用于无新阈值的幅度诊断。所有入选渠道本来就要求两端正值、有效、无歧义归属。因此这个子集不能估计强峰消失概率，也不能据此证明固定 support 在真实实验中成立。
- 保存连续相对强度，不发明 strong/weak 分界。强峰保护可以作为未来模型的明确适用约束；稳定性和边界仍需与观测证据对应。
- 原表同一身份、同一 CE 没有多条保留记录。mask / 不适用 / 峰归属变化不能自动等同于弱峰漏采。未验证检测阈值、同条件技术重复和标准样本，故 weak censoring 为 NOT_VERIFIED。
- 未找到并验证可用于本轮估计的 pixel-level fragment-ratio 重复测量。没有读取或处理原始 MSI。“pixel 波动更小”是待检验假设；混合物比例的空间变化不能直接当成同一分子的谱形波动。
- CE 数据已用于 production adapter 训练，是开发依据。CE30/35/40 的条件变化不等于独立的 library→当前实验 prediction error，也不等于 pixel noise。全谱库误差覆盖、独立预测验证仍 NOT_VERIFIED。

## 下一步的物理合同边界

认可先验证相关扰动、再考虑 open-set 的顺序。可用的起点是保留**同一次实测条件下共同出现的 fragment 向量/CE 轨迹**，而不是任意拼接各 fragment 的极端值；是否采用共享 CE/仪器潜变量、允许哪些插值与未见方向、怎样给出跨身份误差覆盖，尚未冻结。不得用已曝光的 V2 EVAL 来选这些定义。现有记录不足以宣称已经得到覆盖 391 候选的 U_phys；也不能拿“只剩 shared-fragment 信息”当理由填入未经测量的 pixel 噪声或 dropout。

预期模型分开记录系统性相对强度变化、弱峰 censoring、pixel variation 及 measurement noise。首项可从现有成组向量继续构造有限开发模型；其余项只在有直接来源和可用估计时加入，不以 CE 数据替代。技术重复/标准样本/MSI 的具体输入路径及检测信息尚未验证，本轮到此停止，不启动另一轮目录搜索。

对拟议的 separability margin 另保留三条必要定义：

1. U_j 中的谱形须使用固定非零归一化，丰度另置。若允许幅度趋零，cone 包含零，所写 infimum 可平凡地等于零。谱形距离与实验残差比较前必须统一范数、尺度和丰度。
2. cone(union U_-j) 允许同一竞争身份的多个不同谱形同时混合，可能再次放宽物理集合。应明确它是保守松弛，还是允许的真实生成模型；共享实验条件的跨身份约束也须保留。
3. 单列的谱库几何 margin 是诊断，不能单独认证混合观测中的身份或保证 FDR。未来 pilot 仍须检查观测的完整模型相容性及同一观测下的 molecular deletion，并测试 missing-library。

本轮完成的是现有数据能力与联合结构检查，**不是 U_phys 已建成，也不是主线不可能的结论**。V2 结论维持 NO-GO / BOTH_NO_GO；只能说明这版 local 加权没有救回效用，不能外推为所有空间信息无用。主线目标仍为独立验证中 FDP/FDR 风险目标 1% 与 TP retention 至少 40%，60% 仅为强成功。新的小 pilot 需先冻结有证据支持的模型和未曝光开发/验证合同；本轮不自动启动。
