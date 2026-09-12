# 主线 identity robustness pilot — V1

2026-09-12。前置合同已完成；本项直接运行一次完整主线开发 pilot，不再开展 CE 审计。生成侧严格绑定 PHYSICAL_PERTURBATION_CONTRACT_V1，fingerprint `3b2e75c563a0d23ac16db2ce87f1815ae0c3574fac59c3d8e89b794768697df7`。

## 一次封存的六个 case

固定 K=125。复用 V57 matched identities、CAL/HOLD identity pools、175 matched pairs、blocks、healthy real spatial templates、spatial construction 及 abundance assignment；新 realization R71 seed7201 / R72 seed7202，不改变旧 R1–R5 或 V2。R71 为本轮 developmental CAL，R72 为 EVAL；名称中的原 CAL/HOLD 指原身份池，不与本轮评估角色混淆。

| 角色 | Case |
|---|---|
| CAL | SUPPORTED_MISMATCH__CAL_R71_K125 |
| CAL | close_neighbor__HOLD_R71_K125 |
| CAL | relatively_isolated__HOLD_R71_K125 |
| EVAL | SUPPORTED_MISMATCH__CAL_R72_K125 |
| EVAL | close_neighbor__HOLD_R72_K125 |
| EVAL | relatively_isolated__HOLD_R72_K125 |

两个 missing arms 保留原冻结的各五个同名完整 omission，但观测现在也使用本轮 A_target。因此本次为 supported mismatch 与 missing-library 的联合挑战；旧 V2 missing arms 只是 CLEAN omission，不能直接归因比较两版性能。每组仍有125个 truth，缺库 truth 不从分母删去。

## 来源隔离与一次生成

使用79个允许完整端点。donor graph 纳入原清单所有190个身份记录，通过同一身份的全部 CE 和共享 source_file 做传递闭包，再保留含允许端点的26个连通分量。每个分量内排序 pattern IDs，SHA256(`physical_contract_v1_donors|` + 紧凑JSON IDs) 排序，依次分配 MODEL/MODEL/MODEL/CAL/EVAL 五槽循环：16/5/5分量，47/10/22端点。分组 fingerprint `9748b901a42eaccaa64265889d0987b811701a6070efe462d7704695759d0b66`。

每个角色、两个 mapped support set 都有端点的 common CE pair，按最短跨度再按数值排序选择：30→35。三 fragment 的 MODEL/CAL/EVAL 端点数10/2/8，四 fragment 为10/2/1；四 fragment EVAL 只有一个 donor，不伪称大量独立化学误差 realization。

各候选按固定 rule/lipid_name/channel-set 键和 role 哈希选择一个完整目标 donor，同组重复候选共用，所有 pixel 固定。同一角色的三个 case 共享同一完整 A_target。没有 severity 调参、插值、反向扩展、dropout 或 pixel noise。生成采用原 V57→V54 constructor，但生成 context 的 A_solver 指向 A_target，以原单一全局 scalar 直接达到前景 median||B||2=0.6036783456802368。训练使用另行保留的原 nominal context，绝不接收 A_target、X_true 或 donor assignment。

MODEL 字典在独立 `model_patterns.json` 中保存，confidence 核心只接收这个子集、固定 nominal A、B 的原 foreground mean 及用于同名删除的 metadata。CAL/EVAL donor 端点不能进入 inference。输入封存检查同一候选的目标端点与 MODEL/nominal 端点距离；实际变化>1e-10而最近相对距离≤1e-7时停止为 representation-only，不重抽。该检查不是身份筛选结果，不检验 EVAL 性能。

## 身份判据与数值证明

只使用 global foreground mean，不增加 spatial variant。原 production X_hat 的任一同名 candidate 前景均值>0.001 就报告该 molecular identity；所有 solver 候选参与竞争，删除全部同名 nominal/endpoint atoms。

推断 U 是 MODEL 有限端点及 nominal。原 CE v1 L1 LP 仅复用数学求解/数值界限机制，不使用它的 fragment 独立 box。

完整物理可行证据：先解全端点松弛得到下界；单独保存 nominal 库可行拟合。对每个共同 CE pair 解一个松弛，根据每个 identity 分组的最大系数质量选择完整端点（ties按固定顺序），在每候选一个端点的库上重新求非负 L1 系数，选最好可行上界。只做这一轮确定性 rounding/refit，不做结果驱动搜索。这不是离散全局最优声明。

删除模型允许全 MODEL endpoint cone 松弛，包括同一候选端点混合和不同 CE pair 混合；其下界是物理删除问题的保守下界。松弛删除拟合可行，只能标 `RELAXATION_REPLACEABLE`，不能宣称已构造物理替代解释。完整模型的上界必须来自单端点、共同 CE 的原物理 witness。

原单次 LP 时间预算60s、1e-8 outward guards、每个 convex LP primal/dual gap≤1e-6不变。完整物理上界与松弛下界之间的 gap 是模型松弛误差，不是假装某次LP数值失败。任何必需LP或证明失败都弃权；不靠优化器局部失败确认身份。保留每个 full/delete primal/dual 和物理端点选择供独立检查。

数值判定带 gamma=1e-6。入选当且仅当：

```text
full_physical_upper <= epsilon - gamma
AND deleted_relaxation_lower > epsilon + gamma
```

状态优先：NUMERICALLY_UNRESOLVED → FULL_MODEL_INCOMPATIBLE（full relaxation lower>epsilon+gamma）→ THRESHOLD_UNRESOLVED_FULL → RETAINED → RELAXATION_REPLACEABLE → THRESHOLD_UNRESOLVED。数值/模型弃权 truth 保留在原分母。

## CAL 封存与唯一 GO

所有六个 fit 的输入/过程先验证；不在 CAL threshold seal 前读取 EVAL confidence outcomes。CAL epsilon 在所有判定事件及其邻近浮点点中选择：pooled empirical FDP≤1% 下最大 retained TP，ties fewer FP then larger epsilon。没有合格非空集时封存 EMPTY_CALIBRATION，不改 U 或阈值规则。EVAL 只能读取这个 seal。

EVAL GO 必须同时满足：全体报告的 pooled FDP≤1%；aggregate 和三个 challenge 各自 TP retention≥40%、retained set非空；**实际受扰动的支持内 truth** aggregate 和各 challenge retention也≥40%且非空。受扰动 truth 在生成前定义：truth∩69候选支持映射，且相对列变化>1e-10。60% aggregate仅为GO后的强成功。支持内 truth 的子集不虚构类别 FP/FDR；风险始终基于完整 retained set。

报告 raw TP/FP/FN、filtered TP/FP/FN、solver misses、filter-induced loss/fraction、all/reportable-truth recall、TP retention、distinct identities及各弃权状态；支持内与其他 truth 分开。未入选真值不从分母移除；缺库 truth 仍在125 truth中。有限样本、共享donor/身份/模板及CE历史训练暴露限制风险推断，经验FDP不等于总体FDR保证。一次结果只支持本范围的开发结论。

## 执行与保留

主机 westc:55786；新源快照与结果使用明确保留的 `/root/physical_identity_mainline_v1/`，不会覆盖旧 V2/V58/V59。读取时确认系统盘约7GB可用、数据盘约3.6GB；六fits按已有约2.4GB量级加紧凑证明文件预算，逐次检查余量。生产 early-stop 原样保留，3000只是hard cap；无新oracle/sentinel/rho训练环节。

先封存并push源代码、合同及准备设计，再开始GPU。每组结束：独立核对过程/绑定/哈希/记录，下载紧凑产物，核对原字节并commit/push，写入该组handoff ACK，然后才准许下一组训练。EVAL阶段先做过程检查，风险/效用分析等待CALseal。所有unique final arrays/models和绑定checkpoints保留；不做历史清扫。若之后退休任何明确冗余文件，另按既有“先审核/下载/Git、后精确删除/receipt”合同执行，哈希不代替备份。

首次结果完成后独立复核所有证明、CAL选择、EVAL分母和GO，Git保存并停止该版本。成功后安排独立正式验证；失败也不改U、权重、epsilon规则或已曝光EVAL重跑。主线目标仍开放，不能把开发字典表示成功说成真实实验问题已解决。
