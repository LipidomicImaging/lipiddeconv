# 第一 CAL 提前可行性检查 — 用户指示的资源止损

2026-09-12，用户要求先分析已经完成的第一组，若第一组不能通过则节省后续计算。该指示改变执行先后顺序和资源继续决策，不修改原六组科学设计、生产求解、U、LP、gamma、原始分母或正式 CAL/EVAL 规则。

第二组尚未启动；本地交接程序已在认证提示处终止。已完成的第一组保持原始文件，过程审查PASS、正常3000epoch、有限输出；紧凑记录已下载并推送382ee2091773d93d027faecd93b27fde5f22e4a6。此后先暂停其余五组 GPU 拟合，仅运行以下 CPU 检查。

唯一病例：SUPPORTED_MISMATCH__CAL_R71_K125。它是库完整的受控谱形失配场景，比联合缺库少一个挑战，但不能预先保证它数值上一定最容易。原始 solver 报告232个分子身份，其中TP123、FP109、FN2；all/reportable recall98.4%。实际扰动真身份26，全部已被solver报告。40% retention要求全体至少50个TP，同时受扰动子集至少11个TP；所有原分母不变。

复用冻结MODEL-only bank、原 full/delete函数、gamma1e-6、原数值证明和阈值事件规则。只读取第一CAL；在独立 first_case_feasibility 目录保存输入绑定、第一组证据seal、全部数值证明、阈值曲线和报告。这个seal明确属于单组开发检查，绝不写入原结果目录的正式 evidence_seal、calibration_seal 或 evaluation_access，不绕过原score的六组ACK门禁。旧产物不重绑到新seal。以后若恢复正式流程，本次检查不会被偷换成正式CAL/EVAL结果。

提前检查依次回答：

1. 数值证明是否有效。任何未解决数值问题单列TECHNICAL_UNRESOLVED，不宣称物理不可辨或科学失败。
2. 忽略FDP时，是否存在同一个epsilon，让全体TP retention和受扰动TP retention都达到40%。不存在则EARLY_STOP_CAL_FUTILITY。
3. 是否存在同一个epsilon，同时达到上述效用与第一组单独FDP<=1%。不存在则EARLY_STOP_FIRST_CAL_RISK_UTILITY_SCREEN；这是用户指示的开发资源止损，不是原pooled FDR合同的数学不可达证明。
4. 存在则FIRST_CAL_SCREEN_PASS_NOT_EVAL_GO，保留原六组正式CAL/EVAL合同，不将第一CAL的最优值当作正式阈值。

即使第2步失败，也只证明这个CAL病例在当前计算实现下不能过线，不能逻辑推出尚未观察的EVAL必败，更不能证明主线不可能。提前停止时，原EVAL状态保持NOT_RUN；原六组实验标为用户指示的提前开发停止，不伪造正式EVAL NO-GO。

结果报告单独给出原始/筛选TP、FP、FN，solver misses与filter-induced loss，recall、retention、受扰动子集、完整阈值曲线、数值失败、并行计算保守性边界。独立复核保存的证明及计数，再下载/hash/Git。无新GPU、rho、空间权重、CE模型、噪声或删除。
