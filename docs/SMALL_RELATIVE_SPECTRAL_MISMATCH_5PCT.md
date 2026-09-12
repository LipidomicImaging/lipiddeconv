# 固定条件库谱周围的5%独立相对强度失配

2026-09-12。用户明确提出可以在每个碎片离子、残余母离子强度上加入独立Gaussian或其他小噪声，并选择 **5%相对标准差作为主条件**。这是新的用户指定模拟假设。旧CE端点合同及其已关闭结果不变，其余五组旧拟合不恢复。

用户随后明确要求先只跑最简单一组、失败立即停止，并优先使用NNLS。执行范围因此更新为下述单组CPU开发筛查；不启动GPU。该模型不声称已由同CE实测重复估计出真实误差分布。

## 唯一首组：NNLS完整库5%失配

固定case名 `NNLS_FULL_LIBRARY_5PCT__CAL_R71_K125`，实验seed=7301。沿用已保存首组R71的125个真身份、空间图与相对丰度结构，完整391候选库，不缺库。新扰动与旧CE端点无关；已曝光的空间/真值结构仅用于这次配对开发筛查，不作为独立HOLD。所有正峰均按新5%假设处理，单列实际受扰动真身份数。

用float64构造新观测，沿用唯一全局dataset scalar使前景median L2信号范数为0.6036783456802368，所有X_true同步乘同一个scalar；最后一次转float32存储。保存源X、最终X、全局scalar及全部哈希，不改源文件，不作per-identity缩放。先封存设计和输入，再读取NNLS或rho结果。

复用 `run_nnls_solver_baseline.initialize_worker/solve_pixel`：全391列、float64逐前景pixel NNLS、maxiter3910、4个CPU worker、BLAS1线程；背景必须为0。保存float32 X_hat，float64计算前景均值，报告门槛仍为candidate mean>1e-3。

筛选先复用冻结 `src/rho_zero.py`，W=I，观测为原前景平均谱，全候选竞争。保持已有individual candidate rho及同名分子报告聚合：reported aliases的rho取max、X_hat求和。此次不引入group-rho、不把candidate删除改成molecular整体删除，也不加入新的uncertainty score。保存全部candidate/molecular记录、solver misses及filter-induced losses。

同时报告原rho阈值1e-3及完整distinct-value/tie-preserving阈值曲线。本次开发可行性检查选择经验FDP<=1%、非空集合下最大TP的点；并列先少FP、再较高阈值。GO必须FDP<=1%且TP retention>=40%，60%只标强成功。不存在合格点则首组NO-GO，停止后续；数值失败单列技术失败，不算科学不可行。首组最优点不是独立风险校准，也不声称真实FDR已受控。

只允许这一组。无缺库、HOLD、第二seed、K ladder或其他solver自动排队。完成后独立复核输入/过程/计数、下载哈希匹配的紧凑结果并Git保存，再给出停/走结论；通过也不自动启动下一组。旧CE pilot保持关闭。

## 扰动对象与时间结构

参考库始终固定为正确实验条件对应的最佳可用库 `A_lib`。每个脂质在每次实验只有一张固定实际谱 `A_exp`，由参考谱的小幅强度变化生成；所有pixel共用这张实际谱。不同实验使用预先指定的独立seed。

每个物理fragment ion的整个isotope/profile envelope使用一个倍率；残余precursor envelope也使用自己的独立倍率。不能逐profile bin、逐isotope stick分别抽噪声。不同物理离子独立抽样，同一物理离子的candidate aliases共享倍率。随机键包含实验seed及规范identity/adduct/CE/ion信息，不以candidate顺序、列号或删库后索引决定。

覆盖目标是完整名义库的所有已有正强度fragment和precursor成分，不沿用旧CE合同的69个候选或107个候选准入限制。生产准备必须提供完整、明确的物理成分分解并重建A；部分映射不能被默认为其余候选不扰动。若分解缺失，生成函数直接报错。准备阶段须记录实际覆盖数量，不能在尚未应用到生产资产时宣称已实现391/391覆盖。

## 正值Gaussian-derived倍率

使用用户允许的另一种小噪声形式：均值校正的lognormal相对倍率。令 `c=0.05`：

```text
Z ~ Normal(0, 1)
tau = sqrt(log(1 + c*c))
multiplier = exp(tau*Z - tau*tau/2)
```

其期望为1，标准差与均值之比恰好为5%，始终为正。小波动下接近相对Gaussian；它不是强度上的严格加性Gaussian。这里5%是标准差，不是每次变化最多±5%；约95%的倍率落在0.91–1.10之间。尾部没有人为截断，不做结果依赖的裁剪、重抽或dropout。

设 `D_jf` 是离子f对候选列j的完整非负响应，满足 `sum_f D_jf=A_lib[:,j]`：

```text
t_j = sum_f multiplier_jf * D_jf
A_exp[:,j] = t_j * ||A_lib[:,j]||2 / ||t_j||2
```

沿用谱形实验的列L2范数约定，隔离形状与总体尺度。此共同列归一化会耦合各峰，因此独立性与精确5%只适用于归一化前的抽样倍率；不得称最终每个m/z bin仍是独立5%Gaussian。保存每个倍率、列归一化scalar、归一化后实际变化及来源/目标哈希。原始空间图、X_true与production solver不由该函数修改；本函数不实施额外丰度缩放。

## 本次不混入的其他机制

无CE选择错误或跨CE端点；无峰位偏移、新峰、随机删峰、弱峰censoring或pixel噪声。观测仍为 `B_p=A_exp @ X_true,p`。这属于固定库到一次实验的系统性谱形失配，不是从全零背景构造的经验measurement noise。不能把本模型写成经验噪声源AVAILABLE。

缺库测试如后续执行，必须复用原完整观测并只改变solver候选库，不重新抽扰动。缺库不是本次生成函数的一个隐含操作。

## 推断与解释边界

置信度方法只能使用名义库、观测、solver输出和预先冻结的误差假设，不能读取本次真实multiplier、A_exp或truth。生成器的随机manifest是离线provenance，不是可部署输入。

噪声生成分布不等于最坏情形uncertainty set。不得把所有lognormal正倍率的无界支持直接放进必要性检验；也不得把epsilon直接设成0.05。候选竞争、归一化和混合比例决定观测残差尺度。新推断方法、误差预算、case/seed及CAL/EVAL必须在性能结果前封存；同一模型生成/验证的成功只能称假设内性能，不能直接宣称真实DIA MSI的FDR≤1%。

当前主线目标仍为低FDR与有用的individual molecular identity retention。此次文件只确定用户新选择的5%生成假设，不因旧实验失败修改旧分母、阈值或结果，也不提前声称主线已完成。
