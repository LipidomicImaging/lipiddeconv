# Real CE29 identity competition / failure geometry audit

This audit does not estimate real CE29 FDR and does not label real identities as true or false. It tests whether low-confidence identity assignments exhibit stable library-internal replacement geometry consistent with the known false-assignment mechanism observed under synthetic spectral mismatch.

Audited 377 molecular identities from 391 original columns and 15837 foreground pixels. All same-name aliases participate in every deletion; original production results and DEV10 transfer outcomes remain unchanged.

ISTA reports 76; NNLS reports 78; overlap 61; support Jaccard 0.655914. All-alias mean abundance Pearson 0.914196, Spearman 0.741659.

The union contains 5134 directed positive-gain edges. 14 edges are unique top replacements in every eligible frozen processing block, with at least two eligible blocks. Stability uses 64 original 250-pixel blocks (last 87), not the 34 cached spectral folds. Reciprocal strict-stable pairs: 2.

Presence-persistent edges require numerical presence in every eligible block, with at least two eligible blocks, without requiring top replacement. Their reciprocal graph contains 27 maximal cliques of size at least three; largest size 5, identity coverage 0.090186. This clique analysis is separate from strict unique-top stability.

Top five receiver identities account for 0.200738 of eligible block unique-top events. Stable same-class fraction: 0.8571428571428571. These distributions are descriptive; no arbitrary GO cutoff was selected.

Necessity uses unweighted squared residual loss. Abundance redistribution is positive molecular coefficient change. Absorption columns contain raw postfit-residual and residual-change cosines; NNLS KKT can force these near zero. Removed-source versus positive-replacement signal cosine and projection are separately named, and do not imply identity correctness.

Spectral representatives average unit-normalized alias spectra, then normalize. Reduced cosine weights each observed channel by inverse square root of full-library molecular occupancy. Fragment sharing uses only mapped physical fragment envelopes; common means shared by multiple molecular identities and specific means library-exclusive. These are library-sharing definitions, not independently verified chemical diagnostics. Original source fragment labels are retained separately.

Quartile-intersection descriptive anchors: strong 1, weak 1; DESCRIPTIVE_INTERNAL_GRADIENT. No anchors served as labels. V58 MILD CAL R1/R2 comparisons retain one identity per case and report median, IQR, ECDF, Cliff delta, Wasserstein distance and KS statistic; no pooling independence claim or significance-driven conclusion.

Cross-solver edge agreement: NOT_EVALUABLE_CACHE_ABSENT; support and spatial agreement cannot substitute for deleted-ISTA solutions. Decision: SOLVER_STABILITY_FIRST. Q1 PARTIAL, Q2 PARTIAL, Q3 PARTIALLY_SUPPORTED.

The global and per-block means each have their own full-library NNLS full/deleted diagnostic. They are not pixelwise-fit coefficient averages. The existing 34 spectral-fold caches remain source-audited secondary assets and are not spatial evidence. New arrays remain outside lightweight Git; provenance binds all source files and the frozen contract.

The audit does not identify which real molecules are present, measure real identity error or recall, establish decoy exchangeability, or authorize the next experiment. Final independent review is PASS, with full cache coverage and no solver calls. The original two failed reviews and their narrowly scoped corrections are preserved below.

## 最终分析与解释边界

进入审计的是全库377个分子身份（391个候选），不是仅76/78个已报告身份。ISTA/NNLS共同报告61个，ISTA独有15个、NNLS独有17个；报告集合Jaccard为65.59%。全377身份的同名别名合计平均丰度Pearson为0.9142、Spearman为0.7417，不能替代删除边的跨求解器稳定性证据。

5,134条有向边是全图均值与64个原前景顺序块均值中的数值正增益边之并集；14条严格稳定首位替代边占0.2727%。这些边各自有2至64个符合资格的块，只有2条的分母为64；不能将14条都表述为跨全部64块稳定。892条在每个符合资格块持续出现的边不要求首位替代，其双向图有27个至少3身份的极大团，最大5个身份、覆盖34/377个身份（9.02%）。极大团可以相互重叠，不是27个独立竞争家族。原始顺序块不是独立样本，这些团也不是新定义的分子等价组。

前5个接收身份占符合资格块唯一首位替代事件的20.07%，占全图重分配增益的23.53%；结果不支持由极少数接收身份垄断替代。类别关系取决于分母：全部边按条数同类占35.74%，全图按重分配增益加权同类占71.97%，严格稳定首位边同类为12/14（85.71%）。因此，大量弱边跨类，而较强重分配及严格稳定子集偏向同类，不能笼统称所有competition主要同类。

残差变化与首位替代谱的cosine在62个有效全图身份中中位数为3.37e-15，四分位区间为[-2.47e-15, 0.0271]，不支持普遍较强的残差投影。但NNLS的KKT条件会约束活动列与残差接近正交，不能把接近零解释为不存在竞争。decision_summary的global_removed_signal_cosine统计包含全377身份及不活动身份的零值，其中位数0不能当作活动替代子集的吸收强度结论；该量与残差吸收分别命名。

CE29与V58已知FP的几何相似程度为PARTIAL。CE29 ISTA剔除组38个身份的最近谱cosine中位数0.8329，V58两组FP分别为0.8489/0.8504；Cliff delta（V58减CE29）为0.0309/0.0499，Wasserstein距离0.0167/0.0268。但同一CE29组也接近V58 TP（Cliff delta -0.0367/-0.0441），CE29 NNLS保留组的最近谱cosine同样接近FP。双方来自同一391列谱库，静态谱邻近本身不是FP特有机制证据。

共享碎片分布部分接近：CE29 NNLS剔除组29个身份中位数0.2623，V58 FP为0.2439/0.2466，Cliff delta 0.0468/0.0445。替代集中程度则有差异：同一CE29组首位替代占比中位数0.1702、四分位区间[0, 0.3330]，V58 FP中位数均为0、上四分位数为0.1192/0.1061；Cliff delta -0.2793/-0.2936，Wasserstein距离0.1062/0.1128。CE29 ISTA剔除组总重分配C的上四分位数0.00346，也低于V58 FP的0.04142/0.04185。相近的谱邻近和共享通道不能抹去重分配尾部的差异。

原始残差cosine接近机器精度时，即使KS或Cliff delta较大也不能据此作生物学解释。残差变化比较的CE29有效样本只有14，V58分别47/46，不能沿用总组人数38与109/107。强/弱内部锚点各只有1个，仅为单身份描述、不足以建立群体内部梯度。V58只复用了两组已曝光MILD CAL均值缓存，缺少像素块稳定性证据，不能确认机制等价或总体推广性。冻结decision_summary理由中的“two synthetic CAL specimens”指两套具有共享/重复身份的缓存模拟CAL cases，不是两个独立生物样本。

最终路线为SOLVER_STABILITY_FIRST：谱库内部稳定竞争只得到部分支持，competitive decoy几何仅部分成立。当前缺失ISTA删除缓存，跨求解器边一致性保持NOT_EVALUABLE_CACHE_ABSENT。此路线是审计判断，本任务没有启动后续实验，也没有构建decoy、估计真实FDR或新训练/调阈值。

## 复核与保存

67/67诊断正常完成，耗时678.453秒。最终独立复核PASS：核对296个来源文件及哈希、候选顺序与377个同名分组、67个全库模型和25,259次同名整体删除的全部允许列KKT与SSE、全部5,134条边及其块分母、V58的754条身份记录及原34个谱块缓存；无抽样、无求解器重跑。全部科学CSV和图/覆盖/决策JSON相对首次复核前的哈希保持不变。

首次失败validation_report_initial.json来自独立复核程序误将common/specific fragment support的原始强度和归一化；原合同和结果一直是raw sum。第二次失败validation_report_units.json来自独立复核程序把零范数向量的空间cosine记作0；合同和结果一直使用未定义值NA。原复核脚本、两次失败记录及科学输出均保留；两项修正各自先封存并成功push，再复核现有缓存，最终结果见validation_report.json。修正改变复核期望值，不改变科学定义、输入或实验结果。

所有新诊断数组和原生产数组继续保留。final_storage_manifest.json记录本输出目录的精确成员、字节数、哈希与本地保存路径；紧凑CSV/JSON/分析随本case提交到codex/v58-v59-run-records，大数组保留在清单列明的本地存储而不加入轻量Git。没有删除文件。
