# Mem0 与 Hindsight：首组数据表现、指标计算与 LLM-J 区分能力分析

日期：2026-10-24。依据：本机实际代码、冻结输入、已保存逐题回答与裁判记录。本次新增分析仅离线运行，没有新增模型调用，也没有改写原实验成绩。

## 1. 主要结论与数据范围

1. **本轮原始数据评测已经完成：22/22 项，1,130/1,130 条回答，两系统各 565 题。** Hindsight 在原始 LoCoMo、MemoryAgentBench LongMemEval 的规则分数较高；EventQA 接近；FactConsolidation 两者都低；Banking77 是 Mem0 较高。其余切片各只有一道题，只能说明该样本是否答对。
2. **这 1,130 条回答没有运行独立 LLM-J。** 现有 EM、F1、ROUGE、来源 Recall@K、标签正确率等都是代码计算。结果字段叫 `judge` 并不表示调用了模型。
3. **新四档 LLM-J 有补充语义与错误诊断的价值，但尚不能宣称整体准确率更高。** 在另一批已保存的改写 LoCoMo 147 题中，找到三对五项词面指标完全相同、而新 GPT 分类不同的回答；其中一对 GPT、Jev 与助手复核均一致。也找到新裁判错判等价日期的反例。
4. **用户提到的两种旧裁判问题，并非只能由新 LLM-J 识别。** 漏掉一项必答内容，可以由原子要点召回、某些情况下的子串匹配或 ROUGE 召回率识别；完全不含标准答案的非回答，EM/F1/ROUGE 经常已给零分。新逻辑更有价值的地方是区分“完整、部分、缺失、有误”，以及处理词面分数难以表达的语义关系。

全文严格区分以下两批材料：

| 材料 | 范围 | 用途 |
|---|---|---|
| A：本轮完整评测 | 原始 LoCoMo 首组 199 题，其他各类首个完整上下文，共每系统 565 题 | 各数据集表现与当前规则指标 |
| B：历史裁判对照 | 改写后的 LoCoMo 首组 147 题，每系统一份固定回答，共 294 份回答 | 同一回答上的旧二分类、新 GPT、新 Jev、助手复核及离线词面分数 |

**B 的答案、问题范围、回答长度与检索预算不同，不能将其百分比填入 A 的成绩表。** 下文的“助手复核”仍然是 LLM 判断，不是独立人工真值。

## 2. 两系统在各数据集下的表现（材料 A）

下面所有质量分数均为百分比，保留两位小数。EM=框架 exact_match，RL-F1=rougeL_f1；不是官方独立 LLM-J 成绩。

| 数据切片 | 每系统题数 | Mem0 EM | Hindsight EM | Mem0 F1 | Hindsight F1 | Mem0 RL-F1 | Hindsight RL-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoCoMo 原始 conv-26 | 199 | 3.02 | 7.54 | 12.17 | 23.99 | 12.07 | 25.43 |
| MemoryAgentBench EventQA | 100 | 45.00 | 45.00 | 64.35 | 63.33 | 66.42 | 65.48 |
| MemoryAgentBench LongMemEval | 60 | 6.67 | 25.00 | 15.22 | 42.10 | 15.09 | 41.96 |
| FactConsolidation | 100 | 1.00 | 4.00 | 1.00 | 4.00 | 1.00 | 4.00 |
| Banking77 | 100 | 16.00 | 0.00 | 16.00 | 0.00 | 16.00 | 0.00 |
| LongBench-v2 官方首样本 | 1 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| MemBench Simple | 1 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| MemBench Noisy | 1 | 100.00 | 0.00 | 100.00 | 0.00 | 100.00 | 0.00 |
| MemBench Knowledge Update | 1 | 100.00 | 100.00 | 0.00 | 0.00 | 100.00 | 100.00 |
| MemBench HighLevel | 1 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| MemBench RecMultiSession | 1 | 0.00 | 100.00 | 0.00 | 0.00 | 0.00 | 100.00 |

LoCoMo 第一组有 19 个 session、419 条发言；所有历史和 199 道题均已进入完整流程。Hindsight 的 F1 高 11.82 个百分点；MemoryAgentBench LongMemEval 高 26.88 个百分点。这表明**这些配置在这些上下文中**的表现存在差异，不能外推为完整基准上的排名。

EventQA 的 EM 相同，Mem0 F1 高 1.02 个百分点，属于本样本中接近的结果。FactConsolidation 的精确结果只有 1/100 和 4/100，两者仍存在明显困难，不能用“任务执行成功”代替“回答质量合格”。Banking77 两者标签可解析率都为 100%，所以 Hindsight 的 0% 不是所有输出都无法解析；仅凭汇总分还不能将原因唯一归结为记忆、检索或标签映射中的某一环。

LongBench 与 MemBench 为选择题，应以解析后的选项正确率为主。Knowledge Update 与 Hindsight RecMultiSession 的正确选项为 A，因通用词元归一化问题出现 EM=100、F1=0，详见第 3.5 节。

### 2.1 检索来源覆盖

LoCoMo 在相同 K 下的来源 ID Recall@K 如下；这些是**来源标记覆盖率，不是语义事实覆盖率**。

| K | Mem0 | Hindsight |
|---:|---:|---:|
| 1 | 19.33 | 23.01 |
| 5 | 38.16 | 44.75 |
| 10 | 51.95 | 62.35 |

Mem0 自身 K=100 时为 99.49%，Hindsight 自身 K=20 时为 81.01%；K 不同，不能用这两个数直接比较检索优劣。一个记忆条目可关联多个原始发言 ID，因此来源覆盖很高仍可能遗漏真正需要的事实。

MemBench 每类只有一题，下面仅展示相同 K=10 的来源覆盖，帮助理解“找到了相关来源”和“答对”不是同一件事。

| 切片 | Mem0 来源 Recall@10 | Hindsight 来源 Recall@10 |
|---|---:|---:|
| Simple | 100.00 | 100.00 |
| Noisy | 0.00 | 0.00 |
| Knowledge Update | 0.00 | 0.00 |
| HighLevel | 66.67 | 100.00 |
| RecMultiSession | 60.00 | 60.00 |

例如 Knowledge Update 的来源 Recall@10 为零而两者都答对；这说明来源映射指标和最终答案指标不能互相替代，也不能直接据此认定模型在猜测。

LongMemEval 的补充 session/turn Recall@K 因本机缺失原始 canonical gold session 文件未能生成。**缺失不是零分，也不能宣称已得到该项官方检索评测。** 当前 QA 规则分数不受这项报告缺失影响。

### 2.2 影响解释的实验条件

本轮模型为 gpt-4o-mini，嵌入为 text-embedding-3-small、1536 维。Mem0 沿用本仓库 `new_retrieved_facts[:2]`：每个 chunk 最多使用抽取事实列表的前两项；未因本次分析去掉限制。Mem0 检索上限 100，Hindsight 为 20，记忆单位也不同。因此这是当前集成配置的结果，并非同容量、同输入 token 预算的算法对照。

本轮“完整记忆构建”表示历史块全部提交并完成检查点，不代表全部知识被保留。日志仍有模型引用不存在的 UPDATE ID 而跳过动作的情况。EventQA 曾受间歇过滤/网络问题影响，最后通过保留成功输出、完整检查点恢复与有限重试补齐，不再有待补问题；但服务端拒绝根因尚未被最终确认。

本轮每类只取第一个上下文，不计算跨异质任务的统一总分，不给统计显著性结论。耗时涉及已建记忆复用、首次构建、不同重试条件，不能据现有均值给出严格速度或费用排名。

## 3. 指标实际如何计算

本节覆盖材料 A 汇总中出现的全部指标族，也解释新 LLM-J 相关指标。具体数值及原字段保留在 [完整汇总 JSON](<E:/MemoryData/results/full_first_units_20261023/summary.json>)。

### 3.1 共同预处理、参考答案与汇总

EM、Substring、词元 F1 使用归一化 N：先转字符串并小写，删除 ASCII 标点，删除英文冠词 a/an/the，再合并空白。日期、同义词、否定、数字词、人物关系不会被语义归一化。ROUGE 采用本机 rouge_score 0.1.2 的分词、词干化与 LCS，**不是这套冠词删除规则**。

设预测为 p、参考候选为 g₁…gₘ，通用计算对每个指标取 `max_j score(p,g_j)`。所以参考列表被解释为**多个可替代答案（OR）**，不会自动解释为“必须全部回答的要点（AND）”。一条字符串中的“clarinet and violin”也不会自动拆成两个原子要点。

默认 LoCoMo/LongMemEval 后处理分别计算原回答与解析后回答，**每个指标各取较高值**；不同指标可能来自不同文本版本。EventQA、FactConsolidation、标签题、选择题有单独分支，不能假设所有任务都使用相同抽取方式。

聚合时，对每项实际产生的逐题指标取算术平均；质量指标乘 100，字段名含 `_len` 或 `_time` 的计数/耗时保持原单位。没有产生某指标的题目不会自动填零。失败输出本身可能不进入均值，因此需要同时看完成覆盖；本轮已经完成全部计划题目，质量表没有未答题被排除的问题。

计算入口：[归一化及基础分数](<E:/MemoryData/utils/eval_other_utils.py:102>)、[多参考计算](<E:/MemoryData/utils/eval_other_utils.py:753>)、[默认后处理](<E:/MemoryData/utils/eval_other_utils.py:1125>)、[结果平均](<E:/MemoryData/main.py:95>)。

### 3.2 基础答案质量指标

| 字段 | 逐题计算 | 能说明什么与主要边界 |
|---|---|---|
| exact_match（EM） | `1[N(p)=N(g)]`，多参考取最大 | 严格字符串等价；同义改写、解释句、等价日期可能为零 |
| substring_exact_match | `1[N(g) 是 N(p) 的子串]`，多参考取最大 | 可容忍扩写；不检验否定、关系、错误附加内容；空标准串会命中任何预测 |
| f1 | 归一化后空格分词，词元多重集交集大小 c；P=c/预测词数，R=c/参考词数，F1=2PR/(P+R)；c=0 时直接返回零 | 衡量词面重合；长而完整的回答可能低于短而不完整的回答；不是事实正确率 |
| rougeL_f1 | 根据预测/参考的最长公共子序列 L，P=L/预测长度、R=L/参考长度，取调和平均 | 保留词序信息，仍不理解角色与否定；使用 ROUGE 自己的分词/词干化 |
| rougeL_recall | 同上，仅取 L/参考长度 | 对参考词覆盖敏感，较少惩罚回答长；也会奖励“出现了但被否定”的词 |
| rougeLsum_f1、rougeLsum_recall | 按换行句子边界计算 summary-level union-LCS，再得到 F1/recall | 并非新的语义裁判；本轮 22 项汇总中与对应 rougeL 值相同 |

F1 的 yes/no/noanswer 特殊处理会在这类答案不匹配时直接返回零。ROUGE 不保证语义同义词等价，也不是“覆盖了多少真实事实”的比例。

### 3.3 各任务专用字段

| 适用任务与字段 | 计算方法 |
|---|---|
| EventQA：eventqa_recall | 在原回答中做不区分大小写的子串检查，**所有**参考元素都出现才为 1，否则为 0；虽然名称叫 recall，输出是全覆盖二值分，不是命中比例。标准 EM/F1/ROUGE 则在解析回答上计算 |
| Banking77：label_accuracy | 解析标签成功且等于标准标签时为 1；优先用这个字段判断标签正确性 |
| label_format_valid | 能解析出一个标签即为 1，不要求标签正确 |
| label_strict_format | 满足代码定义的严格标签输出格式时为 1 |
| label_strict_accuracy | `label_accuracy AND label_strict_format` |
| FactConsolidation：answer_hit | 原回答中是否包含任一归一化标准答案 |
| parsed_answer_hit、first_line_answer_hit、selected_answer_hit | 分别对解析回答、首个非空行、最终选中的候选做同样子串检查 |
| concise_response | 选中候选的归一化词数不超过 8，即为 1；不保证内容正确 |
| concise_answer_hit | `selected_answer_hit AND concise_response` |
| legacy_exact_match、legacy_f1、legacy_rougeL_recall | 相应任务旧后处理方式算出的对照分；不是另一种模型裁判 |
| LongBench/MemBench：pred、judge | 从原文/解析文本/首尾行等位置提取 A–D 选项（LongBench 也尝试选项正文匹配）；`pred` 为选项，`judge=bool(exact_match)`，都是规则输出 |

FactConsolidation 会从 answer cue、解析回答、首行、首句、末行生成候选，按 `(EM, Substring, F1, RL-recall, RL-F1, -词数)` 的字典序选最佳候选。**选择过程使用标准答案打分**，属于评分端的有利抽取，不是模型实际自行选择了最佳答案。主指标均来自该选中候选。

代码：[任务分支](<E:/MemoryData/utils/eval_other_utils.py:874>)、[Fact 候选选择](<E:/MemoryData/utils/eval_other_utils.py:589>)。

### 3.4 检索、效率与完整性指标

| 字段 | 计算及边界 |
|---|---|
| locomo_recall@K、membench_recall@K | 取前 K 个检索来源 ID 组的并集 Rₖ，除以标准来源集合 G：`|Rₖ∩G|/|G|`。LoCoMo G 来自 evidence，MemBench 来自 target_source_ids；K 是检索组数，不是原始发言数或事实数。缺少标准来源/有效组时可能不产出指标 |
| LongMemEval 补充 turn/session recall_any | 前 K 命中至少一个标准 turn/session 为 1 |
| 同上 recall_all | 前 K 覆盖全部标准 turn/session 为 1 |
| 同上 coverage | 命中标准 turn/session 数除以标准总数；本轮此补充报告缺文件，没有可报告成绩 |
| input_len | 当前两适配器记录回答 API prompt_tokens **再加检索记忆文本的 token 数**；prompt 已含记忆，存在重复计数，不能作为实际计费输入 token；也不包含全部构建阶段调用账单 |
| output_len | 回答 API 的 completion_tokens；不是全部构建/检索模型消耗 |
| memory_construction_time | 进入一次回答的检索前，`当前时刻-agent_start_time`。两系统在每题后重置计时器，因此首题可能包含构建/加载，后续题主要是两题间开销；复用时也不包含此前全部构建。平均值不能直接当作独立完整构建耗时 |
| query_time_len | 同次检索、组织上下文、生成回答等的墙钟耗时，秒；可含网络等待与重试，不是纯模型推理时长 |
| membench_total_overhead_time_s | 该条记录的 memory_construction_time + query_time_len |
| membench_read_time_per_op_s | 该条记录的 query_time_len |
| membench_write_time_per_op_s | memory_construction_time / context_chunk_count；是按块摊销值，不能直接看作每次底层数据库写耗时 |
| completed / expected、成功项数 | 完成题目覆盖与任务执行状态；反映运行完整性，不是答案正确率 |

来源：[来源覆盖](<E:/MemoryData/utils/eval_other_utils.py:822>)、[LongMemEval 补充检索逻辑](<E:/MemoryData/evaluation/longmemeval/memoryagentbench_longmemeval_recall.py:374>)、[Mem0 计时与 token](<E:/MemoryData/utils/agent.py:3070>)、[Hindsight 计时与 token](<E:/MemoryData/utils/agent.py:3178>)。

### 3.5 已验证的评分边界：选项 A 与 AND/OR

**真实结果：** MemBench RecMultiSession 的标准选项为 A，Mem0 回答 D，EM=0、F1=0、Substring=1；Hindsight 回答 A，EM=1、F1=0、Substring=1。原因是 N(A) 变成空字符串，而空串是任何字符串的子串。

离线控制还验证：标准 A、模型文本未能解析出选项时，候选空串与 N(A) 同为空，当前实现甚至可能产生 EM=1。**这是控制例暴露的代码边界；本轮保存的选择题均有有效选项，没有证据表明它抬高了本轮 EM。** 后续应使用独立选项相等比较，并将“未解析出选项”显式判错，不复用英文冠词归一化。本次保留原始成绩，以便追溯。

另一个合成控制：若题意要求同时给出两种乐器，但 gold 写成 `["clarinet","violin"]`，当前通用多参考 max 会让只答 clarinet 的 EM/F1/Substring/ROUGE 全为 1；完整回答反而可能被长度惩罚。EventQA 的 all-elements 检查则能判定不完整。**这说明应明确数据结构中的 OR 别名与 AND 要点，而不是统一改变取最大规则。** 本轮 565 个唯一问题的 gold 实际为 199 个字符串、366 个单元素列表，没有多元素 gold 数组；此控制例不是本轮已经发生的多答案列表失真。

## 4. 到底有多少 LLM-J 或类似评测

计数必须说明范围：模板、维度、后端、已评分记录不能加在一起称为同一种“数量”。

| 范围 | 可核实的数量 | 说明 |
|---|---|---|
| 材料 A 本轮 1,130 条回答 | **独立 LLM-J 运行 0** | 全部为规则评测；没有为这些回答生成新四档裁判标签 |
| MemoryData 专门的外部 Judge 入口 | **1 套 LongMemEval Judge，5 类模板** | 普通事实（多个题型共用）、时间、知识更新、偏好、拒答；不能把模板数当已完成评测数 |
| 相邻旧 LoCoMo 评测 | **1 类宽松二分类协议，两系统各有实现** | 输入问题、标准答案、候选答案，输出 CORRECT/WRONG；role/schema/温度等并非完全一致 |
| 新 answer_coverage_v3 / four_status_v1 | **2 个评分维度，3 个流程阶段** | 答案四档 + 检索原子覆盖；reference 建立是准备阶段，不是第三个独立正确率 |
| 新答案分类后端 | **GPT 与 Jev 两套已保存输出** | 同类四档任务的不同实现，不能视为两种独立事实真值；另有助手复核 |
| 材料 B 已保存标签 | **294 份回答 × 4 个版本 = 1,176 个分类标签** | 旧、新 GPT、新 Jev、助手复核各 294 个；不是 1,176 道独立题，也不是 API 请求计数 |
| 相邻 OmniMemEval 的代码库存（扩展范围） | **7 类协议，覆盖 8 条命名任务路线** | 通用 QA（LoCoMo/LongMemEval 共用）、HaluMem、BEAM rubric、BEAM 排序、BrowseComp-Plus、OmniMath、GDPVal；这些不代表本轮已运行 |

来源：[MemoryData Judge 模板](<E:/MemoryData/evaluation/longmemeval/longmemeval_judge.py:26>)、[相邻框架 Judge 路径清单](<E:/LLM judge/OmniMemEval/LLM_JUDGE_PROMPTS_ZH.md:22>)。

### 4.1 “类似评测”的区别

EM、ROUGE、来源 Recall、Banking77 标签判断、LongBench 的布尔 judge 都不调用 LLM。BEAM 则是混合方案：模型逐 rubric 项给 0/0.5/1，或抽取事件位置，再由代码聚合；它已经能表达部分满足，不能说旧评测都只有二分类。LiveCodeBench/SWE-Bench 的测试或 harness 最终通过判定也不能计入上述 LLM-J。

MemoryData 的独立 LongMemEval Judge 按题型传入问题、参考答案与原始模型回答，不传对话、检索或证据。普通事实模板**已明确要求覆盖全部必需信息，只有子集应判 no**，时间模板还允许特定 off-by-one；知识更新与偏好有不同容忍规则。因此“旧宽松裁判多答题漏判”的结论，特指下面旧 LoCoMo 协议，不能泛化到所有原有 LLM-J。

该独立 Judge 的每题分数是多次运行中的正确比例，汇总 judge_accuracy 是已完成评题的平均正确比例；多次运行可另报整体正确率标准差。调用错误单列，不能自动视作答案错误。其默认裁判模型为 gpt-4o，不能从回答模型 gpt-4o-mini 推断裁判模型也相同。系统要求 JSON CORRECT/WRONG，用户模板要求 yes/no，解析端兼容这些格式；这也是需要记录的实现差异。

## 5. 新逻辑的输入、输出与计算

### 5.1 答案四档：只比较问题、gold、完整原回答

| 类别 | 判定规则 |
|---|---|
| correct | 覆盖全部必答要点，没有可识别的实质错误 |
| partial | 覆盖非空的正确子集，漏掉必答内容，没有实质错误 |
| missing | 完全没有覆盖必答内容，也没有明确错误断言；中性“不知道”可属此类 |
| incorrect | 所问人物、时间、数量、事件、否定等存在实质错误或矛盾；**错误优先于已覆盖多少** |

该版本不提取“核心答案”，不查看原始历史或检索证据，也不逐项核验超出 gold 的附加事实。gold 未提到的额外信息不自动算错，只有从现有输入能识别的错误/矛盾才扣为 incorrect。它允许语义改写与等价日期，但模型未必始终执行正确；同时没有“不确定”类别，困难题会被强制归入四档。

各档比例 = 本档数量 / 成功得到答案标签的数量；同时报告已评覆盖率与执行错误。不能把错误或未评题丢掉后只给一个正确率。Jev 的 0/1/2/3 对应 incorrect/missing/partial/correct，**是类别编码，不是等距质量分**；不应直接取其均值。Jev 的概率/confidence 也未经人工真值校准，不能当成已验证的正确概率。

依据：[答案提示词原文](<E:/LLM judge/mem0_locomo_dual_eval/prompts/answer_direct.txt>)、[版本设计](<E:/LLM judge/mem0_locomo_dual_eval/docs/四档答案与检索覆盖_2026-10-20.md>)、[Jev 准备代码](<E:/LLM judge/mem0_locomo_dual_eval/jev_answer_eval/prepare.py>)。

### 5.2 检索原子覆盖：与答案四档分开

reference 阶段使用问题、gold 与 evidence 附近上下文建立原子单元、required 标记及可替代的 sufficient_sets；不看候选答案或本次检索。检索裁判再读取问题、冻结 reference 与实际检索内容，不看候选答案或旧标签，对每个单元给 covered/partial/missing/contradicted。

- **相关覆盖率**：covered 单元数 / 全部相关单元数。
- **必答覆盖率**：required 中 covered 的单元数 / required 单元数。
- **充分性**：只要任一 sufficient_set 的全部单元均 covered，即 sufficient；否则 insufficient；无法评价时 not_evaluable。
- partial 在覆盖率分子中计 0，**不是 0.5**；先算每题覆盖率，再对可评题取平均，非将所有题的原子单元放在一起求微平均。
- 检索引用 ID 会被校验并回填原文本，但“引用 ID 存在”并不保证它在语义上真的支持单元。

材料 B 的已保存汇总为：

| 系统 | 相关覆盖均值（147 题） | 必答覆盖均值（147 题） | 充分/可判充分性题数 | 充分率 |
|---|---:|---:|---:|---:|
| Mem0 | 80.99% | 83.38% | 110/133 | 82.71% |
| Hindsight | 90.48% | 93.14% | 124/133 | 93.23% |

**覆盖均值的分母是 147；充分率的分母才是 133。** 两者各有 14 题 not_evaluable，不计入充分率，不能一并算作零分或宣称全部覆盖指标只算了 133 题。reference 本身还可能 insufficient，应保留一致参考子集的单独报告。

这些分数仍有预算差异：历史 Mem0 每题 20 条记忆；Hindsight 平均 215.59 条事实及 27.47 个原文块，144/147 题返回了全部 218 条事实。此处不是材料 A 的 K=100/20 配置，也不能据此证明算法在相同检索预算下更优。

来源：[覆盖率聚合代码](<E:/LLM judge/mem0_locomo_dual_eval/src/dual_judge/report.py:14>)、[Mem0 历史汇总](<E:/LLM judge/mem0_locomo_dual_eval/runs/edited_mem0_first147_dual_v3_20261020/reports/summary.json>)、[Hindsight 历史汇总](<E:/LLM judge/hindsight_locomo_dual_eval/comparison/summary.json>)、[实际检索预算审计](<E:/LLM judge/hindsight_locomo_dual_eval/comparison/retrieval_size_analysis_20261021/检索数量与长度分析.md>)。

这种双维度拆分能帮助定位：检索充分但答案 partial/missing/incorrect，说明至少在该裁判判定下，问题不只是未检索到信息；检索不足而答案正确，则需进一步核查外部知识、猜测、参考或覆盖判断。它提供排查线索，不能独自给出因果结论。

## 6. 历史 147 题：新旧判断到底改变了什么（材料 B）

旧 LoCoMo prompt 明确允许“只触及 gold 同一主题”的长回答判对。其宽松性来自协议文字，本身就可能放过遗漏和未真正作答的内容。来源：[Mem0 旧 judge_answer](<E:/LLM judge/mem0_on_edited_locomo0_group1/source_code/run_locomo.py:614>)、[Hindsight 保存的旧提示词](<E:/LLM judge/hindsight_locomo_dual_eval/comparison/original_judge_analysis_20261021/Hindsight_original_judge_prefix.txt>)。

### 6.1 标签分布与裁判一致性

| 系统/裁判 | correct | partial | missing | incorrect | 完整正确率 |
|---|---:|---:|---:|---:|---:|
| Mem0 新 GPT | 90 | 28 | 11 | 18 | 61.22% |
| Mem0 新 Jev | 101 | 14 | 13 | 19 | 68.71% |
| Mem0 助手复核 | 107 | 12 | 17 | 11 | 72.79% |
| Hindsight 新 GPT | 76 | 39 | 14 | 18 | 51.70% |
| Hindsight 新 Jev | 92 | 28 | 15 | 12 | 62.59% |
| Hindsight 助手复核 | 93 | 29 | 16 | 9 | 63.27% |

旧二分类给 Mem0 124/147 正确（84.35%），Hindsight 134/147 正确（91.16%）。其中旧判正确的回答流入新 GPT 四档：

| 系统 | 旧 CORRECT 数 | 新 correct | 新 partial | 新 missing | 新 incorrect |
|---|---:|---:|---:|---:|---:|
| Mem0 | 124 | 90 | 28 | 0 | 6 |
| Hindsight | 134 | 76 | 35 | 10 | 13 |

也就是旧判对中分别有 34、58 条被新 GPT 判为不完整正确，**这是标签迁移，不是已人工证实纠正了 34、58 个错误**。新 GPT 与 Jev 四档完全一致的数量为 Mem0 123/147（83.67%）、Hindsight 116/147（78.91%）；两套模型仍有明显分歧。

历史词面指标则为：

| 系统 | EM | Substring | F1 | RL-F1 | RL-recall |
|---|---:|---:|---:|---:|---:|
| Mem0 | 0.00 | 25.17 | 21.06 | 19.42 | 61.99 |
| Hindsight | 1.36 | 26.53 | 31.63 | 28.90 | 57.33 |

Mem0 新 GPT 判完整正确的 90 条全为 EM=0；Hindsight 的 76 条完整正确中有 74 条 EM=0。自然语言长回答里，EM 因此很难承担唯一质量指标。历史 Hindsight 的 F1 更高，而新四档完整正确率更低，也说明“词面重合高”和“必答内容完整且无误”测量对象不同；此处仍不能以新标签为真值证明哪种排名更准确。

### 6.2 对“其他指标无法区分”的可复查统计

本次对 147 道相同问题逐一配对两系统的固定回答，确认问题与 gold 一致，使用 MemoryData 当前 LoCoMo 后处理重新计算五项指标。不是拿不同问题之间的分数做类比。

新 GPT 对两系统给出不同四档标签的有 **53 对**。在这 53 对中：

| 单项指标 | 两回答得分相同的对数 |
|---|---:|
| EM | 53 |
| Substring | 40 |
| F1 | 4 |
| RL-F1 | 4 |
| RL-recall | 14 |
| 五项同时相同 | **3：q25、q28、q72** |

三对中，仅 q25 的两边分别得到 GPT/Jev/助手复核完全一致的类别。另有 18 对中，新 GPT 判完整正确的回答 F1 反而更低；其中包含新 GPT 本身可能错判的题，所以不能把 18 当作“F1 已被证实错误的数量”。

得分有差异只表示指标能数值区分，并不保证差异方向符合正确性；得分完全相同则说明，仅凭这五个分数不能推出这对回答的四档类别。以上结论针对本次五项指标，不代表所有非 LLM 方法都做不到。

## 7. 逐题证据：优势、重叠能力与反例

下列分数用 0–1 表示；题号为材料 B 的 conv0_qN。完整原文、解释及逐题分数保存在 [例题记录](<E:/MemoryData/results/metric_judge_analysis_20261024/selected_real_examples.json>)。

### 7.1 q57：只答一种乐器，旧裁判放过；并非所有规则指标都失效

问题：What instruments does Melanie play? gold：clarinet and violin。

- Mem0 回答两种乐器并附解释；旧 CORRECT，新 GPT/Jev/助手复核均 correct。
- Hindsight：“Melanie plays the clarinet.”；旧 CORRECT，新三者均 partial。旧理由明确把答出其中一种视作足够。

| 回答 | EM | Substring | F1 | RL-F1 | RL-recall |
|---|---:|---:|---:|---:|---:|
| Mem0 完整但长 | 0 | 1 | 0.1714 | 0.1500 | 1.0000 |
| Hindsight 只含 clarinet | 0 | 0 | 0.3333 | 0.2857 | 0.3333 |

这里新四档能直接指出“部分覆盖”；EM 无法区分，F1/RL-F1 甚至偏好更短的不完整答案。**但 Substring 与 RL-recall 已能区分这两个回答，不能声称它们都无能为力。** 如果使用显式原子要点召回，也可以在无需 LLM 的条件下检查 clarinet、violin 是否齐全；改写、隐含关系与矛盾处理会更难。

### 7.2 q7：同主题但没有 gold，词面指标本来就能给零分

问题是 Caroline 的关系状态，gold 为 Single。Hindsight 回答：“The context does not provide information about Caroline's relationship status.”。旧裁判给 CORRECT，新 GPT/Jev/助手复核均 missing；五项词面指标全为零。

新逻辑能解释这是“未给答案”，旧的同主题容忍过宽。**在是否应算正确这一层，EM/F1/ROUGE 也已拒绝该回答；新逻辑不是唯一能识别它的方法。** 反过来，正确语义改写也可能没有 gold 原词，不能将“没有 gold 字面词”普遍等同于错误。

### 7.3 q25：五项分数全相同，新逻辑区分“错年份”和“缺失”

问题：When did Melanie read the book 'nothing is impossible'? gold：2022。

- Mem0：“Melanie read the book ‘Nothing Is Impossible’ by Tom Oliver on July 12, 2023.”。旧 WRONG，新 GPT/Jev/助手复核均 **incorrect**。
- Hindsight：“The information about Melanie reading the book ‘Nothing is impossible’ is not available in the provided context.”。旧 CORRECT，新三者均 **missing**。

两回答 EM、Substring、F1、RL-F1、RL-recall **全部为 0**。因此五项现成指标无法区分“明确给出错误时间”与“中性没有答案”，而四档分类提供了额外诊断信息。它不是一个“正确与错误被分开”的例子，因为这两条都不完整正确。可专门设计日期抽取与拒答识别规则来复现这种区分，故不能上升为“所有规则方法都不可能”。

### 7.4 q28：等价日期对错分开，但 Jev 做反了

gold：“The Friday before 15 July 2023”。按通常最近前一个星期五解释，15 July 2023 是星期六，对应 14 July。

- Mem0 给 July 14, 2023；新 GPT 与助手复核 correct，Jev incorrect。
- Hindsight 给 July 7, 2023；新 GPT 与助手复核 incorrect，Jev correct；旧二分类两者均 CORRECT。

两回答的五项分数完全相同：EM=0、Substring=0、F1=0.2857、RL-F1=0.3750、RL-recall=0.5000。新 GPT 在这一对中能利用日历语义区分，**Jev 却相反**；不能笼统表述成“新 LLM-J 全都成功”。专门的日期标准化与日历规则同样能处理这类任务，不一定必须使用 LLM。

### 7.5 q38 与 q140：新裁判也会扣错

q38 gold 是“The Tuesday before 20 July 2023”，两回答都给 July 18, 2023，属于等价日期。新 GPT 对 Mem0 判 correct、对 Hindsight 判 incorrect；Jev 两者都判 incorrect；助手复核两者 correct。两份回答的细微措辞不应改变日期等价性。这是不能把更严格的新标签直接当人工真值的反例。

q140 gold 为“Grateful and thankful for her family”，Hindsight 回答含“reflecting gratitude and appreciation for them”；新 GPT 判 partial，Jev 与助手复核 correct。该例提示同义表达也可能被低估。

q72 虽也是五项同分例，但“至少两个”与 gold“三个”的下界、完整性、图注推断导致 GPT/Jev/复核分歧；因此本报告没有把它用作新逻辑稳定正确的证据。

### 7.6 合成控制：只包含相同词不能保证相同含义

问题为举办城市，gold 为 Paris：

| 合成候选 | 按题意应有类别（非模型输出） | EM | Substring | F1 | RL-F1 | RL-recall |
|---|---|---:|---:|---:|---:|---:|
| Paris, not Lyon. | correct | 0 | 1 | 0.5 | 0.5 | 1 |
| Lyon, not Paris. | incorrect | 0 | 1 | 0.5 | 0.5 | 1 |

五项得分完全相同，Substring 和 RL-recall 还同时给满分，因为它们没有理解“not”修饰谁。新 prompt 的规则**旨在**区分这种否定关系，但本次没有调用新 Judge 评这两条，不能将期望标签写成模型实测结果。否定作用域规则或其他语义模型也可能做到；本例只说明这五个词面分数本身信息不足。

## 8. 新 LLM-J 的优势应如何表述

| 情况 | 现有规则是否可能识别 | 新逻辑增加的价值 | 已有证据边界 |
|---|---|---|---|
| 多个必答要点漏一项 | Substring、ROUGE recall、显式要点召回有时可以；F1 可能反向 | 直接报告 partial，允许语义改写 | q57 三种新判断一致，但部分规则也有效 |
| 同主题而未作答 | gold 不出现时，规则经常已经为零 | 将 missing 与实质错误分开 | q7 已实测；不是新方法独占能力 |
| 错年份与不知道 | 五项词面分数可能同为零 | incorrect/missing 诊断区分 | q25 三种新判断一致 |
| 等价日期与错误日期 | 普通词面指标可能同分；专门日期规则可做 | 利用语义/日历推理 | q28 GPT 成功、Jev 失败；q38 还有新裁判反例 |
| 否定或人物关系反转 | 词面命中未必区分；专项语义规则可能区分 | 检验实质矛盾而非关键词出现 | 合成控制展示指标限制，尚无此次模型实测 |
| 检索充分但回答遗漏 | 来源 Recall 不能直接说明事实是否充分 | 独立检索覆盖与答案质量交叉定位 | 有历史双维度记录，但仍依赖 reference 与裁判可靠性 |

由此可以支持的表述是：**新逻辑比宽松“同主题即正确”的旧 LoCoMo 协议更明确，提供比二分类和单一词面分数更细的诊断；现有实例显示了补充区分能力。** 目前不能支持“新模型裁判准确率已显著提高”“所有其他指标都不能区分”“这项能力在所有数据集均成立”。

原子覆盖、部分得分、语义评判在相邻框架中已有实现，不能仅凭新增四档名称宣称方法独有或论文级创新。更可检验的贡献是：明确 AND/OR、错误优先规则、答案/检索分离、冻结参考、可回放输入输出，以及证明这些设计在哪些错误类型上带来可靠增益。

## 9. 若要验证“更准确”，下一步应做什么

这是验证建议，**本次未新增运行**。

1. 冻结同一批问题、标准答案、完整候选回答及输入版本；从本轮 1,130 条回答分层取样，或完整新增 sidecar 评审，保持旧成绩文件不变。选择题/标签题优先用任务专用精确判定，开放题才需要语义裁判。
2. 至少两名独立人工标注者按同一四档说明盲评，再仲裁分歧；单独核查 gold 质量、AND/OR、日期、否定、实体、数量、同义改写、冗余信息与错误附加事实。只看新旧模型分歧不能得到裁判准确率。
3. 对同一模型、版本、温度、输入、schema 运行新旧 prompt 的交叉对照，再另外比较 GPT/Jev。这样才能尽量分离 prompt 改动与后端模型差异；当前历史对照没有完全控制这些条件。
4. 以人工标签计算四档混淆矩阵、每类 precision/recall/F1、macro-F1、完整正确二分类准确率、缺失/错误混淆率与一致性指标；对“模型认为正确但人工不正确”的假阳性重点审查。分类编码不能直接平均成准确率。
5. 对 EM/F1/ROUGE、原子要点召回、日期归一化等比较完整 ROC/阈值表现或分类预测；阈值在开发集确定，在独立测试集验证。现有 53 对标签差异和同分例只是区分性证据，不能替代这一步。
6. 记录执行失败、无法解析、运行覆盖、重复运行稳定性、费用与延迟；检索 Judge 另检验引用文本是否支持单元。若要做置信区间，应考虑问题共享上下文，按上下文聚类，而不是把同一对话的所有问题当独立样本。本轮只有首个上下文，限制了跨场景推断。

评分代码层面优先事项是选择题 A/空串问题、明确 AND/OR 数据结构、分开报告 raw/parsed 分数，以及校正 token 与构建耗时口径。这些修正若执行，应另存新评分版本，与本文原成绩并列，避免悄然改变实验口径。

## 10. 复查入口与本次新增产物

| 文件 | 内容 |
|---|---|
| [主实验汇总](<E:/MemoryData/results/full_first_units_20261023/summary.json>) | 22 项完成覆盖、全部指标、结果路径与 SHA-256 |
| [主实验逐题记录](<E:/MemoryData/results/full_first_units_20261023/per_question.jsonl>) | 1,130 份回答、冻结 gold、逐题分数及来源 |
| [既有完整实验记录](<E:/MemoryData/docs/experiments/2026-10-24_first_unit_evaluation.md>) | 构建、复用、失败恢复与完整性验证 |
| [本次离线分析脚本](<E:/MemoryData/scripts/analyze_metric_judge_discrimination.py>) | 重新调用本地实际评分函数，不调用模型 |
| [分析汇总](<E:/MemoryData/results/metric_judge_analysis_20261024/analysis_summary.json>) | 标签计数、配对统计、输入来源与 SHA-256 |
| [全部 294 份历史回答及规则分数](<E:/MemoryData/results/metric_judge_analysis_20261024/historical_294_metrics.jsonl>) | 原文、四版标签、raw/framework 分数 |
| [147 对逐题对照](<E:/MemoryData/results/metric_judge_analysis_20261024/paired_comparisons.json>) | 同分、分类分歧及 F1 方向检查 |
| [真实例题](<E:/MemoryData/results/metric_judge_analysis_20261024/selected_real_examples.json>) | 文中例题完整内容与裁判理由 |
| [合成控制](<E:/MemoryData/results/metric_judge_analysis_20261024/synthetic_controls.json>) | 否定、日期、AND/OR、选择题 A 评分边界；期望标签不是 LLM 实测 |

复跑本次离线分析（使用已确认的 memorydata 环境；不需要 API）：

```powershell
cd 'E:\MemoryData'
```

```powershell
& 'D:\conda\envs\memorydata\python.exe' 'scripts\analyze_metric_judge_discrimination.py'
```

本报告没有增加原始基准的题数，也没有把历史改写数据的四档分数迁移到原始数据。当前完整评测与新裁判验证分别保留，便于后续在同一批回答上正式比较。
