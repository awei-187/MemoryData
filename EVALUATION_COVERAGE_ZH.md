# MemoryData：已接入方法、数据集与评测指标

核查日期：2026-10-21。依据本地工作目录的现有源码与配置整理；Git HEAD 为 `bdbe698`，工作目录包含未提交的 Hindsight 接入和统一 API 模型修改，以下统计包含这些修改。本次文档更新核对了源码、配置与已有验证产物，没有重新调用模型或运行基准。

## 1. 覆盖概览

MemoryData 通过 `main.py`、方法 YAML 和数据集 YAML，统一不同方法的历史摄取、记忆保存、问答和结果汇总。被测对象既包括记忆框架，也包括长上下文和 RAG 基线。

| 项目 | 当前本地实现 |
| --- | --- |
| 方法入口 | `config/` 下共 **23 个方法预设**：3 个参考基线、3 个序列上下文方法、7 个结构拓扑方法、10 个混合方法 |
| 基准族 | **4 个**：MemoryAgentBench、LoCoMo、LongBench、MemBench |
| LongMemEval 的位置 | 作为 MemoryAgentBench 的一个来源子集接入；另有专门的 Judge 和检索召回后处理 |
| 质量指标 | EM、答案子串命中、Token F1、ROUGE-L/ROUGE-Lsum；部分任务增加标签准确率、选项准确率、证据 Recall@K、LLM Judge |
| 效率指标 | 输入/输出长度、记忆构建时间、问答时间；MemBench 增加每次读写耗时等 |
| 统一接口 | `AgentWrapper.send_message(..., memorizing=True/False)`，分别表示摄取历史与回答问题 |
| 当前模型预设 | 回答及受适配器控制的对话调用默认 `gpt-4o-mini`；嵌入默认 `text-embedding-3-small`、1536 维；通过远程 OpenAI 兼容 API 调用 |

**计数口径：23 是配置预设数，不是 23 个独立的外部记忆产品。** 同一表中含基线、研究方法、本地源码封装和外部服务适配器。“存在接入”也不表示这些方法已在本机全部实测成功。

源码入口：[main.py](main.py)、[方法配置目录](config/)、[AgentWrapper](utils/agent.py)。

## 2. 接入了哪些框架和方法

分类沿用项目 README 的组织方式；运行时由 `utils/agent.py` 分派。下表列出全部 23 个 YAML 入口。

| 分类 | 框架／方法 | 方法配置 | 实现入口与接入形态 |
| --- | --- | --- | --- |
| 参考基线 | Long Context | `reference_long_context_agent.yaml` | `utils/agent.py`；积累历史，直接使用长上下文回答 |
| 参考基线 | Embedding RAG | `reference_embedding_rag.yaml` | `methods/embedding_rag/embedding_retriever.py`；稠密向量检索 |
| 参考基线 | BM25 RAG | `reference_simple_rag_bm25.yaml` | `utils/agent.py`；稀疏词法检索 |
| 序列上下文 | MemAgent | `sequential_memagent.yaml` | `methods/memagent/`；序列记忆方法 |
| 序列上下文 | Mem0 | `sequential_mem0.yaml` | `methods/mem0/source/mem0/` 和 `utils/agent.py`；仓库内 Mem0 实现的封装 |
| 序列上下文 | MemoChat | `sequential_memochat.yaml` | `methods/memochat/memochat_adapter.py`；对话记忆封装 |
| 结构拓扑 | Cognee | `topological_cognee.yaml` | `methods/cognee/source/cognee/`；仓库内图记忆实现 |
| 结构拓扑 | Zep Local | `topological_zep_local.yaml` | `methods/zep_local/main.py`；使用 Graphiti Core 和 Neo4j 的本地图记忆路径，独立于 Zep 云端预设 |
| 结构拓扑 | MemTree | `topological_memtree.yaml` | `methods/memtree/memtree_adapter.py`；树状记忆 |
| 结构拓扑 | GraphRAG | `topological_graph_rag.yaml` | `methods/graph_rag/graph_rag.py`；图检索实现 |
| 结构拓扑 | HippoRAG v2 | `topological_hippo_rag_v2_openai.yaml` | `methods/hipporag/`；当前预设为 OpenAI 兼容接口路径 |
| 结构拓扑 | RAPTOR | `topological_raptor.yaml` | `methods/raptor/raptor.py`；分层聚类、摘要和检索 |
| 结构拓扑 | Zep | `topological_zep.yaml` | `methods/zep/zep.py`；云端记忆集成 |
| 多范式混合 | Letta | `hybrid_letta.yaml` | `methods/letta/source/letta/` 和 `utils/agent.py`；内嵌源码与运行时封装 |
| 多范式混合 | Hindsight | `hybrid_hindsight.yaml` | `methods/hindsight/hindsight_adapter.py`；原生 REST `retain`/`recall`；已有原生服务启动器，服务进程需独立运行；MemoryData 生成最终回答 |
| 多范式混合 | LightMem | `hybrid_lightmem.yaml` | `methods/lightmem/lightmem_adapter.py`；分层记忆 |
| 多范式混合 | SimpleMem | `hybrid_simplemem.yaml` | `methods/simplemem/simplemem_adapter.py`；混合记忆检索 |
| 多范式混合 | MemOS | `hybrid_memos.yaml` | `methods/MemOS/source/src/`；记忆操作系统实现 |
| 多范式混合 | MemoryOS | `hybrid_memoryos.yaml` | `methods/memoryos/memoryos_adapter.py`；另一套独立框架，不能与 MemOS 合并计数 |
| 多范式混合 | A-MEM | `hybrid_a_mem.yaml` | `methods/a_mem/a_mem_adapter.py`；带来源追踪的记忆封装 |
| 多范式混合 | EverOS | `hybrid_everos.yaml` | `methods/everos/everos_adapter.py`；外部服务适配，当前预设指向本地服务 |
| 多范式混合 | Self-RAG | `hybrid_self_rag.yaml` | `methods/self_rag/self_rag.py`；包含检索判断、相关性、支持性和效用判断的 RAG 实现 |
| 多范式混合 | MemoRAG | `hybrid_memo_rag.yaml` | `methods/memorag/`；保留配置入口，但核心记忆模型和检索器依赖本地权重，当前 `api_only: true` 下明确拒绝初始化 |

表中配置均位于 [config/](config/)，方法源码位于 [methods/](methods/)。方法内部对模型、数据库、服务和硬件的依赖不同；统一入口并不消除这些差异。Hindsight 的接入方式和验证范围另见 [docs/hindsight.md](docs/hindsight.md)。

### 2.1 当前统一 API 模型配置

原本本地部署的 Qwen 等模型端点已从这组预设迁出。`main.py` 按自身位置读取根 `.env`，已有进程环境变量优先；`resolve_model_config()` 在初始化／结果命名前解析模型环境变量，并同步适配器内部的模型别名。

| 配置项 | 当前含义 |
| --- | --- |
| `OPENAI_MODEL` | 回答和受适配器控制的记忆抽取／摘要对话模型，默认 `gpt-4o-mini` |
| `OPENAI_EMBEDDING_MODEL` | 默认 `text-embedding-3-small`；Embedding RAG 入口名为 `Embedding_rag_text_embedding_3_small` |
| `OPENAI_EMBEDDING_DIMENSIONS` | 默认 1536，需与实际 API 向量一致 |
| `OPENAI_BASE_URL`、`OPENAI_API_KEY` | 模型服务地址和凭据；示例地址为 `https://api.openai.com/v1`，实际读取 `.env` 或进程环境，不在本文披露真实值 |
| `OPENAI_API_TRUST_ENV` | 默认预设允许模型客户端使用已有代理环境；Hindsight 本地记忆服务 HTTP 连接单独处理 |
| `tokenizer_encoding` | 预设统一为 `o200k_base` |
| `output_dir`、`agent_variant_tag` | 默认回答输出为 `results/outputs/api/<preset>/`；变体标签为 `api-models-v1`，Hindsight 为 `api-models-v1-hindsight-retain-recall-dated`（带时间写入与召回；此前结果用的是 `api-models-v1-hindsight-retain-recall`） |

显式 YAML `base_url` 仍优先于其环境变量；模型变量的默认值也可被实际进程环境覆盖。23 是保留的预设数，**不等于 23 条方法都能在当前纯 API 模式下运行**：MemoRAG 的本地核心模型限制仍在。Long Context 和 BM25 不需要嵌入调用。

这次改造没有统一各方法的 top-k、温度、分块或上下文预算；例如 Mem0 的检索数为 100、Hindsight 为 20、A-MEM／MemTree 为 2。远程模型也不会替代记忆数据库或服务。EverOS、云端 Zep 等服务内部使用什么模型，仍需在对应服务端核对。

更换模型、嵌入维度、API 后端或数据内容后，使用新的 `--artifact_root`，避免续跑逻辑复用旧回答或不兼容记忆。依据：[统一 API 配置指南](docs/api_models.md)、[环境变量示例](.env.example)、[provider_utils.py](utils/provider_utils.py)。

### 2.2 Hindsight 的当前执行链路

```text
历史 → Hindsight 原生 retain／整合 → 独立 bank
问题 → Hindsight 原生 recall → MemoryData 的 gpt-4o-mini 回答 → 原有规则评分
已保存的 LongMemEval 回答 → 独立 Judge 后处理（另行执行）
```

`scripts/run_hindsight_server.py` 在已有 `hindsight` 环境启动原生引擎，默认记忆 API 为 `http://127.0.0.1:8888`，可由 `HINDSIGHT_API_URL` 覆盖；评测仍在 `memorydata` 环境。启动器把共享模型变量映射到服务端抽取／嵌入配置，默认使用 `gpt-4o-mini`、`text-embedding-3-small`（1536 维）和 RRF；显式原生 `HINDSIGHT_API_*` 设置仍可能覆盖。连接已有服务时需自行对齐服务端设置。

当前默认 `recall` 预算为 `mid`、返回 Token 预算 4096、`retrieve_num=20`，同步 retain 后默认等待后台整合。每个评测单元有独立 bank，支持来源追踪、保存／恢复和限定范围的 `--force` 清理。**此适配器不调用原生 `reflect`；服务内部模型处理和最终评分 Judge 是不同角色。**

启动服务及基准命令见 [Hindsight 接入指南](docs/hindsight.md)。本地端口 8888 是记忆引擎入口，不能把它当作 OpenAI 模型端点；也不再要求启动原 Qwen／本地嵌入服务。

### 2.3 已有验证记录与本次核查边界

已有日志记录统一 API 改造后 28 项离线测试通过；下列四份合成历史真实 API 验证摘要均存在，`status=passed`、`resume_unchanged=true`，回答为 `Jasmine tea.`：

- [Hindsight 摘要](results/hindsight_api_smoke/api-env-check/summary.json)
- [Long Context 摘要](results/api_model_smoke/long_context_v2/summary.json)
- [Embedding RAG 摘要](results/api_model_smoke/embedding_rag_v2/summary.json)
- [Mem0 摘要](results/api_model_smoke/mem0_final/summary.json)

这些记录验证真实服务／API 链路、结果保存及恢复，不是正式数据集准确率，也不能证明其他全部方法均已完成真实服务验收。本次仅复核已有记录，没有重新运行这些验证；历史服务状态也不代表现在仍在运行。

## 3. 接入了哪些数据集

公共加载入口是 [benchmark/memoryagentbench/loader.py](benchmark/memoryagentbench/loader.py) 的 `load_eval_data()`。当前只分派下列四个基准族。

### 3.1 MemoryAgentBench

优先读取本地 `datasets/MemoryAgentBench/eval_dataset_collection/` 的 Hugging Face `save_to_disk` 数据；缺失或加载失败时尝试 `ai-hyz/MemoryAgentBench`。加载器支持三个 split，并按 `metadata.source` **精确相等**筛选来源。

| 能力／split | 仓库提供的来源子集 | 代表配置 | 评测内容 |
| --- | --- | --- | --- |
| Accurate_Retrieval | `eventqa_full` | [Eventqa_full.yaml](benchmark/memoryagentbench/Accurate_Retrieval/config/EventQA/Eventqa_full.yaml) | 长历史中的事件信息检索与问答 |
| Accurate_Retrieval | `longmemeval_s*` | [Longmemeval_s.yaml](benchmark/memoryagentbench/Accurate_Retrieval/config/LongMemEval/Longmemeval_s.yaml) | LongMemEval 长期交互记忆；当前配置是该来源的 150k 上下文截断变体 |
| Conflict_Resolution | `factconsolidation_mh_6k` | [Factconsolidation_mh_6k.yaml](benchmark/memoryagentbench/Conflict_Resolution/config/Factconsolidation_mh_6k.yaml) | 冲突事实整合、多跳事实判断 |
| Test_Time_Learning | `icl_banking77_5900shot_balance` | [ICL_banking77.yaml](benchmark/memoryagentbench/Test_Time_Learning/config/ICL/ICL_banking77.yaml) | Banking77 上下文内学习／意图标签分类 |

`longmemeval_s*` 中的星号是实际来源名称的一部分，此处不是通配符。三个 split 的通用加载能力，不代表上游所有子任务都有现成 YAML；本地也没有单独的 Selective Forgetting split 入口。各预设含 `max_test_samples` 限制，例如 EventQA 和 LongMemEval 当前均为 5 个上下文样本，不能把一次默认运行视为全量基准评测。

### 3.2 LoCoMo

- 默认数据：`datasets/LoCoMo/rq1_4cat_600_dist/locomo_4cat_600_dist.json`，即项目采用的四类、600 题子集文件，而不是默认读取原始全部 QA。
- 提供一个综合配置和四个单类配置：1 多跳、2 时间推理、3 开放域、4 单跳。
- 加载器读取会话、问题、答案与 `evidence`，保留对话来源 ID，供严格来源召回评分使用。
- 加载器还支持显式类别包含／排除；代码本身可以处理第 5 类及其 `adversarial_answer`。四类限制来自当前数据与配置，不能据此说加载器永远拒绝第 5 类。

依据：[LoCoMo 配置](benchmark/locomo/config/)、[LoCoMo 加载器](benchmark/locomo/loader.py)。

### 3.3 LongBench

当前 YAML 明确对应 **LongBench v2 的 150 样本比例抽样子集**，虽然项目目录名为 `longbench`。本地默认读取 `datasets/longBench_rep150_proportional/datasets`，split 为 `train`，题型是 A–D 多选项中的单项选择。

支持按长度、难度、领域、子领域和样本 ID 筛选。不能把此接入概括为 LongBench v1 所有任务的完整指标实现。

依据：[LongBench 配置](benchmark/longbench/config/LongBench_rep150_proportional.yaml)、[加载器](benchmark/longbench/loader.py)。

### 3.4 MemBench

默认从 `datasets/MemBench/MemData/FirstAgent/` 读取 JSON。提供五个切片配置：

| 切片 | 评测侧重点 |
| --- | --- |
| `simple` | 简单记忆回忆 |
| `noisy` | 噪声历史下的记忆与问答 |
| `knowledge_update` | 知识更新 |
| `highlevel` | 高层次推理 |
| `RecMultiSession` | 多会话推荐 |

预设主要使用 `FirstAgent` 视角，`roles,events` 分支，变体索引 `0`；各切片还受场景数、样本数和上下文上限约束。评测记录保留 `slice`、`branch`、轨迹、目标来源 ID 等元数据。

依据：[MemBench 配置](benchmark/membench/config/)、[加载器](benchmark/membench/loader.py)。

### 3.5 HaluMem（问答任务）

`feature/halumem` 分支新增。每个用户是一个 context，按官方协议在每个 session 写入后立即回答该 session 的问题（`interleave_queries: true`）。主指标是官方 QA 提示词判分得到的正确／幻觉／遗漏率（默认判分模型 `gpt-4o`，由 `HALUMEM_JUDGE_MODEL` 配置），可按题型和难度汇总；EM/F1/ROUGE 只作辅助。记忆抽取与记忆更新评测尚未接入；MemoRAG 不支持交错提问。详见 [HaluMem 接入说明](docs/halumem.md)。

依据：[HaluMem 配置](benchmark/halumem/config/)、[加载器](benchmark/halumem/loader.py)、[判分器](evaluation/halumem/halumem_judge.py)。

## 4. 具体评测哪些指标

### 4.1 公共回答质量指标

主要实现为 [utils/eval_other_utils.py](utils/eval_other_utils.py) 的 `calculate_metrics()`、`post_process()` 和 `metrics_summarization()`。

| 输出字段 | 含义／计算口径 |
| --- | --- |
| `exact_match` | 答案规范化后完全相等；规范化含小写、去英文标点、去冠词、合并空白 |
| `substring_exact_match` | 规范化后的参考答案是否为预测文本的子串 |
| `f1` | 规范化、空格分词后的词频交集 F1，使用 `Counter`，保留重复词的计数 |
| `rougeL_f1`、`rougeL_recall` | ROUGE-L 的 F1 和 Recall |
| `rougeLsum_f1`、`rougeLsum_recall` | ROUGE-Lsum 的 F1 和 Recall |

多参考答案时，各指标取参考答案中的最高分。默认开放式 QA 后处理还比较原始回答与解析后的回答，并对各指标取较高值，因此不是只对完整原始输出做一次评分。环境缺少 `rouge_score` 时会退回简化 LCS 计算，结果应注明这一条件。

这里的 `f1` 与 ROUGE 的 Recall 是**答案文本重合度**，不是检索证据命中率。

### 4.2 各任务额外指标

| 数据集／任务 | 额外输出 | 实际含义 |
| --- | --- | --- |
| EventQA | `eventqa_recall` | 所有答案元素是否都在原始回答中出现；大小写不敏感。最终是全中为 1、否则为 0，并非部分命中比例 |
| ICL / Banking77 | `label_accuracy` | 提取出的数字标签是否正确 |
| ICL / Banking77 | `label_format_valid`、`label_strict_format`、`label_strict_accuracy` | 能否提取标签、是否满足严格格式、严格格式且正确 |
| FactConsolidation | `answer_hit`、`parsed_answer_hit`、`first_line_answer_hit`、`selected_answer_hit` | 分别在完整回答、解析回答、首行、选定候选中检查参考答案子串命中 |
| FactConsolidation | `concise_response`、`concise_answer_hit` | 选定候选不超过 8 个规范化词；以及该候选同时命中答案 |
| LongBench v2 | `exact_match`，记录中另有 `pred`、`judge` | 提取 A–D 选项后判定正确性；`judge` 在这里是规则布尔值，不是调用 LLM 裁判 |
| MemBench | `exact_match`／汇总 `accuracy` | 提取选项后判定正确性；同样属于规则评分 |
| LoCoMo | `locomo_recall@K` | 前 K 个检索来源组覆盖到的 gold evidence ID 数／gold evidence ID 总数 |
| MemBench | `membench_recall@K` | 前 K 个检索来源组覆盖到的目标步骤 ID 数／目标步骤 ID 总数 |

ICL、FactConsolidation、LongBench 还保留部分 `legacy_*` 指标用于与旧解析规则对照。FactConsolidation 当前会依据参考答案，在多个候选回答片段中选取最佳候选；比较其他实现的分数时必须对齐这一后处理。

LoCoMo 和 MemBench 的 K 通常包含不超过请求深度的 1、5、10，并额外包含实际请求 K。只有存在目标证据、来源组和有效检索深度时才产生相应指标；缺少来源追踪不能直接解释为零召回率。按类别／切片的汇总保留计数，便于检查有效分母。

### 4.3 LongMemEval 专用后处理

这两类结果独立于主结果的文本指标，不能仅看到 `main.py` 输出就断言已经执行了 LLM 裁判。

回答模型迁移没有改变独立 Judge 的默认模型：CLI 仍默认 `gpt-4o`，由 `--judge-model` 显式选择，并读取进程中的 `OPENAI_API_KEY`／`OPENAI_BASE_URL`。它不会自动采用 `OPENAI_MODEL=gpt-4o-mini`；独立评分脚本也不像 `main.py` 那样自动读取根 `.env`。具体加载与运行示例见 [Judge 专项文档](LLM_JUDGE_PROMPTS_ZH.md)。

| 后处理 | 输出与含义 | 入口 |
| --- | --- | --- |
| LLM-as-a-Judge | `judge_accuracy`、`judge_accuracy_std`、按题型准确率及支持／已评分记录数；支持重复裁判。题目类型决定提示词，`*_abs` 问题使用拒答规则 | [evaluate_longmemeval_judge.py](evaluation/longmemeval/evaluate_longmemeval_judge.py)、[longmemeval_judge.py](evaluation/longmemeval/longmemeval_judge.py) |
| 检索召回 | 会话级和轮次级的 `recall_any@K`、`recall_all@K`、`coverage@K`；默认 K 为 1、3、5、10、30、50 | [memoryagentbench_longmemeval_recall.py](evaluation/longmemeval/memoryagentbench_longmemeval_recall.py) |

其中，`recall_any` 表示至少命中一条 gold，`recall_all` 表示覆盖全部 gold，`coverage` 表示覆盖比例。实现通过检索调试文本与原始 LongMemEval 的文本锚点对齐，再计算 session／turn 级指标；这不同于 LoCoMo 的直接来源 ID 匹配。

Judge 默认写 `*.longmemeval_judge.json`，不是覆盖原始回答文件。LongMemEval 召回报告默认写 `*.longmemeval_recall.json`；`main.py` 有运行后生成召回报告的钩子，但缺少所需原始数据或检索调试文件时会给出 skipped／error 诊断，不保证每种方法都有可评分的报告。

### 4.4 效率、开销和分组汇总

| 字段／统计 | 口径 |
| --- | --- |
| `input_len`、`output_len` | 方法返回的输入／输出长度；应结合方法实现和 tokenizer 解释，不能直接视为整个系统所有内部调用的计费 Token |
| `memory_construction_time` | 方法报告的记忆构建耗时，单位秒 |
| `query_time_len` | 方法报告的查询／问答耗时，单位秒，可能同时含检索与生成 |
| `time_cost` | 每次保存时距离本次主运行开始的累计墙钟时间，不是逐题独立耗时 |
| LoCoMo 类别汇总 | 各类样本数、F1 和 Recall@K |
| MemBench 切片汇总 | 按 slice、slice/branch 汇总准确率、Recall@K、长度、上下文规模和耗时 |
| `membench_total_overhead_time_s` | 记忆构建时间 + 本题问答时间；按题求均值时会重复计入同一上下文的构建时间 |
| `membench_read_time_per_op_s` | 本题 `query_time_len` |
| `membench_write_time_per_op_s` | 构建时间／历史块数 |
| MemBench native RT/WT 汇总 | 在 `send_message` 边界测量的读写墙钟时间，输出均值、总时间、操作数；构建写入按上下文去重，读取可能包含最终回答生成 |

主文件 `metrics` 保存逐项原值，`averaged_metrics` 对名字含 `_len` 或 `_time` 的字段保留原单位，其他指标均值乘 100。LongMemEval 独立 Judge 和召回文件使用自身的 0–1 分数约定。对照分数前需统一尺度。

依据：[main.py](main.py) 的 `save_results_to_file()`、`_build_locomo_category_breakdown()`、`_build_membench_breakdown()`、`_build_membench_native_rtwt_sidecar()`。

## 5. 当前覆盖边界与文档差异

1. 当前源码没有可用的 `benchmark/omnimemeval/loader.py`、`benchmark/omnimemeval/runtime.py` 或 `evaluation/omnimemeval/evaluate.py`，公共加载器也没有 `dataset: OmniMemEval` 分派。此前旧对比文档中的五基准移植说明已失效，本次已将 [FRAMEWORK_COMPARISON_ZH.md](FRAMEWORK_COMPARISON_ZH.md) 更新为当前实现对照；BEAM、PersonaMem v2 不计入本项目当前接入列表；HaluMem 已在 `feature/halumem` 分支按新的交错协议重新接入（见 3.5 节），与旧移植无关。遗留 `__pycache__` 也不等于可用源码接入。
2. README 的论文结果图片提到了 DB-Bench，但当前加载入口和 YAML 中未发现对应基准接入，因此未计入四个基准族。
3. `methods/*/source/` 内第三方源码可能自带其他数据集和评测工具；只有接入本项目主加载器／评分流程的部分才列入本文。比如 Cognee 内部的其他 QA adapter 不等于 MemoryData 主入口已接入这些基准。
4. 本文列出的是支持范围和计算规则，没有把既有试跑或论文图表当作“23 种方法 × 全部数据集均已在本机完成评测”的证明。

## 6. 与相邻 OmniMemEval 的区别

MemoryData 统一的是“历史写入／问题回答”的 Agent 接口，包含长上下文、RAG 和记忆方法；相邻 OmniMemEval 的用户记忆线主要统一后端 `add/search` 操作，并将检索、回答和评分拆成阶段，另有 OpenClaw AgentBench 任务评测线。

两边都有 LoCoMo、LongMemEval 和 Mem0 等名字，但所用数据切片、部署方式、回答提示词和后处理可能不同。例如这里的 Mem0 为仓库内实现，OmniMemEval 当前 `mem0` Client 对接平台 V3 REST API。不能直接将同名系统的分数差异归因于评测框架。

相邻项目清单见 [OmniMemEval 说明](<E:/LLM judge/OmniMemEval/EVALUATION_COVERAGE_ZH.md>)。

## LLM Judge 完整输入与 Prompt

具体消息结构、完整英文 prompt、输入来源、返回解析及聚合规则，见 [LLM_JUDGE_PROMPTS_ZH.md](LLM_JUDGE_PROMPTS_ZH.md)。
