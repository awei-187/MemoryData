# REMem 接入

## 来源

- 论文：REMem: Reasoning with Episodic Memory in Language Agents，Yiheng Shu 等（OSU、Intuit AI Research），ICLR 2026 poster，arXiv 2602.13530。
- 官方代码：[intuit-ai-research/REMem](https://github.com/intuit-ai-research/REMem)，论文 arXiv v3 脚注给出该地址；许可证 Apache-2.0。
- 本地副本：`methods/remem/source/REMem-main`，来自 main 分支 zip（SHA-256 `25410f8c7c0b7f1223b303c6df9d0e13667eddfcd8dc94ad63853ffbddc249c5`，2026-11-20 下载）。该目录由 `methods/remem/.gitignore` 排除，不提交；换机器需重新下载并核对哈希。上游代码未做任何修改。

方法概要：索引阶段对每段对话用 LLM 抽取带时间的 gist 和带时间限定的三元组事实，与原文（verbatim）、实体一起组成混合记忆图，gist 之间按嵌入相似度（阈值 0.8）加同义边；问答阶段由 ReAct 式智能体调用 `semantic_retrieve`、`lexical_retrieve`、`find_gist_contexts`、`find_entity_contexts`、`output_answer` 多步检索并作答。

## 为什么用独立环境

REMem 固定 `openai<2`、`numpy==1.26.4`、`torch==2.6.0`、`vllm==0.8.5post1`，而 memorydata 环境是 openai 3.x、numpy 2.x，直接安装会破坏现有方法；vllm 也不支持 Windows 原生安装。因此仿照 Hindsight 的"外部服务"模式：

```text
MemoryData（memorydata 环境）                      REMem 环境
RememAdapter ──JSON lines（stdin/stdout）──> methods/remem/remem_worker.py
  remem_state.json / documents.jsonl              └─ 导入未修改的 src/remem
  native/（上游索引文件，由 worker 写入）
```

- worker 用 `<remem python> -I -X utf8 -u remem_worker.py` 启动，工作目录是该 context 的 `native/`；`-I` 保证 `PYTHONPATH` 和脚本目录不会混入，`-I` 同时忽略 `PYTHONIOENCODING`，因此用 `-X utf8` 让标准输出使用 UTF-8。
- 密钥只通过子进程环境变量传入（`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`MEMORYDATA_EMBEDDING_*`），不进入请求、清单或结果。上游自带的 LLM/嵌入缓存（`native/outputs/memorydata/*.sqlite`）只存响应，不存密钥。
- 上游代码打印到 stdout 的内容被重定向到 `<context>/worker.log`，不影响协议。
- 只走在线 API 路径，不导入 vllm、transformers、sentence_transformers、dspy。环境依赖见 `methods/remem/requirements-remem.txt`。

## 数据流

1. 写入：每个 MemoryData chunk 经 `_prepare_memory_chunk_for_storage` 去掉 `LOCOMO_META` 包装后，追加到 `documents.jsonl`；来源 ID 和时间戳记在 `remem_state.json`。此时不调用模型。
2. `save()`：启动 worker，把全部 chunk 作为 REMem 的 session 传给 `ReMem.index(docs)`（`extract_method=episodic_gist`），完成后检查图、verbatim、gist 均非空，再把清单标为 `ready`。
3. 恢复：清单 `ready` 且 `native/remem_index/graph.pkl` 存在时复用；worker 用同一批文档再次调用 `index()`，上游发现图、嵌入和抽取结果都在，会跳过抽取。
4. 查询：调用上游 `EpisodicGistStrategy._rag_each_query`（即 REMem-I 智能体），问题日期以 `{"date": question_date}` 传入（与上游 LongMemEval 脚本一致）；格式无法识别时记录一次警告并按无日期作答，而不是让该题失败。返回智能体答案和它访问过的全部段落（gist、事实、原文、实体）。
5. 回答：
   - `remem_answer_mode: memorydata`（默认）：智能体访问过的段落按访问顺序交给 MemoryData 的回答模型，用统一模板作答，与其他预设可比。注意智能体仍会完整运行，包括它自己的 `output_answer` 调用，只是该答案不被采用。
   - `remem_answer_mode: native`：直接采用 REMem 智能体的答案。该调用在 worker 内完成，`input_len` 只能按问题加段落的 token 数近似。

## 为对接加的胶水代码（均在 `remem_worker.py`）

- 每个 chunk 变成单条消息的 session：`[{"content": "<时间戳>\n<文本>", "memorydata_doc": "md-doc-…"}]`。上游把消息字典原样存为 verbatim 节点的元数据，因此可以由节点追溯到 chunk。
- 上游 `chunk_by_session` 在消息没有 `date` 字段时会抛 `KeyError`。worker 用一个等价的切分函数替换它：每个 session 一个 chunk，内容与上游单日期 session 的格式一致（日期在首行），只是不加 `role:` 前缀，因为 MemoryData chunk 里已经带有说话人。
- 包装 `FlexibleToolsStrategy._format_result`，在 logs 里额外记录智能体访问过的节点名。检索和回答行为不变。
- 日志：上游 `get_logger` 为每个模块在源码树下打开一个名如 `remem->utils->config_utils.log` 的文件，`>` 在 Windows 文件名中非法，导入即报错。上游在日志器层级上已有处理器时会跳过建文件，因此 worker 在导入前给根日志器配置 stderr 处理器（输出进入 `worker.log`），并把 `LOG_DIR` 指向 context 目录，不改上游代码。

## 来源追踪

- verbatim 节点：直接映射到它的 chunk。
- gist 节点：取同一 chunk 的 verbatim 邻居（gist 之间的同义边只连 gist，不会串到别的 chunk）。
- 事实节点：经它所属 gist 找到 verbatim。
- 实体节点：连接很多 chunk，不作归属，来源组留空。

## 配置

`config/topological_remem.yaml`，`agent_name: Agentic_memory_remem`。

| 字段 | 说明 |
| --- | --- |
| `model_env` / `remem_llm_model` | 抽取、智能体和回答都用 `OPENAI_MODEL`（gpt-4o-mini） |
| `embedding_model_env` / `remem_embedding_model` | `OPENAI_EMBEDDING_MODEL`（text-embedding-3-small，1536 维），经上游的 OpenAI 嵌入客户端调用 |
| `remem_python` / `remem_python_env` | REMem 环境的解释器，默认 `D:/conda/envs/remem/python.exe`，可用 `REMEM_PYTHON` 覆盖 |
| `remem_worker_timeout_seconds` / `remem_index_seconds_per_chunk` | 每次 worker 请求的超时（默认 7200 秒）；建索引（含复用时重新载入）的超时再加上每个 chunk 的额度（默认 300 秒），避免大规模历史被固定上限中断后反复重建 |
| `remem_answer_mode` | `memorydata`（默认）或 `native` |
| `remem_native_config` | 覆盖上游 `BaseConfig`。默认值取自作者 `examples/locomo.py`：`agent_max_steps 3`、`qa_top_k 10`、`linking_top_k 5`、`retrieval_top_k 200`、同义边阈值 0.8、DSPy 过滤提示 `filter_llama3.3-70B-Instruct.json` |
| `retrieve_num` | 只用于输出命名和 recall@k 标注。REMem 的上下文规模由 `qa_top_k` 和 `agent_max_steps` 决定，不截断 |

身份指纹只包含决定索引内容的设置（模型、抽取方式、同义边阈值等）、数据集配置、完整历史和状态目录。智能体步数、`qa_top_k` 等查询期参数变化时直接复用索引；其他变化则拒绝复用，需要 `--force` 或新的 `--artifact_root`。

## 使用

REMem 环境已于 2026-11-20 创建（Python 3.11，torch 2.6.0、numpy 1.26.4、openai 1.109.1、python-igraph 0.11.8、bm25s 0.3.16，`pip check` 无冲突）。在其他机器上重建：

```powershell
C:\Users\wjp\anaconda3\Scripts\conda.exe create -y -p D:\conda\envs\remem python=3.11
```

```powershell
D:\conda\envs\remem\python.exe -m pip install -r methods/remem/requirements-remem.txt
```

运行（与其他方法相同的入口）：

```powershell
D:\conda\envs\memorydata\python.exe main.py --agent_config config/topological_remem.yaml --dataset_config benchmark/locomo/config/Locomo_qa_4cat_600_dist.yaml --max_test_queries_ablation 2
```

## 与原方法的差异和缺口

- **只做过截断历史的冒烟测试**，见下文"真实 API 冒烟测试"，不能代表完整评测分数。
- 模型：论文默认 GPT-4.1-mini + NV-Embed-v2（4096 维，本地 GPU）。这里按统一配置使用 gpt-4o-mini + text-embedding-3-small。上游代码本身支持这两个模型（gpt-4o-mini 也是上游 `BaseConfig` 的默认值），但结果不能视为论文设置的复现。
- 输入粒度：上游 LoCoMo 脚本按原始 session 传入带说话人和日期的消息列表；这里每个 MemoryData chunk（`agent_chunk_size` 4096 token）是一个 session，说话人留在文本里。本分支的 chunk 一般没有单独的时间戳字段，日期只出现在正文里（如 LoCoMo 的 session 标题），由 REMem 的抽取 LLM 从文本中读取；只有带 `timestamp` 的结构化 chunk 才会在首行加日期。
- 只支持整批建索引：所有 chunk 写完后在 `save()` 中一次 `index()`。上游在图已存在时会跳过索引，没有增量接口，因此 HaluMem 式"边写入边提问"（`prepare_for_queries`）未实现。要支持，需要在每批提问前用累积文档完整重建，代价很高。数据集配置带 `interleave_queries` 时，初始化即报错，不会先缓冲历史再在第一题失败。
- 只接入论文方法 `episodic_gist` + 智能体；`openie`、`temporal` 等变体和 REMem-S 单步变体未接入（可通过 `agent_fixed_tools` 近似 REMem-S，但未验证）。
- 上游 `semantic_retrieve` 会对事实文本调用 `eval()`。这是上游行为，在记忆内容来自可信基准数据时风险有限，接入其他数据时需要注意。
- 中断的写入不续传：未到 `ready` 的状态在下次运行时整体丢弃重建。

## 真实 API 冒烟测试（2026-11-20）

只检查真实链路能否走通，不是评测结果。数据为 MemoryAgentBench Accurate_Retrieval 的 `longmemeval_s*` 首个样本，`smoke_max_history_chunks: 3`（88 个 chunk 只取前 3 个），提 2 题；模型为 gpt-4o-mini 和 text-embedding-3-small，结果在 `results/remem_smoke_20261120`（Git 忽略）。

| 检查项 | 结果 |
| --- | --- |
| 索引 | 3 个 chunk 抽出 43 个 gist、67 条事实、72 个实体，图 185 个节点、2624 条边，约 86 秒 |
| 智能体检索 | 每题 3 步，依次调用 `semantic_retrieve`、`find_entity_contexts`、`find_gist_contexts` 等；问题日期生效（日志中时间过滤为 `2023-01-01 to 2023-05-27`） |
| 默认回答（memorydata） | 两题都回答"历史中没有相关信息"，得分 0。证据很可能在被截掉的 85 个 chunk 里，这一结果在预期之内 |
| 复用与原生回答（native） | 第二次运行复用已建索引，上游日志为 `Skipping indexing`；原生答案为 "You did not mention this information." |
| 来源追踪 | MemoryAgentBench 不带来源 ID，来源组为空，属正常；LoCoMo 上的来源追踪只在离线测试中验证过（本地没有 LoCoMo 数据） |

观察到的情况：

- 每题约 111 秒，主要耗在中转服务上：第 1 题智能体 16 秒完成，随后 MemoryData 回答模型的那次调用用了约 95 秒；第 2 题 worker 内对 `/chat/completions` 重试了 3 次。与 `docs/api_models.md` 记录的连接不稳定情况一致，不是接入本身的问题。
- 上游把 LLM 和嵌入响应缓存在 `native/outputs/memorydata/*.sqlite`。对同一索引重复提相同问题会直接命中缓存（第二轮几乎没有网络请求，也几乎不产生费用），结果完全一致；`--force` 删除 context 目录时缓存一并清除。
- 智能体收到的是 MemoryData 已套好模板的问题文本（含答题要求和 `Current Date`），与其他方法收到的输入相同。
- 上游日志中有一条 `Could not determine embedding dimension, using 1536`，维度本身正确，无影响。
- 运行结束时 LongMemEval recall 报告生成失败，原因是本地缺少 `datasets/LongMemEval/.../longmemeval_s_cleaned.json`。这是原有流程对数据文件的要求，与 REMem 无关。
