# HaluMem 接入说明（问答任务）

HaluMem（[MemTensor/HaluMem](https://github.com/MemTensor/HaluMem)，数据 [`IAAR-Shanghai/HaluMem`](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem)，论文 arXiv:2511.03506）评估记忆系统在抽取、更新、问答三个环节的幻觉。MemoryData 目前接入的是**问答任务**；记忆抽取与记忆更新评测尚未接入。

数据与上游仓库均为 **CC BY-NC-ND 4.0**，仅限非商业研究并需署名。数据文件不入库（`datasets/` 已被 Git 忽略）。

## 数据准备

| 版本 | 位置 | 规模 |
| --- | --- | --- |
| Medium | `datasets/HaluMem/HaluMem-Medium.jsonl` | 20 用户、1,387 个 session、3,467 题，每用户约 16 万 token |
| Long | `datasets/HaluMem/HaluMem-Long.jsonl` | 同样的用户与题目，每用户约 100 万 token（含干扰内容） |

本机 Medium 已从 `E:\LLM judge\OmniMemEval\data\halumem\` 复制，SHA-256 为 `486fbc13…5245137aa449c193ac4d29c45634e7`。Long 版需要从 Hugging Face 下载同名文件放到上表位置。

题型分布（Medium）：Memory Boundary 828、Memory Conflict 769、Basic Fact Recall 746、Generalization & Application 746、Multi-hop Inference 198、Dynamic Update 180。

## 协议：逐 session 交错提问

与官方 `eval/eval_memzero.py` 一致：**写入第 N 个 session 后，立即回答该 session 的问题**，再继续写入下一个 session。题目从第一个到最后一个 session 都有分布；Medium 中至少 102 题（按证据原文精确匹配的下限）的证据会在之后被更新，所以“全部写完再问”会让部分参考答案过时。

数据集 YAML 的 `interleave_queries: true` 启用此流程（`main.py` 中的 `process_context_interleaved`）：

- 每个 HaluMem 用户是一个 context。每个 session 按对话轮边界切成不超过 `chunk_size` 字符的块，块不跨 session，每块首行都带 session 头和起止时间。
- 每轮写成 `[时间] 用户姓名: ...` 或 `[时间] Assistant: ...`。姓名用与官方相同的正则，从 `persona_info` 中解析。`persona_info`、`memory_points`、答案和证据都不进入记忆。
- 每题记录 `ask_after_chunk`，即提问前需要写入的块数，也就是该 session 的末尾。`question_date` 取 session 结束时间（ISO 格式，不加时区），Hindsight 会把它作为 recall 的 `query_timestamp`。
- 带 `is_generated_qa_session` 标记的 session 照常写入，但不提问（与上游一致；本地 Medium 数据没有这类 session）。

### 续跑与代价

- **续跑粒度是整个用户**：未完成的用户总是先清理本地与外部状态，再从第一块重建；已答的题保留不再重答。所以中断、`--retry_failed_queries` 都会重放该用户的全部写入。
- **Hindsight 例外**：如果 bank 里已确认写入的文档恰好是某个提问点（或历史末尾）之前的全部历史，且没有更早的未答题，重跑会保留这个 bank，再次等待整合后从该提问点继续写入，不删除、不重写。典型场景是某个提问点 `settle()` 超时，或该组某题回答失败。写入中途失败、配置或历史变了、`--force`、或用 `--retry_failed_queries` 重答更早的题时，仍整体重建。
- 交错运行的提问点很多，建议把 `hindsight_settle_timeout_seconds` 设得比单遍写入更大；超时后重跑同一命令即可接着等。
- 已完成的用户直接跳过。

### 各方法在提问点的处理

| 方法 | 提问点行为 |
| --- | --- |
| Long context、Mem0、LightMem、A-MEM、MemTree、EverOS、MemoryOS、Cognee、Zep Local、MemAgent、MemOS | 写入即可检索，无额外动作 |
| BM25、Embedding、GraphRAG、HippoRAG、RAPTOR、Self-RAG | 每次写入后索引失效，下次提问时重建。GraphRAG、RAPTOR、Self-RAG 每个提问点都要全量重建，成本随 session 数增长 |
| SimpleMem | 提问前 `finalize()` 落库缓冲，并重建全文索引（本机 `memorydata` 环境未装 `lancedb`，这一步尚未实跑验证） |
| MemoChat | 提问前 `save()` 生成摘要 |
| Letta | 提问前 `save_agent()` 刷新查询基线；该组题答完后恢复基线，再继续写入 |
| Hindsight | 提问前 `settle()` 等待后台整合但不标记完成；Medium 共 896 个提问点，每个都等待一次。最终 `save()` 才标记 bank 完成 |
| Zep（云端） | 服务端异步处理写入，提问点不等待，紧接着的提问可能看不到最新 session |
| MemoRAG | 不支持交错，启动时直接报错 |
| REMem、TSM | 不支持交错，在清理状态前直接报错（两者只在 `save()` 时用完整历史整批建索引；代码在 `remem-tsm` 分支，合并后生效） |

计时口径沿用现有处理器：每个提问点第一题的 `memory_construction_time` 是这一段写入的耗时，同组后续题约为 0。

## 运行

进入项目目录：

```powershell
cd 'E:\MemoryData'
```

试点（第一个用户，只答前 5 题）：

```powershell
& 'D:\conda\envs\memorydata\python.exe' main.py --agent_config config/reference_long_context_agent.yaml --dataset_config benchmark/halumem/config/HaluMem_medium_user1.yaml --artifact_root results/halumem_pilot --max_test_queries_ablation 5
```

全量 Medium：把数据集配置换成 `benchmark/halumem/config/HaluMem_medium.yaml`，并去掉 `--max_test_queries_ablation`。全量运行耗时长，应放在后台由监督流程执行。

回答模板使用官方 `PROMPT_MEMZERO`，只把占位符 `{context}` 改名为 `{memories}`。与官方的差异：MemoryData 的统一回答流程会在提示末尾附加真实的 `Current Time`；长上下文、RAG 等方法使用各自的问题模板。

## 判分

```powershell
& 'D:\conda\envs\memorydata\python.exe' -m evaluation.halumem.evaluate_halumem_judge --result_glob "results/halumem_pilot/**/*_results.json"
```

- 提示词逐字采用上游 `eval/eval_tools.py` 的 `EVALUATION_PROMPT_FOR_QUESTION`（提交 `d653749`，见 `evaluation/halumem/UPSTREAM_NOTICE`）；输入为题目、参考答案、证据记忆点和系统回答。
- 判分模型：`--judge-model` 优先，其次是环境变量 `HALUMEM_JUDGE_MODEL`，默认 `gpt-4o`。`temperature 0`；API 地址、密钥和代理设置沿用根 `.env` 中的 `OPENAI_*`。
- 解析方式与上游相同：取回复里第一个 ```json 代码块，`evaluation_result` 必须是 `Correct`、`Hallucination`、`Omission` 之一。格式错误和 API 错误都会重试（`--max-attempts`，默认 3 次）；仍失败的题记为判分失败。
- 输出 `<结果文件名>_halumem_judge.json`，包含总体以及按 `question_type`、`difficulty` 的正确率（C）、幻觉率（H）、遗漏率（O）。分母有两种：
  - `all`：全部题目。回答失败和判分失败都计为非正确。
  - `valid`：只计有效判分的题目。
- 重跑时，复用判分模型相同、回答未变的有效标签；`--overwrite` 全部重判；`--dry-run` 只统计题数，不调用 API。
- `main.py` 不会自动判分，以免意外产生 gpt-4o 费用；运行结束时日志会给出判分命令。EM/F1/ROUGE 仍会计算，但仅作辅助参考。
