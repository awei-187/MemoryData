# TSM 接入

## 来源与可信度

- 论文：Beyond Dialogue Time: Temporal Semantic Memory for Personalized LLM Agents，Miao Su 等 11 人，Findings of ACL 2026，arXiv 2601.07468。
- 代码：[cosmicexotic/TSM](https://github.com/cosmicexotic/TSM)，版本 0.1.1。论文 arXiv v2（2026-09-02 修订）正文摘要末尾写明 "Code is available at https://github.com/cosmicexotic/TSM"，因此这是作者公开的实现；v1（2026-01-12）、arXiv 摘要元数据和 ACL Anthology 页面没有代码链接。仓库于 2026-09-01 新建，只有 3 个提交（v0.1.0、当天被取代的依赖修复 v0.1.1、补引用），README 说明它是不带开发仓库 Git 历史的新仓库，且"未做过真实 API 基准运行，不声称任何分数"；论文分数来自作者未公开的内部开发代码与内部后端（REPRODUCIBILITY.md）。
- 许可证：MIT（派生自 MIT 许可的 YiboZhao624/E-2GraphRAG；依赖 Apache-2.0 的 Graphiti，见其 `NOTICE`、`GRAPHITI_LICENSE`）。
- 本地副本：`methods/tsm/source/TSM-main`，来自 main 分支 zip（SHA-256 `66ba29ba2b45ca2f09b59b8f00501bcd04547d331935a465a77d1673cd004c97`，2026-11-20 下载）。该目录由 `methods/tsm/.gitignore` 排除，不提交。上游代码未做任何修改。

方法概要：按 Zep 的方式用 Graphiti 在 Neo4j 中构建带有效时间的时序知识图；按月切片，对实体嵌入做 GMM 聚类，由 LLM 生成主题摘要和 persona 摘要（snapshots）；检索时用 spaCy/dateparser 解析问题中的时间范围，在原文和摘要上做稠密检索（FAISS），再按"是否落在时间范围内、语义相似度"排序，图中事实作为辅助信号。

## 为什么用独立环境

TSM 固定 `graphiti-core==0.19.0`、`numpy==1.26.4`、`openai==2.24.0`、`spacy==3.8.4` 等，与 memorydata 环境冲突，还需要 Neo4j。做法与 REMem 相同：

```text
MemoryData（memorydata 环境）                     TSM 环境 + Neo4j
TsmAdapter ──JSON lines（stdin/stdout）──> methods/tsm/tsm_worker.py
  tsm_state.json / documents.jsonl               └─ 导入未修改的 TSM-main（main_time.py 的函数）
  native/cache/（tree、graph、snapshots 等）       Neo4j group md_tsm_<uuid>
```

- worker 用 `<tsm python> -I -u tsm_worker.py` 启动；密钥（`OPENAI_API_KEY`、`NEO4J_PASSWORD`）只经子进程环境变量传入；上游的 stdout 输出写入 `<context>/worker.log`。
- LLM 和嵌入地址通过上游支持的 `TSM_LLM_BASE_URL`、`TSM_EMBEDDING_BASE_URL` 指向 `.env` 中的 `OPENAI_BASE_URL`。
- 环境依赖见 `methods/tsm/requirements-tsm.txt`（与上游 `requirements.lock` 相同的版本）。

## 数据流

1. 写入：chunk 去掉 `LOCOMO_META` 包装后追加到 `documents.jsonl`；来源 ID 和时间戳记在 `tsm_state.json`。不调用模型。
2. `save()`：每个 chunk 成为一个 TSM "turn"，格式为 `Date: <时间戳>\n<文本>`（上游按轮读取数据时生成同样的 `Date:` 头）。ISO 时间戳里日期与时间之间的 `T` 换成空格：上游用 `\bYYYY-MM-DD\b` 提取事件时间，`2024-01-01T10:00` 中日期后紧跟 `T` 不构成单词边界，会悄悄落到 2000-01-01 的兜底值（已在真实环境中验证）。worker 依次调用上游 `parallel_build_extract`（Graphiti 抽取 → 聚类建树和 snapshots → FAISS 索引）和 `build_retriever`。完成后检查缓存文件、snapshots、稠密检索器都已生成，才把清单标为 `ready`。
3. 恢复：按上游 `main_time.run_pipeline` 的缓存分支，读取 `tree.json`、`graph.json`、`index.json`、`appearance_count.json`、`text_edge_mapping.json` 重建检索器。上游在这条路径上不传预建索引，会**重新嵌入整棵树**，因此每次恢复都有少量嵌入费用。
4. 查询：调用 `Retriever.query(question, **retriever_kwargs, question_date=…)`。worker 包装了 `_format_results_with_summaries`，额外记录被格式化的叶子 ID（`leaf_<i>` 对应第 i 个 chunk）；返回文本不变。
5. 回答：
   - `tsm_answer_mode: memorydata`（默认）：检索到的 chunk（按 TSM 排序，最多 `retrieve_num` 条）加 snapshot 摘要，交给 MemoryData 的回答模型，用统一模板作答。
   - `tsm_answer_mode: native`：使用上游 `QA_prompt_mirix_improved`（含问题日期）和同一个 LLM 作答，与作者的 LoCoMo/LongMemEval 流程一致。

来源追踪：叶子 chunk 映射到它的 LoCoMo 来源 ID；snapshot 摘要跨多个 chunk，来源组留空。

## Neo4j 与 `--force`

- 每个 context 一个 Graphiti group `md_tsm_<32 位 uuid>`，写入本地清单后才开始构建；清单中还记录 Neo4j 地址和数据库名。
- 未完成的构建在下次运行时先删除自己的 group，再整体重建。
- `--force` 只删除清单记录的 group；如果配置里的 Neo4j 地址或数据库变了，拒绝重置，避免删错库。不属于本适配器格式的 group 名一律拒绝操作。
- 需要已运行的 Neo4j 5.x，密码放在 `.env` 的 `NEO4J_PASSWORD`（变量名可用 `tsm_neo4j_password_env` 修改），不要写进配置。`config/topological_zep_local.yaml` 也使用 `bolt://localhost:7687`；两者用 group 隔离，可以共用同一个实例。

## 配置

`config/topological_tsm.yaml`，`agent_name: Agentic_memory_tsm`。

| 字段 | 说明 |
| --- | --- |
| `model_env` / `tsm_llm_model` | Graphiti 抽取、摘要、原生回答和 MemoryData 回答都用 `OPENAI_MODEL`（gpt-4o-mini，与论文主实验一致） |
| `embedding_model_env` / `tsm_embedding_model` / `tsm_embedding_dimensions` | `OPENAI_EMBEDDING_MODEL`（text-embedding-3-small，1536 维，与论文默认一致） |
| `tsm_python` / `tsm_python_env` | TSM 环境的解释器，默认 `D:/conda/envs/tsm/python.exe`，可用 `TSM_PYTHON` 覆盖 |
| `tsm_neo4j_*` | Neo4j 地址、用户、数据库、密码变量名 |
| `tsm_answer_mode` | `memorydata`（默认）或 `native` |
| `tsm_native_config` | 可覆盖 `cluster`、`entity_clustering`、`retriever_kwargs`、`llm_max_tokens`。默认值取自作者的 `configs/locomo/tsm.yaml`（实体聚类阈值 0.1，`max_chunk_setting 20`，时间过滤开启，`temporal.enable: false`） |

身份指纹包含除检索参数外的全部构建设置、Neo4j 地址和数据库、数据集配置、完整历史和状态目录。只改 `retriever_kwargs` 时复用已建索引。

## 使用

TSM 环境已于 2026-11-20 创建（Python 3.11，graphiti-core 0.19.0、neo4j 5.28.1、spacy 3.8.4、faiss-cpu 1.9.0.post1、openai 2.24.0、numpy 1.26.4，`pip check` 无冲突）。在其他机器上重建：

```powershell
C:\Users\wjp\anaconda3\Scripts\conda.exe create -y -p D:\conda\envs\tsm python=3.11
```

```powershell
D:\conda\envs\tsm\python.exe -m pip install -r methods/tsm/requirements-tsm.txt
```

spaCy 英文模型需要单独下载（上游优先 `en_core_web_lg`，约 400 MB；其次 `en_core_web_sm`，约 12 MB），**目前未安装**：已在真实环境中确认上游此时退回空白管线（`spacy.blank("en")`），时间约束解析能力变弱。

运行：

```powershell
D:\conda\envs\memorydata\python.exe main.py --agent_config config/topological_tsm.yaml --dataset_config benchmark/locomo/config/Locomo_qa_4cat_600_dist.yaml --max_test_queries_ablation 2
```

## 与原方法的差异和缺口

- **尚未做真实 API 运行**：环境已建好，并已在真实环境中打开 worker，确认配置结构被上游 `extract_config_bundle` 接受，`parallel_build_extract`、`build_retriever`、`_format_results_with_summaries`、`clear_group_data`、`build_graph` 的签名与预期一致，未调用模型，也未连接 Neo4j。构建和查询需要真实 LLM、嵌入和 Neo4j，只用替身测过；spaCy 模型未安装，Neo4j 密码尚未配置。首次真实运行前需要一次小规模冒烟测试（会产生 API 费用）。
- **实现来源**：见上文，非论文确认的官方代码，版本 0.1.1，作者自述未做真实 API 基准运行。
- 只用用户消息：上游 TSM 配置 `use_user_only: true`，只用用户发言构图和生成 persona。MemoryData 的 chunk 是多说话人混合文本，无法可靠拆出"用户"，因此关闭该选项、使用整个 chunk。
- 粒度：上游一个叶子是一轮（用户 + 助手）；这里一个叶子是一个 MemoryData chunk（`agent_chunk_size` 4096 token），更粗。需要更接近原设置时可减小 `agent_chunk_size`。
- 时间信息：本分支的 MemoryData chunk 一般是纯文本，没有单独的时间戳字段（只有结构化 chunk 带 `timestamp`），因此通常不加 `Date:` 头，日期留在正文里（如 LoCoMo 的 `Session 1 (1:56 pm on 8 May, 2023)`）。上游从文本中找 `YYYY-MM-DD` 或 `YYYY/MM/DD` 作为 Graphiti 的事件时间，找不到时用 2000-01-01 加序号；LoCoMo 的原始日期格式匹配不上这两种写法，上游自己的 LoCoMo 加载器同样把原始字符串放进 `Date:`，所以这一点与上游一致，但按月切片的 snapshots 实际上会集中在兜底日期，时间相关能力可能明显弱于论文描述。
- 问题日期：上游对 LoCoMo 以"最后一个 session 的次日"作为问题日期；MemoryData 只在数据集提供 `question_date` 时传入（MemoryAgentBench 的 LongMemEval 有，LoCoMo 没有），没有时检索器退回"全局最新 snapshot"。
- 只支持整批构建；HaluMem 式交错提问（`prepare_for_queries`）未实现。上游的增量更新需要重新聚类和重建 snapshots，代价与完整重建相近。
- 恢复时重新嵌入整棵树（上游行为），见上文。

## 真实 API 冒烟测试（2026-11-20，未通过）

数据与 REMem 冒烟相同（MemoryAgentBench `longmemeval_s*` 首个样本，前 3 个 chunk，2 题），Neo4j 为本机 2026.09 企业版，spaCy 模型未安装。三次尝试都停在 Graphiti 抽取的第一步，**没有产生任何模型调用**，Neo4j 中没有写入数据。发现的都是发布版代码与其自身固定依赖 `graphiti-core==0.19.0`（`REPRODUCIBILITY.md` 所列）之间的不兼容：

1. `GraphitiExtractor._initialize_graphiti` 调用 `LLMConfig(..., request_timeout=...)`，而 0.19.0 的 `LLMConfig` 没有这个参数，直接 `TypeError`。已在 worker 中加兼容垫片：参数不被支持时接收并丢弃，代价是 Graphiti 改用 OpenAI SDK 的默认请求超时，而不是发布版设想的 120 秒。
2. `extract_graph_compatible` 给每个 `RawEpisode` 传入固定的 `uuid`（`uuid5`），而 0.19.0 的 `add_episode_bulk` 把带 uuid 的 episode 当作"已存在、按 uuid 读出"，在新分组上必然 `NodeNotFoundError`。已在 worker 中加第二个兼容垫片：仅当 `add_episode_bulk` 会按 uuid 读取已有 episode 时，构造 `RawEpisode` 前丢弃传入的 uuid，由 Graphiti 新建。发布版读图时按 `chunk_<i>` 名称取回库中的 uuid，不依赖传入值，因此功能上等价，但会失去重复写入的幂等性（MemoryData 每次构建都先清空本 context 的分组，不依赖这一点）；之后的聚类、snapshots、稠密索引和查询从未在真实环境中运行过，可能还有其他不兼容。

结论：TSM 发布版 0.1.1 按发布状态无法完成构建（与其 README"未做真实 API 基准运行"的自述一致）。两处不兼容现都已用垫片绕过，上游源码未改；带垫片的重跑待 Neo4j 再次可用后进行（重启后 `localhost:7687` 拒绝连接）。

另外，第一次失败时暴露了桥接代码的两个问题，已修复并加了离线测试：存活 worker 登记表原为弱引用，`main.py` 因异常退出时 worker 可能已被回收而得不到关闭；worker 在协议循环结束后可能被上游留下的非守护线程卡住、不退出。现在登记表持强引用，worker 在循环结束后强制退出，并在 `worker.log` 首行记录 PID。修复前遗留的 worker 进程已随重启结束，旧结果目录已删除。
