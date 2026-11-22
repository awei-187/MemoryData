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

- worker 用 `<tsm python> -I -X utf8 -u tsm_worker.py` 启动（`-I` 会忽略 `PYTHONIOENCODING`，靠 `-X utf8` 保证上游输出 ✅ 和中文日志不出错）；密钥（`OPENAI_API_KEY`、`NEO4J_PASSWORD`）只经子进程环境变量传入；上游的 stdout 输出写入 `<context>/worker.log`。
- LLM 和嵌入地址通过上游支持的 `TSM_LLM_BASE_URL`、`TSM_EMBEDDING_BASE_URL` 指向 `.env` 中的 `OPENAI_BASE_URL`。
- 环境依赖见 `methods/tsm/requirements-tsm.txt`（与上游 `requirements.lock` 相同的版本）。

## 数据流

1. 写入：chunk 去掉 `LOCOMO_META` 包装后追加到 `documents.jsonl`；来源 ID 和时间戳记在 `tsm_state.json`。不调用模型。
2. `save()`：每个 chunk 成为一个 TSM "turn"，格式为 `Date: <时间戳>\n<文本>`（上游按轮读取数据时生成同样的 `Date:` 头）。ISO 时间戳里日期与时间之间的 `T` 换成空格：上游用 `\bYYYY-MM-DD\b` 提取事件时间，`2024-01-01T10:00` 中日期后紧跟 `T` 不构成单词边界，会悄悄落到 2000-01-01 的兜底值（已在真实环境中验证）。worker 依次调用上游 `parallel_build_extract`（Graphiti 抽取 → 聚类建树和 snapshots → FAISS 索引）和 `build_retriever`。完成后检查缓存文件、snapshots、稠密检索器都已生成、叶子数等于 chunk 数、FAISS 向量数等于节点数且没有全零向量，才把清单标为 `ready`（上游多处出错只记日志、以零向量等兜底继续）。
3. 恢复：按上游 `main_time.run_pipeline` 的缓存分支，读取 `tree.json`、`graph.json`、`index.json`、`appearance_count.json`、`text_edge_mapping.json` 重建检索器。上游在这条路径上不传预建索引，会**重新嵌入整棵树**，因此每次恢复都有少量嵌入费用。
4. 查询：调用 `Retriever.query(question, **retriever_kwargs, question_date=…)`。worker 包装了 `_format_results_with_summaries`，额外记录被格式化的叶子 ID（修补后 `leaf_<i>` 对应第 i 个 chunk，见下文）；返回文本不变。
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
| `tsm_worker_timeout_seconds` / `tsm_index_seconds_per_chunk` | 每次 worker 请求的超时（默认 7200 秒）；建图和复用时的 `reopen` 再加上每个 chunk 的额度（默认 600 秒）。冒烟中建图约 120 秒/块（并发 4），完整 LongMemEval 样本 88 块约需 3 小时，固定上限会中断构建并在重跑时重付全部抽取费用 |
| `tsm_neo4j_*` | Neo4j 地址、用户、数据库、密码变量名 |
| `tsm_answer_mode` | `memorydata`（默认）或 `native` |
| `tsm_graphiti_concurrency` | Graphiti 并发模型请求上限，经 `SEMAPHORE_LIMIT` 传给 worker（上游默认 20，嵌套并发实际更多）。默认配置设 4：约 40 个并发的长抽取请求在共享中转上全部超时。只影响速度，不计入身份指纹 |
| `agent_variant_tag` | `…-release-0.1.1-mdpatch1`，标明结果来自打了 MemoryData 修补的发布版 |
| `tsm_native_config` | 可覆盖 `cluster`、`entity_clustering`、`retriever_kwargs`、`llm_max_tokens`。默认值取自作者的 `configs/locomo/tsm.yaml`（实体聚类阈值 0.1，`max_chunk_setting 20`，时间过滤开启，`temporal.enable: false`） |

身份指纹包含修补清单（`RELEASE_PATCHES`）、除检索参数外的全部构建设置、Neo4j 地址和数据库、数据集配置、完整历史和状态目录。只改 `retriever_kwargs` 时复用已建索引。

## 使用

TSM 环境已于 2026-11-20 创建（Python 3.11，graphiti-core 0.19.0、neo4j 5.28.1、spacy 3.8.4、faiss-cpu 1.9.0.post1、openai 2.24.0、numpy 1.26.4，`pip check` 无冲突）。在其他机器上重建：

```powershell
conda create -y -p D:\conda\envs\tsm python=3.11
```

```powershell
D:\conda\envs\tsm\python.exe -m pip install -r methods/tsm/requirements-tsm.txt
```

spaCy 英文模型需要单独下载（上游优先 `en_core_web_lg`，约 400 MB；其次 `en_core_web_sm`，约 12 MB），**目前未安装**：已在真实环境中确认上游此时退回空白管线（`spacy.blank("en")`），时间约束解析能力变弱。

运行：

```powershell
D:\conda\envs\memorydata\python.exe main.py --agent_config config/topological_tsm.yaml --dataset_config benchmark/locomo/config/Locomo_qa_4cat_600_dist.yaml --max_test_queries_ablation 2
```

## 对发布版的兼容处理与修补

所有改动都在 worker 进程里替换上游函数，`methods/tsm/source` 中的上游文件未改。

与其固定依赖 `graphiti-core==0.19.0` 的不兼容（只改调用方式，不改方法）：

1. `LLMConfig(..., request_timeout=...)`：0.19.0 不认识该参数（`TypeError`）。接收并丢弃，Graphiti 改用 OpenAI SDK 默认超时，而非发布版设想的 120 秒。
2. `RawEpisode(uuid=<uuid5>)`：0.19.0 的 `add_episode_bulk` 把带 uuid 的 episode 当作已存在去读取（`NodeNotFoundError`）。不传 uuid；发布版读图时按名称 `chunk_<i>` 取回 uuid，功能等价，只失去重复写入的幂等性（MemoryData 每次构建前都清空本 context 的分组）。
3. `OpenAIClient` 默认给每次结构化调用带 GPT-5 专用参数 `reasoning`/`verbosity`，gpt-4o-mini 返回 400。对非推理模型（不以 gpt-5、o1、o3、o4 开头）设为 None，模型不变。

发布版的两处逻辑缺陷（修补会改变发布版的运行结果，结果标签因此带 `mdpatch1`，修补清单计入身份指纹）：

1. **稠密索引恒为零向量**：`parallel_build_extract` 是异步函数，却在其中同步调用 `build_dense_index`；`embed_documents_batch` 再自建事件循环 `run_until_complete`，在已运行的循环里必然失败（`Cannot run the event loop while another loop is running`），上游捕获后静默写入零向量。上游自己的入口 `asyncio.run(run_pipeline(...))` 同样如此，非时序入口 `main.py` 也一样。修补：调用方处在运行中的事件循环时，把同一函数放到独立线程执行（与上游 `graphiti_extractor._run_coroutine` 的做法相同），算法不变；并在建完后拒绝零向量索引。
2. **叶子只剩最后一个时间窗口**：`build_tree_clustering_time.py` 的按月循环里把参数 `text_chunks` 重绑为本窗口 episode 的文本、每个窗口的叶子都从 `leaf_0` 编号，`all_caches.update(cache)` 让后面的窗口覆盖前面的。历史跨月时多数原文对检索不可见。同仓库的非时序 `build_tree_clustering.py` 为每个输入块各建一个叶子（注释"为所有 text_chunks 创建 leaf 节点，不管是否被实体引用"）。修补：树建好后按非时序版的规则重建叶子（`leaf_i` = 第 i 个输入块，叶子文本取法沿用时序版），重写 `tree.json`；snapshots 与实体索引不变。已核对：时间过滤按文本内容匹配叶子，不依赖叶子编号；snapshots 里各窗口的 `cache` 检索时不读取。

缺陷来源：GitHub 仓库 2026-09-01 新建，只有 3 个提交，`build_tree_clustering_time.py` 只出现在首个提交里。上游 Level 1 离线验收（`scripts/synthetic_e2e.py`）只跑非时序的 `main.py` 并用哈希嵌入，`tests/test_release_contracts.py` 只检查配置，完整 TSM 的建树和建索引在发布前从未执行。论文分数来自未公开的内部代码，无法判断内部代码是否有同样问题；缺陷一所在的桥接代码注释为 "Public async embedder interface, bridged into this synchronous stage."，应是发布时为公开接口新写的。

## 与原方法的差异和缺口

- **实现来源与修补**：见上文。论文主方法是时序版（`main_time.py`）；`no_temporal`、`no_persona`、`traditional`、`pure_rag` 是消融，本接入未提供。论文 Table 3（LongMemEval_S，GPT-4o-mini）：完整 TSM 74.80，w/o Temporal 72.80，w/o Persona/Summary 73.40。
- 只用用户消息：上游 TSM 配置 `use_user_only: true`，只用用户发言构图和生成 persona。MemoryData 的 chunk 是多说话人混合文本，无法可靠拆出"用户"，因此关闭该选项、使用整个 chunk。
- 粒度：上游一个叶子是一轮（用户 + 助手）；这里一个叶子是一个 MemoryData chunk（`agent_chunk_size` 4096 token），更粗。需要更接近原设置时可减小 `agent_chunk_size`。
- 时间信息：本分支的 MemoryData chunk 一般是纯文本，没有单独的时间戳字段（只有结构化 chunk 带 `timestamp`），因此通常不加 `Date:` 头，日期留在正文里（如 LoCoMo 的 `Session 1 (1:56 pm on 8 May, 2023)`）。上游从文本中找 `YYYY-MM-DD` 或 `YYYY/MM/DD` 作为 Graphiti 的事件时间，找不到时用 2000-01-01 加序号；LoCoMo 的原始日期格式匹配不上这两种写法，上游自己的 LoCoMo 加载器同样把原始字符串放进 `Date:`，所以这一点与上游一致，但按月切片的 snapshots 实际上会集中在兜底日期，时间相关能力可能明显弱于论文描述。MemoryAgentBench 的 LongMemEval chunk 正文带 `2022/11/17` 形式的日期，能被识别。
- 问题日期：上游对 LoCoMo 以"最后一个 session 的次日"作为问题日期；MemoryData 只在数据集提供 `question_date` 时传入（MemoryAgentBench 的 LongMemEval 有，LoCoMo 没有），没有时检索器退回"全局最新 snapshot"；格式无法识别时记录一次警告并按无日期处理。
- spaCy 英文模型未安装，上游退回空白管线，问题中的时间约束解析变弱。
- 只支持整批构建；HaluMem 式交错提问（`prepare_for_queries`）未实现，数据集配置带 `interleave_queries` 时初始化即报错。上游的增量更新需要重新聚类和重建 snapshots，代价与完整重建相近。
- 恢复时重新嵌入整棵树（上游行为），见上文。

## 真实 API 冒烟测试（2026-11-20 至 11-21，通过）

数据与 REMem 冒烟相同（MemoryAgentBench `longmemeval_s*` 首个样本，88 个 chunk 中的前 3 个，2 题），Neo4j 为本机 2026.09 企业版，模型 gpt-4o-mini、text-embedding-3-small。共 10 次尝试，依次暴露并处理了：上述 3 处 Graphiti 不兼容；约 40 个并发请求在中转上全部超时（加 `tsm_graphiti_concurrency`）；上游聚类阶段 `print("✅ …")` 在 GBK 管道上报错（worker 加 `-X utf8`）；上述 2 处发布版逻辑缺陷；以及桥接代码自身的进程问题（见下）。另有两次因 Neo4j 未运行而在连接阶段失败。

最终一次（`results/tsm_smoke_20261121`，exit 0，共 384 秒）：

- 构建 371 秒：Graphiti 抽取约 305 秒（89 个实体、43 条关系，其中一次 `NodeResolutions` 输出校验失败后重试成功）；时间范围 2022-11-17 至 2023-01-15，按月 3 个窗口，第 2 个窗口无实体，生成 2 个 snapshots；3 个叶子对应 3 个 chunk；稠密索引 3 个非零向量。
- 查询每题约 2–3.5 秒，检索返回 3 个叶子并按问题日期选中 1 个 snapshot。
- 两题得分为 0：两题的答案线索（工作时长、鱼缸）都不在前 3 个 chunk 里（关键词检索确认），截断历史下无法作答，与 REMem 冒烟相同，分数不可与完整评测比较。
- LongMemEval 召回报告失败，原因是本地缺 `datasets/LongMemEval/.../longmemeval_s_cleaned.json`，与 TSM 无关（REMem 冒烟同样）。
- 修补前的第 8 次运行（`results/tsm_smoke_20261120d`）流程同样跑完，但树里只有 1 个叶子、稠密索引为零向量，即发布版原样的结果；它在 Neo4j 中的分组仍保留，可对该目录用 `--force` 清理。

桥接代码在冒烟中暴露并已修复的问题（均有离线测试）：

- 存活 worker 登记表原为弱引用，`main.py` 因异常退出时可能关不掉 worker；现为强引用，并在 `worker.log` 首行记录 PID。
- worker 结束时可能被上游的非守护线程卡住；现在协议循环结束后强制退出。
- 强制退出会跳过 multiprocessing 的清理：上游每次构建都通过 `Timer()` 启动 `mp.Manager` 子进程，其工作目录就是 context 目录，会成为孤儿进程并锁住目录。现在 worker 退出前先结束自己的 multiprocessing 子进程；父进程正常关闭时等待 worker 自行退出，强制结束时在 Windows 上用 `taskkill /T` 结束整棵进程树。
