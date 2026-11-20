# 记忆系统接入说明

本文说明 MemoryData 如何接入一个记忆系统，以已接入的 Hindsight 为范例。内容基于提交 `d0e18dc`（`hindsight-api-integration` 分支）。

## 1. 运行链路

```text
main.py
  ├─ setup_configs_and_directories()      读取 Agent/Dataset YAML，resolve_model_config() 解析 *_env，
  │                                        应用消融参数，生成输出路径
  ├─ create_agent_and_fetch_data()        ConversationCreator 按 agent_chunk_size 分块，返回
  │                                        (start_time, 各 context 的 chunk 列表, 各 context 的问答对)
  └─ 对每个 context：
       generate_agent_save_folder()        每个 context 一个本地状态目录
       initialize_and_memorize_agent()     --force 时先清外部状态再删本地目录；
                                           构造 AgentWrapper；判断能否复用已保存状态：
                                             能 → agent.load_agent()
                                             否 → 逐块 send_message(chunk, memorizing=True)，再 agent.save_agent()
       对每个问题：agent.send_message(query, memorizing=False, query_id, context_id, eval_metadata)
                   返回 {"output", "input_len", "output_len", "memory_construction_time", "query_time_len", ...}
```

方法类型完全由 YAML 中的 `agent_name` 子串决定（`AgentWrapper._is_agent_type` 是 `agent_type in self.agent_name`）。因此新方法名不能与已有子串冲突，例如含 `rag`、`mem0`、`zep`、`a_mem` 的名字会被误判。

## 2. 接入一个方法要改的位置

| 位置 | 作用 | Hindsight 的做法 |
| --- | --- | --- |
| `config/<分类>_<方法>.yaml` | `agent_name`、回答模型、分块、检索数、方法私有参数 | `config/hybrid_hindsight.yaml`，`agent_name: Agentic_memory_hindsight` |
| `utils/provider_utils.resolve_model_config` | 把 `model_env`、`embedding_model_env`、`embedding_dim_env` 解析成实际模型名，并同步方法私有别名字段 | 别名 `hindsight_llm_model`、`hindsight_embeddings_model`、`hindsight_embeddings_dimensions` |
| `AgentWrapper._initialize_agent_by_type` | 按名字构造适配器 | `_initialize_hindsight_agent`：`HindsightAdapter(agent_config, dataset_config, self.agent_save_to_folder)` + 回答用 OpenAI 客户端 |
| `AgentWrapper.send_message` | 预处理（如取出 chunk 的 `timestamp`），并把记忆方法路由到 `_handle_memory_agent` | 在列表里加入 `"hindsight"`；记忆写入时暂存时间戳 |
| `_handle_memory_agent` → `_handle_<方法>_agent` | `memorizing=True` 写入；`False` 检索并回答 | 写入调用 `add_chunk`；检索调用 `retrieve`，再用 `_fit_memories_for_answer` 截断、`_generate_answer_from_memories` 由 MemoryData 的回答模型作答，`_attach_locomo_recall_metadata` 附来源 |
| `save_agent` / `load_agent` | 写入结束后的持久化；复用时的恢复 | 委托 `hindsight.save()`（等待后台整合、标记 ready）/ `hindsight.load()`（确认远端 bank 仍存在） |
| `initialize_and_memorize_agent` 中"能否复用" | 决定 load 还是重建 | 以本地清单 `ready` 为准；只差整合未完成时只等待、不重写 |
| `_reset_external_persistent_state` | `--force` 时清理外部状态 | `HindsightAdapter.reset_saved` 只删除本地清单记录的 UUID bank |
| `utils/conversation_creator.MEMORY_AGENT_NAME_HINTS` | 允许该方法设置 `agent_chunk_size` | 加入 `"hindsight"` |
| `initialization._apply_chunk_size_ablation`、`_generate_output_name_tag` | 消融时改 `agent_chunk_size`；输出名包含 `k{retrieve_num}_chunk{...}` | 两处方法名列表加入 `"hindsight"` |
| `benchmark/memoryagentbench/prompts/benchmark_templates.AGENT_TYPE_MAPPING` | 选择记忆/提问模板 | `("hindsight", "agentic_memory_agent")`，放在 `rag` 之前 |
| `AgentWrapper._supports_locomo_recall` | 是否在写入时保留 `LOCOMO_META` 来源标注并计算来源召回率 | 加入；适配器在本地清单中保存 `document_id → source_ids`，检索结果按此还原 |
| `main.BACKFILL_LONGMEMEVAL_RECALL_METHODS` | LongMemEval 召回调试回填 | 加入 `("hindsight", "Hindsight")` |
| 交错提问钩子 `prepare_for_queries` / `finish_query_batch` | 逐 session 交错提问（HaluMem） | 本分支尚无这两个钩子，它们在 `feature/halumem` 分支上；新适配器应能在写入未结束时多次进入检索（见第 4 节） |

## 3. Hindsight 范例的要点

文件：`methods/hindsight/hindsight_adapter.py`、`methods/hindsight/api_config.py`、`config/hybrid_hindsight.yaml`、`scripts/run_hindsight_server.py`、`tests/test_hindsight.py`、`docs/hindsight.md`。

- **外部服务适配**：Hindsight 作为独立服务运行，MemoryData 只通过 REST（`httpx`）调用 retain/recall，不导入其服务端代码。`api_config.server_environment` 把根 `.env` 的 `OPENAI_*` 映射为服务端变量，仅传给服务进程。
- **隔离**：每个 context 一个 `md-hs-<uuid>` bank，UUID 写进本地清单后才开始写入，避免误用或误删他人数据。
- **本地状态清单** `hindsight_state.json`：协议名、身份指纹、bank id、`ready`、`document_id → source_ids`；不含密钥，原子写入（临时文件 + `os.replace`）。
- **身份指纹**：对端点、bank 配置、数据集配置、完整历史（`_hindsight_history_fingerprint`，由 `initialize_and_memorize_agent` 注入）和状态目录做 SHA-256。只影响检索的参数（top-k、预算）不计入，修改后可复用；配置或历史变化则拒绝复用，要求 `--force` 或新 `--artifact_root`。路径按仓库相对路径计算，整体移动仓库不失效。
- **保存与恢复**：写入失败绝不标记 ready；下次运行整体重建。`save()` 等待后台整合后才置 ready，超时保留已确认的写入，重跑时只继续等待。
- **`--force`**：先删除清单记录的远端 bank，再删本地目录；端点变化时拒绝重置。
- **来源追踪**：LoCoMo 的来源 ID 存在清单而不是记忆文本里；observation 通过 `source_facts` 递归追溯。缺失时保留空组，不按文本猜。
- **时间戳**：chunk 的 `timestamp` 作为写入时间；问题的 `question_date` 规范为 ISO 后作为 `query_timestamp`，不臆造时区。
- **最终回答**：由 MemoryData 的回答模型（`model_env: OPENAI_MODEL`）按统一模板生成，不调用原生 `reflect`，文档中明确这不是原生完整问答协议。
- **离线测试**：`tests/test_hindsight.py` 用假后端（同一个 `Backend` 既直接注入 `httpx.MockTransport`，也通过本地回环 HTTP 服务供 `main.py` 全流程使用），覆盖密钥不落盘、复用/拒绝复用、重建、`--force`、超时恢复、时间戳、来源追踪，不调用任何真实模型。

## 4. 新方法接入清单

1. `methods/<方法>/`：适配器类，构造参数 `(agent_config, dataset_config, state_dir)`，提供 `add_chunk(content, source_ids, timestamp)`、`retrieve(query, query_timestamp)`、`save()`、`load()`、`reset_saved(agent_config, state_dir)`；状态清单不含密钥、带身份指纹。
2. `config/<分类>_<方法>.yaml`：`agent_name` 选不与现有子串冲突的名字；`model_env: OPENAI_MODEL`、`embedding_model_env: OPENAI_EMBEDDING_MODEL`、`embedding_dim_env`、`api_key_env`、`base_url_env`；方法私有的模型字段在 `resolve_model_config` 中登记为别名。
3. `AgentWrapper`：初始化、`send_message` 列表、`_handle_memory_agent` 分派、`_handle_<方法>_agent`、`save_agent`/`load_agent`、`_supports_locomo_recall`。
4. `initialization.py`：复用判断、`--force` 清理、两处方法名列表；`conversation_creator.MEMORY_AGENT_NAME_HINTS`；`AGENT_TYPE_MAPPING`。
5. 交错提问：适配器不应假设"全部写完才查询"；若原方法需要整理（如聚类、整合）才能检索，应提供可重复调用的整理步骤，供 `prepare_for_queries` 调用。
6. `tests/test_<方法>.py`：假后端，不调用真实模型；`docs/<方法>.md`：说明协议、与原方法的差异和缺口。

对共享文件（`utils/agent.py`、`utils/initialization.py` 等）只做最小增量修改，方法逻辑尽量放在新增文件中，以减少与其他分支的合并冲突。
