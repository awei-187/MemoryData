# Hindsight 接入 / Hindsight integration

本接入在现有 `main.py`、Agent YAML、Dataset YAML 及结果格式下新增 Hindsight 方法，不恢复此前已回退的 OmniMemEval 接入。

API 合约依据本地 `E:\AML\hindsight`，来源仓库为 [vectorize-io/hindsight](https://github.com/vectorize-io/hindsight)，检查版本 `1e427025b4d4c01e8ad6385dd04e02102886a5c4`，Python 客户端/OpenAPI 版本 `0.10.1`。MemoryData 仅实现 REST 适配器，没有修改外部源码，也不需要安装 Hindsight Python SDK 或复制服务端。

本机原 Hindsight 实际向上读取 `E:\AML\.env`。这份文件最初原样复制到 MemoryData 根 `.env`，随后补充了统一模型环境变量；原有密钥和地址保留，文件由 Git 忽略。本次配置使用远程 API：`gpt-4o-mini` 负责记忆抽取与最终回答，`text-embedding-3-small` 提供 1536 维嵌入，使用 RRF 而非本地神经重排模型。全部预设现使用同一组模型变量，见 [统一 API 配置](api_models.md)。

## 评测协议

```text
原有数据集历史 → 原生 Hindsight retain → 独立 bank 中的记忆
问题 → 原生 Hindsight recall → MemoryData 回答模型 → 原有评分流程
```

- Hindsight 保留原生事实抽取、多路检索、重排和 observation 整合；这些操作使用服务端配置的模型。
- 最终回答使用 MemoryData 所选 Agent 配置的模型、模板、上下文预算及评分；本版本不调用原生 `reflect`，不能将结果宣称为 Hindsight 原生完整问答协议的复现。
- 原有 MemoryAgentBench、LoCoMo、LongBench、MemBench 的配置继续使用。Hindsight 在模板、分块、路径、保存/恢复、`--force` 和 LongMemEval recall debug 分派中均被识别为记忆方法；后续统一 API 改造更新了其他方法的模型预设，未改变数据集接口。
- 正确答案和评分标注不写入记忆。LoCoMo 来源 ID 保存在文档映射中，不把内部 `LOCOMO_META` 包装作为记忆内容；召回原始事实通过 `document_id` 还原来源，observation 通过返回的 `source_facts` 追溯。缺少来源映射时保留空组，不按文本猜测证据。

## 前置条件

使用已有 Python 3.11 环境 `D:\conda\envs\memorydata` 及 MemoryData 原有依赖。新增适配器直接依赖 `httpx>=0.27,<1`，已加入两个原依赖清单；本机已安装，无需额外下载。

需要两类已运行服务：

| 服务 | 默认地址 | 配置位置 |
| --- | --- | --- |
| Hindsight API | `http://127.0.0.1:8888` | `hindsight_base_url`；`HINDSIGHT_API_URL` 可覆盖 |
| OpenAI 兼容模型 API | 根 `.env` 中的 `OPENAI_BASE_URL`，默认模型 `gpt-4o-mini` | `base_url_env`、`api_key_env`、`model` |

Hindsight API 地址是服务根地址，不加 `/v1` 或 `/v1/default`，也不是回答模型地址。远程模型 API 提供推理；本地 Hindsight 引擎仍负责数据库、写入、检索和整合。无需启动本地 Qwen `9908` / 嵌入 `9009`。本项目的启动器可使用已有 `hindsight` 环境和本地源码运行原生服务，也可以连接另一台电脑的 Hindsight 服务。

Hindsight 鉴权可选，密钥读取 `HINDSIGHT_API_KEY`（可通过配置更换变量名）；模型密钥读取 `OPENAI_API_KEY`。真实远程 API 需要有效密钥，不能使用 `local` 占位值。适配器的本地 Hindsight HTTP 客户端绕开代理；模型客户端通过 `model_api_trust_env: true` 使用现有代理环境。本机验证中，远程 API 直连超时，经现有代理正常响应。

## 使用方式

以下为当前 Windows 工作区的单行 PowerShell 命令，其他机器需要调整路径。根 `.env` 需包含有效的 `OPENAI_API_KEY`、`OPENAI_BASE_URL`。

```powershell
cd 'E:\MemoryData'
```

先使用已有 Hindsight 专用环境启动原生记忆引擎，保持该进程运行；此环境与 MemoryData 评测环境分开。启动器将根 `.env` 的模型 API 配置映射到 Hindsight 原生变量，数据库与服务日志保存在 `results/hindsight_server/`，不安装或下载本地模型。已提供 `HINDSIGHT_API_DATABASE_URL` 时使用该数据库，否则使用本机已有 pg0 运行时创建独立数据库。

```powershell
& 'D:\conda\envs\hindsight\python.exe' -u scripts/run_hindsight_server.py
```

随后在另一个 PowerShell 中使用现有 LoCoMo 配置，先只验证 1 道问题：

```powershell
& 'D:\conda\envs\memorydata\python.exe' main.py --agent_config config/hybrid_hindsight.yaml --dataset_config benchmark/locomo/config/Locomo_qa_4cat_600_dist.yaml --artifact_root results/hindsight_smoke --max_test_queries_ablation 1
```

此命令仍摄取所选首个对话单元在该数据集配置下的完整历史，不等于只写入一个 chunk。`max_test_queries_ablation` 只限制问答数量，不会被视为历史变更；提高问题数量可以复用已完成的首个单元记忆。

仅需检查真实 API 链路时，可使用一份很小的合成历史，完成写入、整合、检索、问答、原评分及再次恢复；这不构成正式数据集效果验收：

```powershell
& 'D:\conda\envs\memorydata\python.exe' scripts/smoke_hindsight_api.py
```

停止由上述启动器运行的默认服务（保留数据库文件）：

```powershell
New-Item -ItemType File -Force -Path 'E:\MemoryData\results\hindsight_server\stop.request'
```

使用 MemoryAgentBench 的 LongMemEval S 配置：

```powershell
& 'D:\conda\envs\memorydata\python.exe' main.py --agent_config config/hybrid_hindsight.yaml --dataset_config benchmark/memoryagentbench/Accurate_Retrieval/config/LongMemEval/Longmemeval_s.yaml --artifact_root results/hindsight_lme --max_test_queries_ablation 1
```

更换 `--dataset_config` 即可使用原有其他基准，不需要修改加载器或评分入口。数据仍需按原项目要求自行准备。长任务如需实时观察，可在独立 Windows PowerShell 窗口中执行这些命令。

## 配置与预算

默认配置：`config/hybrid_hindsight.yaml`。

| 字段 | 默认值 / 作用 |
| --- | --- |
| `agent_name` | `Agentic_memory_hindsight` |
| `model` / `model_env` | 默认 `gpt-4o-mini`，共享 `OPENAI_MODEL` 可覆盖，实际模型名写入结果 |
| `base_url_env` / `api_key_env` | `OPENAI_BASE_URL` / `OPENAI_API_KEY`，从根 `.env` 读取 |
| `model_api_trust_env` | 默认 `true`，由共享 `OPENAI_API_TRUST_ENV` 控制模型请求是否使用现有代理 |
| `agent_chunk_size` | 4096 字符，沿用 MemoryData 分块规则；支持 `--chunk_size_ablation` |
| `retrieve_num` | 20，原生召回排名返回后再截取前 k 项 |
| `hindsight_recall_budget` | `mid`；可选 `low` / `mid` / `high`，原生检索计算预算 |
| `hindsight_recall_max_tokens` | 4096，原生召回 Token 预算，与 k 是不同维度 |
| `hindsight_recall_types` | `[world, experience, observation]` |
| `hindsight_timeout_seconds` | 单次 HTTP 请求超时，默认 300 秒 |
| `hindsight_wait_for_consolidation` | 默认 `true`；全部历史写入后等待后台整合结束 |
| `hindsight_settle_timeout_seconds` | 默认 300 秒，等待整合的截止时间；超时后重跑同一命令会继续等待，可调大此值 |
| `hindsight_legacy_identity_params` | 可选，仅用于复用 2026-11-19 前保存、且当时检索/等待参数与当前不同的 bank，见下文 |
| `hindsight_bank_config` | 创建 bank 时传给原生 API 的可选 bank 设置，默认 `{}`，使用服务端默认值 |
| `hindsight_memory_answer_prompt` | 可选，覆盖原有统一回答模板 |
| `hindsight_llm_model` | 原生引擎的抽取模型，默认 `gpt-4o-mini` |
| `hindsight_embeddings_model` / `hindsight_embeddings_dimensions` | 默认 `text-embedding-3-small` / 1536 |
| `hindsight_reranker_provider` | 默认 `rrf`，与现有 E:\AML API 配置一致，不调用本地神经重排模型 |
| `input_length_limit`、`model_context_window` | 最终回答输入预算，独立于原生 recall 预算 |

API 使用同步 `retain`（`async: false`），每个 chunk 有独立文档 ID，`update_mode: replace`。不自动重试同步写入，避免超时后重复插入。写入失败、返回异步结果或不确认成功时，不写“历史完成”标记。

默认保存前查询 `/stats?refresh=true`，检查待执行/执行中操作和待整合事实；后台失败或超时会阻止问答。若实验明确要求不等待，可设 `hindsight_wait_for_consolidation: false`，但应在实验说明中报告这种协议差异。服务端禁用自动整合而又存在待整合事实时，需先对齐服务端配置或显式选择不等待。

历史有结构化时间时传给 retain；缺失时使用上游支持的 `timestamp: unset`。原始对话中的日期、角色和顺序仍保留在内容中。问题元数据提供 `question_date` 时传给 recall，其他情况使用服务原生行为；回答模板保持 MemoryData 原有时间提示规则。

## 状态、恢复与清理

每个 MemoryData 评测单元创建独立 `md-hs-<UUID>` bank。UUID 和文档来源映射保存在原 `results/agents/.../exp_<index>/hindsight_state.json`，回答结果仍位于 Agent 的 `output_dir/<dataset>/..._results.json`；`--artifact_root` 沿用原路径重定位规则。

- 完整历史与设置兼容时恢复原 bank；恢复前检查远端 bank 是否存在，缺失时要求 `--force`。
- 中途写入失败的状态不用于问答，下次会清理该单元记录的 bank 并重建整段历史。
- **整合超时不重写**：若全部 chunk 已确认写入、只是等待后台整合超时，重跑同一命令时会核对本地文档 ID/来源映射与当前完整历史一致、远端文档数一致且无失败操作，然后只继续等待整合，不删除 bank、不重新 retain。可同时调大 `hindsight_settle_timeout_seconds`。后台操作失败或文档数不符时仍拒绝，需要 `--force`。MemBench 的写入耗时在等待整合前已保存，续等后照常报告。
- **身份指纹只覆盖记忆内容**：服务地址、bank 设置、数据集配置、完整历史、状态目录和是否等待整合。`retrieve_num`、`hindsight_recall_*` 只在检索时生效，`hindsight_settle_timeout_seconds` 只是等待时长，修改它们会直接复用原 bank，便于做检索预算消融。修改等待开关、bank 设置或历史仍会被拒绝。
- `--force` 在删除本地状态前清理该单元记录的 UUID bank，不清空整个 Hindsight 服务，也不触碰其他用户 bank。
- 本地清单包含配置/历史指纹，不保存 API Key。配置或历史不兼容时拒绝复用。Endpoint 变化时不猜测旧 bank 所属服务，需使用旧配置清理，或改用新的 artifact 根目录。
- **已有回答文件的续跑仍遵循原 MemoryData 行为**：完成的单元会直接跳过。更换数据内容、模型或后端设置时，应使用新 `--artifact_root`，或显式 `--force`；不能用同一个已有回答文件宣称完成了新配置实验。
- 单独删除本地状态目录不能清除远端 bank；清理时保留清单并使用 `--force`。
- **整体移动仓库不影响复用**：指纹里位于仓库内的绝对路径（状态目录、数据集配置中的文件路径、快照来源目录）按仓库相对路径计算，因此可以把整个 MemoryData 文件夹移到别处。仓库外的路径仍按绝对路径计算；把状态目录复制到仓库内的其他位置仍会被拒绝，需要用 `hindsight_snapshot_origin` 显式声明来源。
- 同一规则（`utils/repo_paths.py`）也用于 Mem0 分块断点（`utils/mem0_checkpoint.py`）的身份指纹，覆盖 `artifact_root`、`output_dir` 与数据文件路径；旧断点的兼容方式与下条相同，同样不改写断点文件。
- **旧状态兼容**：2026-11-19 之前保存的状态使用绝对路径指纹。在原位置可直接复用；仓库移动后，需在根 `.env` 或进程环境中用 `MEMORYDATA_LEGACY_REPO_ROOTS` 声明原仓库根目录（多个用 `;` 分隔），例如本机 `E:\LLM judge\MemoryData`。匹配时按原样复用，不改写状态文件，快照来源记录的 SHA-256 因此保持有效。不经 `main.py` 或未加载根 `.env` 的脚本需在进程环境中自行设置。
- **旧状态与检索参数**：2026-11-19 前（含当日仓库相对路径版本）保存的状态，其指纹还包含当时的 `retrieve_num`、`hindsight_recall_max_tokens`、`hindsight_recall_budget`、`hindsight_recall_types`、`hindsight_settle_timeout_seconds`。当前值与当时相同可直接复用；不同时，用 `hindsight_legacy_identity_params` 声明当时的值（字典或字典列表，键名同上，未写的键取当前值），例如 `{retrieve_num: 20, hindsight_settle_timeout_seconds: 36000}`。这只用于核对旧指纹，检索仍使用当前值；声明值与旧指纹对不上时仍拒绝复用。新保存的状态不需要此项。

主结果沿用六个字段和现有指标含义。`input_len` 等计数保持共享 Agent 约定；不会把这些计数宣称为 Hindsight 服务端内部抽取、嵌入、整合、重排的完整 Token/费用统计。

## 验证 / Validation

无模型验证：

```powershell
Start-Process powershell.exe -WindowStyle Normal -ArgumentList '-NoProfile -NoExit -ExecutionPolicy Bypass -File "E:\MemoryData\scripts\verify_hindsight.ps1"'
```

验证脚本使用当前 `memorydata` 环境，日志为 `results/logs/hindsight/tests.log`，状态为 `verify.status`。测试覆盖原生请求合约、同步写入、后台整合失败/超时、文档来源、observation 追溯、bank 隔离、保存恢复、部分摄取重建、配置/历史不兼容和安全清理。

2026-10-20 本机验证：15 项测试全部通过（15.608 秒，无跳过），包括本地上游 SDK 合约对照、旧 Long Context 基线主流程以及 LongMemEval debug 路径。原有 22 个 Agent YAML 均与 Git HEAD 一致，另新增 1 个 Hindsight 配置；语法、文档相对链接和 `git diff --check` 均通过。

集成测试通过真实本地 HTTP 测试服务运行现有 `main.py`、LoCoMo 加载器、AgentWrapper、OpenAI SDK、原评分和结果保存，并验证部分问答恢复、完成题目不重复调用及 `--force`。服务响应和 Tokenizer 使用测试替身，不能作为模型准确率或真实 Hindsight 质量结果。

本地上游合约测试通过 `HINDSIGHT_SOURCE_PATH` 找到 Hindsight 源码（当前默认 `E:/AML/hindsight`），直接调用其生成的请求模型 `from_dict` / `to_dict` 检查线协议；不需要加载服务器。在没有该源码的机器上，这一项会明确跳过，其余离线测试仍可运行。

最初使用本地 Qwen 预设时，`8888` / `9908` 均拒绝连接。现已切换为根 `.env` 中的远程模型 API：对话和嵌入调用均返回 HTTP 200，模型分别为 `gpt-4o-mini`、`text-embedding-3-small`，向量维度 1536。脱敏检查记录在 `results/logs/hindsight/api_configuration.json`；合成数据真实链路结果由 `scripts/smoke_hindsight_api.py` 另行记录，不能作为正式基准准确率。

本次 API 改造验收已完成：使用独立数据库临时启动原生 Hindsight，通过远程 API 对一份小型合成历史执行完整 `main.py` 写入、后台整合、检索、问答、原指标计算及再次恢复。返回预期答案 `Jasmine tea.`，恢复前后回答结果文件字节一致。记录为 `results/hindsight_api_smoke/api-env-check/summary.json`，模式标记为 `real_service_remote_api_synthetic_input`。验证后已停止临时服务并保留数据库；正式运行需启动上述默认服务或指定已有服务地址。

API 配置改造新增了环境映射、缺失凭据报错和回答模型覆盖检查，离线测试共 18 项；日志为 `results/logs/hindsight/tests_api_profile.log`，状态为 `verify_api_profile.status`。原始 15 项验证记录保留作为前一阶段结果。

后续统一模型改造更新了原有 22 个预设及 Hindsight 预设，统一使用根 `.env` 的模型变量，完整离线测试增至 28 项。上述“原有 YAML 与 HEAD 一致”是 Hindsight 首次接入阶段的历史验证记录；当前配置和真实 API 验证结果以 [统一 API 指南](api_models.md) 为准。

## English summary

Select `config/hybrid_hindsight.yaml` through the existing `main.py --agent_config ... --dataset_config ...` interface. Native synchronous retain, recall and consolidation settling use one owned bank per context. MemoryData produces and scores answers; native reflect is outside this protocol. The root `.env`, copied from the source workspace, supplies remote OpenAI-compatible API credentials. Defaults use gpt-4o-mini, text-embedding-3-small (1536 dimensions), and RRF; no local model serving is needed. Run the native memory service with `scripts/run_hindsight_server.py` in the existing hindsight environment; run benchmarks in memorydata. Port 8888 remains the memory-engine API. API model requests can use the existing system proxy. Offline tests and the tiny real-service synthetic smoke are separate from formal benchmark quality evaluation.
