# 统一远程模型配置

`config/` 下的 23 个预设使用相同的远程模型配置：`gpt-4o-mini` 负责回答及适配器支持的记忆抽取、摘要等对话调用，`text-embedding-3-small` 提供 1536 维嵌入。无需启动原先的 9908–9911 / 9009–9010 模型服务。长上下文和 BM25 不需要调用嵌入模型。

本次同时替换了预设中的 Qwen2.5、OpenRouter 模型名、LightMem 的 MiniLM 和 MemoChat 的 hashing 嵌入，避免它们继续使用另一套默认模型。保留原方法名称和入口；Embedding RAG 的内部名称改为已支持的 `Embedding_rag_text_embedding_3_small`。

## 根目录 .env

已有 `.env` 时直接编辑，不要用示例文件覆盖密钥。没有时复制 [.env.example](../.env.example)，填写可用的 OpenAI 兼容 API 地址和密钥。模型名必须是服务实际支持的名称，正确拼写为 `text-embedding-3-small`。

| 环境变量 | 用途 |
| --- | --- |
| `OPENAI_API_KEY` | 对话和嵌入 API 密钥；仅在创建客户端时读取 |
| `OPENAI_BASE_URL` | 对话和嵌入的 API 根地址，通常以 `/v1` 结尾 |
| `OPENAI_MODEL` | 默认 `gpt-4o-mini`，同步到方法内部对话模型别名 |
| `OPENAI_EMBEDDING_MODEL` | 默认 `text-embedding-3-small`，同步到内部 embedder 别名 |
| `OPENAI_EMBEDDING_DIMENSIONS` | 默认 1536，必须与 API 实际输出维度一致；各适配器保留原生维度请求方式，本次不提供跨适配器统一缩维协议 |
| `OPENAI_API_TRUST_ENV` | 默认 `true`，允许模型请求使用现有 `HTTP_PROXY`、`HTTPS_PROXY`、`NO_PROXY` |

`main.py` 按自身路径读取项目根 `.env`，已有进程环境变量优先。预设中的 `model_env`、`embedding_model_env`、`embedding_dim_env` 指定上述变量名；实际解析后的模型名保存在结果配置中，密钥不会被解析写回配置。显式填写 YAML `base_url` 时仍优先使用该地址。没有这些模型 `*_env` 字段的自定义旧配置保留原有行为。

LightMem、A-MEM、SimpleMem、Mem0、MemOS、Letta、Zep Local、Hindsight 等各自的模型字段和向量维度同步使用这套设置。Hindsight 服务由本仓库启动器启动时，其原生抽取和嵌入也读取这些变量；已有服务需自行重启并对齐配置。显式设置原生 `HINDSIGHT_API_*` 模型变量仍可覆盖服务设置，正式比较时应检查它们是否存在。

启用 `*_env` 字段后，各方法旧的专用环境变量不再生效：`LIGHTMEM_MODEL`、`LIGHTMEM_EMBEDDING_MODEL`、`LIGHTMEM_EMBEDDING_DIMENSION` 由统一模型设置取代，`LIGHTMEM_BASE_URL`、`SIMPLEMEM_BASE_URL` 由共享的 `OPENAI_BASE_URL` 取代（YAML 中显式填写的 `lightmem_base_url`、`simplemem_base_url` 仍优先）。这些变量若仍被设置，运行时会记录一次 `... is set but ignored` 警告。

`.env` 已被 Git 忽略；本机原密钥和 API 地址已保留，修改前备份位于忽略的 `results/env-backups/`。示例文件只有占位值。

## 原有运行方式

进入项目：

```powershell
cd 'E:\MemoryData'
```

例如使用 Embedding RAG 和原 LoCoMo 配置：

```powershell
& 'D:\conda\envs\memorydata\python.exe' main.py --agent_config config/reference_embedding_rag.yaml --dataset_config benchmark/locomo/config/Locomo_qa_4cat_600_dist.yaml --artifact_root results/api_eval_01 --max_test_queries_ablation 1
```

更换 `--agent_config` 即可选择其他方法。新预设的默认回答输出位于 `results/outputs/api/<preset>/`，记忆方法使用 `api-models-v1` 变体标签，避免复用此前的记忆状态。更换 `.env` 中的模型、维度、API 后端或数据内容时，必须换一个新的 `--artifact_root`；原有回答续跑逻辑不会自动判定所有实验变更。

## 范围和公平性

- 本次统一模型、API 来源和 Tokenizer（`o200k_base`），保留原数据加载、`send_message`、结果格式和评分流程。
- `retrieve_num`、温度、分块大小、上下文预算和方法内部多路检索参数保持原值，并非全部一致。例如 A-MEM / MemTree 的 k 为 2、Mem0 为 100、Hindsight 为 20，多数其他方法为 10。开展严格对比前应另行制定并记录这些实验参数。
- Hindsight、EverOS、Zep、Neo4j 等记忆服务或数据库与模型推理服务是不同组件；使用远程模型并不会替代这些组件。EverOS 和云端 Zep 的服务内部模型不受本地 Agent YAML 自动控制，需要在服务端另行核对；本地配置仅控制框架实际创建的客户端。
- **MemoRAG 的核心记忆模型和检索器依赖本地模型**。其最终回答模型配置虽已切换，但 `api_only: true` 会在初始化时明确拒绝运行，避免自动下载本地权重。要运行原算法仍需另行准备本地模型并显式关闭此限制。本次没有把它替换成另一种算法。
- 独立 Judge 的模型配置与回答模型配置分开；既有 LongMemEval Judge 的默认值未随本次回答模型迁移改变。

## 简单验证

离线配置契约与 Hindsight 回归测试：

```powershell
& 'D:\conda\envs\memorydata\python.exe' -m unittest discover -s tests -p 'test_*.py' -v
```

使用很小的合成历史验证真实 API、主流程保存和再次恢复（不依赖正式数据集，也不构成准确率排名）：

```powershell
& 'D:\conda\envs\memorydata\python.exe' scripts/smoke_hindsight_api.py --agent_config config/reference_embedding_rag.yaml --artifact_root results/api_model_smoke/manual_embedding_rag
```

将配置改为 `config/reference_long_context_agent.yaml` 或 `config/sequential_mem0.yaml` 可检查另外两条路径，每次使用新的测试目录。Hindsight 的服务启动与验证见 [接入指南](hindsight.md)。其余方法本次仅检查配置兼容性，不宣称全部完成真实服务验收。

本机验证已完成：28 项离线测试通过；Long Context、Embedding RAG、Mem0 均通过根 `.env` 的真实远程 API 完成合成历史问答、原评分、结果保存及再次恢复，答案均为 `Jasmine tea.`，恢复前后结果文件字节一致。三条链路的记录分别位于 `results/api_model_smoke/long_context_v2/summary.json`、`embedding_rag_v2/summary.json`、`mem0_final/summary.json`；完整离线日志位于 `results/logs/api-models/offline-tests-final.log`。首次检查暴露的 Windows YAML 编码和 Mem0 强制禁用代理问题均已修复，并加入回归测试。
