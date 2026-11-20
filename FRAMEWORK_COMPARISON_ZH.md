# MemoryData 与 OmniMemEval：当前执行流程、模型配置和评分差异

更新日期：2026-10-21（Asia/Shanghai）。依据当前本地源码、配置和已有验证摘要。MemoryData HEAD 为 `bdbe698`，另有未提交的 Hindsight／统一 API 修改；OmniMemEval HEAD 为 `a5c8b4b`。本次只更新文档，没有重新调用模型或运行基准。

本文替换旧版中已失效的 `memorydata_omnimemeval_v1` 五基准移植说明和本地 Qwen 运行示例。当前 MemoryData 没有相应加载／运行／评分源码入口，不能继续按旧命令执行。

## 1. 当前覆盖范围

| 方面 | MemoryData | OmniMemEval |
| --- | --- | --- |
| 方法／后端 | 23 个方法预设，含 Long Context、Embedding RAG、BM25 和记忆系统；当前 MemoRAG 在纯 API 配置下拒绝初始化 | User Memory 注册 15 个后端入口，文档按 Zep/Graphiti 合并称 14 种方案 |
| 用户记忆基准 | MemoryAgentBench、LoCoMo、LongBench v2 子集、MemBench，共四个基准族；LongMemEval 是 MemoryAgentBench 的来源子集 | LoCoMo、LongMemEval、BEAM、PersonaMem v2、HaluMem |
| 真实 Agent 任务 | 当前主入口没有移入 OmniMemEval 的 AgentBench 任务线 | OpenClaw + EvoAgentBench 五个任务域，另有五份记忆插件生命周期配置 |
| 统一边界 | `AgentWrapper.send_message(..., memorizing=True/False)` 统一历史写入和回答 | Client 的 add/search 等操作统一记忆后端，检索、回答、裁判分阶段执行 |
| 当前模型设置 | 预设统一迁移到远程 OpenAI 兼容 API，默认 `gpt-4o-mini` 和 `text-embedding-3-small`（1536 维） | 维持本项目 ANSWER/EVAL 和 domain 配置，不自动读取 MemoryData 的 `.env` |

两种框架都包含多个可选实现，预设／注册数量不能直接当作完成本机验收的方法数量。详细清单见 [MemoryData 覆盖说明](EVALUATION_COVERAGE_ZH.md)、[OmniMemEval 覆盖说明](<E:/LLM judge/OmniMemEval/EVALUATION_COVERAGE_ZH.md>)。

## 2. MemoryData 当前如何执行

```text
main.py 加载根 .env 和两个 YAML
→ 解析模型环境变量及适配器模型别名
→ 加载数据、组织历史和问答
→ 恢复或初始化 Agent
→ 摄取历史、保存记忆
→ send_message(question, memorizing=False)
→ 方法检索／组织上下文／生成回答
→ 规则评分、保存结果、按基准生成附加报告
→ LongMemEval LLM Judge 需要另行执行
```

`main.py` 按其自身路径加载 `.env`，已有进程环境优先。新预设的模型变量如下：

| 变量 | 用途 |
| --- | --- |
| `OPENAI_MODEL` | 默认 `gpt-4o-mini`，控制回答和适配器支持的抽取／摘要等调用 |
| `OPENAI_EMBEDDING_MODEL` | 默认 `text-embedding-3-small` |
| `OPENAI_EMBEDDING_DIMENSIONS` | 默认 1536，必须匹配实际向量 |
| `OPENAI_BASE_URL`、`OPENAI_API_KEY` | 远程模型端点和有效凭据；本文不记录真实地址或密钥 |
| `OPENAI_API_TRUST_ENV` | 允许模型请求使用已有代理环境 |

Tokenizer 预设为 `o200k_base`，默认回答输出为 `results/outputs/api/<preset>/`，变体标签为 `api-models-v1`（Hindsight 为 `api-models-v1-hindsight-retain-recall`）。模型、向量维度、API 后端或数据改变时使用新的 `--artifact_root`；续跑不能替代实验身份核查。

这次统一没有把各方法的检索数、温度、分块、上下文预算统一为相同值。MemoRAG 仍需要本地核心记忆模型和检索器，当前 `api_only: true` 下明确拒绝运行；远程最终回答模型不能代替它的核心算法。

依据：[main.py](main.py)、[provider_utils.py](utils/provider_utils.py)、[AgentWrapper](utils/agent.py)、[统一 API 配置指南](docs/api_models.md)。

## 3. Hindsight 在两边如何接入

| 项目 | MemoryData 当前接入 | OmniMemEval 原有 Client |
| --- | --- | --- |
| 选择方式 | `config/hybrid_hindsight.yaml` | `--lib hindsight` 和后端环境文件 |
| 写入 | 原生同步 retain；按评测单元隔离 bank，默认等待后台整合 | retain／批量异步写入等由 Client 和环境变量控制 |
| 检索与回答 | recall 检索后由 MemoryData 回答模型生成；当前不调用 reflect | recall + 外部 ANSWER 模型，或 reflect 直接生成答案；以各 benchmark 的已实现分支为准 |
| 当前示例检索预算 | `mid`、4096 recall Token、top-k 20 | `.env.hindsight` 为 `high`、32768 recall Token，并配置原始 chunk Token 预算 |
| 引擎模型 | 本仓库启动器将共享模型配置映射到原生服务；显式原生环境变量可能覆盖，已有服务需另行对齐 | 由所连接的 Hindsight 服务配置，不等于 ANSWER 或 EVAL 模型 |
| 最终评分 | 沿用所选基准原评分；LongMemEval 独立 Judge 另行运行 | 生成或直接返回的答案保存后，由该基准的 EVAL 流程评分 |

MemoryData 的 [服务启动器](scripts/run_hindsight_server.py) 使用已有 `hindsight` 环境运行原生引擎，默认服务根地址 `http://127.0.0.1:8888`；评测使用 `memorydata` 环境。引擎仍需要数据库，但抽取／嵌入由远程 API 提供，默认 RRF，不需要启动原本的本地 Qwen 或嵌入模型服务。8888 是记忆引擎地址，不是 OpenAI 模型地址。

Hindsight 的来源追踪、bank 隔离、保存恢复及 `--force` 清理边界见 [接入指南](docs/hindsight.md)。即使两边都叫 Hindsight，也不能忽略 retain 异步处理、整合等待、recall 预算和 reflect 模式差异。

## 4. 回答模型与 Judge 必须分开说明

| 角色 | MemoryData | OmniMemEval |
| --- | --- | --- |
| 回答模型 | 默认 `OPENAI_MODEL=gpt-4o-mini`，配置可覆盖 | User Memory 用 `ANSWER_MODEL`；Hindsight 示例为 `gpt-4.1-mini`；部分直接回答模式由后端生成 |
| 嵌入模型 | 默认 `text-embedding-3-small`、1536 维 | 取决于后端部署或 API 产品，不一定由评测端控制 |
| 用户 QA Judge | LongMemEval 独立 CLI 默认仍为 `gpt-4o`；通过 `--judge-model` 显式选择 | 使用 `EVAL_MODEL`；Hindsight 示例为 `gpt-4o-mini` |
| Judge 端点／密钥 | `OPENAI_BASE_URL`、`OPENAI_API_KEY`；独立脚本不会自动执行 main.py 的 `.env` 加载 | `EVAL_BASE_URL`、`EVAL_API_KEY`，与 ANSWER 分离 |
| AgentBench Judge | 当前没有对应任务线 | IR 用 JUDGE 配置，数学／GDPVal 用各 domain 的 evaluation 配置；现有 `.env.agent` 仍含 Qwen 示例 |

所以“切换为 OpenAI API”不表示两个框架、全部服务内部模型和所有裁判已相同。MemoryData 的回答迁移也没有自动把 Judge 默认模型改成 `gpt-4o-mini`。

## 5. LLM Judge 输入与 prompt 的核心差异

| 项目 | MemoryData LongMemEval Judge | OmniMemEval User Memory Judge |
| --- | --- | --- |
| 输入 | canonical 问题、标准答案、原结果 output | 保存的 question、golden_answer、answer；BEAM 则使用 rubric |
| 历史和检索证据 | 不传给 QA Judge | QA Judge 同样不传完整历史、search_context 或 evidence |
| prompt 选择 | 通用事实、时间、更新、偏好、拒答五套；题型／`*_abs` ID 决定 | LoCoMo 与 LongMemEval 共用宽松通用模板；HaluMem 专用模板；BEAM 分原子项／事件排序模板 |
| 关键判分差异 | 通用事实要求信息完整，时间题显式容忍 off-by-one，更新／偏好／拒答分别定义 | 通用模板要求宽松判断“同主题”；HaluMem 要求关键事实、最新状态且不编造 |
| 返回 | system 要 JSON 标签，user 仍写 yes/no；解析接受两组并有全文子串回退 | QA 通过正则提取 label JSON；BEAM 输出 score 或事件位置，由代码再计算分数 |
| 重复裁判 | 对同一答案重复评分后取均值 | QA 同样重复评分；BEAM 按原子项聚合；PersonaMem 是多次答案生成而非额外 Judge |

这些裁判主要评价参考答案正确性，不能直接解释为基于原始记忆逐事实核查的忠实度。Hindsight 的 reflect 是生成待评答案，也不是 Judge。

完整输入、英文 prompt、解析与失败回退见 [MemoryData Judge 专项说明](LLM_JUDGE_PROMPTS_ZH.md)、[OmniMemEval Judge 专项说明](<E:/LLM judge/OmniMemEval/LLM_JUDGE_PROMPTS_ZH.md>)。本次模型迁移未改变这两套 prompt 的原文。

## 6. 运行示例

以下为本机 Windows PowerShell 单行命令，前提是已经在 MemoryData 根 `.env` 配好有效远程 API 凭据。已有 `.env` 时不要用示例覆盖。

```powershell
cd 'E:\MemoryData'
```

在已有 `hindsight` 环境中启动原生记忆引擎，保持运行；如果已连接其他服务，按接入指南配置地址：

```powershell
& 'D:\conda\envs\hindsight\python.exe' -u scripts/run_hindsight_server.py
```

在另一个 PowerShell 会话运行 LoCoMo 小规模问答：

```powershell
& 'D:\conda\envs\memorydata\python.exe' main.py --agent_config config/hybrid_hindsight.yaml --dataset_config benchmark/locomo/config/Locomo_qa_4cat_600_dist.yaml --artifact_root results/hindsight_openai_20261021 --max_test_queries_ablation 1
```

问题数限制不等于只摄取一个历史块；实际数据文件仍需准备。LongMemEval 独立 Judge 的显式环境加载和模型选择命令见 [专项说明第 0 节](LLM_JUDGE_PROMPTS_ZH.md)。不要使用已删除的 `benchmark/omnimemeval` 配置或 `evaluation.omnimemeval` 模块。

OmniMemEval 的入口仍是其 `scripts/run_*_eval.sh` 与各 Python 阶段脚本；Bash 运行器需要相应 shell，不能直接当 PowerShell 脚本。

## 7. 已验证到什么程度

已存在的 [统一 API 指南](docs/api_models.md) 记录 28 项离线测试通过。当前复核了四份真实服务／远程 API 合成历史摘要：

- [Hindsight](results/hindsight_api_smoke/api-env-check/summary.json)
- [Long Context](results/api_model_smoke/long_context_v2/summary.json)
- [Embedding RAG](results/api_model_smoke/embedding_rag_v2/summary.json)
- [Mem0](results/api_model_smoke/mem0_final/summary.json)

四份均为 `status=passed`、`resume_unchanged=true`，回答 `Jasmine tea.`。这是写入／检索／问答、保存和恢复验证，不是正式基准准确率，未证明全部方法已实测，也未证明历史启动的服务现在仍在运行。本次文档更新没有重新执行这些验证或 Judge。
