# OmniMemEval 五基准使用指南

协议标记为 `memorydata_omnimemeval_v1`，数据定义与评分提示词来源于 OmniMemEval `a5c8b4b`。本接入在 MemoryData 的记忆写入和问答流程下使用五个基准，不能与 OmniMemEval 公开排名直接等同。参见 [变更汇总](omnimemeval_changes.md)、[验证记录](omnimemeval_validation.md) 和 [英文详细说明](omnimemeval.md)。

## 1. 环境、数据和模型服务

本机已创建 `D:\conda\envs\memorydata`（Python 3.11），并安装 `requirements-omnimemeval.txt`。该清单覆盖接入与 Long Context、Embedding RAG、Mem0 三种验收方法；其他方法仍需各自依赖。

以下是当前 Windows 工作区可直接执行的命令。换机器时调整项目路径、Conda 可执行文件及环境路径；脚本中的环境默认值也需要同步调整。长任务在独立可见的 Windows PowerShell 中执行，日志保存在 `results/logs/omnimemeval/`。

尚未创建环境时运行：

```powershell
Start-Process powershell.exe -WindowStyle Normal -ArgumentList '-NoProfile -NoExit -ExecutionPolicy Bypass -File "E:\LLM judge\MemoryData\scripts\setup_omnimemeval.ps1"'
```

数据准备依赖相邻的 [OmniMemEval 仓库](https://github.com/MemTensor/OmniMemEval)，上游来源版本为 `a5c8b4b`。默认布局：

```text
E:\LLM judge\
├── MemoryData\
└── OmniMemEval\
    └── data\
```

默认数据已在本机准备完成。其他机器需要先取得相邻仓库，再执行其准备脚本；本项目的包装脚本会依次调用五个上游脚本：

```powershell
Start-Process powershell.exe -WindowStyle Normal -ArgumentList '-NoProfile -NoExit -ExecutionPolicy Bypass -File "E:\LLM judge\MemoryData\scripts\prepare_omnimemeval.ps1"'
```

数据不随 Git 提交。若放在其他目录，修改数据集 YAML 的 `data_root`；相对路径按 MemoryData 根目录解析。大规模变体只提供配置和加载能力，需另外准备对应上游文件。

本接入没有下载模型权重或部署推理服务。现有配置指向以下 OpenAI 兼容端点：

| 用途 | 当前默认模型 | 当前默认地址 |
| --- | --- | --- |
| Long Context / Embedding RAG 回答 | Qwen3-8B | `http://127.0.0.1:9908/v1` |
| Mem0 回答与记忆处理 | Qwen3-8B | `http://127.0.0.1:9909/v1` |
| 嵌入 | Qwen3-Embedding-4B | `http://127.0.0.1:9009/v1` |
| Judge | Qwen3-8B | `http://127.0.0.1:9908/v1` |

修改所选 Agent YAML 的 `model`、`base_url`、`embedding_base_url` 等字段可连接其他已部署服务；Judge 配置独立位于 `evaluation/omnimemeval/config/qwen3_8b.yaml`。远程部署时将地址改为服务器的可达地址，保持模型名、嵌入维度与实际服务一致。`smoke_omnimemeval.py` 的就绪检查按上述默认模型和端口编写，远程验收也需调整该检查。

API Key 默认读取 `OPENAI_API_KEY`。对不校验密钥的本地服务可设为 `local`；设置密钥不能解决端口无服务导致的连接拒绝。

## 2. 选择数据集与方法

所有新增配置位于 `benchmark/omnimemeval/config/`，使用 `dataset: OmniMemEval` 和唯一的 `sub_dataset`。新 LoCoMo 原始数据与旧 600 题子集分开；新 LongMemEval 与 MemoryAgentBench 中的同名任务也分开。

| 基准 | sub_dataset | 小样本配置 | 默认完整配置 | 其他完整配置 |
| --- | --- | --- | --- | --- |
| LoCoMo，默认类别 1–4 | `omni_locomo` | `locomo_smoke.yaml` | `locomo_default.yaml` | — |
| LongMemEval | `omni_longmemeval` | `longmemeval_smoke.yaml` | `longmemeval_s.yaml` | `longmemeval_oracle.yaml`、`longmemeval_m.yaml` |
| PersonaMem v2 | `omni_personamem_v2` | `personamem_v2_smoke.yaml` | `personamem_v2_32k.yaml` | `personamem_v2_128k.yaml` |
| HaluMem | `omni_halumem` | `halumem_smoke.yaml` | `halumem_medium.yaml` | `halumem_long.yaml` |
| BEAM | `omni_beam` | `beam_smoke.yaml` | `beam_100k.yaml` | `beam_500k.yaml`、`beam_1m.yaml`、`beam_10m.yaml` |

常用方法配置保持原样：

| 方法 | --agent_config |
| --- | --- |
| Long Context | `config/reference_long_context_agent.yaml` |
| Embedding RAG | `config/reference_embedding_rag.yaml` |
| Mem0 | `config/sequential_mem0.yaml` |

小样本配置选择原始顺序中首个有效单元，摄取该单元的完整历史，最多回答 3 题；LongMemEval 单元只有 1 题。BEAM 小样本包含普通 rubric、事件排序和第三种维度的问题，按来源顺序保留。完整配置将 `max_test_samples`、`max_queries_per_unit` 设为 `0`，表示不限制数量，不能直接作为本机的运行规模建议。

可复制配置后修改：

| 字段 | 作用 |
| --- | --- |
| `data_root` | 数据根目录，默认 `../OmniMemEval/data` |
| `sample_ids` | 稳定样本 ID 列表，如 LoCoMo 的 `conv-26`、PersonaMem 的 `persona:7`；筛选不重排、不重新编号 |
| `max_test_samples` | 最多处理多少个评测单元，0 为全部 |
| `max_queries_per_unit` | 每单元最多多少题，0 为全部 |
| `generation_max_length` | 回答 Token 上限；默认 QA 256、PersonaMem 32、BEAM 2048 |
| `chunk_size` | 历史分块的字符预算，默认 4096 |

HaluMem 固定采用完整用户历史写入后问答。正确答案、evidence、rubric、`memory_points` 和答案解释只用于评测；PersonaMem 候选选项正常进入问题，正确选项标签不进入回答提示。

## 3. 运行与恢复

进入项目：

```powershell
cd 'E:\LLM judge\MemoryData'
```

运行 LoCoMo 小样本，按服务要求设置密钥：

```powershell
$env:OPENAI_API_KEY='local'; & 'D:\conda\envs\memorydata\python.exe' main.py --agent_config config/reference_long_context_agent.yaml --dataset_config benchmark/omnimemeval/config/locomo_smoke.yaml --artifact_root results/omni_runs
```

使用方式与旧基准相同：仍由 `main.py` 接收 Agent YAML 和 Dataset YAML，复用 `AgentWrapper.send_message()`。公共加载入口仍提供 `{"data": ...}` 样本接口，兼容 `ConversationCreator`；新主运行分支逐单元加载、执行并释放，以限制大数据集的内存占用。

本次结果保留原六个字段：`agent_config`、`dataset_config`、`data`、`metrics`、`time_cost`、`averaged_metrics`，并增加协议、指纹及摄取/截断信息。主结果质量均值沿用原项目百分数；后述专用评分文件使用 0–1 分数。

主结果默认位置：

```text
<artifact_root>/outputs/<Agent配置中的输出命名空间>/OmniMemEval/<sub_dataset>/<运行指纹>/benchmark_results.json
<artifact_root>/agents/omnimemeval/<sub_dataset>/<运行指纹>/<样本指纹>/
```

`--artifact_root` 重定位标准 `results/` 根目录；自定义 `output_dir` 遵循原项目路径规则。以运行后打印的 `Results:` 路径为准。

同一命令再次运行会恢复已完成的回答和记忆状态，不重复处理成功题目。`--retry_failed_queries` 重试失败回答，`--force` 重建该运行。数据、配置及相关实现变化会产生不同运行指纹；已有状态身份不匹配时拒绝复用，部分完成的历史摄取会重建。各单元状态隔离。

所有历史先进入记忆系统；最终回答上下文可能因模型预算裁剪。结果记录 `ingested_context_chars`、`ingested_chunk_count`、`ingested_session_count` 以及 `answer_context_budget`、`answer_context_truncated`，摄取量与回答输入预算不能混用。

## 4. 专用评分

将以下占位路径替换为上一步实际结果路径：

```powershell
& 'D:\conda\envs\memorydata\python.exe' -m evaluation.omnimemeval.evaluate '替换为实际的benchmark_results.json路径' --judge_config evaluation/omnimemeval/config/qwen3_8b.yaml
```

| 基准 | v1 主指标 |
| --- | --- |
| LoCoMo | 上游提示词下的 Judge Accuracy，按类别汇总 |
| LongMemEval | 上游提示词下的 Judge Accuracy，按题型汇总 |
| PersonaMem v2 | 上游选项提取规则的 Accuracy，按偏好类型汇总，不调用 Judge |
| HaluMem | QA Judge Accuracy，按问题类型与难度汇总 |
| BEAM | 逐 rubric 项 Nugget Score；事件排序使用 Kendall tau-b × 覆盖率 |

评分另存为 `<结果文件名>.omnimemeval_scores.json`，不覆盖回答。相同命令恢复已有有效评分；若需要更换裁判配置或显式重算，添加 `--overwrite`。默认温度 0、每题 1 次、并发 2、应用层最多重试 2 次，保留裁判原始响应、解析结果及内容/提示词指纹。

主分数按有效评分计算。无法解析或冲突的标签记为失败，缺失评分不填零；汇总同时报告总题数、成功回答数、有效评分数、失败数、跳过数和覆盖率。原文本指标以 `memorydata_aux_` 命名，仅作为辅助，不能称为 OmniMemEval 同口径 F1。无可靠证据映射时不输出 Recall，HaluMem v1 不评估抽取/更新操作质量。原 LongMemEval Judge 入口保留。

## 5. 无模型验证与真实验收

不需要部署模型的验证：

```powershell
Start-Process powershell.exe -WindowStyle Normal -ArgumentList '-NoProfile -NoExit -ExecutionPolicy Bypass -File "E:\LLM judge\MemoryData\scripts\verify_omnimemeval.ps1"'
```

该检查运行 23 项测试，包括五个基准 × 三种方法的 15 个模拟 HTTP 组合。实际执行 `main.py`、Agent 适配器、SDK、FAISS/Qdrant、SQLite、结果保存、评分和恢复；模型 HTTP 响应与 Tokenizer 使用替身。它验证链路，不代表真实模型效果。日志为 `results/logs/omnimemeval/tests.log`，组合报告为 `simulated_validation.json`，状态为 `verify.status`。

部署好默认模型服务后，真实小样本验收：

```powershell
Start-Process powershell.exe -WindowStyle Normal -ArgumentList '-NoProfile -NoExit -ExecutionPolicy Bypass -File "E:\LLM judge\MemoryData\scripts\smoke_omnimemeval.ps1"'
```

真实验收调用五个基准 × 三种方法，完成后再次运行问答和评分以检查恢复，报告为 `results/omnimemeval_smoke/summary.json`。截至本次交付，三端口没有模型服务，15 个真实组合均为 `blocked`；没有真实模型准确率。默认数据共 750 个单元、10,907 题已全部解析通过。

原有 22 个 Agent 配置未改动。其余 19 个方法只完成模板兼容检查，未进行在线认证。未下载大规模变体，不开展全量排名，也未接入 AgentBench 或新增记忆框架。完整证据范围见 [验证记录](omnimemeval_validation.md)，运行日志和下载数据不上传 Git。
