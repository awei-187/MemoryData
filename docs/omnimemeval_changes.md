# 本次变更汇总：OmniMemEval 接入

协议：`memorydata_omnimemeval_v1`。上游来源：OmniMemEval `a5c8b4b`。使用步骤见 [中文指南](omnimemeval_zh.md) / [English guide](omnimemeval.md)，验证证据见 [验证记录](omnimemeval_validation.md)。

## 新增功能

- 接入 LoCoMo 原始数据（类别 1–4）、LongMemEval、PersonaMem v2、HaluMem、BEAM。新增 17 个配置：12 个完整变体与 5 个小样本配置；默认准备 LoCoMo、LongMemEval S、PersonaMem benchmark/32K、HaluMem Medium、BEAM 100k。
- 在 `benchmark/omnimemeval/` 增加独立加载器、模板和逐单元运行模块。保留原始角色、历史时间、session、题目 ID 与评分元数据；PersonaMem 选项采用与上游一致的确定性排列，BEAM 按 JSONL 逐行读取。
- 继续使用 `main.py --agent_config ... --dataset_config ...`，公共加载接口兼容 `ConversationCreator`，复用现有 Agent 写入/问答接口和结果字段。
- 使用数据、配置和相关实现指纹隔离运行与样本记忆，保留稳定 ID，支持恢复、失败重试与显式重建。记录历史摄取量和最终回答上下文截断情况。
- 新增 `evaluation/omnimemeval/evaluate.py` 独立评分入口：QA Judge、PersonaMem 选项提取、BEAM Nugget 与事件排序。评分结果单独保存，保留原响应、指纹、有效覆盖率和失败记录，支持恢复与重算。
- 移植必要上游原始提示词和纯评分逻辑，保留 Apache-2.0 许可与来源说明。严格解析二元标签，避免把 `INCORRECT` 误判为 `CORRECT`。
- 新增最小依赖清单、Windows 环境/数据准备/验证脚本、中英文说明及自动化测试。

## 对原项目的兼容调整

没有删除或移动原有文件；原 22 个 Agent YAML 配置未改动，原四类基准目录与原 LongMemEval Judge 保留。已有代码的接入点为：

| 文件 | 调整 |
| --- | --- |
| `main.py` | 优先分派 `dataset: OmniMemEval`，进入逐单元执行；原运行路径保留 |
| `benchmark/memoryagentbench/loader.py` | 为新增基准提供统一 `{"data": ...}` 公共加载接口 |
| `benchmark/memoryagentbench/prompts/benchmark_templates.py` | 新增模板分派 |
| `utils/conversation_creator.py` | 仅对新增基准允许有效短历史、单题单答案与 0 表示不限样本 |
| `utils/agent.py` | 新基准使用数据时间、完整 RAG 摄取与回答预算记录，避免注入运行日期 |
| `methods/embedding_rag/embedding_retriever.py` | 可选返回上下文预算，旧调用默认返回格式保留 |
| `utils/eval_other_utils.py` | 新基准评分分派与辅助指标前缀，避免误入旧同名数据集分支 |

此外更新中英文 README 的概览、用法及目录。Mem0 的 SQLite/Qdrant/客户端资源在新运行路径中逐单元释放，修复 Windows 文件锁问题。回答结果保留 `time_cost` 与原百分数均值约定；专用评分文件使用 0–1 分数。

## 验证与交付边界

- Python 3.11 环境及所需依赖已安装，依赖检查通过。
- 默认五份数据全量解析通过：750 个评测单元、10,907 题。
- 23 项自动化测试通过，包含 15 个实际方法适配器的模拟 HTTP 组合、旧路径回归、评分/恢复和接口结构检查。模拟分数不代表模型效果。
- 真实模型验收未通过执行条件：默认端口 9908、9909、9009 无服务，15 个组合标记为 `blocked`；没有下载模型权重、部署服务或生成真实模型准确率。
- 其他 19 个方法仅完成模板兼容检查。大规模变体仅提供配置，未开展全量实验；不包含 AgentBench、新记忆框架、HaluMem 抽取/更新操作级评分。

Git 交付包含源码、YAML、脚本、测试、文档和上游许可。下载数据、模型权重、Conda 环境、API Key、运行结果、日志及缓存不提交。
