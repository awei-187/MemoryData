# Mem0 每个 chunk 只保留前两条事实：来源与依据核查

核查日期：2026-10-21（Asia/Shanghai）。本次仅检查资料，没有修改事实截断逻辑或重新运行评测。

## 结论

已确认：固定截断来自 MemoryData 随仓库发布的 Mem0 源码，官方 OpenDataBox/MemoryData 仓库也包含相同语句。它不是本次本机评测新增的参数。

未找到：为何选择前两条、为何阈值为 2、是否用于统一成本预算、是否仅用于调试，以及针对这个阈值的效果或成本消融依据。

因此，现有证据只能支持“公开实现中存在未说明依据的硬编码截断”。不能据此确认作者有意这样设置，也不能直接断定它是调试遗留或实现错误。此前关于调试、节约开销的说法属于推测，未获得本次文献核查的支持。

## 1. 本地项目与 Git 历史

- 位置：`methods/mem0/source/mem0/memory/main.py:227`。
- 语句：`new_retrieved_facts = new_retrieved_facts[:2]`。
- 执行顺序：先调用模型并解析完整 facts 列表，再取前两条，随后进入逐条嵌入、旧记忆检索与记忆动作生成。
- 选择方式：列表切片，代码没有在此处按重要性、置信度或与未来问题的相关性重新排序。
- 配置：`config/sequential_mem0.yaml` 没有该阈值字段；回答阶段的 `retrieve_num: 100` 不控制写入前的这次切片。
- `git blame` 与 `git log --all -S 'new_retrieved_facts[:2]'` 均指向提交 `c63391c128e33eedb91115edf689f12acf4bbc63`，时间 2026-06-24 01:11:58 +0800，提交说明仅为 `upload code`。该文件随整份代码上传，提交没有说明截断动机。
- 对本地项目的 README、中文 README、docs、配置及源码检索没有发现针对前两条事实的解释。全项目精确短语检索除该语句外，仅出现了其他系统的无关去重说明。

本地 Git remote 为 `awei-187/MemoryData`。论文实际指向的官方仓库是 `OpenDataBox/MemoryData`；本次另外核查了官方公开文件，避免只依据本机副本判断。

官方代码同样包含切片及 `truncated facts count` 调试打印，但没有解释原因：

- [官方 Mem0 memory/main.py](https://github.com/OpenDataBox/MemoryData/blob/main/methods/mem0/source/mem0/memory/main.py)
- [官方原始源码](https://raw.githubusercontent.com/OpenDataBox/MemoryData/main/methods/mem0/source/mem0/memory/main.py)

## 2. README 引用的论文

论文为 *Are We Ready For An Agent-Native Memory System?*，arXiv:2606.24775v1，2026-06-23，PDF 共 14 页。核查了 HTML、PDF 的相关正文、图表、结论和末尾；没有独立附录解释这个设置。

- [论文摘要与版本记录](https://arxiv.org/abs/2606.24775)
- [论文 HTML](https://arxiv.org/html/2606.24775v1)
- [论文 PDF](https://arxiv.org/pdf/2606.24775)

| 检查位置 | 与本问题有关的内容 | 能否解释前两条截断 |
|---|---|---|
| §3.2 Memory Extraction | 介绍 Mem0 将交互转为独立事实 | 未规定每个 chunk 的事实条数 |
| §4.5 Memory Operation Cost | 比较操作延迟与效用，讨论维护范围与成本 | 未说明用前两条事实限制 Mem0 成本，也未说明阈值 2 的来源 |
| §5.1 / Table 3 | 列出 Mem0 Default 与 Graph Store | 不是事实数量阈值消融 |
| §5.2 / Table 4 | 比较 MemoChat、MemOS、LightMem 的抽取变体 | 没有 Mem0 前 N 条或不限条数的对照 |
| PDF 第 12 页 Finding 7 | 提出 Late Filtering Principle，主张写入时保留上下文、避免激进过滤 | 不能作为固定保留前两条的理由 |

论文中有关于成本、信息保留和抽取粒度的一般讨论，但没有将它们与该源码语句建立联系。论文也没有单独报告这项截断的影响。公开代码包含该语句，不足以单独证明论文每项结果都实际使用了它。

## 3. 公开 issue / PR

通过 GitHub 连接器检索该官方仓库的全部状态 issue 与 PR，并读取返回条目的正文和普通评论：

- [Issue #2](https://github.com/OpenDataBox/MemoryData/issues/2)：LongMemEval 与 MemoryAgentBench 重构数据的关系；作者回复解释数据来源，没有讨论事实截断。
- [PR #1](https://github.com/OpenDataBox/MemoryData/pull/1)：新增 Nova Memory 预设；相关讨论不涉及此限制。
- [PR #3](https://github.com/OpenDataBox/MemoryData/pull/3)：Mem0 Qwen3.5 / Vast.ai 启动与部署；正文没有解释两条事实截断，普通评论为空。
- [PR #4](https://github.com/OpenDataBox/MemoryData/pull/4)：新增 MemTether；没有解释此限制。

未找到针对此语句的作者说明。检索结论限于本次可访问的公开资料，不包括作者私有实验脚本、未公开交流或历史未发布材料。

## 4. 对现有评测的影响

上次原始 LoCoMo 第一组的运行日志证明：首个 chunk 的模型输出包含 11 条事实，截断后只传递 2 条。18 个 chunk 最终在 Qdrant 中持久化了 36 条记忆。这表明该限制在本次运行中确实生效。

由代码流程可以推断，截断会减少后续逐事实处理量并丢弃候选事实；但“会减少处理量”不等于“作者就是为了节约开销而设计它”。要解释作者动机仍需作者确认。

现有分数应继续标为当前受限实现的结果。若后续研究截断影响，应保留现有运行，再在独立 artifact 根目录中移除或参数化该限制、重建全部记忆并进行对照实验；不能仅提高检索 k，也不能复用现有 36 条记忆库。

## 5. Mem0 官方上游对照

另行核查了 `mem0ai/mem0` 的 `mem0/memory/main.py`：

- [v0.1.100](https://github.com/mem0ai/mem0/blob/v0.1.100/mem0/memory/main.py)：解析 `facts` 后直接遍历完整的 `new_retrieved_facts`，没有 `[:2]`。
- [v0.1.114](https://github.com/mem0ai/mem0/blob/v0.1.114/mem0/memory/main.py)：同样遍历完整事实列表，没有固定取两条的切片；同步和异步路径均已检查。
- [v1.0.10](https://github.com/mem0ai/mem0/blob/v1.0.10/mem0/memory/main.py)：这是 MemoryData requirements.txt 声明的安装依赖版本，其事实处理循环也没有该两条截断。安装依赖版本不等于 MemoryData 实际调用的内置源码版本。
- [当前 main](https://github.com/mem0ai/mem0/blob/main/mem0/memory/main.py)：实现已经更新，抽取后对记忆列表进行批量嵌入和逐条处理，没有该固定前两条限制。

在已核查的官方版本中，这项截断不属于 Mem0 上游实现；它是 MemoryData 发布副本相对于这些上游版本的差异。这里没有穷举 Mem0 的全部历史提交，也未确认 MemoryData 最初复制自哪个确切上游 commit。正常的模型抽取、去重和 ADD/UPDATE/DELETE/NONE 决策仍可能减少最终记忆数，与无条件截取前两条是不同机制。
