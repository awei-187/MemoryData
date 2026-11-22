# 有时间预算的冒烟评测

`scripts/run_benchmark_pilot.py` 用于检查真实输入经过记忆写入、检索、回答、评分的完整链路，不能代替完整历史评测。

- 输入为已有评测 manifest 和对应的 `input/*_selected_history_and_qa.json`；源数据和旧结果不修改。
- 每类使用首个上下文的前 2 个 chunk、最多 2 题。新配置显式记录 `smoke_max_history_chunks: 2`，原数据来源 ID 保持不变。
- `smoke_max_history_chunks` 只影响显式设置了该字段的运行；没有该字段时保留原有完整历史行为。未重新解释旧的 `context_max_length`，避免悄悄改变既有评测协议。
- Mem0 原有每块前两条事实限制仍保留。
- 每项 360 秒、整轮 1800 秒；Hindsight 整合额外等待上限 180 秒。这些是检查预算，超时不能记作 0 分，也不能宣称性能问题已修复。
- 各任务顺序执行，每项 Hindsight 使用独立服务和数据库；任务成功、报错或超时后均请求启动器关闭原生引擎及数据库，保留文件。停机未确认时不再启动下一项。
- 模型连接错误立即停止后续任务；单项内容过滤或整合超时单独记录，服务停止后才可继续其他独立任务。所有异常输出 `event=attention` 供监督立即报告。
- 用户可在本轮输出目录创建 `stop.request`；当前评测子进程会退出，Hindsight 服务随之停止，后续任务标记未运行。
- 启动前要求 `network_selection.json` 中有通过聊天和嵌入检查的明确路由选择；不关闭 TLS 证书校验，不替换数据原文来绕过内容过滤。
- 新运行不能覆盖已有 `suite.status.json`；运行开始时冻结非敏感源码及 SHA-256。

本次运行目录：`results/other_benchmarks_smoke_20261022`。原始零重试探测中直连全部超时、代理间歇失败；使用 SDK 现有最多两次重试设置后，代理连续两轮聊天和嵌入均成功。因此选择代理，但仍将连接稳定性列为未彻底解决的问题。

使用已确认的 memorydata 环境，单行 PowerShell 命令：

```powershell
& 'D:\conda\envs\memorydata\python.exe' -u scripts/run_benchmark_pilot.py --source_manifest results/other_benchmarks_small_20261022/manifest.json --run_dir results/other_benchmarks_smoke_20261022
```

本轮修复验证：`tests.test_artifact_paths`、`tests.test_smoke_history_limit`、`tests.test_benchmark_pilot`、`tests.test_hindsight` 共 30 项通过，涵盖非法文件名、显式历史限制及来源保留、超时/停止后回收真实子进程、服务停机握手、Hindsight 未完成整合不标 ready 及诊断持久化。
