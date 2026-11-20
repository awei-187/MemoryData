"""Write the reproducible experiment notebook from audited local evidence."""
from collections import Counter
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/full_first_units_20261023'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary, current = read(RUN / 'summary.json'), read(RUN / 'suite.status.json')
    assert summary['suite_status_sha256'] == sha(RUN / 'suite.status.json')
    assert current['phase'] == 'success' and summary['case_counts'] == {'success': 22}
    assert summary['successful_answers'] == summary['expected_answers'] == 1130
    assert sum(summary['case_counts'].values()) == 22
    repro = read(RUN / 'mem0_eventqa_reproduction_20261024.json')
    assert repro['status'] == 'success'
    attempts = [read(path) for path in sorted(RUN.glob('suite.attempt_*.status.json'))] + [current]
    versions = {name: importlib.metadata.version(name) for name in
                ('openai', 'httpx', 'qdrant-client', 'pandas', 'pyarrow', 'rouge-score')}
    native_versions = json.loads(subprocess.check_output([
        r'D:\conda\envs\hindsight\python.exe', '-c',
        'import platform,json,importlib.metadata; print(json.dumps({"python":platform.python_version(),'
        '"openai":importlib.metadata.version("openai"),"httpx":importlib.metadata.version("httpx")}))'], text=True))
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).splitlines()
    evidence = {
        'generated_at': datetime.now().astimezone().isoformat(), 'python': platform.python_version(),
        'python_executable': r'D:\conda\envs\memorydata\python.exe',
        'native_python_executable': r'D:\conda\envs\hindsight\python.exe',
        'platform': platform.platform(), 'architecture': platform.machine(), 'packages': versions,
        'native_environment': native_versions, 'repository_head': head,
        'native_source_head_verified_20261024': '1e427025b4d4c01e8ad6385dd04e02102886a5c4',
        'dirty_path_count': len(dirty), 'active_full_suite_processes_final_host_check': 0,
        'files': {name: sha(RUN / name) for name in ('manifest.json', 'suite.status.json', 'summary.json',
            'REPORT.md', 'per_question.jsonl', 'mem0_eventqa_reproduction_20261024.json',
            'mem0_eventqa.log', 'suite_resume_7.log', 'suite_resume_8.log',
            'suite_resume_9.log', 'suite_resume_10.log', 'suite_resume_11.log',
            'suite_resume_12.log', 'suite_resume_13.log', 'suite_resume_14.log',
            'tail_completion_audit.json', 'tail_resume_tests.log',
            'mem0_eventqa_route_diagnostic_20261024.json', 'mem0_checkpoint_production_validation.json',
            'model_availability_20261024.json',
            'SERVICE_REVIEW_REQUEST.md', 'MEM0_CHUNK_RECOVERY.md', 'mem0_checkpoint_tests.log')},
        'source_files': {str(path.relative_to(ROOT)): sha(path) for path in
            (ROOT / 'scripts/run_full_first_units.py', ROOT / 'scripts/diagnose_mem0_eventqa.py',
             ROOT / 'scripts/summarize_full_first_units.py', Path(__file__).resolve())}}
    for name in ('utils/mem0_checkpoint.py', 'utils/model_api_diagnostics.py', 'utils/initialization.py',
                 'methods/mem0/source/mem0/llms/openai.py', 'tests/test_mem0_checkpoint.py',
                 'scripts/audit_tail_completion.py', 'scripts/verify_mem0_eventqa_checkpoint.py',
                 'scripts/preflight_mem0_tail.py', 'tests/test_full_first_units.py'):
        evidence['source_files'][name] = sha(ROOT / name)
    (RUN / 'experiment_record_metadata.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    result_table = (RUN / 'REPORT.md').read_text(encoding='utf-8').split('| 数据切片', 1)[1].split('\n\n', 1)[0]
    result_table = '| 数据切片' + result_table
    scope = []
    by_task = {case['task']: case for case in summary['cases'] if case['method'] == 'hindsight'}
    for task in ('locomo', 'eventqa', 'longmemeval', 'factconsolidation', 'banking77', 'longbench',
                 'membench_simple', 'membench_noisy', 'membench_knowledge_update', 'membench_highlevel', 'membench_recmultisession'):
        case = by_task[task]
        scope.append(f"| {task} | {case['chunk_count']} | {case['expected_questions']} |")
    timeline = []
    for number, attempt in enumerate(attempts, 1):
        counts = Counter(case['phase'] for case in attempt['cases'])
        timeline.append(f"| {attempt.get('attempt_number', number)} | {attempt['started_at']} | {attempt.get('ended_at', '—')} | {counts.get('success', 0)} | {counts.get('failed', 0)} | {counts.get('blocked', 0)} | {counts.get('not_run', 0)} |")
    links = []
    for title, name in (('完整范围与来源指纹', 'manifest.json'), ('全部分数与逐项核验', 'summary.json'),
                        ('逐题成功问答', 'per_question.jsonl'), ('请求可靠性修复记录', 'RELIABILITY_NOTES.md'),
                        ('原第85块单次复验', 'mem0_eventqa_reproduction_20261024.json'),
                        ('第7次完整重跑日志', 'mem0_eventqa.log'), ('环境与证据哈希', 'experiment_record_metadata.json')):
        links.append(f'- [{title}](<{(RUN / name).as_posix()}>)')
    # The current main log belongs to attempt 8; the older attempt remains archived.
    links = [line.replace('第7次完整重跑日志', '当前最新重跑日志') for line in links]
    for title, name in (('服务端核查单（未发送）', 'SERVICE_REVIEW_REQUEST.md'),
                        ('分块恢复协议与测试', 'MEM0_CHUNK_RECOVERY.md'),
                        ('生产检查点副本恢复验证', 'mem0_checkpoint_production_validation.json')):
        links.append(f'- [{title}](<{(RUN / name).as_posix()}>)')
    text = f'''# Mem0 与 Hindsight 首个完整上下文评测实验记录

记录时间：{evidence['generated_at']}（Asia/Shanghai）。最终状态：**全部完成，第14次调度结束于2026-10-24 23:27:22**。

## 1. 实验目的与最终状态

在 MemoryData 统一评测框架中，对 Mem0、Hindsight 分别运行各已准备基准切片的第一个完整上下文/场景及其全部问题。共 11 个切片、每系统 565 题、22 个组合、1,130 条计划回答。

**22/22 个组合通过完整覆盖核验，1,130/1,130 条回答有效。** Hindsight 完成565/565，Mem0完成565/565，无剩余失败或受阻项目。最后补齐的Mem0 EventQA已完整构建131块、回答100题，EM45.00、F1 64.35、ROUGE-L F1 66.42。

本轮通过一致检查点及逐题恢复补齐了Mem0 EventQA。第12次完成完整记忆并获得98条有效回答；第13次补回query76，第14次补回query10。其余99条回答逐项保持不变，原21项成功结果文件SHA-256也完全一致。以下保留历史失败阶段作为审计记录，不代表最终仍未完成。低分与运行失败分开：例如Hindsight Banking77的0%是有效质量成绩，没有为了提高分数重跑成功题。

## 2. 数据选择与实验协议

| 切片 | 完整历史块数 | 每系统全部问题数 |
|---|---:|---:|
{chr(10).join(scope)}

LoCoMo 使用原始 conv-26，19 个 session、419 条发言、199 题，包含全部问题类别。MemoryAgentBench 选其各任务首条完整上下文；MemBench 选各类首分支、首完整场景，并保留其原有一个末尾问题。LongBench 使用官方 LongBench-v2 第一条完整样本：本机原 rep150 子集缺失，不能称为该子集的复现。

所有正式运行均不设置 smoke_max_history_chunks，CLI --max_test_queries_ablation 0 表示不限问题，max_test_samples=1 表示首上下文。旧配置/tag 的 max10 字样仅为兼容已保存记忆身份，实际题数按冻结输入与逐题 ID 核验。本记录不纳入两块历史、两题的冒烟成绩。

聊天模型 gpt-4o-mini，嵌入模型 text-embedding-3-small、1536 维，服务为用户已授权的 api.relayrouter.ai。无独立 LLM Judge。Mem0 使用本仓库实现、每块事实抽取后保留前两条，检索上限 100；Hindsight 使用原生 retain / consolidation / recall，检索上限 20，由 MemoryData 完成最终回答，不使用 reflect。这些是两系统各自原配置，不是相同检索预算的消融对比。

## 3. 环境与版本

- MemoryData Python：{evidence['python']}，`{evidence['python_executable']}`。
- Hindsight 原生 Python：{native_versions['python']}，`{evidence['native_python_executable']}`；原生 OpenAI SDK {native_versions['openai']}、httpx {native_versions['httpx']}。
- 本机平台：{evidence['platform']}，{evidence['architecture']}。推理调用外部服务；本实验未测量 GPU 性能、实际服务端硬件、账单费用或碳排放。
- MemoryData 依赖：{', '.join(name + ' ' + version for name, version in versions.items())}。
- 仓库基线 HEAD：`{head}`。工作区存在 {len(dirty)} 个变更/未跟踪路径，包含此前工作；HEAD 不能单独代表执行源码。正式各次源码存于 source_snapshot* 和 code_sha256*.json，环境与当前脚本哈希另见 experiment_record_metadata.json。
- 本机 Hindsight 源码 HEAD（2026-10-24只读核验）：`1e427025b4d4c01e8ad6385dd04e02102886a5c4`；本项目未修改外部原生仓库。相关原生文件哈希另见 native_policy_fix_audit.json、native_llm_connect_fix_audit.json。

## 4. 构建、复用与运行轨迹

完整记忆先构建/加载，再检索和回答。只复用经过内容指纹、配置身份或结果哈希检查的完整记忆/结果；不将未完成库当作完整历史，不混用冒烟库。403条来自已完成的完整旧结果、30条为完整记忆下的有效旧回答，本轮新增697条成功回答。

Hindsight 待整合的 Banking77/EventQA 分别已完整导入 26/131 个文档：复制已停止的原 PostgreSQL 数据库，逐文件 SHA-256 一致后，在副本中等待原生整合就绪，不重新 retain、不手工标 ready。bank 按 UUID 隔离。早期 Mem0 无可靠分块 checkpoint 的部分库先归档，再重建；不能仅按日志进度截取后续历史继续追加。第8次新增并启用经过测试的一致分块检查点，后续可恢复已提交的53块。

下表为每次调度结束时的累计状态，包含复用结果，不代表该次新跑了全部项目。时间为各状态原样记录；包含模型等待、恢复与服务启停，不能当作单系统构建耗时或严格吞吐对比。

| 调度 | started_at | ended_at | 成功 | 失败 | 受阻 | 未运行 |
|---:|---|---|---:|---:|---:|---:|
{chr(10).join(timeline)}

## 5. 失败诊断、修复与复跑

| 问题 | 证据与归因 | 处理及验证 | 最终状态 |
|---|---|---|---|
| Windows 路径含 `*` 等非法字符 | 数据任务名会进入输出路径/集合名；属本机兼容问题 | 统一编码非法字符，保留原始 ID；路径与真实 Qdrant 回归 | 已完成受影响实验 |
| Hindsight 旧 profile 接口 410 | 原生版本接口差异 | 仅在 410 时回退 config，其他错误仍报；适配器回归 | 已完成 |
| LongMemEval 日期含斜线和星期 | 原生时间参数格式要求与数据格式不同 | 转为等价 ISO 日期；不改原评测问题或附加时区 | 已完成 |
| 保存的 Hindsight 仍在整合 | 原轮短预算耗尽，不代表导入一定失败 | 校验完整文档映射后，仅继续整合；每项最多12小时、整合最多10小时 | 已完成 |
| TLS ConnectTimeout / Unexpected EOF | SDK 堆栈显示经当前本机代理握手超时/提前断连；不是答案错误 | 显式连接预算30秒，SDK最大重试由4增至10；原生聊天使用官方连接变量、保持自己的有限重试循环；失败题有限补问、失败任务单项续跑 | 相关失败均补齐 |
| Mem0 UPDATE 使用不存在的 ID | 模型给出映射外编号，原实现 catch 后跳过该动作 | 记录并保留原算法，不猜测 UUID，不重采样以提高分数；成功日志仍可能含动作错误 | 属质量限制，非进程失败 |
| Mem0 EventQA content_filter | 历史上多次导入或问答遭遇HTTP400；失败阶段详下文 | 原配置分块恢复及仅补问失败题，保留证据，不删块/换模型/改写模板 | 最终100题已补齐；服务端间歇拒绝根因未确定 |
| Mem0 中断后重复构建 | 旧实现无可验证分块恢复点 | 新增关闭数据库的一致快照、逐文件哈希、身份检查和原子指针；真实部分写入回滚测试 | 已修复，生产53块可恢复 |

连接预检曾显示代理路径4/4完成握手、未认证返回401，直连4/4超时；没有更改本机代理设置或禁用 TLS 校验。真实 SDK + MockTransport 验证有限握手重试和400不重试，原生配置验证连接30秒且原请求读写预算保留；此次调度修改后6项覆盖/可靠性回归通过。详尽日志见 RELIABILITY_NOTES.md；不能把有限重试当作消除了网络波动。

### EventQA 本轮复验的可核查细节

1. 上次完整失败：2026-10-24 02:43 左右，84/131 已处理，进入第85块事实抽取时报400。旧日志与部分库已经归档至 attempt_history/attempt_6，旧最终报告归档至 attempt_history/final_reports_before_attempt7。
2. 本次诊断：{repro['time']}，第85块原文、相同模型/提示词模板单次调用成功，不写记忆、不自动重试；原模板动态时间正常刷新。原块 SHA-256：`{repro['chunk_sha256']}`。这不是字节完全相同的HTTP请求重放，也不能证明上次过滤的具体触发词或策略。
3. 完整重跑：08:23:55 启动第7次调度；从归档后的空白库处理原131块。08:40:26 在69/131已处理后、第70块的事实抽取再次返回400 content_filter，未进入100题问答。请求ID `20261024084020902527217ctzOWQWk`。本次与上次失败块不同，不能断言“修复第85块就能修复全流程”。
4. 本地代码能正确收到并记录明确服务拒绝。没有证据支持某个本地参数错误导致这一400；服务内部过滤机制/具体触发原因仍未确定。本轮没有通过改写输入、略过块或关闭过滤来生成“完整成功”成绩。
5. 上述为第7次结束时的历史状态。后续按原协议逐步恢复，最终于第14次补齐。将来若改变服务、模型或输入，应单独标记为另一实验条件，不能覆盖本记录；本次成功不意味着已经确定或根治服务端间歇过滤原因。

### 按用户要求执行的第8次修复验证

- 官方网关说明已查阅并链接于 SERVICE_REVIEW_REQUEST.md；当前服务为聚合网关，但本次响应不足以确定实际上游。未登录控制台、调用管理接口、获取新管理令牌或向服务商发消息。
- 16:04原第70块单次诊断得到200，仅见server=nginx和x-oneapi-request-id；没有上游渠道或过滤类别。诊断仍使用原文、同一模型及模板，模板动态时间正常刷新。
- 16:39通过正常模型API只读核查/v1/models，当前gpt-4o-mini与text-embedding-3-small均有条目、owned_by=openai；这只是网关公布的模型目录，不能证明某个推理请求实际使用的上游路由。证据见model_availability_20261024.json。
- 实现 opt-in `--mem0_checkpoint`：每块完成后关闭Qdrant/SQLite，同步保存库与来源映射；校验历史/配置/关键源码及逐文件哈希，再原子提交指针。中断后先归档不确定运行库、恢复最后提交快照，不直接追加到部分写入。保留原算法及前两条事实限制。
- 29项联合回归通过；其中真实Qdrant/SQLite验证部分写入后回滚、继续时无重复历史、破坏快照与改变输入/配置会被拒绝、指针原子提交中断不会破坏上一代。诊断只记录允许字段，不记录认证头或完整请求文本。最终针对诊断增强的5项测试再次通过。
- 第8次于16:14:34启动，仅重跑Mem0 EventQA。16:33:15在完成53块后，第54块事实抽取返回400 content_filter；请求ID `202610241633052443895835C6shcCY`。API错误记录仍只有server=nginx，无渠道或结构化过滤类别；未进行100题问答。
- 生产检查点 `complete=false`、已提交53/131块；在独立副本中校验并恢复成功，得到98个Qdrant点、102行历史，SQLite integrity_check=ok。没有修改原生产检查点，没有模型调用。后续可从第54块恢复。
- 第8次结束时服务端拒绝仍在，本地恢复已修复并验证，未以跳块或更换服务混入原成绩。后续补齐情况见第12至14次记录。

### 用户再次要求重跑后的第9至11次续跑

- 19:03第54块独立调用成功，网关请求ID `202610241903195317422086lo6mYHs`；该诊断未写入记忆，不计作完整实验成功。
- 第9次19:11:51从53块恢复；19:21:35在提交83块后，第84块事实抽取返回400 content_filter，网关请求ID `2026102419212616395231it6e9EbF`。此为最新明确的内容过滤拒绝记录。
- 主智能体明确最多再续跑两次，同一块连续被过滤则停止，保持原模型、原文、模板和路由。第10次19:24:05从83块恢复，19:38:43推进至130块；最后第131块的embedding请求APIConnectionError，未获得HTTP响应。
- 第11次19:39:54从130块恢复；19:41:53仍在第131块的chat请求APIConnectionError，底层SSL UNEXPECTED_EOF_WHILE_READING。没有HTTP状态或服务端请求ID，不能把它解释为又一次content_filter。本轮有限续跑已用完，没有继续追加。
- 第11次结束时生产检查点complete=false、committed_chunks=130。独立副本恢复为237个Qdrant点、255行历史，SQLite integrity_check=ok；当时100题尚未开始。下一次由第131块恢复，没有重做前130块。
- 当时21项成功回答文件再次核验未改变；服务核查需分别处理间歇内容拒绝和网络中断，不能混为一个错误。

### 最终完成：第12至14次

- 用户明确要求完成剩余内容。23:07原服务、原代理路径的聊天及嵌入最小预检均成功。续跑前补充修复旧结果读取不应把gold列表无条件取[0]的问题；命中第二参考答案的回归证明保留全gold后评分与首次规则一致，连同checkpoint共10项测试通过。测试首次因fixture缺少debug字段报错，补全fixture后通过，未修改生产评分规则。
- 第12次23:10:37从已验证130块恢复，完成第131块并提交complete=true。100题均尝试，其中98成功，query10与76因内容过滤失败，保留失败状态而不计作有效回答。
- 第13次23:23:22复用完整记忆和98成功回答，仅补问两条失败题；query76成功，query10仍content_filter（请求ID20261024232404369918946bqJQ8XFW）。没有为提高分数重问成功题。
- 第14次23:26:30仅补问最后query10，23:27:22成功结束。最终100/100无失败/空输出；与第13次归档对比，其余99个output完全不变。最终EventQA结果SHA-256为 `be4745038d4be062a3c57ee55354248098b63d75b410bd6875b362f097852868`。
- 完整检查点再次在独立副本中恢复验证：131/131、complete=true，239个Qdrant点、257行历史，SQLite integrity_check=ok；未改生产库、未调用模型。所有22项现已成功，原21项结果SHA不变；完整问题与标准答案核验覆盖全部1130条。
- 99条有效时的中间均值只基于99题，最终本记录严格以100题分母报告，不混用中间分数；失败日志、旧结果和每次源码快照均保留。

## 6. 质量成绩

分数单位为百分比，来自框架已有规则。选择题优先看 EM；Banking77 的 label_accuracy 与 EM 相同。以下是首上下文完整问题集成绩，不是全数据集总体成绩。

{result_table}

LoCoMo和LongMemEval的本次F1均为Hindsight高于Mem0；Banking77是Mem0高于Hindsight。EventQA两者EM均45.00，F1为Mem0 64.35、Hindsight63.33，相差约1.02个百分点。MemBench各类只有一题，不据此作稳定排名或显著性结论。未跨不同任务求一个总平均分，也未进行置信区间或多随机种子实验。

## 7. 完整性核验与限制

- 22项成功文件逐项检查SHA-256、完整query_id集合、无重复、context_id=0、问题ID/文本、标准答案及非空输出；主要指标与逐题均值一致。原21项结果及EventQA最后补答前的99条有效输出均验证不变，详tail_completion_audit.json。
- 框架续跑把118条单元素gold列表转成了字符串，审核确认全部语义等价、没有多答案丢失。逐题JSONL同时保留answer与frozen_gold。
- Mem0完整历史输入不等于完整信息保留：前两条事实上限与无效UPDATE跳过会影响质量，报告保留当前日志计数而非宣称每条动作成功。
- 标准词元F1会去掉英文冠词a/an/the，单字母A可能被归一化为空，因此存在EM=100但F1=0；本轮未改评分，应以选择题EM解释这些行。
- LongMemEval原始gold session文件在本机缺失，补充Recall@K生成诊断失败；60题QA规则评分正常，不能宣称获得官方Recall@K或官方LLM评审成绩。
- 单个上下文/场景的结果不能外推全数据集；源输入、模型设置及各次请求策略不同，不能将累计墙钟时间用于严格性能排名。
- 最终本机检查无本轮评测进程。summary核验的5个原生服务状态均stopped且数据库保留；本轮7份Hindsight保存状态ready=true且文档数等于完整历史块数。第7次仅运行Mem0，未启动新Hindsight服务。

## 8. 文件、复现与后续条件

{chr(10).join(links)}

冻结输入在结果目录input/，配置在configs/，结果原文件路径和哈希见summary.json。原始来源URL、版本与指纹见manifest.json。每个文件均不包含模型密钥；.env不写入文档或快照。

重新核验并生成成绩汇总（不调用模型）：

```powershell
& 'D:\\conda\\envs\\memorydata\\python.exe' scripts/summarize_full_first_units.py
```

重新生成本实验记录（不调用模型）：

```powershell
& 'D:\\conda\\envs\\memorydata\\python.exe' scripts/write_first_unit_experiment_record.py
```

最终保留与补齐审计（不调用模型）：

```powershell
& 'D:\\conda\\envs\\memorydata\\python.exe' scripts/audit_tail_completion.py
```

本次没有剩余项目需要续跑。若将来做新的模型/输入/服务条件实验，应使用独立产物目录，并保留这里的冻结配置和结果。
'''
    path = ROOT / 'docs/experiments/2026-10-24_first_unit_evaluation.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    linked = [Path(target) for target in re.findall(r'\]\(<([^>]+)>\)', text)]
    for target in linked:
        assert target.is_file(), target
    from dotenv import dotenv_values
    secrets = [value for key, value in dotenv_values(ROOT / '.env').items()
               if value and len(value) >= 8 and any(word in key.upper() for word in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD'))]
    for artifact in {path, *linked, *(RUN / name for name in evidence['files'])}:
        content = artifact.read_text(encoding='utf-8')
        assert not any(value in content for value in secrets), 'Credential found in ' + str(artifact)
    evidence['document_checks'] = {'linked_files_exist': True, 'configured_secrets_absent_in_record_and_linked_evidence': True}
    (RUN / 'experiment_record_metadata.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    integrity = {str(artifact.relative_to(ROOT)): sha(artifact) for artifact in
                 (path, RUN / 'experiment_record_metadata.json', RUN / 'REPORT.md', RUN / 'summary.json', RUN / 'per_question.jsonl')}
    (RUN / 'experiment_record_integrity.json').write_text(json.dumps(integrity, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'record': str(path), **summary['case_counts'], 'answers': summary['successful_answers']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
