# MemoryData：LLM-as-Judge 输入结构与完整 Prompt

核查日期：2026-10-21；已结合 Hindsight 接入及统一远程 OpenAI 模型配置复核。以下英文 prompt 从 Python AST 的字符串常量提取，保留原文；本次更新没有执行评测或调用模型。

## 0. 本次模型迁移与 Judge 的关系

| 角色 | 当前配置 | 与 Judge 的区别 |
| --- | --- | --- |
| MemoryData 最终回答 | 23 个预设默认 `gpt-4o-mini`，由 `OPENAI_MODEL` 覆盖，通过远程 OpenAI 兼容 API 调用 | 被评分的回答来自此模型 |
| 记忆抽取／摘要、嵌入 | 适配器支持的对话调用同步共享模型；嵌入默认 `text-embedding-3-small`、1536 维 | 负责构建／检索记忆，不是最终评分 |
| Hindsight 引擎 | 原生 retain／recall；本仓库启动器映射共享模型配置，默认 RRF；已有服务需另行对齐 | 引擎处理记忆，MemoryData 生成回答；本适配器不调用 reflect |
| LongMemEval Judge | **仍默认 `gpt-4o`**，由 `--judge-model` 显式设置；读取 `OPENAI_API_KEY`／`OPENAI_BASE_URL` | 不自动跟随 `OPENAI_MODEL`，不因新增 Hindsight 改变 prompt 或输入字段 |

根 `.env` 自动加载发生在 `main.py`；独立 Judge 入口没有相同的加载代码。评分时需先将配置加载到当前 Python 进程。真实 API 需有效凭据，旧命令中的 `OPENAI_API_KEY=local` 仅适合不校验密钥的本地测试服务，不能用于这套远程模型配置。

例如在项目目录中，显式选择 `gpt-4o-mini` 作裁判（这是命令的选择，不是修改后的 CLI 默认值）：

```powershell
cd 'E:\MemoryData'
```

```powershell
& 'D:\conda\envs\memorydata\python.exe' -c "from dotenv import load_dotenv; import runpy; load_dotenv('.env', override=False); runpy.run_path('evaluation/longmemeval/evaluate_longmemeval_judge.py', run_name='__main__')" '替换为实际的LongMemEval结果文件路径' --judge-model gpt-4o-mini --num-runs 1 --sidecar-suffix openai_api_20261021
```

需以实际文件路径替换占位参数；`--sidecar-suffix` 区分该次裁判产物，模型／prompt 改变时不要直接沿用旧评分。示例没有在命令行打印凭据，也没有在本次文档更新中执行。默认裁判、五类 prompt、消息构造和解析代码均未随回答模型迁移改变。

共享配置细节见 [docs/api_models.md](docs/api_models.md)，Hindsight 运行方式见 [docs/hindsight.md](docs/hindsight.md)。

## 1. 实际有哪些 Judge

当前统一评测链路中，专用的外部 LLM Judge 是 **LongMemEval 结果后处理**。LoCoMo、MemBench、LongBench 等主结果中的 EM/F1/ROUGE/Recall 是规则计算，不能因为记录里有 `judge` 字段就认定调用了大模型。Self-RAG 等方法内部的相关性判断属于被测算法，不是此处的基准最终评分器；第三方内嵌源码的评估工具也不自动属于统一评分入口。

入口为 [evaluate_longmemeval_judge.py](evaluation/longmemeval/evaluate_longmemeval_judge.py)，核心为 [longmemeval_judge.py](evaluation/longmemeval/longmemeval_judge.py) 的 `extract_supported_longmemeval_rows()`、`build_judge_messages()` 和 `LongMemEvalJudgeClient.judge_once()`。

## 2. 数据到裁判输入的映射

| 裁判字段 | 实际来源 | 是否进入模型消息 |
| --- | --- | --- |
| `question` | 通过 `eval_metadata.question_id` 匹配原始 LongMemEval 后取得 canonical `question`；仅没有 ID 时回退问题文本匹配 | 是 |
| `golden_answer` | canonical 数据中的 `answer`，不是直接信任结果文件里的答案字段 | 是；按题型标为 Correct Answer、Rubric 或 Explanation |
| `generated_answer` | 原始结果行的 `output`；不是 `parsed_output` | 是 |
| `question_id` | 题目 ID；以 `_abs` 结尾时强制归为 abstention | 用于选 prompt，不单独发送 |
| `question_type` | canonical 题型优先，回退评测元数据 | 用于选 prompt，不单独发送 |
| 历史、检索上下文、evidence、记忆内容、方法名 | 不被 `build_judge_messages()` 读取 | 否 |

这是**有参考答案的单回答正确性评分**，不让裁判先读完整历史自行确定事实，也不比较两种系统的答案。

实际请求结构如下；`USER_PROMPT` 是第 4 节某个模板填充后的字符串：

```json
{
  "model": "<--judge-model，默认 gpt-4o>",
  "messages": [
    {
      "role": "system",
      "content": "You are an expert evaluator for LongMemEval. Return JSON only with a single key: label. Valid labels are \"CORRECT\" or \"WRONG\"."
    },
    {
      "role": "user",
      "content": "<USER_PROMPT：按题型选择后填充 question / golden_answer / generated_answer>"
    }
  ],
  "temperature": 0
}
```
调用 `AsyncOpenAI.chat.completions.create()`；没有在请求中设置 `response_format`、JSON Schema 或工具函数约束，JSON 主要依靠 prompt 约束。

## 3. System prompt 原文

```text
You are an expert evaluator for LongMemEval. Return JSON only with a single key: label. Valid labels are "CORRECT" or "WRONG".
```
## 4. 按题型选择的五个 User prompt 原文

| 题型 | 模板 |
| --- | --- |
| `single-session-user`、`single-session-assistant`、`multi-session` | 4.1 通用事实 |
| `temporal-reasoning` | 4.2 时间推理 |
| `knowledge-update` | 4.3 知识更新 |
| `single-session-preference` | 4.4 个性化偏好 |
| `abstention` 或 ID 以 `_abs` 结尾 | 4.5 拒答 |

不支持的题型会触发异常，不是静默使用统一默认模板。

### 4.1 通用事实：要求答案信息完整

```text
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Question: {question}

Correct Answer: {golden_answer}

Model Response: {generated_answer}

Is the model response correct? Answer yes or no only.
```

### 4.2 时间推理：天数等计算容忍 off-by-one

```text
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.

Question: {question}

Correct Answer: {golden_answer}

Model Response: {generated_answer}

Is the model response correct? Answer yes or no only.
```

### 4.3 知识更新：包含旧信息时，只要给出所需更新答案仍可判对

```text
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

Question: {question}

Correct Answer: {golden_answer}

Model Response: {generated_answer}

Is the model response correct? Answer yes or no only.
```

### 4.4 偏好：正确回忆并使用个人信息即可，不要求命中 rubric 所有点

```text
I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.

Question: {question}

Rubric: {golden_answer}

Model Response: {generated_answer}

Is the model response correct? Answer yes or no only.
```

### 4.5 拒答：是否识别问题无法回答

```text
I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.

Question: {question}

Explanation: {golden_answer}

Model Response: {generated_answer}

Does the model correctly identify the question as unanswerable? Answer yes or no only.
```
## 5. 返回、解析与聚合

预期返回 `{"label":"CORRECT"}` 或 `{"label":"WRONG"}`。当前 user prompt 却要求 yes/no，与 system 的 JSON/CORRECT/WRONG 指令存在格式不一致；不能在展示源码时把这个矛盾改写掉。

当前解析器先尝试 JSON 的 label（接受 CORRECT/WRONG/YES/NO）；失败时按 CORRECT、WRONG、YES、NO 顺序在整个输出中做子串查找。随后 CORRECT/YES 转为 1，WRONG/NO 转为 0。完全无法解析会报评分错误。

这种回退不做严格词边界／唯一标签验证；例如正文同时含多种标签时，或 `INCORRECT` 含有 `CORRECT` 子串时，可能错误归类。这里描述现有实现，没有修改评分代码。

`--num-runs N` 会针对同一组输入重复 N 次调用，并不是多数投票：

- 每题 `judge_mean = 正确次数 / N`。
- `judge_accuracy` 为已完整评分题目的 `judge_mean` 均值。
- `judge_accuracy_std` 为各次运行整体正确率的总体标准差。
- 跳过生成失败或无法匹配 canonical 数据的记录；无完整评分记录时准确率为 null。
- 独立保存 `judge_labels`、`parsed_label`、`raw_judge_response` 等，输出 `*.longmemeval_judge.json`。

## 6. 与 OmniMemEval 的直接差异

MemoryData 的通用事实模板要求所需信息完整；时间模板放宽 off-by-one；更新、偏好和拒答有各自规则。相邻 OmniMemEval 的 LongMemEval 当前使用通用 QA 模板，明确要求对“触及同一主题”的答案宽松评分，也没有相同的逐题型分派。即使 Judge 模型相同，两者也不是同一评分协议。

完整对照及 OmniMemEval 的 HaluMem、BEAM、AgentBench prompt 见 [OmniMemEval Judge 文档](<E:/LLM judge/OmniMemEval/LLM_JUDGE_PROMPTS_ZH.md>)。覆盖总表见 [EVALUATION_COVERAGE_ZH.md](EVALUATION_COVERAGE_ZH.md)。
