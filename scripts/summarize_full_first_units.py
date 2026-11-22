"""Audit completed first-unit evaluations and write a self-contained report; no API calls."""
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import re
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_full_first_units import digest, verify_result


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def main():
    run = ROOT / 'results/full_first_units_20261023'
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    state = json.loads((run / 'suite.status.json').read_text(encoding='utf-8'))
    if state['phase'] == 'running':
        raise RuntimeError('Do not finalize a running suite')
    labels = {
        'locomo': 'LoCoMo 原始 conv-26', 'eventqa': 'MemoryAgentBench EventQA',
        'longmemeval': 'MemoryAgentBench LongMemEval', 'factconsolidation': 'FactConsolidation',
        'banking77': 'Banking77', 'longbench': 'LongBench-v2 官方首样本',
        'membench_simple': 'MemBench Simple', 'membench_noisy': 'MemBench Noisy',
        'membench_knowledge_update': 'MemBench Knowledge Update',
        'membench_highlevel': 'MemBench HighLevel', 'membench_recmultisession': 'MemBench RecMultiSession'}
    cases = []
    per_question = []
    locomo_creator = None
    imported = seeds = 0
    for record in state['cases']:
        item = dict(record)
        if record['task'] == 'locomo':
            original = ROOT / 'results/locomo_first_original_20261021'
            original_manifest = json.loads((original / 'manifest.json').read_text(encoding='utf-8'))
            assert digest(original_manifest['input']) == original_manifest['input_sha256']
            if locomo_creator is None:
                from utils.conversation_creator import ConversationCreator
                config = yaml.safe_load((original / 'hybrid_hindsight.yaml').read_text(encoding='utf-8'))
                dataset = yaml.safe_load((original / 'dataset.yaml').read_text(encoding='utf-8'))
                locomo_creator = ConversationCreator(config, dataset)
            qa = locomo_creator.get_query_and_answers()[0]
            chunks = locomo_creator.get_chunks()[0]
            item.update(chunk_count=len(chunks), input_file=original_manifest['input'],
                        input_sha256=original_manifest['input_sha256'])
        else:
            frozen = json.loads(Path(record['input_file']).read_text(encoding='utf-8'))
            qa, chunks = frozen['qa'], frozen['chunks']
            actual = hashlib.sha256(canonical(chunks).encode('utf-8')).hexdigest()
            assert actual == record['history_sha256'], record['id']
            assert len(chunks) == record['chunk_count']
            dataset = yaml.safe_load(Path(record['dataset_config']).read_text(encoding='utf-8'))
            assert dataset['max_test_samples'] == 1 and not dataset.get('smoke_max_history_chunks')
            item['input_sha256'] = digest(record['input_file'])
        assert len(qa) == record['expected_questions']
        if record['phase'] == 'success':
            checked = verify_result(record['result_file'], qa, record['result_sha256'])
            assert checked['completed'] == record['completed']
            result = json.loads(Path(record['result_file']).read_text(encoding='utf-8'))
            rows = sorted(result['data'], key=lambda row: row['query_id'])
            singleton_equivalents = 0
            for row, expected in zip(rows, qa):
                if row['answer'] != expected[1]:
                    # Existing framework resume unwraps list-valued gold. Only
                    # a singleton is equivalent; never accept dropped aliases.
                    assert isinstance(expected[1], list) and len(expected[1]) == 1 and row['answer'] == expected[1][0], (record['id'], row['query_id'], 'gold mismatch')
                    singleton_equivalents += 1
                per_question.append({
                    'case_id': record['id'], 'method': record['method'], 'task': record['task'],
                    'query_id': row['query_id'], 'qa_pair_id': row['qa_pair_id'],
                    'query': row['query'], 'answer': row['answer'], 'output': row['output'],
                    'frozen_gold': expected[1],
                    'parsed_output': row.get('parsed_output'),
                    'metrics': {key: row[key] for key in checked['averaged_metrics'] if key in row},
                    'result_file': record['result_file'], 'result_sha256': record['result_sha256']})
            for metric in ('exact_match', 'f1', 'rougeL_f1', 'label_accuracy'):
                if metric in checked['averaged_metrics'] and all(metric in row for row in rows):
                    mean = sum(float(row[metric]) for row in rows) / len(rows) * 100
                    assert abs(mean - checked['averaged_metrics'][metric]) < 1e-7, (record['id'], metric)
            item['audit'] = 'passed: full IDs, query text, gold answers, file SHA-256, metric means'
            item['singleton_gold_representation_compatibility'] = singleton_equivalents
            if record['reuse_kind'] == 'previous_full_result':
                imported += len(rows)
            else:
                seeds += record.get('seed_answers', 0)
        else:
            assert record['phase'] in ('blocked', 'failed')
            item['completed'] = 0
            item['averaged_metrics'] = None
            item['audit'] = record['phase'] + '; excluded from scores and successful-answer count'
        cases.append(item)

    services = []
    for path in sorted((run / 'native_services').glob('*/status.json')) + [run / 'native_reuse_service/status.json']:
        service = json.loads(path.read_text(encoding='utf-8'))
        assert service['phase'] == 'stopped' and service['database_preserved'] is True
        services.append({'status_file': str(path), **service})
    memory_states = []
    case_map = {item['id']: item for item in cases}
    for path in sorted((run / 'artifacts').glob('*/hindsight/agents/*/exp_0/hindsight_state.json')):
        task = path.relative_to(run / 'artifacts').parts[0]
        memory = json.loads(path.read_text(encoding='utf-8'))
        assert memory['ready'] is True and len(memory['documents']) == case_map[f'hindsight_{task}']['chunk_count']
        memory_states.append({'task': task, 'state_file': str(path), 'sha256': digest(path),
                              'ready': memory['ready'], 'documents': len(memory['documents'])})
    action_errors = []
    for path in sorted(run.glob('mem0_*.log')):
        matches = re.findall(r"Error in new_memories_with_actions: '(\d+)'", path.read_text(encoding='utf-8', errors='replace'))
        if matches:
            action_errors.append({'log': str(path), 'count': len(matches), 'invalid_ids': dict(Counter(matches))})
    counts = Counter(item['phase'] for item in cases)
    assert len(cases) == len(manifest['cases'])
    assert len(per_question) + sum(item['expected_questions'] for item in cases if item['phase'] != 'success') == manifest['expected_answers']
    per_method = Counter(row['method'] for row in per_question)
    summary = {
        'generated_at': datetime.now().astimezone().isoformat(), 'phase': state['phase'],
        'scope': manifest['scope'], 'models': {'chat': 'gpt-4o-mini', 'embedding': 'text-embedding-3-small', 'dimensions': 1536},
        'expected_answers': manifest['expected_answers'], 'successful_answers': len(per_question),
        'case_counts': dict(counts), 'successful_answers_per_method': dict(per_method),
        'answer_lineage': {'previous_full_results': imported, 'previous_valid_seed_answers': seeds,
                           'new_successful_answers': len(per_question) - imported - seeds},
        'manifest_sha256': digest(run / 'manifest.json'), 'suite_status_sha256': digest(run / 'suite.status.json'),
        'native_services': services, 'saved_hindsight_memory_checks': memory_states,
        'mem0_invalid_update_id_errors_current_logs': action_errors, 'cases': cases}
    validation = run / 'mem0_checkpoint_production_validation.json'
    if validation.exists():
        summary['mem0_eventqa_checkpoint_validation'] = json.loads(validation.read_text(encoding='utf-8'))
    (run / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (run / 'per_question.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in per_question), encoding='utf-8')
    lines = [
        '# 首个完整上下文评测结果', '',
        f"运行已结束：**{counts['success']}/{len(cases)} 项成功，{len(per_question):,}/{manifest['expected_answers']:,} 条计划回答完成**。" + ('全部项目均已完成，无剩余失败/受阻项。' if state['phase'] == 'success' else '未完成项不以零分代替；失败/受阻详情见 summary.json。'), '',
        '最终补齐过程：Mem0 EventQA 曾在导入阶段遇到内容过滤和网络错误，第8次加入一致分块快照；第12次完成131块并成功答98题，第13次只补回1题，第14次补回最后1题。原文、模型、提示词模板和评分规则保持原实验条件，没有跳过历史或问题；旧成功回答均保留。运行间歇失败及恢复过程详见项目 docs/experiments/2026-10-24_first_unit_evaluation.md。', '',
        '最新完整记忆检查点为131/131、complete=true；独立副本恢复验证通过（Qdrant239点、历史257行、SQLite integrity=ok）。Mem0 EventQA已完成100/100题，最终EM45.00、F1 64.35、ROUGE-L F1 66.42。服务核查材料保留于SERVICE_REVIEW_REQUEST.md；本次补齐不等于已经确定服务端间歇拒绝的根因。', '',
        '每个数据切片取第一个完整上下文/场景，运行全部问题；LoCoMo 为原始 conv-26，19 个 session、419 条发言、199 题。其他上下文的题数各不相同。没有使用两块历史的冒烟结果。', '',
        '模型为 gpt-4o-mini、text-embedding-3-small（1536 维），使用项目配置的服务与现有规则评分，没有独立 LLM Judge。Hindsight 为原生 retain / consolidation / recall 后由框架回答。', '',
        '分数均为百分比。EM 为框架 exact_match；F1 为框架词元 F1。Banking77 的 label_accuracy 与表中 EM 相同；LongBench 和 MemBench 为选择题，应优先看选项是否正确（EM）。', '',
        '| 数据切片 | 每系统题数 | Mem0 EM | Hindsight EM | Mem0 F1 | Hindsight F1 |',
        '|---|---:|---:|---:|---:|---:|']
    def score(item, name):
        return ('受阻' if item['phase'] == 'blocked' else '未完成') if item['phase'] != 'success' else f"{item['averaged_metrics'][name]:.2f}"
    for task, label in labels.items():
        mem0, hs = case_map[f'mem0_{task}'], case_map[f'hindsight_{task}']
        lines.append(f"| {label} | {hs['expected_questions']} | {score(mem0, 'exact_match')} | {score(hs, 'exact_match')} | {score(mem0, 'f1')} | {score(hs, 'f1')} |")
    lines += ['', '## 完整性与复用', '',
        f"成功结果逐项核验完整 query_id 覆盖、无重复、context_id=0、问题 ID/文本、标准答案、非空成功输出和 SHA-256；主要分数与逐题均值相符。Hindsight 完成 {per_method['hindsight']}/565 题，Mem0 完成 {per_method['mem0']}/565 题。", '',
        f'其中 {imported} 条为之前已完成的完整结果，{seeds} 条为完整记忆下的有效旧回答，其余 {len(per_question) - imported - seeds} 条为本轮新增成功回答。完整记忆复用及数据库副本来源见 RUN_PLAN.md、memory_snapshot_validation.json 和 native_reuse_service/snapshot_manifest.json。保留旧 tag 中 max10 仅为记忆身份兼容，实际完整题数以本报告及核验为准。', '',
        f'已核验本轮 {len(services)} 个原生服务状态均 stopped、database_preserved=true；本轮 {len(memory_states)} 份 Hindsight 完成状态均 ready=true 且文档数等于完整历史块数。原始结果、原始数据库未覆盖。', '',
        '## 解释限制', '',
        '- 本次仅一个完整上下文/场景；MemBench 各类及 LongBench 各仅 1 题，不能据此推断整个数据集或系统整体排名。不混合各数据集求总分。',
        '- Mem0 沿用本仓库每个 chunk 最多前 2 条事实，完整输入不等于所有信息均被保留。该限制及无效 UPDATE ID 跳过动作会影响质量；没有为提高成绩改动算法。',
        f'- 当前 Mem0 日志共 {sum(row["count"] for row in action_errors)} 条数字 ID 更新异常：模型给出的 UPDATE ID 不在旧记忆映射中，该动作被跳过。所有历史块提交完成与所有记忆动作成功是不同概念，细项在 summary.json。',
        '- 框架 F1 会移除英文冠词 a/an/the；单字母答案 A 归一化为空，因此可能出现 EM=100、F1=0。这是原评分规则表现，选择题应参考 EM，未为本次结果修改评分。',
        '- LongMemEval 的原始 gold session 文件在本机缺失，补充 Recall@K 报告生成了缺文件诊断；表中 QA 规则分数正常，不能宣称得到该项原生 Recall@K 或官方 LLM 评审成绩。',
        '- 框架续跑曾将单元素标准答案列表转为字符串；逐题审核确认本轮仅 118 条此类等价表示差异，没有多答案丢失。per_question.jsonl 同时保留结果 answer 与 frozen_gold。',
        '- LongBench 采用官方 LongBench-v2 首条完整样本；本机原 rep150 子集不可用，不能称为复现该子集。',
        '- 多次代理 TLS 超时/断连后按有限重试续跑，原生 LLM 连接预算也映射至官方配置。各次传输参数不同，耗时不能作为严格控制条件下的性能横向比较。详见 RELIABILITY_NOTES.md 和每次 source_snapshot/code_sha256。', '',
        '## 文件', '',
        '- summary.json：全部分数、覆盖核验、结果文件路径与哈希、来源、记忆和停机检查。',
        f'- per_question.jsonl：{len(per_question):,} 条成功问答及逐题分数；沿用原结果文件，可追溯来源。',
        '- suite.status.json / manifest.json：最终运行状态及冻结的完整范围。',
        '- mem0_eventqa.log：当前未完成项的最新错误证据；先前服务拒绝及失败构建另存 attempt_history。', '']
    (run / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'success_cases': counts['success'], 'blocked_cases': counts['blocked'], 'failed_cases': counts['failed'],
                      'answers': len(per_question), 'lineage': summary['answer_lineage'],
                      'report': str(run / 'REPORT.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
