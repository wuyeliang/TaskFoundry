# 自动化评测说明

## 1. 目的

项目内置的 `scripts/evaluate_models.py` 适合做交付前的自动化烟雾测试，用来并排比较：

- 原始 base 模型
- 蒸馏后的 student 模型
- teacher 模型

## 2. benchmark 格式

评测脚本不再绑定内置测试集，运行时需要自行提供 benchmark JSON 文件。

建议每条记录包含：

```json
{
  "id": "case-001",
  "instruction": "Question or task",
  "input": "",
  "reference": "Reference answer",
  "keywords": ["keyword-a", "keyword-b"],
  "system": "Optional system prompt override"
}
```

其中：

- `reference` 用于相似度计算
- `keywords` 用于关键词覆盖率计算
- `system` 可选；如不传则使用脚本内默认系统提示词

## 3. 运行方式

```bash
cd /root/taskfoundry
docker run --rm --gpus all --ipc=host --network host \
  -v /root/taskfoundry:/workspace \
  -w /app \
  hiyouga/llamafactory:0.9.4 \
  python /workspace/scripts/evaluate_models.py \
  --benchmark /workspace/benchmarks/your_benchmark.json \
  --output-dir /workspace/eval_outputs/manual_eval \
  --max-new-tokens 192 \
  --temperature 0.0 \
  --model base_1_7::/workspace/modelscope/models/Qwen__Qwen3-1.7B::- \
  --model student_1_7::/workspace/modelscope/models/Qwen__Qwen3-1.7B::/workspace/outputs/student-distill-lora \
  --model teacher_4b::/workspace/modelscope/models/Qwen__Qwen3-4B-Instruct-2507::/workspace/outputs/teacher-lora
```

## 4. 输出文件

`--output-dir` 下会生成：

- `*_results.json`
- `summary.md`
- `comparison.xlsx`

用途：

- `summary.md`：快速查看总体指标
- `comparison.xlsx`：逐题并排审阅
- `*_results.json`：方便程序化后处理

## 5. 当前指标

脚本默认计算：

- `avg_keyword_hit_rate`
- `avg_reference_similarity`
- `risk_hint_rate`

这些指标的定位是“快速筛查”，不是最终业务结论。

## 6. 评测建议

更稳妥的验收方式通常是：

1. 先看自动指标有没有明显退化
2. 再看 `comparison.xlsx` 做逐题对比
3. 最后让业务侧或领域专家做抽样复核

如果你在做高风险领域，建议额外加入：

- 边界题
- 拒答题
- 证据不足题
- 干扰项较多的复杂题
