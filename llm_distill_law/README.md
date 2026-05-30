# TaskFoundry

中文 | [English](#english)

## 中文

TaskFoundry 是一个基于 LLaMA-Factory 的领域蒸馏工程模板，用来把较大的
teacher 模型通过 LoRA 微调和 response-based distillation 压缩到更小的
student 模型。

这个仓库刻意保留成一个“可复用骨架”：

- 用 ModelScope 下载模型和示例数据
- 将原始数据整理成 Alpaca 格式
- 完成 teacher SFT -> teacher 生成 -> student distill 的两阶段流水线
- 输出 JSON / Markdown / Excel 自动化评测报告
- 支持离线恢复和离线打包

仓库默认附带的是一套可运行示例。实际使用时，你通常只需要替换：

- 数据集来源
- 系统提示词
- 模型路径
- 评测基准
- 训练批大小和显存相关参数

## 目录结构

```text
llm_distill_law/
  configs/                  # 训练配置
  data/                     # 数据集注册信息
  docs/                     # 部署、评测、架构文档
  scripts/                  # 下载、预处理、蒸馏、评测脚本
  run_all.sh                # 端到端训练入口
  run_chat.sh               # WebChat 验证入口
  offline_restore.sh        # 离线恢复入口
  OFFLINE_MANIFEST.md       # 离线包清单说明
```

## Quick Start

前提：

- 已安装 NVIDIA 驱动和 Docker
- Docker 支持 `--gpus all`
- 本地可用 `hiyouga/llamafactory:0.9.4`

运行完整流水线：

```bash
cd /root/taskfoundry
bash run_all.sh
```

训练完成后启动 student 模型验证：

```bash
cd /root/taskfoundry
bash run_chat.sh
```

## Pipeline

`run_all.sh` 默认执行以下步骤：

1. 下载 teacher / student 模型与示例数据集
2. 将原始数据转换成 Alpaca 格式
3. 训练 teacher LoRA
4. 使用 teacher 生成蒸馏监督数据
5. 训练 student LoRA

默认输出文件：

- `data/domain_train_alpaca.json`
- `data/distill_prompts.json`
- `data/domain_teacher_aligned.json`
- `outputs/teacher-lora/`
- `outputs/student-distill-lora/`

## Evaluation

评测脚本：

`scripts/evaluate_models.py`

运行时需要自行提供 benchmark JSON 文件。每条样本建议包含：

```json
{
  "id": "case-001",
  "instruction": "Your domain question",
  "input": "",
  "reference": "Expected answer or reference answer",
  "keywords": ["keyword-a", "keyword-b"],
  "system": "Optional system prompt override"
}
```

示例命令：

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

输出包括：

- `*_results.json`
- `summary.md`
- `comparison.xlsx`

## Config Notes

当前默认配置偏向高显存单卡。如果换成显存更小的环境，建议优先降低：

- `per_device_train_batch_size`
- `per_device_eval_batch_size`
- `cutoff_len`

并打开或保留：

- `gradient_checkpointing: true`

相关文件：

- `configs/qwen3_teacher_lora_sft.yaml`
- `configs/qwen3_student_distill_lora_sft.yaml`

## Offline Usage

离线恢复入口：

```bash
bash offline_restore.sh
```

相关说明：

- `OFFLINE_MANIFEST.md`
- `docs/DEPLOYMENT.md`

## Documents

- `docs/DEPLOYMENT.md`
- `docs/TECHNICAL_ARCHITECTURE.md`
- `docs/EVALUATION.md`

## English

TaskFoundry is a domain distillation project template built on top of
LLaMA-Factory. It helps you compress a larger teacher model into a smaller
student model through LoRA tuning and response-based distillation.

The repository is intentionally kept as a reusable template:

- download models and example data from ModelScope
- convert raw data into Alpaca format
- run a two-stage teacher SFT -> teacher generation -> student distillation pipeline
- produce JSON / Markdown / Excel evaluation reports
- support offline restore and offline packaging

The bundled setup is only a runnable example. In real projects, you will usually
replace:

- the dataset source
- the system prompt
- the model paths
- the evaluation benchmark
- the batch sizes and memory-related settings

### Structure

```text
llm_distill_law/
  configs/
  data/
  docs/
  scripts/
  run_all.sh
  run_chat.sh
  offline_restore.sh
  OFFLINE_MANIFEST.md
```

### Run

```bash
cd /root/taskfoundry
bash run_all.sh
```

```bash
cd /root/taskfoundry
bash run_chat.sh
```

### Evaluation

Use `scripts/evaluate_models.py` with your own benchmark JSON file. The script
loads one or more local models, generates answers, and exports:

- per-model JSON reports
- `summary.md`
- `comparison.xlsx`

### Notes

The default configs target a higher-memory single-GPU setup. On smaller-memory
GPUs, reduce micro-batch sizes and keep gradient checkpointing enabled.
