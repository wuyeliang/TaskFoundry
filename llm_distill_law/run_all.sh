#!/usr/bin/env bash
set -euo pipefail

# 端到端蒸馏流水线入口。
#
# 执行顺序：
#   1. 从 ModelScope 下载 teacher/student 模型与示例数据集。
#   2. 把原始数据转换成 LLaMA-Factory 可直接读取的 Alpaca 格式。
#   3. 用领域数据对 4B teacher 做第一阶段 LoRA SFT。
#   4. 用微调后的 teacher 生成 response-based 蒸馏监督数据。
#   5. 用 teacher 输出对 1.7B student 做第二阶段 LoRA SFT。
#
# 默认假设脚本在宿主机 /root/taskfoundry 下执行，Docker 会把该目录挂载到
# 容器内 /workspace。若项目路径不同，可通过 WORKDIR=/path/to/project 覆盖。

IMAGE="${IMAGE:-hiyouga/llamafactory:0.9.4}"
WORKDIR="${WORKDIR:-/root/taskfoundry}"

# 第一步：下载模型和数据集。离线环境中如果文件已经存在，脚本会直接复用。
docker run --rm --gpus all --ipc=host \
  --network host \
  -v "${WORKDIR}:/workspace" \
  -w /app \
  "${IMAGE}" \
  python /workspace/scripts/download_modelscope_assets.py

# 第二步：数据清洗、去重、截断，并生成 teacher 蒸馏用 prompts。
docker run --rm --gpus all --ipc=host \
  --network host \
  -v "${WORKDIR}:/workspace" \
  -w /app \
  "${IMAGE}" \
  bash -lc 'python /workspace/scripts/prepare_domain_dataset.py \
    --output /workspace/data/domain_train_alpaca.json \
    --teacher-prompts /workspace/data/distill_prompts.json \
    --max-samples 80000 \
    --teacher-prompt-samples 30000'

# 第三步：训练 4B teacher 的 LoRA adapter。
# 具体 batch size、学习率、LoRA rank 等参数在 configs/qwen3_teacher_lora_sft.yaml。
docker run --rm --gpus all --ipc=host \
  --network host \
  -v "${WORKDIR}:/workspace" \
  -w /app \
  "${IMAGE}" \
  llamafactory-cli train /workspace/configs/qwen3_teacher_lora_sft.yaml

# 第四步：用 4B teacher + LoRA 对 distill_prompts.json 生成对齐答案。
# 输出 domain_teacher_aligned.json 是 student 蒸馏训练的监督数据。
docker run --rm --gpus all --ipc=host \
  --network host \
  -v "${WORKDIR}:/workspace" \
  -w /app \
  "${IMAGE}" \
  python /workspace/scripts/generate_teacher_responses.py \
    --base-model /workspace/modelscope/models/Qwen__Qwen3-4B-Instruct-2507 \
    --adapter /workspace/outputs/teacher-lora \
    --input /workspace/data/distill_prompts.json \
    --output /workspace/data/domain_teacher_aligned.json

# 第五步：训练 1.7B student 的 LoRA adapter。
# 这里的训练目标不是原始数据答案，而是 teacher 重新生成的 response。
docker run --rm --gpus all --ipc=host \
  --network host \
  -v "${WORKDIR}:/workspace" \
  -w /app \
  "${IMAGE}" \
  llamafactory-cli train /workspace/configs/qwen3_student_distill_lora_sft.yaml
