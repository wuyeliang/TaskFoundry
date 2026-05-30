#!/usr/bin/env bash
set -euo pipefail

# 启动 LLaMA-Factory WebChat，用于手动验证蒸馏后的 1.7B student。
#
# 默认加载：
#   - 基座模型：/workspace/modelscope/models/Qwen__Qwen3-1.7B
#   - LoRA adapter：/workspace/outputs/student-distill-lora
#
# 如果需要验证 4B teacher 或其他 adapter，可以在命令前覆盖 MODEL/ADAPTER：
#   MODEL=/workspace/modelscope/models/Qwen__Qwen3-4B-Instruct-2507 \
#   ADAPTER=/workspace/outputs/teacher-lora \
#   ./run_chat.sh

IMAGE="${IMAGE:-hiyouga/llamafactory:0.9.4}"
WORKDIR="${WORKDIR:-/root/taskfoundry}"
MODEL="${MODEL:-/workspace/modelscope/models/Qwen__Qwen3-1.7B}"
ADAPTER="${ADAPTER:-/workspace/outputs/student-distill-lora}"

# -it 是交互式 webchat 所需；--network host 让容器服务直接暴露到宿主机网络。
# qwen3_nothink 与训练模板保持一致，避免推理时出现训练阶段没见过的格式差异。
docker run --rm -it --gpus all --ipc=host \
  --network host \
  -v "${WORKDIR}:/workspace" \
  -w /app \
  "${IMAGE}" \
  llamafactory-cli webchat \
    --model_name_or_path "${MODEL}" \
    --adapter_name_or_path "${ADAPTER}" \
    --template qwen3_nothink \
    --finetuning_type lora \
    --trust_remote_code true
