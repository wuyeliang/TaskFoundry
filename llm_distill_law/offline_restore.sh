#!/usr/bin/env bash
set -euo pipefail

# 离线恢复脚本。
#
# 适用场景：目标机器没有公网，不能 pip/docker pull/modelscope download。
# 输入来自 /root/taskfoundry_bundle：
#   1. llamafactory-0.9.4.tar：Docker 镜像归档。
#   2. taskfoundry_assets.tar.zst：项目代码、模型、数据集等资产。
#   3. TARGET_DIR：恢复到哪个宿主机目录，默认 /root。
#
# 恢复完成后，目标机器应出现：
#   /root/taskfoundry
#   /root/taskfoundry/modelscope/models/...
#   /root/taskfoundry/modelscope/datasets/...

IMAGE_TAR="${1:-/root/taskfoundry_bundle/llamafactory-0.9.4.tar}"
ASSET_TAR="${2:-/root/taskfoundry_bundle/taskfoundry_assets.tar.zst}"
TARGET_DIR="${3:-/root}"

# Docker 是唯一强依赖；训练、推理和数据处理都在镜像内完成。
if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but not found" >&2
  exit 1
fi

# 提前检查归档是否存在，避免 docker load 或 tar 执行到一半才失败。
if [[ ! -f "${IMAGE_TAR}" ]]; then
  echo "image tar not found: ${IMAGE_TAR}" >&2
  exit 1
fi

if [[ ! -f "${ASSET_TAR}" ]]; then
  echo "asset tar not found: ${ASSET_TAR}" >&2
  exit 1
fi

echo "[1/3] Loading Docker image from ${IMAGE_TAR}"
docker load -i "${IMAGE_TAR}"

echo "[2/3] Extracting project assets to ${TARGET_DIR}"
mkdir -p "${TARGET_DIR}"

# 优先使用 tar 的 zstd 解压参数；如果 tar 不支持，则显式调用 zstd -dc。
if command -v zstd >/dev/null 2>&1; then
  tar --use-compress-program=unzstd -xf "${ASSET_TAR}" -C "${TARGET_DIR}"
else
  zstd -dc "${ASSET_TAR}" | tar -xf - -C "${TARGET_DIR}"
fi

chmod +x "${TARGET_DIR}/taskfoundry/run_all.sh" \
  "${TARGET_DIR}/taskfoundry/run_chat.sh" \
  "${TARGET_DIR}/taskfoundry/scripts/"*.py

echo "[3/3] Verifying offline assets"

# 关键文件校验：模型 config、基础配置、Docker 镜像都存在才认为恢复成功。
test -f "${TARGET_DIR}/taskfoundry/modelscope/models/Qwen__Qwen3-4B-Instruct-2507/config.json"
test -f "${TARGET_DIR}/taskfoundry/modelscope/models/Qwen__Qwen3-1.7B/config.json"
test -f "${TARGET_DIR}/taskfoundry/data/dataset_info.json"
docker image inspect hiyouga/llamafactory:0.9.4 >/dev/null

# 给出下一步命令，方便离线恢复人员直接启动流水线。
echo "Offline restore finished."
echo "Next:"
echo "  cd ${TARGET_DIR}/taskfoundry"
echo "  ./run_all.sh"
