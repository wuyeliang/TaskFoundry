#!/usr/bin/env python3
"""Download the models and example dataset required by the pipeline.

这个脚本负责把训练流程的“原始依赖”落到本地目录：

1. teacher 模型：用于第一阶段 LoRA 微调。
2. student 模型：用于第二阶段 response-based 蒸馏。
3. 示例数据集文件：后续会被转换成 LLaMA-Factory 的 Alpaca 格式。

脚本默认在 Docker 容器里的 /workspace/modelscope 下工作。宿主机通过
run_all.sh 把项目目录挂载到 /workspace，因此下载结果实际会保存在项目目录
的 modelscope/ 子目录，便于后续离线打包和复用。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
from modelscope import snapshot_download
from modelscope.hub.api import HubApi


def download_dataset_file(repo_id: str, file_name: str, output_dir: Path) -> Path:
    """Download one dataset file from a ModelScope dataset repository.

    ModelScope 的模型可以直接使用 snapshot_download；数据集单文件下载这里用
    HubApi 先拿到真实下载 URL，再用 requests 流式写入本地，避免一次性把大文件
    读入内存。
    """
    namespace, dataset_name = repo_id.split("/", 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / file_name

    # 幂等保护：文件已存在且非空时直接复用，方便断点重跑整个 run_all.sh。
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"dataset exists: {output_path}")
        return output_path

    api = HubApi()
    url = api.get_dataset_file_url(
        file_name=file_name,
        dataset_name=dataset_name,
        namespace=namespace,
        revision="master",
    )
    print(f"downloading dataset {repo_id}/{file_name} -> {output_path}")

    # stream=True 可以边下载边落盘，适合在内存较小的离线准备机上执行。
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    print(f"dataset downloaded: {output_path} ({output_path.stat().st_size} bytes)")
    return output_path


def download_model(model_id: str, output_root: Path) -> Path:
    """Download a model snapshot and expose it through a stable local path.

    snapshot_download 通常会使用自己的缓存目录结构。为了让配置文件可以稳定引用
    /workspace/modelscope/models/Qwen__xxx，这里把模型名里的 "/" 替换为 "__"。
    如果 ModelScope 返回的真实缓存目录不是这个目标目录，就创建符号链接。
    """
    target = output_root / model_id.replace("/", "__")

    # config.json 是 Transformers 模型目录的核心文件；它存在通常说明模型已就绪。
    if (target / "config.json").exists():
        print(f"model exists: {target}")
        return target

    print(f"downloading model {model_id} -> {target}")
    path = snapshot_download(model_id, cache_dir=str(output_root))
    downloaded = Path(path)

    # 统一路径，避免 YAML 中引用 ModelScope 的内部 cache hash 目录。
    if downloaded != target:
        if target.exists():
            return target
        target.symlink_to(downloaded, target_is_directory=True)
    print(f"model downloaded: {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    # root 是容器内路径；宿主机项目目录会被 run_all.sh 挂载成 /workspace。
    parser.add_argument("--root", default="/workspace/modelscope")
    parser.add_argument("--teacher-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--student-model", default="Qwen/Qwen3-1.7B")
    # 默认数据集只是一个可运行示例；公开模板可以替换成任意垂直领域数据集。
    parser.add_argument("--dataset", default="AI-ModelScope/DISC-Law-SFT")
    parser.add_argument("--dataset-file", default="DISC-Law-SFT-Pair.csv")
    args = parser.parse_args()

    root = Path(args.root)
    models_root = root / "models"
    datasets_root = root / "datasets" / args.dataset.replace("/", "__")

    # 下载顺序没有强依赖，但顺序执行能让日志更清楚，失败时也容易定位卡在哪个资产。
    teacher = download_model(args.teacher_model, models_root)
    student = download_model(args.student_model, models_root)
    dataset = download_dataset_file(args.dataset, args.dataset_file, datasets_root)

    # 输出固定标签，便于部署时从日志里快速确认最终路径。
    print("ASSET_PATHS")
    print(f"teacher_model={teacher}")
    print(f"student_model={student}")
    print(f"dataset_file={dataset}")


if __name__ == "__main__":
    main()
