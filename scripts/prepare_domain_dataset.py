#!/usr/bin/env python3
"""Convert a raw domain dataset into LLaMA-Factory Alpaca JSON files.

脚本会尽量兼容常见字段命名，保留高信号问答样本，以 `(instruction, input)`
为键去重，并把结果写成少于 100k 条的 Alpaca 记录。

本项目使用它做两类输出：

1. domain_train_alpaca.json：第一阶段 4B teacher LoRA 微调数据。
2. distill_prompts.json：只保留 instruction/input/system，供微调后的 teacher
   生成 response-based 蒸馏答案。

转换目标是 LLaMA-Factory 的 Alpaca 格式，每条数据包含：
instruction、input、output、system。
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, IterableDataset, load_dataset


DEFAULT_SYSTEM = (
    "你是一名严谨的领域助手。回答应尽量准确、清晰，"
    "说明适用条件、边界和风险；不确定时应明确说明信息不足。"
)


def normalize_text(value: Any) -> str:
    """Convert arbitrary cell values to a compact one-line string.

    原始 CSV/JSON 字段可能包含 None、换行、连续空格或制表符。这里统一压缩空白，
    可以减少重复样本，也能避免训练时出现大量无意义空白 token。
    """
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def pick_first(row: dict[str, Any], names: list[str]) -> str:
    """Return the first non-empty value among candidate field names.

    不同开源 SFT 数据集的字段命名不统一，例如问题可能叫 question/query/prompt，
    答案可能叫 answer/response/output。用候选字段列表可以兼容更多数据源。
    """
    for name in names:
        if name in row:
            text = normalize_text(row[name])
            if text:
                return text
    return ""


def row_to_alpaca(row: dict[str, Any], system_prompt: str) -> dict[str, str] | None:
    """Map one raw dataset row into one Alpaca record.

    返回 None 表示该样本质量不足或无法识别问答字段，需要跳过。过滤策略偏保守：
    宁可少一些样本，也避免把空答案、乱码、过短样本和超长异常样本带进训练。
    """
    instruction = pick_first(
        row,
        [
            "instruction",
            "question",
            "query",
            "prompt",
            "input",
            "ask",
            "user",
            "title",
        ],
    )
    answer = pick_first(
        row,
        [
            "output",
            "answer",
            "response",
            "chosen",
            "assistant",
            "content",
            "target",
        ],
    )

    # 有些数据集使用 chat messages 格式，而不是平铺的 question/answer 字段。
    # 这里提取 user/human 作为问题，assistant/gpt 作为答案。
    if not answer and "messages" in row:
        messages = row["messages"]
        if isinstance(messages, list):
            user_parts, assistant_parts = [], []
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role") or msg.get("from")
                content = normalize_text(msg.get("content") or msg.get("value"))
                if role in {"user", "human"}:
                    user_parts.append(content)
                elif role in {"assistant", "gpt"}:
                    assistant_parts.append(content)
            instruction = instruction or "\n".join(part for part in user_parts if part)
            answer = "\n".join(part for part in assistant_parts if part)

    # 基础质量控制：字段缺失、乱码、太短或过长的样本都跳过。
    if not instruction or not answer:
        return None
    if "\ufffd" in instruction or "\ufffd" in answer:
        return None
    if len(instruction) < 6 or len(answer) < 20:
        return None
    if len(instruction) > 4000 or len(answer) > 6000:
        return None

    return {
        "instruction": instruction,
        "input": "",
        "output": answer,
        "system": system_prompt,
    }


def iter_rows(dataset: DatasetDict | Dataset | IterableDataset):
    """Yield rows from any HuggingFace datasets container shape."""
    if isinstance(dataset, DatasetDict):
        for split in dataset.values():
            for row in split:
                yield row
    else:
        for row in dataset:
            yield row


def main() -> None:
    parser = argparse.ArgumentParser()
    # dataset="csv" 时从 --data-file 读取本地 CSV；也可以传 HuggingFace 或
    # ModelScope 对齐后的其他垂直领域数据集名称。
    parser.add_argument("--dataset", default="csv")
    parser.add_argument(
        "--data-file",
        default="/workspace/modelscope/datasets/AI-ModelScope__DISC-Law-SFT/DISC-Law-SFT-Pair.csv",
        help="CSV/JSON/JSONL file. The default points to the bundled example dataset.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-prompts", required=True)
    parser.add_argument("--max-samples", type=int, default=80000)
    parser.add_argument("--teacher-prompt-samples", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    args = parser.parse_args()

    # 固定随机种子保证离线复现实验时样本顺序和抽样结果一致。
    random.seed(args.seed)

    # datasets.load_dataset 会根据输入类型选择 CSV/JSON 解析器；其他名称则按数据集
    # ID 处理。默认路径来自 download_modelscope_assets.py 下载的示例数据。
    if args.dataset == "json":
        ds = load_dataset("json", data_files=args.data_file)
    elif args.dataset == "csv":
        ds = load_dataset("csv", data_files=args.data_file)
    else:
        ds = load_dataset(args.dataset)

    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for row in iter_rows(ds):
        item = row_to_alpaca(dict(row), args.system)
        if item is None:
            continue

        # 以 instruction/input 去重；同一个问题重复出现会放大训练偏置。
        key = (item["instruction"], item["input"])
        if key in seen:
            continue
        seen.add(key)
        records.append(item)

    # 打乱后截取，避免只拿到原始数据文件前部某一类题型。
    random.shuffle(records)
    records = records[: args.max_samples]

    # 蒸馏 prompts 不包含 output，因为 output 要由 LoRA 微调后的 teacher 重新生成。
    prompt_records = [
        {
            "instruction": item["instruction"],
            "input": item["input"],
            "system": item["system"],
        }
        for item in records[: args.teacher_prompt_samples]
    ]

    output = Path(args.output)
    prompt_output = Path(args.teacher_prompts)
    output.parent.mkdir(parents=True, exist_ok=True)
    prompt_output.parent.mkdir(parents=True, exist_ok=True)

    # ensure_ascii=False 保留中文可读性，便于人工抽查样本质量。
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    prompt_output.write_text(json.dumps(prompt_records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {len(records)} alpaca records -> {output}")
    print(f"wrote {len(prompt_records)} distillation prompts -> {prompt_output}")


if __name__ == "__main__":
    main()
