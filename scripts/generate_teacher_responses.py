#!/usr/bin/env python3
"""Generate response-based distillation targets with a LoRA-tuned teacher.

白盒蒸馏的第二步需要让“大一些且已领域微调”的 teacher 先回答一批 prompt，
再把这些回答作为 student 的监督标签。本脚本完成：

1. 加载 teacher 基座模型。
2. 挂载第一阶段训练得到的 LoRA adapter。
3. 读取 distill_prompts.json。
4. 批量生成 teacher response。
5. 写出 domain_teacher_aligned.json，供 student 做 LoRA SFT。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def batched(items, size: int):
    """Yield fixed-size batches without copying the whole dataset repeatedly."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    # base-model 是 teacher 的原始基座；adapter 是第一阶段 LoRA 输出目录。
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    # 生成阶段显存占用与 batch-size、max-new-tokens 强相关。
    # 高显存单卡可适当增大 batch-size；显存紧张时保持默认 2 更稳。
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=768)

    # 低温度让 teacher 输出更稳定，适合做监督标签；如果希望答案更多样可以略升高。
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    # tokenizer 负责把 messages 渲染成 Qwen3 chat template。
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # device_map="auto" 会自动把模型放到可用 GPU；单卡场景通常就是 cuda:0。
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # PeftModel.from_pretrained 不会合并权重，而是在推理时叠加 LoRA delta。
    # 这样速度略有开销，但磁盘占用小，也保留了 adapter 可替换性。
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    prompts = json.loads(Path(args.input).read_text(encoding="utf-8"))
    outputs = []
    for batch in tqdm(list(batched(prompts, args.batch_size))):
        messages = []
        for item in batch:
            content = item["instruction"]
            if item.get("input"):
                content += "\n\n补充信息：\n" + item["input"]

            # system prompt 来自数据准备阶段，用于约束 teacher 的领域回答风格。
            messages.append(
                [
                    {"role": "system", "content": item.get("system", "")},
                    {"role": "user", "content": content},
                ]
            )

        # enable_thinking=False 对应 qwen3_nothink 模板，避免生成额外思考过程。
        texts = [
            tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for msg in messages
        ]

        # padding=True 让同一批 prompt 可以并行推理；随后整体搬到模型所在设备。
        encoded = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
            )

        # generated 包含 prompt + response，这里切掉 prompt 部分，只保留 teacher 答案。
        response_tokens = generated[:, encoded["input_ids"].shape[1] :]
        responses = tokenizer.batch_decode(response_tokens, skip_special_tokens=True)
        for item, response in zip(batch, responses, strict=True):
            # 输出仍然保持 Alpaca 字段结构，student 阶段可直接被 dataset_info.json 引用。
            outputs.append(
                {
                    "instruction": item["instruction"],
                    "input": item.get("input", ""),
                    "output": response.strip(),
                    "system": item.get("system", ""),
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 缩进 JSON 便于人工抽查 teacher 质量；大文件场景也可以改成 JSONL 节省空间。
    output.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(outputs)} distilled records -> {output}")


if __name__ == "__main__":
    main()
