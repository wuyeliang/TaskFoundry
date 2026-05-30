#!/usr/bin/env python3
"""Run a benchmark against one or more local models.

这个脚本适合项目交付后的“快速验收”：

1. 读取用户提供的测试集。
2. 按顺序加载一个或多个模型（可带 LoRA）。
3. 逐题生成回答。
4. 计算轻量自动指标：
   - 关键词覆盖率
   - 与参考答案的字符级相似度
   - 是否包含风险提示用语
5. 输出 JSON 明细和 Markdown 摘要，方便继续人工复核。

注意：
这些自动指标只能做烟雾测试，不等价于严格业务正确性评估。
"""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


RISK_HINT_TOKENS = [
    "建议咨询专业律师",
    "建议咨询律师",
    "需结合具体情况",
    "结合具体事实",
    "以法院认定为准",
    "以生效裁判为准",
    "需结合证据",
]


@dataclass
class ModelSpec:
    name: str
    base_model: str
    adapter: str | None


def parse_model_spec(raw: str) -> ModelSpec:
    """Parse CLI model spec in the form name::base_model::adapter_or_dash."""
    parts = raw.split("::")
    if len(parts) != 3:
        raise ValueError(
            f"invalid --model value: {raw!r}; expected name::base_model::adapter_or_dash"
        )

    name, base_model, adapter = (part.strip() for part in parts)
    if not name or not base_model:
        raise ValueError(f"invalid --model value: {raw!r}")

    if adapter in {"", "-", "none", "null"}:
        adapter = None

    return ModelSpec(name=name, base_model=base_model, adapter=adapter)


def load_benchmark(path: Path) -> list[dict[str, object]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("benchmark file must contain a JSON list")
    return records


def build_prompt(item: dict[str, object]) -> list[dict[str, str]]:
    system_prompt = str(
        item.get(
            "system",
            "你是一名严谨的领域助手。回答应尽量准确、清晰，说明适用条件、边界和风险；"
            "不确定时应明确说明信息不足。",
        )
    )
    content = str(item["instruction"])
    if item.get("input"):
        content += "\n\n补充信息：\n" + str(item["input"])

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def keyword_hit_rate(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    hits = sum(1 for token in keywords if token and token in answer)
    return hits / len(keywords)


def similarity_score(answer: str, reference: str) -> float:
    return SequenceMatcher(None, answer, reference).ratio()


def has_risk_hint(answer: str) -> bool:
    return any(token in answer for token in RISK_HINT_TOKENS)


def sanitize_sheet_name(value: str) -> str:
    """Trim invalid Excel worksheet characters and length."""
    invalid = set('[]:*?/\\')
    cleaned = "".join("_" if ch in invalid else ch for ch in value).strip()
    cleaned = cleaned or "Sheet"
    return cleaned[:31]


def excel_column_name(index: int) -> str:
    """Convert a zero-based column index to Excel letters."""
    label = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        label = chr(65 + remainder) + label
    return label


def excel_cell(value: object) -> str:
    """Serialize a value as an inline string cell."""
    text = "" if value is None else str(value)
    return (
        '<c t="inlineStr">'
        f"<is><t xml:space=\"preserve\">{xml_escape(text)}</t></is>"
        "</c>"
    )


def excel_row(values: list[object]) -> str:
    cells = "".join(excel_cell(value) for value in values)
    return f"<row>{cells}</row>"


def write_minimal_xlsx(path: Path, sheets: list[tuple[str, list[list[object]]]]) -> None:
    """Write a small .xlsx file without third-party dependencies.

    只实现当前报告需要的最小功能：多个 worksheet、纯文本单元格。
    生成出的文件可直接被 Excel/WPS/LibreOffice 打开。
    """
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for index in range(len(sheets)):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{index + 1}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    workbook_sheets = []
    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    root_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>',
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>',
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>',
        "</Relationships>",
    ]

    for index, (sheet_name, _rows) in enumerate(sheets, start=1):
        workbook_sheets.append(
            f'<sheet name="{xml_escape(sanitize_sheet_name(sheet_name))}" sheetId="{index}" r:id="rId{index}"/>'
        )
        workbook_rels.append(
            f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>'
        )

    workbook_rels.append(
        f'<Relationship Id="rId{len(sheets) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    )
    workbook_rels.append("</Relationships>")

    workbook_xml = "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
            "<sheets>",
            *workbook_sheets,
            "</sheets>",
            "</workbook>",
        ]
    )

    styles_xml = "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>',
            '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>',
            '<borders count="1"><border/></borders>',
            '<cellStyleXfs count="1"><xf/></cellStyleXfs>',
            '<cellXfs count="1"><xf xfId="0"/></cellXfs>',
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>',
            "</styleSheet>",
        ]
    )

    core_xml = "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">',
            "<dc:title>Model Evaluation Report</dc:title>",
            "<dc:creator>Codex</dc:creator>",
            "</cp:coreProperties>",
        ]
    )

    app_xml = "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
            'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">',
            "<Application>Codex</Application>",
            "</Properties>",
        ]
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "\n".join(content_types))
        archive.writestr("_rels/.rels", "\n".join(root_rels))
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", "\n".join(workbook_rels))
        archive.writestr("xl/styles.xml", styles_xml)
        archive.writestr("docProps/core.xml", core_xml)
        archive.writestr("docProps/app.xml", app_xml)

        for index, (_sheet_name, rows) in enumerate(sheets, start=1):
            max_columns = max((len(row) for row in rows), default=1)
            dimension = f"A1:{excel_column_name(max_columns - 1)}{max(len(rows), 1)}"
            sheet_xml = [
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
                f'<dimension ref="{dimension}"/>',
                "<sheetData>",
            ]
            for row in rows:
                sheet_xml.append(excel_row(row))
            sheet_xml.extend(["</sheetData>", "</worksheet>"])
            archive.writestr(f"xl/worksheets/sheet{index}.xml", "\n".join(sheet_xml))


def build_comparison_rows(reports: list[dict[str, object]]) -> list[dict[str, object]]:
    """Pivot per-model JSON into one row per benchmark item."""
    if not reports:
        return []

    ordered_models = [str(report["summary"]["model_name"]) for report in reports]
    by_id: dict[str, dict[str, object]] = {}

    for report in reports:
        model_name = str(report["summary"]["model_name"])
        for item in report["results"]:
            row = by_id.setdefault(
                str(item["id"]),
                {
                    "id": item["id"],
                    "instruction": item["instruction"],
                    "reference": item["reference"],
                    "keywords": item["keywords"],
                },
            )
            row[f"{model_name}_answer"] = item["answer"]
            row[f"{model_name}_keyword_hit_rate"] = item["keyword_hit_rate"]
            row[f"{model_name}_reference_similarity"] = item["reference_similarity"]
            row[f"{model_name}_has_risk_hint"] = item["has_risk_hint"]

    rows = []
    for row_id in sorted(by_id):
        row = by_id[row_id]
        row["model_order"] = ordered_models
        rows.append(row)
    return rows


def generate_answers(
    spec: ModelSpec,
    benchmark: list[dict[str, object]],
    template: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, object]:
    """Generate answers and compute simple summary metrics for one model."""
    tokenizer = AutoTokenizer.from_pretrained(spec.base_model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        spec.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = base_model
    if spec.adapter:
        model = PeftModel.from_pretrained(base_model, spec.adapter)
    model.eval()

    rows: list[dict[str, object]] = []
    keyword_scores: list[float] = []
    similarity_scores: list[float] = []
    risk_hits = 0

    for item in tqdm(benchmark, desc=f"evaluating {spec.name}", leave=False):
        messages = build_prompt(item)
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False if template == "qwen3_nothink" else True,
        )
        encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
            )

        answer = tokenizer.decode(
            generated[0, encoded["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()

        keywords = [str(token) for token in item.get("keywords", [])]
        reference = str(item.get("reference", ""))
        kw_score = keyword_hit_rate(answer, keywords)
        sim_score = similarity_score(answer, reference)
        risk_flag = has_risk_hint(answer)

        keyword_scores.append(kw_score)
        similarity_scores.append(sim_score)
        risk_hits += int(risk_flag)

        rows.append(
            {
                "id": item["id"],
                "instruction": item["instruction"],
                "reference": reference,
                "keywords": keywords,
                "answer": answer,
                "keyword_hit_rate": round(kw_score, 4),
                "reference_similarity": round(sim_score, 4),
                "has_risk_hint": risk_flag,
                "answer_length": len(answer),
            }
        )

    # 主动释放显存，便于顺序评测多个模型。
    del model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total = len(rows) or 1
    summary = {
        "model_name": spec.name,
        "base_model": spec.base_model,
        "adapter": spec.adapter,
        "template": template,
        "num_examples": len(rows),
        "avg_keyword_hit_rate": round(sum(keyword_scores) / total, 4),
        "avg_reference_similarity": round(sum(similarity_scores) / total, 4),
        "risk_hint_rate": round(risk_hits / total, 4),
    }
    return {"summary": summary, "results": rows}


def write_markdown_report(path: Path, benchmark_name: str, reports: list[dict[str, object]]) -> None:
    lines = [
        f"# {benchmark_name} 评测摘要",
        "",
        "| 模型 | 样本数 | 关键词覆盖率 | 参考答案相似度 | 风险提示命中率 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for report in reports:
        summary = report["summary"]
        lines.append(
            "| {model_name} | {num_examples} | {avg_keyword_hit_rate:.4f} | {avg_reference_similarity:.4f} | {risk_hint_rate:.4f} |".format(
                **summary
            )
        )

    lines.extend(["", "## 说明", ""])
    lines.append("这些指标适合做自动化烟雾测试，仍建议结合 JSON 明细进行人工复核。")

    comparison_rows = build_comparison_rows(reports)
    if comparison_rows:
        lines.extend(["", "## 逐题对比", ""])
        for row in comparison_rows:
            lines.append(f"### {row['id']}")
            lines.append("")
            lines.append(f"**问题**：{row['instruction']}")
            lines.append("")
            lines.append(f"**参考答案**：{row['reference']}")
            lines.append("")
            lines.append(f"**关键词**：{', '.join(row['keywords'])}")
            lines.append("")
            for model_name in row["model_order"]:
                answer = row.get(f"{model_name}_answer", "")
                keyword_rate = row.get(f"{model_name}_keyword_hit_rate", "")
                similarity = row.get(f"{model_name}_reference_similarity", "")
                risk_flag = row.get(f"{model_name}_has_risk_hint", "")
                lines.append(f"#### {model_name}")
                lines.append("")
                lines.append(
                    f"- 关键词覆盖率：`{keyword_rate}` 参考答案相似度：`{similarity}` 风险提示：`{risk_flag}`"
                )
                lines.append(f"- 回答：{answer}")
                lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_excel_report(path: Path, benchmark_name: str, reports: list[dict[str, object]]) -> None:
    """Write summary and per-question comparison into a minimal .xlsx workbook."""
    summary_rows: list[list[object]] = [
        ["benchmark", benchmark_name],
        [],
        ["model_name", "num_examples", "avg_keyword_hit_rate", "avg_reference_similarity", "risk_hint_rate"],
    ]
    for report in reports:
        summary = report["summary"]
        summary_rows.append(
            [
                summary["model_name"],
                summary["num_examples"],
                summary["avg_keyword_hit_rate"],
                summary["avg_reference_similarity"],
                summary["risk_hint_rate"],
            ]
        )

    comparison_rows = build_comparison_rows(reports)
    detail_header = ["id", "instruction", "reference", "keywords"]
    model_order = comparison_rows[0]["model_order"] if comparison_rows else []
    for model_name in model_order:
        detail_header.extend(
            [
                f"{model_name}_keyword_hit_rate",
                f"{model_name}_reference_similarity",
                f"{model_name}_has_risk_hint",
                f"{model_name}_answer",
            ]
        )

    detail_rows: list[list[object]] = [detail_header]
    for row in comparison_rows:
        detail_row: list[object] = [
            row["id"],
            row["instruction"],
            row["reference"],
            ", ".join(row["keywords"]),
        ]
        for model_name in row["model_order"]:
            detail_row.extend(
                [
                    row.get(f"{model_name}_keyword_hit_rate", ""),
                    row.get(f"{model_name}_reference_similarity", ""),
                    row.get(f"{model_name}_has_risk_hint", ""),
                    row.get(f"{model_name}_answer", ""),
                ]
            )
        detail_rows.append(detail_row)

    sheets = [
        ("summary", summary_rows),
        ("comparison", detail_rows),
    ]
    write_minimal_xlsx(path, sheets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        required=True,
        help="JSON benchmark file containing instruction/reference/keywords.",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/eval_outputs",
        help="Directory for JSON detail files and Markdown summary.",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec in the form name::base_model::adapter_or_dash",
    )
    parser.add_argument("--template", default="qwen3_nothink")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = load_benchmark(benchmark_path)

    reports: list[dict[str, object]] = []
    for raw_spec in args.model:
        spec = parse_model_spec(raw_spec)
        report = generate_answers(
            spec=spec,
            benchmark=benchmark,
            template=args.template,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        reports.append(report)

        model_output = output_dir / f"{spec.name}_results.json"
        model_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote model report -> {model_output}")

    summary_path = output_dir / "summary.md"
    write_markdown_report(summary_path, benchmark_path.stem, reports)
    print(f"wrote markdown summary -> {summary_path}")

    excel_path = output_dir / "comparison.xlsx"
    write_excel_report(excel_path, benchmark_path.stem, reports)
    print(f"wrote excel comparison -> {excel_path}")


if __name__ == "__main__":
    main()
