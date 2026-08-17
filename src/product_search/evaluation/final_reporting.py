"""Atomic final-evaluation reports and dependency-free SVG charts."""

from __future__ import annotations

import csv
import html
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

SYSTEM_LABELS = {
    "lexical": "Lexical",
    "semantic": "Semantic",
    "hybrid": "Hybrid",
    "reranked_hybrid": "Reranked hybrid",
}

FINAL_OUTPUT_FILENAMES = (
    "final_test_metrics.json",
    "final_test_metrics.csv",
    "final_test_per_query_metrics.csv",
    "final_comparison.md",
    "final_engine.json",
    "final_system_comparison.svg",
    "final_ndcg_distribution.svg",
    "final_latency_comparison.svg",
)


def write_final_report_files(
    report: Mapping[str, object],
    aggregate_rows: Sequence[Mapping[str, object]],
    per_query_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> dict[str, str]:
    """Write the immutable final report family after every evaluation has succeeded."""

    resolved = output_dir.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    paths = {name: resolved / name for name in FINAL_OUTPUT_FILENAMES}
    existing = sorted(name for name, path in paths.items() if path.exists())
    if existing:
        raise FileExistsError(
            f"final evaluation outputs already exist and are immutable: {existing}"
        )

    systems = _mapping(report["systems"])
    final_engine = _mapping(report["final_engine"])
    _write_json_atomic(paths["final_test_metrics.json"], report)
    _write_csv_atomic(paths["final_test_metrics.csv"], aggregate_rows)
    _write_csv_atomic(paths["final_test_per_query_metrics.csv"], per_query_rows)
    _write_text_atomic(paths["final_comparison.md"], _comparison_markdown(report, aggregate_rows))
    _write_json_atomic(paths["final_engine.json"], final_engine)
    _write_text_atomic(
        paths["final_system_comparison.svg"],
        _grouped_bar_svg(
            title="Held-out judged-candidate ranking quality",
            y_label="Macro metric",
            values={
                system: {
                    "nDCG@5": _metric(systems, system, "ndcg_at_5"),
                    "nDCG@10": _metric(systems, system, "ndcg_at_10"),
                }
                for system in systems
            },
        ),
    )
    _write_text_atomic(
        paths["final_ndcg_distribution.svg"],
        _box_plot_svg(
            title="Per-query judged-candidate nDCG@10 distribution",
            values=_per_query_values(per_query_rows, "ndcg_at_10"),
        ),
    )
    _write_text_atomic(
        paths["final_latency_comparison.svg"],
        _grouped_bar_svg(
            title="Warm-process full-catalog query latency",
            y_label="Milliseconds",
            values={
                system: {
                    "Median": _latency(systems, system, "median"),
                    "p95": _latency(systems, system, "p95"),
                }
                for system in systems
            },
        ),
    )
    return {key.removesuffix(Path(key).suffix): path.name for key, path in paths.items()}


def _comparison_markdown(
    report: Mapping[str, object], aggregate_rows: Sequence[Mapping[str, object]]
) -> str:
    final_engine = _mapping(report["final_engine"])
    hardware = _mapping(report["hardware"])
    lines = [
        "# Final held-out retrieval comparison",
        "",
        (
            "This is the one-time held-out test comparison. All configurations were frozen from "
            "train/validation work before test-query metrics were calculated."
        ),
        "",
        "| System | nDCG@5 | nDCG@10 | P@5 | P@10 | R@5 | R@10 | MRR@10 | Median ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {label} | {ndcg5:.6f} | {ndcg10:.6f} | {p5:.6f} | {p10:.6f} | "
            "{r5:.6f} | {r10:.6f} | {mrr:.6f} | {median:.3f} | {p95:.3f} |".format(
                label=SYSTEM_LABELS[str(row["system"])],
                ndcg5=_as_float(row["ndcg_at_5"]),
                ndcg10=_as_float(row["ndcg_at_10"]),
                p5=_as_float(row["precision_at_5"]),
                p10=_as_float(row["precision_at_10"]),
                r5=_as_float(row["recall_at_5"]),
                r10=_as_float(row["recall_at_10"]),
                mrr=_as_float(row["mrr_at_10"]),
                median=_as_float(row["median_latency_ms"]),
                p95=_as_float(row["p95_latency_ms"]),
            )
        )
    lines.extend(
        [
            "",
            f"Selected default: **{SYSTEM_LABELS[str(final_engine['system'])]}** "
            f"(`{final_engine['selected_search_mode']}`).",
            "",
            str(final_engine["selection_rationale"]),
            "",
            "Metrics in the table are macro-averaged over the controlled judged-candidate test "
            "evaluation. Exact and Partial are binary-relevant. Latency is warm-process, "
            "end-to-end `search(query, top_k=10)` over the full catalog; artifact/model loading "
            "is excluded.",
            "",
            "Full-catalog precision and nDCG are intentionally omitted because WANDS does not "
            "judge every retrieved product. Unjudged products remain unknown, not irrelevant.",
            "",
            "Charts: `final_system_comparison.svg`, `final_ndcg_distribution.svg`, and "
            "`final_latency_comparison.svg`.",
            "",
            "Hardware: {processor}; {logical} logical CPUs; {platform}.".format(
                processor=hardware["processor"],
                logical=hardware["logical_cpu_count"],
                platform=hardware["platform"],
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _grouped_bar_svg(*, title: str, y_label: str, values: Mapping[str, Mapping[str, float]]) -> str:
    width, height = 920, 520
    left, right, top, bottom = 90, 30, 65, 105
    plot_width = width - left - right
    plot_height = height - top - bottom
    systems = list(values)
    series = list(next(iter(values.values()))) if values else []
    maximum = max((value for group in values.values() for value in group.values()), default=1.0)
    scale_max = maximum * 1.12 if maximum > 0 else 1.0
    colors = ("#2563eb", "#f59e0b", "#10b981", "#8b5cf6")
    group_width = plot_width / max(1, len(systems))
    bar_width = min(62.0, group_width * 0.62 / max(1, len(series)))
    parts = [_svg_header(width, height), _svg_text(width / 2, 30, title, 20, anchor="middle")]
    for tick in range(6):
        fraction = tick / 5
        y = top + plot_height * (1 - fraction)
        tick_value = scale_max * fraction
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" '
            'stroke="#d1d5db" stroke-width="1" />'
        )
        parts.append(_svg_text(left - 10, y + 4, f"{tick_value:.2f}", 12, anchor="end"))
    for system_index, system in enumerate(systems):
        center = left + group_width * (system_index + 0.5)
        offset = -bar_width * len(series) / 2
        for series_index, series_name in enumerate(series):
            value = float(values[system][series_name])
            bar_height = plot_height * value / scale_max
            x = center + offset + series_index * bar_width
            y = top + plot_height - bar_height
            color = colors[series_index % len(colors)]
            parts.append(
                f'<rect x="{x + 2:.2f}" y="{y:.2f}" width="{bar_width - 4:.2f}" '
                f'height="{bar_height:.2f}" fill="{color}" rx="2" />'
            )
            parts.append(_svg_text(x + bar_width / 2, y - 6, f"{value:.3f}", 10, anchor="middle"))
        parts.append(
            _svg_text(center, height - bottom + 24, SYSTEM_LABELS[system], 12, anchor="middle")
        )
    legend_x = left
    for series_index, series_name in enumerate(series):
        x = legend_x + series_index * 150
        parts.append(
            f'<rect x="{x}" y="{height - 35}" width="14" height="14" '
            f'fill="{colors[series_index % len(colors)]}" />'
        )
        parts.append(_svg_text(x + 20, height - 23, series_name, 12))
    parts.append(_svg_text(18, top + plot_height / 2, y_label, 12, anchor="middle", rotate=-90))
    parts.append("</svg>\n")
    return "".join(parts)


def _box_plot_svg(*, title: str, values: Mapping[str, Sequence[float]]) -> str:
    width, height = 920, 500
    left, right, top, bottom = 90, 30, 65, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    systems = list(values)
    group_width = plot_width / max(1, len(systems))
    parts = [_svg_header(width, height), _svg_text(width / 2, 30, title, 20, anchor="middle")]
    for tick in range(6):
        metric = tick / 5
        y = top + plot_height * (1 - metric)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" '
            'stroke="#d1d5db" stroke-width="1" />'
        )
        parts.append(_svg_text(left - 10, y + 4, f"{metric:.1f}", 12, anchor="end"))
    colors = ("#2563eb", "#f59e0b", "#10b981", "#8b5cf6")
    for index, system in enumerate(systems):
        data = np.asarray(values[system], dtype=np.float64)
        if data.size == 0:
            continue
        minimum, q1, median, q3, maximum = np.percentile(data, [0, 25, 50, 75, 100])
        center = left + group_width * (index + 0.5)
        box_width = min(100.0, group_width * 0.48)

        def y_position(value: float) -> float:
            return top + plot_height * (1 - value)

        color = colors[index % len(colors)]
        parts.append(
            f'<line x1="{center:.2f}" y1="{y_position(maximum):.2f}" '
            f'x2="{center:.2f}" y2="{y_position(minimum):.2f}" stroke="{color}" stroke-width="2" />'
        )
        parts.append(
            f'<rect x="{center - box_width / 2:.2f}" y="{y_position(q3):.2f}" '
            f'width="{box_width:.2f}" height="{y_position(q1) - y_position(q3):.2f}" '
            f'fill="{color}" fill-opacity="0.25" stroke="{color}" stroke-width="2" />'
        )
        parts.append(
            f'<line x1="{center - box_width / 2:.2f}" y1="{y_position(median):.2f}" '
            f'x2="{center + box_width / 2:.2f}" y2="{y_position(median):.2f}" '
            f'stroke="{color}" stroke-width="3" />'
        )
        for endpoint in (minimum, maximum):
            parts.append(
                f'<line x1="{center - box_width / 4:.2f}" y1="{y_position(endpoint):.2f}" '
                f'x2="{center + box_width / 4:.2f}" y2="{y_position(endpoint):.2f}" '
                f'stroke="{color}" stroke-width="2" />'
            )
        parts.append(
            _svg_text(center, height - bottom + 28, SYSTEM_LABELS[system], 12, anchor="middle")
        )
    parts.append(_svg_text(18, top + plot_height / 2, "nDCG@10", 12, anchor="middle", rotate=-90))
    parts.append("</svg>\n")
    return "".join(parts)


def _per_query_values(
    rows: Sequence[Mapping[str, object]], metric_name: str
) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["system"]), []).append(_as_float(row[metric_name]))
    return grouped


def _metric(systems: Mapping[str, object], system: str, metric_name: str) -> float:
    judged = _mapping(_mapping(systems[system])["judged_candidate_evaluation"])
    return float(judged[metric_name])


def _latency(systems: Mapping[str, object], system: str, statistic: str) -> float:
    full_catalog = _mapping(_mapping(systems[system])["full_catalog_known_relevant_evaluation"])
    latency = _mapping(full_catalog["latency_ms_at_10"])
    return float(latency[statistic])


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        '<rect width="100%" height="100%" fill="#ffffff" />'
    )


def _svg_text(
    x: float,
    y: float,
    value: object,
    size: int,
    *,
    anchor: str = "start",
    rotate: int | None = None,
) -> str:
    transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Segoe UI,Arial,sans-serif" '
        f'font-size="{size}" fill="#111827" text-anchor="{anchor}"{transform}>'
        f"{html.escape(str(value))}</text>"
    )


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    temporary = path.with_name(f"{path.name}.part")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output_file:
            if fieldnames:
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f"{path.name}.part")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _mapping(value: object) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], value)


def _as_float(value: object) -> float:
    return float(cast(Any, value))
