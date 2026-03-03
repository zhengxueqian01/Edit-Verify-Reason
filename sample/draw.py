import os
import csv
import json
import random
import math
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent


# 读取单个 csv，返回元信息和数据
def parse_csv(csv_path: Path) -> Dict[str, Any]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = list(csv.reader(f))
    if len(reader) < 3:
        return {}

    # 第一行: 主题(title/theme)、x轴label、单位
    title_or_theme = (reader[0][0] or "").strip()
    x_label = (reader[0][1] or "").strip() if len(reader[0]) > 1 else ""
    unit = (reader[0][2] or "").strip() if len(reader[0]) > 2 else ""

    # 第二行跳过 (例如 trend,stable_rising)
    # 第三行: 第一列是 x 轴字段名(一般是 Year)，后面是系列名
    header_row = reader[2]
    if len(header_row) < 3:
        # 至少需要: Year + 2 个类别
        return {}

    x_field = header_row[0].strip()
    series_names = [c.strip() for c in header_row[1:]]

    # 第四行开始是真实数据
    data_rows = reader[3:]
    xs: List[float] = []
    series_data: Dict[str, List[float]] = {name: [] for name in series_names}

    for row in data_rows:
        if not row or len(row) < 2:
            continue
        # 读取 x
        try:
            x_val = float(row[0])
        except ValueError:
            # 非数值年份则跳过该行
            continue
        xs.append(x_val)

        # 读取各 series
        for idx, name in enumerate(series_names, start=1):
            val = None
            if idx < len(row):
                cell = row[idx].strip()
                if cell != "":
                    try:
                        val = float(cell)
                    except ValueError:
                        val = None
            series_data[name].append(val if val is not None else float("nan"))

    return {
        "title_or_theme": title_or_theme,
        "x_label": x_label or x_field,
        "unit": unit,
        "x_field": x_field,
        "series_names": series_names,
        "xs": xs,
        "series_data": series_data,
    }


def plot_line_chart(
    out_png: Path,
    out_svg: Path,
    title: str,
    x_label: str,
    y_label: str,
    xs: List[float],
    series_data: Dict[str, List[float]],
    series_plotted: List[str],
    excluded_series: List[str],
) -> None:
    plt.figure(figsize=(8, 5))

    # 真实要画的线
    for name in series_plotted:
        ys = series_data.get(name, [])
        if len(ys) != len(xs):
            continue
        plt.plot(xs, ys, marker="o", label=name)

    # Fix x-axis padding so ticks don't extend beyond actual data range.
    xs_clean = [v for v in xs if isinstance(v, (int, float)) and not (v != v)]
    if xs_clean:
        x_min, x_max = min(xs_clean), max(xs_clean)
        if x_min == x_max:
            pad = 1.0
        else:
            pad = (x_max - x_min) * 0.05
        plt.xlim(x_min - pad, x_max + pad)
        # Keep ticks within actual years but avoid overcrowding.
        if all(float(v).is_integer() for v in xs_clean):
            years = sorted(set(int(v) for v in xs_clean))
            if len(years) <= 8:
                ticks = years
            else:
                step = max(1, math.ceil(len(years) / 8))
                ticks = years[::step]
                if ticks[-1] != years[-1]:
                    ticks.append(years[-1])
            plt.xticks(ticks)

    # 计算 y 轴范围：包含没画的线的数据，但不画出来
    all_y_vals = []
    for name in list(series_plotted) + list(excluded_series):
        ys = series_data.get(name, [])
        if len(ys) != len(xs):
            continue
        for v in ys:
            if isinstance(v, (int, float)) and not (v != v):  # 过滤 NaN
                all_y_vals.append(v)
    if all_y_vals:
        y_min, y_max = min(all_y_vals), max(all_y_vals)
        # 给上下各留一点空白
        margin = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
        plt.ylim(y_min - margin, y_max + margin)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_svg)
    plt.close()


def build_metadata(
    chart_id: str,
    chart_type: str,
    title: str,
    x_label: str,
    y_label: str,
    x_field: str,
    theme: str,
    field_name: str,
    unit: str,
    series_plotted: List[str],
    excluded_series: List[str],
    xs: List[float],
    source_csv: str,
) -> Dict[str, Any]:
    xs_clean = [v for v in xs if isinstance(v, (int, float))]
    x_min = min(xs_clean) if xs_clean else None
    x_max = max(xs_clean) if xs_clean else None
    data_summary = {
        "x_min": x_min,
        "x_max": x_max,
        "num_points": len(xs),
        "num_series_plotted": len(series_plotted),
    }

    meta = {
        "id": chart_id,
        "type": chart_type,
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "x_field": x_field,
        "theme": theme,
        "field_name": field_name,
        "unit": unit,
        "series_plotted": series_plotted,
        "excluded_series": excluded_series,
        "data_summary": data_summary,
        "source_csv": source_csv,
    }
    return meta


def select_valid_csvs(csv_dir: Path, max_samples: int = 100) -> List[Path]:
    """在保持原有排序的前提下，只选择前 max_samples 个 *有效* CSV 用于生成样本。

    有效条件与原来 process_csv_folder 中一致：
    - parse_csv 能够成功解析且返回非空 dict；
    - 至少包含两个 series。
    """
    csv_files = sorted(csv_dir.glob("*.csv"))
    selected: List[Path] = []

    for csv_path in csv_files:
        parsed = parse_csv(csv_path)
        if not parsed:
            continue
        series_names: List[str] = parsed["series_names"]
        if len(series_names) < 2:
            continue
        selected.append(csv_path)
        if len(selected) >= max_samples:
            break

    return selected


def build_index_mapping_from_selected(selected_csvs: List[Path]) -> Dict[str, str]:
    """根据已筛选出的有效 CSV，按顺序分配 000-099 连续编号。

    这里不再基于所有 csv 文件构建完整 raw_id->index_id，而是：
    - 先选出最多 100 个有效 csv；
    - 按文件名（数字）排序后，依次编号 000, 001, ...。
    这样可以保证：
    - 最终只会生成 100 个样本；
    - index_id 从 000 连续到 0XX，不会出现空洞。
    """
    # 先基于 stem 做排序（与原逻辑保持一致的数字优先排序）
    def _stem_key(p: Path):
        stem = p.stem
        return int(stem) if stem.isdigit() else stem

    sorted_selected = sorted(selected_csvs, key=_stem_key)
    mapping: Dict[str, str] = {}
    for i, p in enumerate(sorted_selected):
        raw_id = p.stem
        index_id = f"{i:03d}"
        mapping[raw_id] = index_id
    return mapping


def process_csv_folder():
    base_dir = BASE_DIR
    csv_dir = base_dir / "csv"

    # 先筛选出最多 100 个有效 csv
    selected_csvs = select_valid_csvs(csv_dir, max_samples=100)
    if not selected_csvs:
        return

    # 基于这些 selected_csvs 构建 raw_id -> 连续 index_id(000..099) 映射
    id_mapping = build_index_mapping_from_selected(selected_csvs)

    random.seed()  # 使用系统随机种子

    # 仅遍历 selected_csvs，而不是所有 csv_files
    for csv_path in selected_csvs:
        parsed = parse_csv(csv_path)
        if not parsed:
            continue

        series_names: List[str] = parsed["series_names"]
        if len(series_names) < 2:
            continue

        # 随机选择排除的 series
        excluded = random.choice(series_names)
        series_plotted = [s for s in series_names if s != excluded]

        stem = csv_path.stem  # 原始 id，如 "1049"
        index_id = id_mapping.get(stem)
        if index_id is None:
            # 理论上不会发生，如果发生就跳过
            continue

        # 输出目录: task6-line/<index_id>/
        sample_dir = base_dir / index_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        png_path = sample_dir / f"{index_id}.png"
        svg_path = sample_dir / f"{index_id}.svg"

        theme = parsed["title_or_theme"]
        x_label = parsed["x_label"]
        unit = parsed["unit"]
        y_label = f"{theme} ({unit})" if unit else theme
        title = theme
        field_name = theme

        plot_line_chart(
            out_png=png_path,
            out_svg=svg_path,
            title=title,
            x_label=x_label,
            y_label=y_label,
            xs=parsed["xs"],
            series_data=parsed["series_data"],
            series_plotted=series_plotted,
            excluded_series=[excluded],
        )

        # 组织与 task5 类似的 json：写入每个类别的完整数据
        xs = parsed["xs"]
        years_list = [float(v) for v in xs]
        all_series_data: Dict[str, List[float]] = {}
        for name, vals in parsed["series_data"].items():
            all_series_data[name] = vals

        # 只在 excluded_series_data 中写入被排除系列的名字列表，不再写入数据
        excluded_series_data = [excluded]

        sample_json = {
            "id": f"sample_{index_id}",
            "raw_id": stem,
            "type": "line_chart",
            "theme": theme,
            "field_name": field_name,
            "unit": unit,
            "title": title,
            "x_label": x_label,
            "y_label": y_label,
            "x_field": parsed["x_field"],
            "years": years_list,
            "series": all_series_data,
            # "excluded_series_data": excluded_series_data,
            "series_plotted": series_plotted,
            "excluded_series": [excluded],
            "source_csv": csv_path.name,
        }

        json_path = sample_dir / f"{index_id}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(sample_json, f, ensure_ascii=False, indent=2)

        print(f"生成 task6 样本 {index_id} 于 {sample_dir}")


if __name__ == "__main__":
    process_csv_folder()
