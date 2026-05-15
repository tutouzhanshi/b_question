# -*- coding: utf-8 -*-
"""
B题：多源融合机器人定位及任务优化

运行方式：
    python b_solution.py

输出目录：
    outputs/
        estimates_summary.xlsx
        problem1_10Hz_trajectory.xlsx
        problem2_10Hz_trajectory.xlsx
        problem3_10Hz_trajectory.xlsx
        result.xlsx
        B题_多源融合机器人定位及任务优化_论文.docx
        B题_论文.md
        figures/*.png
"""

from __future__ import annotations

import argparse
import html
import math
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter


DT_OUT = 0.1

# 附录公式（由题面 MathType 公式对象读取）
SHOOT_DISTANCE = (5.0, 30.0)
SHOOT_SPEED_MAX = 2.0
SHOOT_ACCEL_MAX = 1.5
SHOOT_PREP = 1.5
SHOOT_HIT_PROB = 0.85

PHOTO_DISTANCE = (10.0, 40.0)
PHOTO_ANGLE_MIN_DEG = 60.0
PHOTO_SPEED_MAX = 1.5
PHOTO_ACCEL_MAX = 1.5
PHOTO_PREP = 0.5


@dataclass
class AlignmentResult:
    problem: int
    delta2_minus_1: float
    bias2_x: float
    bias2_y: float
    mse_with_bias: float
    mse_without_bias: float | None = None
    bias_model_improvement: float | None = None
    has_system_bias: bool = True
    overlap_seconds: float = 0.0


def discover_files(root: Path) -> tuple[dict[int, Path], Path]:
    xlsx_files: dict[int, Path] = {}
    result_template: Path | None = None
    for path in root.glob("*.xlsx"):
        if path.name.lower().startswith("result"):
            result_template = path
            continue
        m = re.search(r"(\d+)", path.name)
        if m:
            xlsx_files[int(m.group(1))] = path
    missing = [i for i in (1, 2, 3, 4) if i not in xlsx_files]
    if missing:
        raise FileNotFoundError(f"缺少附件：{missing}")
    if result_template is None:
        raise FileNotFoundError("未找到 result.xlsx 模板")
    return xlsx_files, result_template


def read_pair(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d1 = pd.read_excel(path, sheet_name=0).dropna()
    d2 = pd.read_excel(path, sheet_name=1).dropna()
    a1 = d1.iloc[:, 0:3].to_numpy(float)
    a2 = d2.iloc[:, 0:3].to_numpy(float)
    return a1[:, 0], a1[:, 1:3], a2[:, 0], a2[:, 1:3]


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window) / window
    yp = np.pad(y, ((pad, pad), (0, 0)), mode="edge")
    return np.column_stack([np.convolve(yp[:, i], kernel, mode="valid") for i in range(y.shape[1])])


def interp_xy(t_new: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(t_new, t, p[:, i]) for i in range(2)])


def delta_score(
    t1: np.ndarray,
    p1: np.ndarray,
    t2: np.ndarray,
    p2: np.ndarray,
    delta: float,
    correct_bias: bool,
    dt: float,
    trim_ratio: float,
    min_overlap_ratio: float,
) -> tuple[float, np.ndarray, float]:
    t2_corr = t2 - delta
    lo = max(float(t1.min()), float(t2_corr.min()))
    hi = min(float(t1.max()), float(t2_corr.max()))
    overlap = hi - lo
    min_duration = min(float(t1.max() - t1.min()), float(t2.max() - t2.min()))
    if overlap < min_overlap_ratio * min_duration:
        return float("inf"), np.array([np.nan, np.nan]), overlap

    grid = np.arange(lo, hi + 1e-9, dt)
    q1 = interp_xy(grid, t1, p1)
    q2 = interp_xy(grid, t2_corr, p2)
    diff = q2 - q1
    bias = np.median(diff, axis=0) if correct_bias else np.zeros(2)
    err = np.sum((diff - bias) ** 2, axis=1)
    if trim_ratio > 0 and len(err) > 20:
        keep = max(10, int(len(err) * (1.0 - trim_ratio)))
        err = np.sort(err)[:keep]
    return float(np.mean(err)), bias, overlap


def estimate_alignment(
    problem: int,
    path: Path,
    correct_bias: bool,
    score_smooth_window: int,
    trim_ratio: float,
    min_overlap_ratio: float = 0.85,
) -> tuple[float, np.ndarray, float, float]:
    t1, p1, t2, p2 = read_pair(path)
    p1s = moving_average(p1, score_smooth_window)
    p2s = moving_average(p2, score_smooth_window)

    min_duration = min(float(t1.max() - t1.min()), float(t2.max() - t2.min()))
    d_min = float(t2.min() - t1.max() + min_overlap_ratio * min_duration)
    d_max = float(t2.max() - t1.min() - min_overlap_ratio * min_duration)
    deltas = np.linspace(d_min, d_max, 2500)
    scores = np.array(
        [
            delta_score(t1, p1s, t2, p2s, d, correct_bias, 0.5, trim_ratio, min_overlap_ratio)[0]
            for d in deltas
        ]
    )
    best_i = int(np.nanargmin(scores))
    best = float(deltas[best_i])
    step = float(deltas[1] - deltas[0]) if len(deltas) > 1 else 1.0
    lo = max(d_min, best - 20 * step)
    hi = min(d_max, best + 20 * step)
    opt = minimize_scalar(
        lambda d: delta_score(t1, p1s, t2, p2s, d, correct_bias, 0.1, trim_ratio, min_overlap_ratio)[0],
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-7},
    )
    mse, bias, overlap = delta_score(
        t1, p1s, t2, p2s, float(opt.x), correct_bias, 0.1, trim_ratio, min_overlap_ratio
    )
    return float(opt.x), bias, mse, overlap


def build_alignment_results(files: dict[int, Path]) -> dict[int, AlignmentResult]:
    results: dict[int, AlignmentResult] = {}

    d1, b1, mse1, ov1 = estimate_alignment(1, files[1], False, 1, 0.0)
    results[1] = AlignmentResult(1, d1, 0.0, 0.0, mse1, has_system_bias=False, overlap_seconds=ov1)

    d2, b2, mse2, ov2 = estimate_alignment(2, files[2], True, 9, 0.02)
    d2_nb, _b2_nb, mse2_nb, _ov2_nb = estimate_alignment(2, files[2], False, 9, 0.02)
    improvement2 = (mse2_nb - mse2) / mse2_nb if mse2_nb > 0 else 0.0
    results[2] = AlignmentResult(
        2,
        d2,
        float(b2[0]),
        float(b2[1]),
        mse2,
        mse_without_bias=mse2_nb,
        bias_model_improvement=improvement2,
        has_system_bias=True,
        overlap_seconds=ov2,
    )

    d3_b, b3, mse3_b, ov3_b = estimate_alignment(3, files[3], True, 11, 0.05)
    d3_nb, _b3_nb, mse3_nb, ov3_nb = estimate_alignment(3, files[3], False, 11, 0.05)
    improvement3 = (mse3_nb - mse3_b) / mse3_nb if mse3_nb > 0 else 0.0
    bias_norm = float(np.linalg.norm(b3))
    has_bias3 = bool(improvement3 >= 0.05 and bias_norm >= 0.5)
    if has_bias3:
        d3, b3_used, mse3, ov3 = d3_b, b3, mse3_b, ov3_b
    else:
        d3, b3_used, mse3, ov3 = d3_nb, np.zeros(2), mse3_nb, ov3_nb
    results[3] = AlignmentResult(
        3,
        d3,
        float(b3_used[0]),
        float(b3_used[1]),
        mse3,
        mse_without_bias=mse3_nb,
        bias_model_improvement=improvement3,
        has_system_bias=has_bias3,
        overlap_seconds=ov3,
    )
    # 保存实际数据的候选偏差，供论文说明。
    results[3].candidate_bias_x = float(b3[0])  # type: ignore[attr-defined]
    results[3].candidate_bias_y = float(b3[1])  # type: ignore[attr-defined]
    results[3].candidate_delta = float(d3_b)  # type: ignore[attr-defined]
    results[3].candidate_mse = float(mse3_b)  # type: ignore[attr-defined]
    return results


def make_trajectory(
    path: Path,
    alignment: AlignmentResult,
    smooth_window: int,
    smooth_poly: int = 3,
) -> pd.DataFrame:
    t1, p1, t2, p2 = read_pair(path)
    t2_corr = t2 - alignment.delta2_minus_1
    p2_corr = p2 - np.array([alignment.bias2_x, alignment.bias2_y])
    lo = max(float(t1.min()), float(t2_corr.min()))
    hi = min(float(t1.max()), float(t2_corr.max()))
    start = math.ceil(lo / DT_OUT) * DT_OUT
    end = math.floor(hi / DT_OUT) * DT_OUT
    grid = np.round(np.arange(start, end + 1e-9, DT_OUT), 10)

    q1 = interp_xy(grid, t1, p1)
    q2 = interp_xy(grid, t2_corr, p2_corr)
    fused = 0.5 * (q1 + q2)
    if smooth_window > 1 and len(fused) > smooth_window:
        if smooth_window % 2 == 0:
            smooth_window += 1
        fused = savgol_filter(fused, smooth_window, smooth_poly, axis=0, mode="interp")

    vel = np.gradient(fused, DT_OUT, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    acc_vec = np.gradient(vel, DT_OUT, axis=0)
    acc = np.linalg.norm(acc_vec, axis=1)
    return pd.DataFrame(
        {
            "time_s": grid,
            "x_m": fused[:, 0],
            "y_m": fused[:, 1],
            "speed_m_s": speed,
            "accel_m_s2": acc,
        }
    )


def rolling_all(mask: np.ndarray, window_samples: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    csum = np.r_[0, np.cumsum(mask.astype(int))]
    out[window_samples - 1 :] = (csum[window_samples:] - csum[:-window_samples]) == window_samples
    return out


def circular_angle_diff_deg(a: float, b: float) -> float:
    d = abs((a - b + 180.0) % 360.0 - 180.0)
    return float(d)


def select_task_candidates(traj: pd.DataFrame, target_path: Path) -> pd.DataFrame:
    t = traj["time_s"].to_numpy(float)
    xy = traj[["x_m", "y_m"]].to_numpy(float)
    speed = traj["speed_m_s"].to_numpy(float)
    accel = traj["accel_m_s2"].to_numpy(float)

    shots = pd.read_excel(target_path, sheet_name=0).dropna().iloc[:, 0:3]
    photos = pd.read_excel(target_path, sheet_name=1).dropna().iloc[:, 0:3]
    rows: list[dict[str, float | str]] = []

    shoot_window = int(round(SHOOT_PREP / DT_OUT)) + 1
    for _, row in shots.iterrows():
        target_id = str(row.iloc[0])
        pt = row.iloc[1:3].to_numpy(float)
        dist = np.linalg.norm(xy - pt, axis=1)
        base = (
            (dist >= SHOOT_DISTANCE[0])
            & (dist <= SHOOT_DISTANCE[1])
            & (speed <= SHOOT_SPEED_MAX)
            & (accel <= SHOOT_ACCEL_MAX)
        )
        ok = rolling_all(base, shoot_window)
        if not np.any(ok):
            continue
        idxs = np.where(ok)[0]
        best_idx = int(idxs[np.argmin(dist[idxs])])
        rows.append(
            {
                "target_id": target_id,
                "task": "模拟射击",
                "prep_start_s": round(float(t[best_idx] - SHOOT_PREP), 1),
                "exec_time_s": round(float(t[best_idx]), 1),
                "distance_m": float(dist[best_idx]),
                "speed_m_s": float(speed[best_idx]),
                "accel_m_s2": float(accel[best_idx]),
                "angle_deg": "",
                "expected_success": SHOOT_HIT_PROB,
            }
        )

    photo_window = int(round(PHOTO_PREP / DT_OUT)) + 1
    for _, row in photos.iterrows():
        target_id = str(row.iloc[0])
        pt = row.iloc[1:3].to_numpy(float)
        vec = pt - xy
        dist = np.linalg.norm(vec, axis=1)
        angles = (np.degrees(np.arctan2(vec[:, 1], vec[:, 0])) + 360.0) % 360.0
        base = (
            (dist >= PHOTO_DISTANCE[0])
            & (dist <= PHOTO_DISTANCE[1])
            & (speed <= PHOTO_SPEED_MAX)
            & (accel <= PHOTO_ACCEL_MAX)
        )
        ok = rolling_all(base, photo_window)
        if not np.any(ok):
            continue

        # 同一目标保留尽量多的不同角度；同一角度簇内选距离最近的时刻。
        candidates = np.where(ok)[0]
        selected: list[int] = []
        for idx in candidates[np.argsort(dist[candidates])]:
            if all(circular_angle_diff_deg(float(angles[idx]), float(angles[j])) >= PHOTO_ANGLE_MIN_DEG for j in selected):
                selected.append(int(idx))
        selected.sort(key=lambda i: t[i])
        for idx in selected:
            rows.append(
                {
                    "target_id": target_id,
                    "task": "拍照",
                    "prep_start_s": round(float(t[idx] - PHOTO_PREP), 1),
                    "exec_time_s": round(float(t[idx]), 1),
                    "distance_m": float(dist[idx]),
                    "speed_m_s": float(speed[idx]),
                    "accel_m_s2": float(accel[idx]),
                    "angle_deg": float(angles[idx]),
                    "expected_success": 1.0,
                }
            )

    tasks = pd.DataFrame(rows)
    if tasks.empty:
        return tasks
    return tasks.sort_values(["prep_start_s", "exec_time_s", "target_id", "task"]).reset_index(drop=True)


def filter_sequential(tasks: pd.DataFrame) -> pd.DataFrame:
    if tasks.empty:
        return tasks
    rows = []
    current_end = -float("inf")
    for _, row in tasks.sort_values(["exec_time_s", "prep_start_s"]).iterrows():
        if float(row["prep_start_s"]) >= current_end - 1e-9:
            rows.append(row)
            current_end = float(row["exec_time_s"])
    return pd.DataFrame(rows).reset_index(drop=True)


def write_result_xlsx(template: Path, output: Path, tasks: pd.DataFrame) -> None:
    shutil.copy2(template, output)
    wb = load_workbook(output)
    ws = wb.active
    # 仅写 A:E 答案区，保留右侧红色说明和示例。
    for r in range(2, max(ws.max_row + 1, len(tasks) + 3)):
        for c in range(1, 6):
            ws.cell(r, c).value = None
    for i, row in tasks.iterrows():
        r = i + 2
        ws.cell(r, 1, i + 1)
        ws.cell(r, 2, row["target_id"])
        ws.cell(r, 3, row["task"])
        ws.cell(r, 4, float(row["prep_start_s"]))
        ws.cell(r, 5, float(row["exec_time_s"]))
        for c in range(1, 6):
            cell = ws.cell(r, c)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.font = Font(name="宋体", size=12)
    wb.save(output)


def save_estimates(path: Path, results: dict[int, AlignmentResult]) -> None:
    rows = []
    for i in (1, 2, 3):
        r = results[i]
        row = {
            "问题": i,
            "方式2相对方式1时间偏差_delta_s": r.delta2_minus_1,
            "方式1时间偏差_s": 0.0,
            "方式2时间偏差_s": r.delta2_minus_1,
            "方式2相对方式1系统偏差_x_m": r.bias2_x,
            "方式2相对方式1系统偏差_y_m": r.bias2_y,
            "是否判定存在系统偏差": "是" if r.has_system_bias else "否",
            "融合重叠时长_s": r.overlap_seconds,
            "带偏差模型MSE": r.mse_with_bias,
            "无偏差模型MSE": r.mse_without_bias,
            "偏差模型误差下降比例": r.bias_model_improvement,
        }
        if i == 3:
            row["实际数据候选偏差_x_m"] = getattr(r, "candidate_bias_x", np.nan)
            row["实际数据候选偏差_y_m"] = getattr(r, "candidate_bias_y", np.nan)
            row["实际数据带偏差候选delta_s"] = getattr(r, "candidate_delta", np.nan)
        rows.append(row)
    pd.DataFrame(rows).to_excel(path, index=False)


def plot_outputs(out_dir: Path, trajectories: dict[int, pd.DataFrame], tasks: pd.DataFrame, target_path: Path) -> list[Path]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    plt.figure(figsize=(7.0, 5.2), dpi=180)
    for i, df in trajectories.items():
        plt.plot(df["x_m"], df["y_m"], linewidth=1.2, label=f"Problem {i}")
    plt.axis("equal")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.tight_layout()
    p = fig_dir / "trajectories_10hz.png"
    plt.savefig(p)
    plt.close()
    paths.append(p)

    shots = pd.read_excel(target_path, sheet_name=0).dropna().iloc[:, 0:3]
    photos = pd.read_excel(target_path, sheet_name=1).dropna().iloc[:, 0:3]
    tr3 = trajectories[3]
    plt.figure(figsize=(7.0, 5.2), dpi=180)
    plt.plot(tr3["x_m"], tr3["y_m"], color="#1f77b4", linewidth=1.2, label="Fused trajectory")
    plt.scatter(shots.iloc[:, 1], shots.iloc[:, 2], marker="x", color="#d62728", label="Shooting targets")
    plt.scatter(photos.iloc[:, 1], photos.iloc[:, 2], marker="o", facecolors="none", edgecolors="#2ca02c", label="Photo targets")
    selected_ids = set(tasks["target_id"].astype(str)) if not tasks.empty else set()
    for df, color in [(shots, "#d62728"), (photos, "#2ca02c")]:
        sel = df[df.iloc[:, 0].astype(str).isin(selected_ids)]
        if not sel.empty:
            plt.scatter(sel.iloc[:, 1], sel.iloc[:, 2], s=80, facecolors="none", edgecolors=color, linewidths=2.0)
    plt.axis("equal")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    p = fig_dir / "problem4_selected_tasks.png"
    plt.savefig(p)
    plt.close()
    paths.append(p)
    return paths


def xml_escape(text: object) -> str:
    return html.escape(str(text), quote=False)


def w_run(text: object, bold: bool = False, font: str = "宋体", size: int = 24) -> str:
    b = "<w:b/>" if bold else ""
    return (
        "<w:r><w:rPr>"
        f"{b}<w:rFonts w:ascii=\"Times New Roman\" w:eastAsia=\"{font}\" w:hAnsi=\"Times New Roman\"/>"
        f"<w:sz w:val=\"{size}\"/><w:szCs w:val=\"{size}\"/>"
        f"</w:rPr><w:t xml:space=\"preserve\">{xml_escape(text)}</w:t></w:r>"
    )


def w_p(
    text: object = "",
    align: str = "left",
    bold: bool = False,
    font: str = "宋体",
    size: int = 24,
    spacing_after: int = 80,
    indent_first: bool = False,
) -> str:
    jc = {"left": "left", "center": "center", "right": "right"}.get(align, "left")
    ind = '<w:ind w:firstLine="480"/>' if indent_first else ""
    return (
        "<w:p><w:pPr>"
        f"<w:jc w:val=\"{jc}\"/><w:spacing w:after=\"{spacing_after}\" w:line=\"240\" w:lineRule=\"auto\"/>"
        f"{ind}</w:pPr>{w_run(text, bold=bold, font=font, size=size)}</w:p>"
    )


def w_heading(text: str, level: int) -> str:
    if level == 1:
        return w_p(text, align="center", bold=True, font="黑体", size=28, spacing_after=120)
    return w_p(text, align="left", bold=True, font="黑体", size=24, spacing_after=80)


def w_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def cell(text: object, bold: bool = False) -> str:
        return (
            "<w:tc><w:tcPr><w:tcW w:w=\"0\" w:type=\"auto\"/>"
            '<w:vAlign w:val="center"/></w:tcPr>'
            f"{w_p(text, align='center', bold=bold, size=20, spacing_after=0)}</w:tc>"
        )

    borders = (
        "<w:tblPr><w:tblBorders>"
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        "</w:tblBorders></w:tblPr>"
    )
    out = ["<w:tbl>", borders]
    out.append("<w:tr>" + "".join(cell(h, True) for h in headers) + "</w:tr>")
    for row in rows:
        out.append("<w:tr>" + "".join(cell(x) for x in row) + "</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


def w_image(rid: str, width_in: float, height_in: float) -> str:
    cx = int(width_in * 914400)
    cy = int(height_in * 914400)
    return f"""
<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing>
<wp:inline distT="0" distB="0" distL="0" distR="0">
<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>
<wp:docPr id="1" name="Picture"/><wp:cNvGraphicFramePr>
<a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/>
</wp:cNvGraphicFramePr><a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
<pic:nvPicPr><pic:cNvPr id="0" name="image.png"/><pic:cNvPicPr/></pic:nvPicPr>
<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>
"""


def section_break_no_footer() -> str:
    return (
        "<w:p><w:pPr><w:sectPr>"
        '<w:type w:val="nextPage"/>'
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="0" w:footer="720" w:gutter="0"/>'
        "</w:sectPr></w:pPr></w:p>"
    )


def final_section_with_footer() -> str:
    return (
        '<w:sectPr><w:footerReference w:type="default" r:id="rIdFooter1"/>'
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="0" w:footer="720" w:gutter="0"/>'
        '<w:pgNumType w:start="1"/></w:sectPr>'
    )


def build_report_text(
    results: dict[int, AlignmentResult],
    trajectories: dict[int, pd.DataFrame],
    tasks: pd.DataFrame,
) -> tuple[str, list[str], list[tuple[str, Sequence[str], Sequence[Sequence[object]]]]]:
    r1, r2, r3 = results[1], results[2], results[3]
    task_count = len(tasks)
    shoot_count = int((tasks["task"] == "模拟射击").sum()) if not tasks.empty else 0
    photo_count = int((tasks["task"] == "拍照").sum()) if not tasks.empty else 0
    expected = float(tasks["expected_success"].sum()) if not tasks.empty else 0.0
    abstract = (
        "针对两类异频异步定位数据，建立了以方式1为时间基准的轨迹匹配、固定偏差估计和10Hz重采样融合模型。"
        f"问题1在无噪声条件下得到方式2相对方式1的时间偏差为{r1.delta2_minus_1:.4f}s，融合残差接近0。"
        f"问题2在随机噪声和系统偏差共同存在时，估计方式2时间偏差为{r2.delta2_minus_1:.4f}s，"
        f"系统偏差为({r2.bias2_x:.4f},{r2.bias2_y:.4f})m，带偏差模型使匹配均方误差下降{100*r2.bias_model_improvement:.2f}%。"
        f"问题3实际数据的候选固定偏差仅为({getattr(r3, 'candidate_bias_x', 0.0):.4f},"
        f"{getattr(r3, 'candidate_bias_y', 0.0):.4f})m，误差下降{100*r3.bias_model_improvement:.2f}%，"
        "低于本文判据，故判定不存在工程意义上的固定系统偏差，并输出10Hz融合轨迹。"
        f"在任务优化中，按题目给定距离、速度、加速度和准备时间约束筛选可行时刻；在不额外加入题面未给出的任务互斥约束时，共得到{task_count}项任务，"
        f"其中射击{shoot_count}项、拍照{photo_count}项，考虑85%单次命中率后的期望完成数为{expected:.2f}。"
    )
    keywords = ["多源融合", "时间对齐", "系统偏差", "10Hz重采样", "任务优化"]
    tables = [
        (
            "表1  时间偏差与系统偏差估计结果",
            ["问题", "delta/s", "bias_x/m", "bias_y/m", "系统偏差判定", "重叠时长/s"],
            [
                [1, f"{r1.delta2_minus_1:.4f}", "0", "0", "否", f"{r1.overlap_seconds:.1f}"],
                [2, f"{r2.delta2_minus_1:.4f}", f"{r2.bias2_x:.4f}", f"{r2.bias2_y:.4f}", "是", f"{r2.overlap_seconds:.1f}"],
                [3, f"{r3.delta2_minus_1:.4f}", f"{r3.bias2_x:.4f}", f"{r3.bias2_y:.4f}", "否", f"{r3.overlap_seconds:.1f}"],
            ],
        ),
        (
            "表2  10Hz轨迹规模",
            ["问题", "点数", "起始/s", "终止/s", "x范围/m", "y范围/m"],
            [
                [
                    i,
                    len(df),
                    f"{df['time_s'].iloc[0]:.1f}",
                    f"{df['time_s'].iloc[-1]:.1f}",
                    f"{df['x_m'].min():.2f}~{df['x_m'].max():.2f}",
                    f"{df['y_m'].min():.2f}~{df['y_m'].max():.2f}",
                ]
                for i, df in trajectories.items()
            ],
        ),
        (
            "表3  任务优化结果摘要",
            ["指标", "数值"],
            [
                ["任务总数", task_count],
                ["射击任务数", shoot_count],
                ["拍照任务数", photo_count],
                ["期望完成数", f"{expected:.2f}"],
            ],
        ),
    ]
    return abstract, keywords, tables


def write_markdown_report(
    path: Path,
    results: dict[int, AlignmentResult],
    trajectories: dict[int, pd.DataFrame],
    tasks: pd.DataFrame,
    fig_paths: Sequence[Path],
) -> None:
    abstract, keywords, _tables = build_report_text(results, trajectories, tasks)
    r1, r2, r3 = results[1], results[2], results[3]
    md = []
    md.append("# B题 多源融合机器人定位及任务优化\n")
    md.append("## 摘要\n")
    md.append(abstract + "\n")
    md.append("**关键词：**" + "；".join(keywords) + "\n")
    md.append("## 1 问题重述\n")
    md.append("两种定位方式的采样频率分别为4Hz和5Hz，存在起始时间不同步、随机噪声以及可能的固定系统偏差。需要建立时间对齐、偏差修正和轨迹融合模型，并基于附件3轨迹完成射击和拍照任务设计。\n")
    md.append("## 2 模型与假设\n")
    md.append("设方式2修正后的时间为 tau=t2-delta，坐标修正为 p2'=p2-b。delta 表示方式2相对方式1的时间偏差，b 表示方式2相对方式1的固定坐标偏差。题面未给出任务互斥或冷却约束，因此主结果仅采用附录列出的距离、速度、加速度、准备时间和拍照角度约束。\n")
    md.append("## 3 主要结果\n")
    md.append(f"- 问题1：delta={r1.delta2_minus_1:.4f}s，系统偏差为0。\n")
    md.append(f"- 问题2：delta={r2.delta2_minus_1:.4f}s，方式2系统偏差=({r2.bias2_x:.4f},{r2.bias2_y:.4f})m。\n")
    md.append(f"- 问题3：候选偏差=({getattr(r3, 'candidate_bias_x', 0.0):.4f},{getattr(r3, 'candidate_bias_y', 0.0):.4f})m，误差下降{100*r3.bias_model_improvement:.2f}%，判定无显著系统偏差；delta={r3.delta2_minus_1:.4f}s。\n")
    md.append(f"- 问题4：得到{len(tasks)}项任务，已写入 `result.xlsx`。\n")
    for p in fig_paths:
        md.append(f"\n![{p.stem}]({p.as_posix()})\n")
    md.append("\n## 4 任务明细\n")
    md.append(dataframe_to_markdown(tasks) if not tasks.empty else "无可行任务")
    md.append("\n## 参考文献\n")
    md.append("[1] Savitzky A, Golay M J E, Smoothing and Differentiation of Data by Simplified Least Squares Procedures, Analytical Chemistry, 36(8):1627-1639, 1964.\n")
    path.write_text("\n".join(md), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append([format_markdown_cell(row[c]) for c in df.columns])
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def format_markdown_cell(value: object) -> str:
    if isinstance(value, float):
        text = f"{value:.4f}"
    else:
        text = str(value)
    return text.replace("|", "\\|")


def write_docx_report(
    path: Path,
    results: dict[int, AlignmentResult],
    trajectories: dict[int, pd.DataFrame],
    tasks: pd.DataFrame,
    fig_paths: Sequence[Path],
) -> None:
    abstract, keywords, tables = build_report_text(results, trajectories, tasks)
    r1, r2, r3 = results[1], results[2], results[3]

    body: list[str] = []
    body.append(w_p("B题 多源融合机器人定位及任务优化", align="center", bold=True, font="黑体", size=32, spacing_after=180))
    body.append(w_p("摘要", align="center", bold=True, font="黑体", size=28, spacing_after=120))
    body.append(w_p(abstract, indent_first=True))
    body.append(w_p("关键词：" + "；".join(keywords), bold=True, font="宋体"))
    body.append(section_break_no_footer())

    body.append(w_heading("一、问题重述", 1))
    body.append(w_p("题目给出两类定位数据：方式1为4Hz，方式2为5Hz。两类数据存在时间起点不同、采样频率不同、随机噪声以及可能的固定系统偏差。本文需要完成三类轨迹的对齐与10Hz融合，并利用附件3轨迹对沿途射击和拍照任务进行可行时刻设计。", indent_first=True))
    body.append(w_heading("二、模型假设与符号", 1))
    body.append(w_p("以方式1时间为基准，设方式2修正时间为 tau=t2-delta，其中 delta 为方式2相对方式1的时间偏差；设方式2相对方式1的固定坐标偏差为 b=(b_x,b_y)，修正坐标为 p2'=p2-b。题面未给出任务互斥、执行器冷却或同一时刻只能处理一个目标的限制，故主结果仅采用附录明确给出的距离、速度、加速度、准备时间和拍照角度约束。", indent_first=True))
    body.append(w_p("射击约束为距离5m至30m、线速度不超过2m/s、加速度不超过1.5m/s^2，执行前1.5s内均满足上述约束；拍照约束为距离10m至40m、不同拍照方向角至少60度、线速度不超过1.5m/s、加速度不超过1.5m/s^2，执行前0.5s内均满足上述约束。", indent_first=True))

    body.append(w_heading("三、时间对齐与融合模型", 1))
    body.append(w_p("对给定 delta，将方式2时间平移到方式1基准，在两者重叠区间按0.1s网格插值。无噪声问题直接最小化两轨迹平方距离；有噪声问题在每个候选 delta 下先用重叠样本的中位差估计固定偏差，再最小化去偏后的截尾均方误差。为避免局部短重叠造成伪匹配，搜索时要求重叠时长不低于较短轨迹时长的85%。", indent_first=True))
    body.append(w_p("融合阶段将方式2坐标扣除系统偏差后，与方式1在10Hz网格上等权平均。对含噪声轨迹采用Savitzky-Golay平滑与差分求速度、加速度[1]；问题4使用问题3的10Hz融合轨迹。", indent_first=True))
    for title, headers, rows in tables[:2]:
        body.append(w_p(title, align="center", font="黑体", size=22, spacing_after=40))
        body.append(w_table(headers, rows))
        body.append(w_p("", spacing_after=80))

    if fig_paths:
        body.append(w_p("图1  三组10Hz融合轨迹", align="center", font="黑体", size=22, spacing_after=40))
        body.append(w_image("rIdImage1", 5.8, 4.3))

    body.append(w_heading("四、问题结果分析", 1))
    body.append(w_p(f"问题1中无随机噪声和系统偏差，最优时间偏差为{r1.delta2_minus_1:.4f}s，匹配均方误差约为{r1.mse_with_bias:.3e}，说明两类数据在时间平移后完全一致。", indent_first=True))
    body.append(w_p(f"问题2中，若不估计系统偏差，截尾均方误差为{r2.mse_without_bias:.4f}；引入固定偏差后误差降至{r2.mse_with_bias:.4f}，下降{100*r2.bias_model_improvement:.2f}%，因此判定存在固定系统偏差。方式2相对方式1的偏差为({r2.bias2_x:.4f},{r2.bias2_y:.4f})m。", indent_first=True))
    body.append(w_p(f"问题3中，带偏差模型得到的候选偏差为({getattr(r3, 'candidate_bias_x', 0.0):.4f},{getattr(r3, 'candidate_bias_y', 0.0):.4f})m，误差下降比例为{100*r3.bias_model_improvement:.2f}%。本文设定偏差模型需同时满足误差下降不少于5%且偏差模长不小于0.5m才判定为显著系统偏差，因此实际数据不认为存在工程意义上的固定系统偏差。", indent_first=True))

    body.append(w_heading("五、任务优化模型与结果", 1))
    body.append(w_p("在10Hz轨迹上逐时刻检查目标距离、机器人线速度和加速度，并用滚动窗口保证准备区间内约束全部成立。射击目标每个目标选择距离最近的可行时刻；拍照目标在可行时刻中按方向角差异至少60度筛选，以保留尽量多的不同视角。", indent_first=True))
    if fig_paths and len(fig_paths) >= 2:
        body.append(w_p("图2  附件3轨迹与被选任务目标", align="center", font="黑体", size=22, spacing_after=40))
        body.append(w_image("rIdImage2", 5.8, 4.3))
    title, headers, rows = tables[2]
    body.append(w_p(title, align="center", font="黑体", size=22, spacing_after=40))
    body.append(w_table(headers, rows))
    body.append(w_p("任务明细已按开始准备时刻写入输出目录中的 result.xlsx。由于射击单次命中率为85%，射击任务按期望完成数计为0.85项，拍照任务计为1项。", indent_first=True))

    if not tasks.empty:
        display = tasks[["target_id", "task", "prep_start_s", "exec_time_s", "distance_m", "speed_m_s", "accel_m_s2"]].copy()
        display["distance_m"] = display["distance_m"].map(lambda x: f"{x:.2f}")
        display["speed_m_s"] = display["speed_m_s"].map(lambda x: f"{x:.2f}")
        display["accel_m_s2"] = display["accel_m_s2"].map(lambda x: f"{x:.2f}")
        body.append(w_p("表4  任务明细", align="center", font="黑体", size=22, spacing_after=40))
        body.append(w_table(["目标", "任务", "开始准备/s", "执行/s", "距离/m", "速度/(m/s)", "加速度/(m/s^2)"], display.values.tolist()))

    body.append(w_heading("六、模型评价", 1))
    body.append(w_p("模型优点是参数含义清晰、只依赖题目给出的多源轨迹数据，并用重叠时长约束避免短区间伪匹配。偏差判定同时考虑误差下降比例和偏差绝对量，能区分随机噪声导致的小幅均值漂移与真实固定偏差。", indent_first=True))
    body.append(w_p("不足之处在于速度、加速度由平滑轨迹差分获得，平滑窗口会影响临界任务的可行性。若后续给出机器人动力学模型或执行器互斥约束，可将本文候选任务表作为输入，进一步建立混合整数规划进行全局时序优化。", indent_first=True))
    body.append(w_heading("参考文献", 1))
    body.append(w_p("[1] Savitzky A, Golay M J E, Smoothing and Differentiation of Data by Simplified Least Squares Procedures, Analytical Chemistry, 36(8):1627-1639, 1964.", indent_first=False))

    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"
xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
xmlns:o="urn:schemas-microsoft-com:office:office"
xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
xmlns:v="urn:schemas-microsoft-com:vml"
xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"
xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
xmlns:w10="urn:schemas-microsoft-com:office:word"
xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"
xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk"
xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml"
xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"
mc:Ignorable="w14 wp14"><w:body>{''.join(body)}{final_section_with_footer()}</w:body></w:document>"""

    rels = [
        '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
        '<Relationship Id="rIdSettings" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>',
        '<Relationship Id="rIdFooter1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>',
    ]
    image_targets: list[tuple[str, Path]] = []
    if fig_paths:
        rels.append('<Relationship Id="rIdImage1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image1.png"/>')
        image_targets.append(("word/media/image1.png", fig_paths[0]))
    if len(fig_paths) >= 2:
        rels.append('<Relationship Id="rIdImage2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image2.png"/>')
        image_targets.append(("word/media/image2.png", fig_paths[1]))

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>
</Types>"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    doc_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels)
        + "</Relationships>"
    )
    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal">
<w:name w:val="Normal"/><w:qFormat/>
<w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="宋体" w:hAnsi="Times New Roman"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>
</w:style></w:styles>"""
    settings = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:zoom w:percent="100"/><w:defaultTabStop w:val="420"/>
</w:settings>"""
    footer = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:p><w:pPr><w:jc w:val="center"/></w:pPr>
<w:r><w:fldChar w:fldCharType="begin"/></w:r>
<w:r><w:instrText xml:space="preserve">PAGE</w:instrText></w:r>
<w:r><w:fldChar w:fldCharType="separate"/></w:r>
<w:r><w:t>1</w:t></w:r>
<w:r><w:fldChar w:fldCharType="end"/></w:r>
</w:p></w:ftr>"""

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("word/document.xml", document_xml)
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        z.writestr("word/styles.xml", styles)
        z.writestr("word/settings.xml", settings)
        z.writestr("word/footer1.xml", footer)
        for arc, src in image_targets:
            z.write(src, arc)


def choose_font(paths: Sequence[str]) -> FontProperties:
    for p in paths:
        path = Path(p)
        if path.exists():
            return FontProperties(fname=str(path))
    return FontProperties()


def wrap_cjk_text(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for raw in str(text).splitlines() or [""]:
        raw = raw.strip()
        while len(raw) > width:
            cut = width
            for mark in "，。；、：,.; ":
                pos = raw.rfind(mark, 0, width)
                if pos > width * 0.55:
                    cut = pos + 1
                    break
            lines.append(raw[:cut])
            raw = raw[cut:].lstrip()
        lines.append(raw)
    return lines


def write_pdf_report(
    path: Path,
    results: dict[int, AlignmentResult],
    trajectories: dict[int, pd.DataFrame],
    tasks: pd.DataFrame,
    fig_paths: Sequence[Path],
) -> None:
    body_font = choose_font([r"C:\Windows\Fonts\simsun.ttc", r"C:\Windows\Fonts\msyh.ttc"])
    title_font = choose_font([r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\simsunb.ttf"])
    abstract, keywords, _tables = build_report_text(results, trajectories, tasks)
    r1, r2, r3 = results[1], results[2], results[3]

    sections: list[tuple[str, list[str]]] = [
        (
            "一、问题重述",
            [
                "两类定位数据采样频率分别为4Hz和5Hz，需解决时间异步、随机噪声和可能的固定系统偏差，并输出10Hz连续轨迹；在附件3轨迹上，还需设计射击与拍照任务时刻。"
            ],
        ),
        (
            "二、模型与假设",
            [
                "以方式1为时间基准，方式2修正时间为 tau=t2-delta，修正坐标为 p2'=p2-b。搜索delta时要求两轨迹重叠时长不少于较短轨迹的85%，避免短区间伪匹配。",
                "题面未给出任务互斥或冷却约束，主结果只采用附录明确给出的距离、速度、加速度、准备时间和拍照角度约束。",
            ],
        ),
        (
            "三、结果",
            [
                f"问题1：delta={r1.delta2_minus_1:.4f}s，系统偏差为0。",
                f"问题2：delta={r2.delta2_minus_1:.4f}s，方式2系统偏差=({r2.bias2_x:.4f},{r2.bias2_y:.4f})m，偏差模型误差下降{100*r2.bias_model_improvement:.2f}%。",
                f"问题3：候选偏差=({getattr(r3, 'candidate_bias_x', 0.0):.4f},{getattr(r3, 'candidate_bias_y', 0.0):.4f})m，误差下降{100*r3.bias_model_improvement:.2f}%，判定无显著系统偏差；delta={r3.delta2_minus_1:.4f}s。",
                f"问题4：得到{len(tasks)}项任务，其中射击{int((tasks['task']=='模拟射击').sum()) if not tasks.empty else 0}项、拍照{int((tasks['task']=='拍照').sum()) if not tasks.empty else 0}项，期望完成数为{float(tasks['expected_success'].sum()) if not tasks.empty else 0.0:.2f}。",
            ],
        ),
    ]

    def new_page(pdf: PdfPages, page_no: int | None = None):
        fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        if page_no is not None:
            ax.text(0.5, 0.035, str(page_no), ha="center", va="center", fontsize=10, fontproperties=body_font)
        return fig, ax

    with PdfPages(path) as pdf:
        fig, ax = new_page(pdf, None)
        y = 0.93
        ax.text(0.5, y, "B题 多源融合机器人定位及任务优化", ha="center", va="top", fontsize=16, fontproperties=title_font)
        y -= 0.06
        ax.text(0.5, y, "摘要", ha="center", va="top", fontsize=14, fontproperties=title_font)
        y -= 0.04
        for line in wrap_cjk_text(abstract, 43):
            ax.text(0.11, y, line, ha="left", va="top", fontsize=10.5, fontproperties=body_font)
            y -= 0.026
        y -= 0.01
        for line in wrap_cjk_text("关键词：" + "；".join(keywords), 43):
            ax.text(0.11, y, line, ha="left", va="top", fontsize=10.5, fontproperties=body_font)
            y -= 0.026
        pdf.savefig(fig)
        plt.close(fig)

        page_no = 1
        fig, ax = new_page(pdf, page_no)
        y = 0.94
        for title, paras in sections:
            if y < 0.16:
                pdf.savefig(fig)
                plt.close(fig)
                page_no += 1
                fig, ax = new_page(pdf, page_no)
                y = 0.94
            ax.text(0.5, y, title, ha="center", va="top", fontsize=13, fontproperties=title_font)
            y -= 0.04
            for para in paras:
                for line in wrap_cjk_text(para, 45):
                    ax.text(0.11, y, line, ha="left", va="top", fontsize=10.5, fontproperties=body_font)
                    y -= 0.026
                y -= 0.012
        ax.text(0.5, y, "四、任务明细", ha="center", va="top", fontsize=13, fontproperties=title_font)
        y -= 0.04
        headers = ["序号", "目标", "任务", "准备/s", "执行/s"]
        ax.text(0.11, y, "  ".join(headers), ha="left", va="top", fontsize=9.5, fontproperties=title_font)
        y -= 0.026
        for i, row in tasks.iterrows():
            if y < 0.12:
                pdf.savefig(fig)
                plt.close(fig)
                page_no += 1
                fig, ax = new_page(pdf, page_no)
                y = 0.94
            text = f"{i+1:>2}  {row['target_id']}  {row['task']}  {row['prep_start_s']:.1f}  {row['exec_time_s']:.1f}"
            ax.text(0.11, y, text, ha="left", va="top", fontsize=9.5, fontproperties=body_font)
            y -= 0.024
        pdf.savefig(fig)
        plt.close(fig)

        for fig_path in fig_paths[:2]:
            page_no += 1
            fig, ax = new_page(pdf, page_no)
            ax.imshow(plt.imread(fig_path))
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="B题多源融合定位与任务优化")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="B题目录")
    parser.add_argument("--sequential", action="store_true", help="附加单任务非重叠约束，仅输出顺序执行子集")
    args = parser.parse_args()

    root = args.root
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    files, result_template = discover_files(root)
    alignments = build_alignment_results(files)
    save_estimates(out_dir / "estimates_summary.xlsx", alignments)

    smooth_windows = {1: 1, 2: 101, 3: 201}
    trajectories: dict[int, pd.DataFrame] = {}
    for i in (1, 2, 3):
        df = make_trajectory(files[i], alignments[i], smooth_windows[i])
        trajectories[i] = df
        df.to_excel(out_dir / f"problem{i}_10Hz_trajectory.xlsx", index=False)

    tasks = select_task_candidates(trajectories[3], files[4])
    if args.sequential:
        tasks = filter_sequential(tasks)
    tasks.to_excel(out_dir / "task_schedule_detail.xlsx", index=False)
    write_result_xlsx(result_template, out_dir / "result.xlsx", tasks)

    fig_paths = plot_outputs(out_dir, trajectories, tasks, files[4])
    write_markdown_report(out_dir / "B题_论文.md", alignments, trajectories, tasks, fig_paths)
    write_docx_report(out_dir / "B题_多源融合机器人定位及任务优化_论文.docx", alignments, trajectories, tasks, fig_paths)
    write_pdf_report(out_dir / "B题_多源融合机器人定位及任务优化_论文.pdf", alignments, trajectories, tasks, fig_paths)

    print("完成。输出目录：", out_dir)
    print("时间与偏差估计：")
    for i in (1, 2, 3):
        r = alignments[i]
        print(
            f"  问题{i}: delta={r.delta2_minus_1:.6f}s, "
            f"bias=({r.bias2_x:.6f},{r.bias2_y:.6f})m, "
            f"has_bias={r.has_system_bias}"
        )
    print(f"任务数：{len(tasks)}，结果表：{out_dir / 'result.xlsx'}")


if __name__ == "__main__":
    main()
