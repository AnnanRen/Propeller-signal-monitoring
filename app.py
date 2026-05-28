from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from src.pipeline import list_events, preview_auto_band, run_pipeline
from src.data_io import get_bundle_overview
from src.plotting import (
    PlotParams,
    SaveOptions,
    plot_azimuth_mask,
    plot_azimuth_spectrogram,
    plot_azimuth_stability,
    plot_confidence_map,
    plot_lofar,
    plot_snr_curve,
    plot_spectrogram,
    plot_waveform,
)


PROJECT_ROOT = Path(__file__).resolve().parent

PLOT_OPTIONS = [
    "波形图",
    "时频谱图",
    "LOFAR图",
    "SNR曲线图(HYD)",
    "方位角遮罩谱",
    "方位角谱图",
    "方位角R值谱",
    "方位置信度图",
]

STATUS_STYLE = {
    "idle": ("● 未开始", "status-idle"),
    "running": ("◉ 进行中", "status-running"),
    "success": ("● 已完成", "status-success"),
    "error": ("● 失败", "status-error"),
}


def _resolve_dir(path_text: str, default_name: str) -> Path:
    text = (path_text or "").strip()
    if not text:
        return PROJECT_ROOT / default_name
    p = Path(text)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _save_uploaded_files(uploaded_files, data_dir: Path) -> list[str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for uf in uploaded_files:
        target = data_dir / uf.name
        target.write_bytes(uf.getbuffer())
        saved.append(uf.name)
    return saved


def _build_plot_flags(selected_items: list[str]) -> dict[str, bool]:
    mapping = {
        "波形图": "plot_waveform",
        "时频谱图": "plot_spectrogram",
        "LOFAR图": "plot_lofar",
        "SNR曲线图(HYD)": "plot_snr",
        "方位角遮罩谱": "plot_azimuth_mask",
        "方位角谱图": "plot_azimuth",
        "方位角R值谱": "plot_azimuth_stability",
        "方位稳定性图": "plot_azimuth_stability",
        "方位置信度图": "plot_azimuth_confidence",
    }
    return {value: (label in selected_items) for label, value in mapping.items()}


def _sync_run_state() -> None:
    st.session_state.setdefault("run_status", "idle")
    st.session_state.setdefault("run_message", "等待运行")
    st.session_state.setdefault("last_run_at", "-")
    st.session_state.setdefault("has_run_result", False)
    st.session_state.setdefault("run_info", None)
    st.session_state.setdefault("run_merge_all", False)
    st.session_state.setdefault("run_plot_flags", {})
    st.session_state.setdefault("run_plot_config", {})


def _status_badge(status: str) -> str:
    label, css_class = STATUS_STYLE.get(status, STATUS_STYLE["idle"])
    return f"<span class='status-pill {css_class}'>{label}</span>"


def _step_title(step_name: str, status: str, can_continue: bool) -> str:
    continue_text = "<span class='step-next'>可继续</span>" if can_continue else ""
    parts = step_name.split(" ", 2)
    if len(parts) >= 3 and parts[0] == "Step":
        num = parts[1]
        label = parts[2]
        formatted = f"<span class='step-number'>{num}</span> {label}"
    else:
        formatted = step_name
    return f"{formatted} {_status_badge(status)} {continue_text}"


def _render_summary_cards(run_info: dict) -> None:
    cols = st.columns(6, gap="small")
    cards = [
        ("事件ID", str(run_info["event_id"])),
        ("分量", str(run_info["component"])),
        ("频段", str(run_info["selected_band"])),
        ("UTC起始", str(run_info["utc_start_iso"])),
        ("输出文件", str(len(run_info["output_files"]))),
        ("运行状态", "成功"),
    ]
    for c, (label, value) in zip(cols, cards):
        c.markdown(
            f"<div class='summary-card'><div class='summary-title'>{label}</div>"
            f"<div class='summary-value'>{value}</div></div>",
            unsafe_allow_html=True,
        )


def _group_output_images(output_paths: list[Path]) -> dict[str, list[Path]]:
    groups = {
        "合并图": [],
        "波形": [],
        "SNR": [],
        "时频": [],
        "LOFAR": [],
        "遮罩": [],
        "方位角": [],
        "稳定性": [],
        "置信度": [],
    }
    for p in output_paths:
        name = p.name.lower()
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        if name.endswith("_all.png") or name.endswith("_all.jpg") or name.endswith("_all.jpeg"):
            groups["合并图"].append(p)
        elif "waveform" in name:
            groups["波形"].append(p)
        elif "spectrogram" in name:
            groups["时频"].append(p)
        elif "lofar" in name:
            groups["LOFAR"].append(p)
        elif "azimuth_mask" in name:
            groups["遮罩"].append(p)
        elif "azimuth_stability" in name:
            groups["稳定性"].append(p)
        elif "azimuth_confidence" in name or "confidence" in name:
            groups["置信度"].append(p)
        elif "snr" in name:
            groups["SNR"].append(p)
        elif "azimuth" in name:
            groups["方位角"].append(p)
    return groups


def _show_images(paths: list[Path]) -> None:
    if not paths:
        st.info("该分组暂无可显示图像。")
        return
    for p in paths:
        st.image(str(p), caption=p.name, use_container_width=True)


def _show_logs_and_downloads(run_info: dict, output_paths: list[Path]) -> None:
    result_payload = run_info["result"]
    preprocess_report = result_payload.get("preprocess_report", {})
    st.markdown("**运行日志**")
    st.text(f"- 事件：{run_info['event_id']}")
    st.text(f"- 频段：{run_info['selected_band']}")
    st.text(f"- UTC起始时间：{run_info['utc_start_iso']}")
    st.text(f"- 分量：{run_info['component']}")
    st.text(f"- 时间裁切：{result_payload.get('time_slice_s')}")
    st.text(f"- 预处理：{preprocess_report}")
    st.text(f"- SNR噪声窗：{result_payload.get('snr_windows', {}).get('noise_window_s')}")
    st.text(f"- SNR噪声窗来源：{result_payload.get('snr_noise_window_source')}")
    st.text(f"- 时区偏移：{result_payload.get('timezone_offset_hours', 8)}")
    if result_payload.get("snr_hyd_db") is not None:
        st.text(f"- HYD SNR(dB)：{float(result_payload['snr_hyd_db']):.2f}")
    st.text(f"- 输出文件数量：{len(run_info['output_files'])}")

    st.markdown("**输出文件列表与下载**")
    if not output_paths:
        st.warning("未找到输出文件。")
        return

    for fp in output_paths:
        st.write(fp.name)
        st.download_button(
            label=f"下载：{fp.name}",
            data=fp.read_bytes(),
            file_name=fp.name,
            mime="application/octet-stream",
            key=f"download-{fp.name}",
        )


def _render_running_banner(container, event_id: str, component: str) -> None:
    container.markdown(
        "<div class='run-modal'>"
        "<div class='run-modal-title'>处理中，请勿切换参数</div>"
        f"<div class='run-modal-body'>当前事件：{event_id} | 分量：{component}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_temp_plot(panel: str, run_info: dict, plot_flags: dict, plot_cfg: dict) -> None:
    result = run_info["result"]
    component = run_info["component"]
    threshold = float(plot_cfg.get("confidence_threshold", 0.6))
    show_utc = result.get("utc_start")

    params = PlotParams(
        font_name=str(plot_cfg.get("plot_font_name", "Helvetica")),
        dpi=int(plot_cfg.get("plot_dpi", 300)),
        figsize=(float(plot_cfg.get("plot_fig_width", 7.2)), float(plot_cfg.get("plot_fig_height", 3.2))),
        cmap_spec=str(plot_cfg.get("plot_cmap_spec", "viridis")),
        cmap_lofar=str(plot_cfg.get("plot_cmap_lofar", "plasma")),
        cmap_azi=str(plot_cfg.get("plot_cmap_azi", "hsv")),
        cmap_stability=str(plot_cfg.get("plot_cmap_stability", "RdYlBu_r")),
        cmap_confidence=str(plot_cfg.get("plot_cmap_confidence", "magma")),
        freq_min=float(run_info["selected_band"][0]),
        freq_max=float(run_info["selected_band"][1]),
        linewidth_waveform=float(plot_cfg.get("plot_linewidth_waveform", 0.4)),
        grid_alpha=float(plot_cfg.get("plot_grid_alpha", 0.2)),
        timezone_offset_hours=int(plot_cfg.get("timezone_offset_hours", 8)),
    )
    save_opts = SaveOptions(save=False)

    panel_to_flag = {
        "wave": "plot_waveform",
        "spec": "plot_spectrogram",
        "lofar": "plot_lofar",
        "snr": "plot_snr",
        "mask": "plot_azimuth_mask",
        "azi": "plot_azimuth",
        "stab": "plot_azimuth_stability",
        "conf": "plot_azimuth_confidence",
    }
    if panel == "wave":
        fig, _ = plot_waveform(
            result["t_sec"],
            result["signals"][component],
            component,
            params,
            save_opts,
            normalize=bool(plot_cfg.get("normalize_waveform", True)),
            utc_start=show_utc,
            noise_window_s=result.get("snr_windows", {}).get("noise_window_s"),
            noise_window_source=result.get("snr_noise_window_source"),
            show_noise_window=True,
        )
    elif panel == "spec":
        fig, _ = plot_spectrogram(
            result["t_spec"],
            result["f_hz"],
            result["spectrogram_db"][component],
            component,
            params,
            save_opts,
            utc_start=show_utc,
        )
    elif panel == "lofar":
        fig, _ = plot_lofar(
            result["t_spec"],
            result["f_hz"],
            result["lofar"][component],
            component,
            params,
            save_opts,
            utc_start=show_utc,
        )
    elif panel == "mask":
        fig, _ = plot_azimuth_mask(
            result["t_spec"],
            result["f_hz"],
            result["azimuth_masked"],
            threshold=threshold,
            plot_params=params,
            save_opts=save_opts,
            utc_start=show_utc,
        )
    elif panel == "azi":
        fig, _ = plot_azimuth_spectrogram(
            result["t_spec"],
            result["f_hz"],
            result["azimuth_deg"],
            params,
            save_opts,
            utc_start=show_utc,
        )
    elif panel == "stab":
        fig, _ = plot_azimuth_stability(
            result["t_spec"],
            result["f_hz"],
            result["azimuth_stability"],
            params,
            save_opts,
            utc_start=show_utc,
        )
    elif panel == "snr":
        if result.get("snr_series") and ("HYD" in result["snr_series"]):
            fig, _ = plot_snr_curve(
                result["t_spec"],
                result["snr_series"]["HYD"],
                "HYD",
                params,
                save_opts,
                utc_start=show_utc,
                noise_window_s=result.get("snr_windows", {}).get("noise_window_s"),
                noise_window_source=result.get("snr_noise_window_source"),
                show_noise_window=True,
            )
        else:
            st.info("当前运行没有可用的 SNR 序列。")
            return
    else:
        fig, _ = plot_confidence_map(
            result["t_spec"],
            result["f_hz"],
            result["confidence"],
            plot_params=params,
            save_opts=save_opts,
            utc_start=show_utc,
        )

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _panel_flag_notice(flag_key: str, plot_flags: dict) -> None:
    if bool(plot_flags.get(flag_key, False)):
        st.caption("已勾选：会输出/会参与合并")
    else:
        st.caption("未勾选：仅查看，不输出")


def main() -> None:
    st.set_page_config(page_title="螺旋桨实验数据处理", layout="wide", page_icon="⚙️")
    st.markdown(
        """
        <style>
        /* ============================================================
           DESIGN TOKENS — Swiss Modernism × Scientific Dashboard
           ============================================================ */
        :root {
            --p: #1E3A5F;
            --p-light: #2B5280;
            --p-lighter: #E8EEF5;
            --s: #2563EB;
            --s-light: #DBEAFE;
            --a: #059669;
            --a-light: #D1FAE5;
            --bg: #F7F8FA;
            --surface: #FFFFFF;
            --text: #0F172A;
            --text2: #475569;
            --muted: #94A3B8;
            --bdr: #E2E8F0;
            --bdr-light: #F1F5F9;
            --ok: #059669;
            --warn: #D97706;
            --err: #DC2626;
            --r-sm: 8px;
            --r-md: 12px;
            --r-lg: 16px;
            --r-full: 9999px;
            --sh-sm: 0 1px 3px rgba(0,0,0,0.04);
            --sh-md: 0 4px 16px rgba(0,0,0,0.06);
            --sh-lg: 0 8px 32px rgba(0,0,0,0.08);
            --font: 'Inter', 'Helvetica Neue', 'Segoe UI', system-ui, -apple-system, sans-serif;
            --mono: 'JetBrains Mono', 'SF Mono', 'Fira Code', 'Consolas', monospace;
        }

        /* === BASE === */
        .stApp { background: var(--bg); }
        .block-container { padding: 1.25rem 1.5rem 1.5rem 1.5rem; max-width: 1340px; }

        h1 {
            font-family: var(--font);
            font-size: 1.375rem;
            font-weight: 700;
            color: var(--p);
            letter-spacing: -0.015em;
            margin-bottom: 0.15rem;
            padding-left: 14px;
            border-left: 4px solid var(--p);
        }
        h2, h3 { font-family: var(--font); font-weight: 600; color: var(--text); }
        .stCaption { color: var(--text2); font-size: 0.875rem; }

        /* === EXPANDER / STEP CARDS === */
        .stExpander {
            background: var(--surface);
            border: 1px solid var(--bdr);
            border-radius: var(--r-md) !important;
            margin-bottom: 0.85rem;
            box-shadow: var(--sh-sm);
            transition: box-shadow 0.2s ease;
        }
        .stExpander:hover { box-shadow: var(--sh-md); }
        .stExpander > details > summary {
            padding: 0.7rem 1rem !important;
            font-family: var(--font);
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--text);
            border-radius: var(--r-md) var(--r-md) 0 0 !important;
            background: var(--surface);
            transition: background 0.15s ease;
        }
        .stExpander > details > summary:hover { background: #FAFBFC; }
        .stExpander > details > div {
            padding: 0.25rem 1rem 0.75rem 1rem;
            background: var(--surface);
            border-radius: 0 0 var(--r-md) var(--r-md) !important;
        }

        /* === BUTTONS === */
        .stButton > button {
            font-family: var(--font);
            font-weight: 600;
            font-size: 0.875rem;
            border-radius: var(--r-sm);
            padding: 0.45rem 1.25rem;
            transition: all 0.18s ease;
            border: 1px solid transparent;
        }
        .stButton > button[kind="primary"] {
            background: var(--p);
            color: #fff;
            border-color: var(--p);
        }
        .stButton > button[kind="primary"]:hover {
            background: var(--p-light);
            border-color: var(--p-light);
            box-shadow: 0 2px 8px rgba(30,58,95,0.25);
        }
        .stButton > button[kind="secondary"] {
            background: var(--surface);
            color: var(--p);
            border-color: var(--bdr);
        }
        .stButton > button[kind="secondary"]:hover {
            background: var(--p-lighter);
            border-color: var(--p);
        }

        /* === FIELDS === */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {
            border-radius: var(--r-sm);
            border: 1px solid var(--bdr);
            font-family: var(--font);
            font-size: 0.875rem;
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus {
            border-color: var(--p);
            box-shadow: 0 0 0 3px rgba(30,58,95,0.1);
        }
        .stSelectbox > div > div {
            border-radius: var(--r-sm);
        }
        .stCheckbox label { font-family: var(--font); font-weight: 500; }
        .stRadio > div { gap: 0.35rem; }

        /* === TABS === */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            border-bottom: 2px solid var(--bdr);
        }
        .stTabs [data-baseweb="tab"] {
            font-family: var(--font);
            font-weight: 500;
            font-size: 0.85rem;
            padding: 0.45rem 0.9rem;
            border-radius: var(--r-sm) var(--r-sm) 0 0;
        }
        .stTabs [aria-selected="true"] {
            color: var(--p);
            border-bottom: 2px solid var(--p);
        }

        /* === SUMMARY CARDS === */
        .summary-card {
            background: var(--surface);
            border: 1px solid var(--bdr);
            border-radius: var(--r-md);
            padding: 10px 12px;
            min-height: 72px;
            box-shadow: var(--sh-sm);
            transition: box-shadow 0.2s ease;
        }
        .summary-card:hover { box-shadow: var(--sh-md); }
        .summary-title {
            font-family: var(--font);
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--text2);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 6px;
        }
        .summary-value {
            font-family: var(--mono);
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--p);
            overflow-wrap: anywhere;
        }

        /* === STATUS PILLS === */
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            margin-left: 8px;
            padding: 2px 10px;
            border-radius: var(--r-full);
            font-family: var(--font);
            font-size: 0.75rem;
            font-weight: 600;
            border: 1px solid transparent;
            transition: all 0.2s ease;
        }
        .status-idle  { background: #F1F5F9; color: #475569; border-color: #E2E8F0; }
        .status-running { background: #FFFBEB; color: #B45309; border-color: #FDE68A; animation: pulse-border 1.5s ease-in-out infinite; }
        .status-success { background: #ECFDF5; color: #047857; border-color: #A7F3D0; }
        .status-error   { background: #FEF2F2; color: #B91C1C; border-color: #FECACA; }
        @keyframes pulse-border {
            0%, 100% { border-color: #FDE68A; }
            50% { border-color: #F59E0B; }
        }

        /* === STEP NUMBER BADGE === */
        .step-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 24px;
            height: 24px;
            background: var(--p);
            color: #fff;
            border-radius: var(--r-sm);
            font-family: var(--mono);
            font-size: 0.78rem;
            font-weight: 700;
            margin-right: 2px;
            vertical-align: middle;
        }

        /* === STEP NEXT INDICATOR === */
        .step-next {
            display: inline-flex;
            align-items: center;
            gap: 3px;
            margin-left: 10px;
            color: var(--p);
            font-family: var(--font);
            font-size: 0.78rem;
            font-weight: 600;
        }
        .step-next::before {
            content: '→';
            font-size: 0.9rem;
        }

        /* === RUN CARD === */
        .run-card {
            background: var(--surface);
            border: 1px solid var(--bdr);
            border-left: 5px solid var(--p);
            border-radius: var(--r-md);
            padding: 10px 14px;
            margin: 0.3rem 0 0.6rem 0;
            box-shadow: var(--sh-sm);
        }
        .run-card b { color: var(--text); font-weight: 600; }

        /* === RUNNING BANNER === */
        .run-modal {
            border: 1.5px solid var(--warn);
            background: #FFFBEB;
            border-radius: var(--r-md);
            padding: 12px 14px;
            margin-bottom: 0.6rem;
            box-shadow: 0 4px 20px rgba(217,119,6,0.12);
            animation: shimmer-bg 2s ease-in-out infinite;
        }
        @keyframes shimmer-bg {
            0%, 100% { background: #FFFBEB; }
            50% { background: #FFF7ED; }
        }
        .run-modal-title {
            font-family: var(--font);
            font-weight: 700;
            color: #92400E;
            margin-bottom: 4px;
            font-size: 0.95rem;
        }
        .run-modal-body {
            color: #78350F;
            font-size: 0.85rem;
            font-family: var(--mono);
        }

        /* === SUCCESS / ERROR BANNERS === */
        .stSuccess, .stError, .stWarning, .stInfo {
            font-family: var(--font);
            border-radius: var(--r-sm);
            font-weight: 500;
        }

        /* === METRIC OVERRIDE === */
        [data-testid="stMetricValue"] {
            font-family: var(--mono);
            font-weight: 700;
            color: var(--p);
        }
        [data-testid="stMetricLabel"] {
            font-family: var(--font);
            font-weight: 600;
            color: var(--text2);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        /* === IMAGES === */
        .stImage {
            border-radius: var(--r-sm);
            overflow: hidden;
            box-shadow: var(--sh-sm);
        }

        /* === SCROLLBAR === */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--bdr); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--muted); }

        /* === DOWNLOAD BUTTONS === */
        .stDownloadButton > button {
            font-family: var(--font);
            font-size: 0.8rem;
            font-weight: 500;
            border-radius: var(--r-sm);
            padding: 0.25rem 0.75rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    _sync_run_state()
    st.session_state.setdefault("window_length_s", 2.0)
    st.session_state.setdefault("overlap", 0.5)
    st.session_state.setdefault("stability_window", 15)
    st.session_state.setdefault("stability_step", 5)
    st.session_state.setdefault("confidence_threshold", 0.6)
    st.session_state.setdefault("plot_items", ["波形图", "时频谱图", "方位角遮罩谱"])

    st.markdown(
        "<div style='display:flex;align-items:center;gap:14px;margin-bottom:0.35rem;'>"
        "<span style='font-size:1.7rem;line-height:1;'>⚙️</span>"
        "<div>"
        "<h1 style='margin:0;padding:0;border:none;'>螺旋桨实验数据处理</h1>"
        "<p style='margin:0;color:#64748B;font-size:0.85rem;font-weight:500;'>"
        "四步流程 · 配置 → 运行 → 查看结果</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    step1_open = True
    step2_open = False
    step3_open = False
    step4_open = st.session_state.has_run_result

    status2 = "idle"
    status3 = "idle"
    status4 = "idle"

    with st.expander(_step_title("Step 1 数据与事件", "success", can_continue=True), expanded=step1_open):
        top1, top2, top3 = st.columns([1.1, 1.1, 1.0], gap="small")
        with top1:
            data_dir_text = st.text_input("数据目录路径", value="data")
        with top2:
            output_dir_text = st.text_input("结果目录路径", value="results")
        with top3:
            show_upload = st.checkbox("显示上传 SAC 面板", value=False)
            if show_upload:
                uploaded_files = st.file_uploader(
                    "上传一个或多个 .sac 文件",
                    type=["sac"],
                    accept_multiple_files=True,
                )
                if st.button("保存上传文件", use_container_width=True):
                    if not uploaded_files:
                        st.warning("未选择文件。")
                    else:
                        saved = _save_uploaded_files(uploaded_files, _resolve_dir(data_dir_text, "data"))
                        st.success(f"已保存 {len(saved)} 个文件：{', '.join(saved)}")
                        st.rerun()

        data_dir = _resolve_dir(data_dir_text, "data")
        output_dir = _resolve_dir(output_dir_text, "results")
        data_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        events = list_events(data_dir)
        if not events:
            st.error("在数据目录中未找到完整事件（需同前缀的 BH1/BH2/BHZ/HYD 四个文件）。")
            return

        event_ids = [e.event_id for e in events]
        event_col, manual_col, comp_col = st.columns([1.0, 1.1, 0.9], gap="small")
        with event_col:
            event_id = st.selectbox("事件选择", event_ids, index=0)
        with manual_col:
            manual_input = st.text_input("事件ID/文件名覆盖（可选）", value="")
        with comp_col:
            component = st.selectbox("主要分量", ["BHZ", "BH1", "BH2", "HYD"], index=0)
        input_path = manual_input.strip() if manual_input.strip() else event_id

        selected_bundle = next(e for e in events if e.event_id == event_id)
        overview = get_bundle_overview(selected_bundle)
        show_overview = st.checkbox("显示事件概况", value=False)
        if show_overview:
            ov_col1, ov_col2, ov_col3 = st.columns(3)
            ov_col1.metric("采样率 (Hz)", f"{overview['fs_hz']:.2f}")
            ov_col2.metric("时长 (s)", f"{overview['duration_s']:.1f}")
            ov_col3.metric("时长 (h)", f"{overview['duration_hours']:.2f}")
            if overview["warnings"]:
                for w in overview["warnings"]:
                    st.warning(w)

    status2 = "success"
    step2_open = True

    with st.expander(_step_title("Step 2 参数配置", status2, can_continue=True), expanded=step2_open):
        common1, common2, common3 = st.columns([1.2, 1.2, 1.0], gap="small")
        with common1:
            band_min = st.number_input("分析频段最小值 (Hz)", min_value=0.01, value=1.0, step=0.1)
            band_max = st.number_input("分析频段最大值 (Hz)", min_value=0.02, value=30.0, step=0.1)
            if band_max <= band_min:
                st.error("分析频段最大值必须大于最小值。")
                return
            selected_band = (float(band_min), float(band_max))

            st.number_input("时窗长度 (秒)", min_value=0.2, step=0.1, key="window_length_s")
            st.slider("重叠比例", min_value=0.0, max_value=0.95, step=0.05, key="overlap")

        with common2:
            enable_demean = st.checkbox("去均值", value=True)
            enable_detrend = st.checkbox("去趋势", value=True)
            apply_orientation = st.checkbox("方位角矫正", value=True)
            orientation_deg = st.number_input("方位矫正角度 (度，逆时针为正)", value=0.0, step=0.1)
            normalize_waveform = st.checkbox("波形标准化显示", value=True)
            timezone_offset_hours = st.number_input("时区偏移 (小时)", min_value=-12, max_value=14, value=8, step=1)
            use_downsample = st.checkbox("启用降采样", value=False)
            target_fs = None
            if use_downsample:
                target_fs = st.number_input("目标采样率 (Hz)", min_value=1.0, value=50.0, step=10.0)

        with common3:
            slice_mode = st.radio("时间裁切模式", ["不裁切 (全时段)", "手动裁切", "自动分段"], index=0)
            time_slice_s = None
            auto_slice_length_s = None
            if slice_mode == "手动裁切":
                slice_start = st.number_input("裁切起始时间 (秒)", min_value=0.0, value=0.0, step=1.0)
                slice_end = st.number_input("裁切结束时间 (秒)", min_value=1.0, value=300.0, step=1.0)
                if slice_end <= slice_start:
                    st.error("裁切结束时间必须大于起始时间。")
                    return
                time_slice_s = (float(slice_start), float(slice_end))
            elif slice_mode == "自动分段":
                auto_slice_length_s = st.number_input("每段时长 (秒)", min_value=10.0, value=300.0, step=10.0)

            st.number_input("稳定性窗口大小", min_value=2, step=1, key="stability_window")
            st.number_input("稳定性步长", min_value=1, step=1, key="stability_step")
            st.slider("置信度阈值", min_value=0.0, max_value=1.0, step=0.05, key="confidence_threshold")
            compute_snr = st.checkbox("启用SNR计算", value=False)
            snr_auto_noise_window_s = st.number_input("自动噪声窗长度 (秒)", min_value=5.0, value=60.0, step=1.0)
            use_manual_noise = st.checkbox("手动噪声窗覆盖", value=False)
            snr_noise_window_s = None
            if use_manual_noise:
                snr_noise_start = st.number_input("手动噪声窗起始时间 (秒)", min_value=0.0, value=0.0, step=1.0)
                snr_noise_end = st.number_input("手动噪声窗结束时间 (秒)", min_value=0.1, value=60.0, step=1.0)
                if snr_noise_end <= snr_noise_start:
                    st.error("噪声窗结束时间必须大于起始时间。")
                    return
                snr_noise_window_s = (float(snr_noise_start), float(snr_noise_end))

        auto_col1, auto_col2 = st.columns([1.2, 1.2], gap="small")
        with auto_col1:
            use_auto_band = st.checkbox("使用自动频段推荐（覆盖上方手动频段）", value=False)
            if use_auto_band:
                selected_band = None
                st.caption("运行时自动推荐高能频段")
        with auto_col2:
            if use_auto_band:
                if st.button("预览推荐频段", use_container_width=True):
                    with st.spinner("正在计算频段推荐..."):
                        try:
                            preview = preview_auto_band(
                                input_path=input_path,
                                data_dir=data_dir,
                                window_length_s=float(st.session_state.window_length_s),
                                overlap=float(st.session_state.overlap),
                                time_slice_s=time_slice_s,
                                auto_slice_length_s=auto_slice_length_s,
                                enable_demean=enable_demean,
                                enable_detrend=enable_detrend,
                                apply_orientation=apply_orientation,
                                orientation_deg=orientation_deg,
                                target_fs=target_fs,
                            )
                            st.success(f"推荐频段: {preview['recommended'][0]:.1f} – {preview['recommended'][1]:.1f} Hz")
                            st.caption(f"采样率: {preview['fs_hz']:.1f} Hz")
                            if preview["candidates"]:
                                st.caption("候选频段: " + ", ".join(f"{lo:.1f}–{hi:.1f} Hz" for lo, hi in preview["candidates"]))
                        except Exception as exc:
                            st.error(f"频段预览失败: {exc}")

        out1, out2 = st.columns([1.6, 1.0], gap="small")
        with out1:
            st.multiselect("输出图类型", PLOT_OPTIONS, key="plot_items")
            if "方位稳定性图" in st.session_state.plot_items:
                st.session_state.plot_items = [i for i in st.session_state.plot_items if i != "方位稳定性图"]
            if not st.session_state.plot_items:
                st.error("至少选择一种输出图类型。")
                return
        with out2:
            format_items = st.multiselect("输出文件格式", ["png", "pdf"], default=["png", "pdf"])
            if not format_items:
                st.error("至少选择一种输出文件格式。")
                return
            formats = tuple(format_items)
            save_plots = st.checkbox("保存结果文件", value=True)
            merge_all_plots = st.checkbox("合并所有图片", value=True)

        plot_flags = _build_plot_flags(st.session_state.plot_items)
        compute_snr_effective = bool(compute_snr) or bool(plot_flags["plot_snr"])

        plot_font_name = "Helvetica"
        plot_dpi = 300
        plot_grid_alpha = 0.2
        plot_fig_width = 7.2
        plot_fig_height = 3.2
        plot_linewidth_waveform = 0.4
        plot_cmap_spec = "viridis"
        plot_cmap_lofar = "plasma"
        plot_cmap_azi = "hsv"
        plot_cmap_stability = "RdYlBu_r"
        plot_cmap_confidence = "magma"

        show_advanced_plot = st.checkbox("显示高级参数（绘图）", value=False)
        if show_advanced_plot:
            a1, a2, a3 = st.columns(3)
            with a1:
                plot_font_name = st.text_input("字体名称", value=plot_font_name)
            with a2:
                plot_dpi = st.number_input("图像 DPI", min_value=72, value=plot_dpi, step=1)
            with a3:
                plot_grid_alpha = st.slider("网格透明度", min_value=0.0, max_value=1.0, value=plot_grid_alpha, step=0.05)

            a4, a5, a6 = st.columns(3)
            with a4:
                plot_fig_width = st.number_input("图宽（英寸）", min_value=2.0, value=plot_fig_width, step=0.1)
            with a5:
                plot_fig_height = st.number_input("图高（英寸）", min_value=2.0, value=plot_fig_height, step=0.1)
            with a6:
                plot_linewidth_waveform = st.number_input("波形线宽", min_value=0.1, value=plot_linewidth_waveform, step=0.1)

            cmap_options = ["viridis", "plasma", "magma", "cividis", "inferno", "turbo", "hsv", "RdYlBu_r"]
            cma, cmb, cmc, cmd, cme = st.columns(5)
            with cma:
                plot_cmap_spec = st.selectbox("时频色图", cmap_options, index=cmap_options.index(plot_cmap_spec))
            with cmb:
                plot_cmap_lofar = st.selectbox("LOFAR色图", cmap_options, index=cmap_options.index(plot_cmap_lofar))
            with cmc:
                plot_cmap_azi = st.selectbox("方位角色图", cmap_options, index=cmap_options.index(plot_cmap_azi))
            with cmd:
                plot_cmap_stability = st.selectbox("稳定性色图", cmap_options, index=cmap_options.index(plot_cmap_stability))
            with cme:
                plot_cmap_confidence = st.selectbox("置信度色图", cmap_options, index=cmap_options.index(plot_cmap_confidence))

    status3 = st.session_state.run_status
    step3_open = True
    with st.expander(_step_title("Step 3 运行控制", status3, can_continue=True), expanded=step3_open):
        running_banner = st.empty()
        status_label = STATUS_STYLE.get(st.session_state.run_status, STATUS_STYLE["idle"])[0]
        st.markdown(
            "<div class='run-card' style='display:flex;gap:28px;flex-wrap:wrap;'>"
            f"<span><b>运行状态</b>&nbsp;&nbsp;{status_label}</span>"
            f"<span><b>状态说明</b>&nbsp;&nbsp;{st.session_state.run_message}</span>"
            f"<span><b>最近运行</b>&nbsp;&nbsp;{st.session_state.last_run_at}</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        run_clicked = st.button("开始运行", type="primary", use_container_width=True)

        if run_clicked:
            st.session_state.run_status = "running"
            st.session_state.run_message = "处理中，请勿切换参数"
            st.session_state.last_run_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _render_running_banner(running_banner, event_id, component)
            with st.spinner("正在执行数据处理，请稍候..."):
                try:
                    run_info = run_pipeline(
                        input_path=input_path,
                        output_dir=output_dir,
                        data_dir=data_dir,
                        component=component,
                        selected_band=selected_band,
                        time_slice_s=time_slice_s,
                        auto_slice_length_s=auto_slice_length_s,
                        window_length_s=float(st.session_state.window_length_s),
                        overlap=float(st.session_state.overlap),
                        enable_demean=bool(enable_demean),
                        enable_detrend=bool(enable_detrend),
                        apply_orientation=bool(apply_orientation),
                        orientation_deg=float(orientation_deg),
                        stability_window=int(st.session_state.stability_window),
                        stability_step=int(st.session_state.stability_step),
                        confidence_threshold=float(st.session_state.confidence_threshold),
                        save_plots=bool(save_plots),
                        formats=formats,
                        plot_waveform=plot_flags["plot_waveform"],
                        plot_spectrogram=plot_flags["plot_spectrogram"],
                        plot_lofar=plot_flags["plot_lofar"],
                        plot_azimuth=plot_flags["plot_azimuth"],
                        plot_azimuth_stability=plot_flags["plot_azimuth_stability"],
                        plot_azimuth_mask=plot_flags["plot_azimuth_mask"],
                        plot_confidence=plot_flags["plot_azimuth_confidence"],
                        plot_snr=plot_flags["plot_snr"],
                        compute_snr=bool(compute_snr_effective),
                        snr_noise_window_s=snr_noise_window_s,
                        snr_auto_noise_window_s=float(snr_auto_noise_window_s),
                        target_fs=target_fs,
                        merge_all_plots=bool(merge_all_plots),
                        normalize_waveform=bool(normalize_waveform),
                        plot_font_name=plot_font_name,
                        plot_dpi=int(plot_dpi),
                        plot_fig_width=float(plot_fig_width),
                        plot_fig_height=float(plot_fig_height),
                        plot_cmap_spec=plot_cmap_spec,
                        plot_cmap_lofar=plot_cmap_lofar,
                        plot_cmap_azi=plot_cmap_azi,
                        plot_cmap_stability=plot_cmap_stability,
                        plot_cmap_confidence=plot_cmap_confidence,
                        plot_linewidth_waveform=float(plot_linewidth_waveform),
                        plot_grid_alpha=float(plot_grid_alpha),
                        timezone_offset_hours=int(timezone_offset_hours),
                    )
                    running_banner.empty()
                    st.session_state.run_status = "success"
                    st.session_state.run_message = "运行完成"
                    st.session_state.has_run_result = True
                    st.session_state.run_merge_all = bool(merge_all_plots)
                    st.session_state.run_plot_flags = dict(plot_flags)
                    st.session_state.run_plot_config = {
                        "confidence_threshold": float(st.session_state.confidence_threshold),
                        "normalize_waveform": bool(normalize_waveform),
                        "plot_font_name": plot_font_name,
                        "plot_dpi": int(plot_dpi),
                        "plot_fig_width": float(plot_fig_width),
                        "plot_fig_height": float(plot_fig_height),
                        "plot_cmap_spec": plot_cmap_spec,
                        "plot_cmap_lofar": plot_cmap_lofar,
                        "plot_cmap_azi": plot_cmap_azi,
                        "plot_cmap_stability": plot_cmap_stability,
                        "plot_cmap_confidence": plot_cmap_confidence,
                        "plot_linewidth_waveform": float(plot_linewidth_waveform),
                        "plot_grid_alpha": float(plot_grid_alpha),
                        "timezone_offset_hours": int(timezone_offset_hours),
                    }
                    st.session_state.run_info = run_info
                    st.success("运行完成。")
                except Exception as exc:
                    running_banner.empty()
                    st.session_state.run_status = "error"
                    st.session_state.run_message = "运行失败，请检查参数或输入数据"
                    st.session_state.has_run_result = False
                    st.session_state.run_info = None
                    st.error("运行失败。")
                    st.exception(exc)

    status4 = "success" if st.session_state.has_run_result else "idle"
    step4_open = st.session_state.has_run_result
    with st.expander(_step_title("Step 4 结果查看", status4, can_continue=False), expanded=step4_open):
        if not st.session_state.has_run_result or not st.session_state.run_info:
            st.info("等待运行。请在 Step 3 点击“开始运行”。")
            return

        run_info = st.session_state.run_info
        _render_summary_cards(run_info)

        output_paths = [Path(p) for p in run_info["output_files"]]
        output_paths = sorted(output_paths, key=lambda p: p.name)
        grouped = _group_output_images(output_paths)
        merged_mode = bool(st.session_state.get("run_merge_all", False))
        run_plot_flags = st.session_state.get("run_plot_flags", {})
        run_plot_config = st.session_state.get("run_plot_config", {})

        tab_all, tab_wave, tab_snr, tab_spec, tab_lofar, tab_mask, tab_azi, tab_stab, tab_conf, tab_log = st.tabs(
            ["合并图", "波形", "SNR", "时频", "LOFAR", "遮罩", "方位角", "稳定性", "置信度", "日志与下载"]
        )

        with tab_all:
            _show_images(grouped["合并图"])
        with tab_wave:
            _panel_flag_notice("plot_waveform", run_plot_flags)
            if merged_mode:
                _render_temp_plot("wave", run_info, run_plot_flags, run_plot_config)
            else:
                if grouped["波形"]:
                    _show_images(grouped["波形"])
                else:
                    _render_temp_plot("wave", run_info, run_plot_flags, run_plot_config)
        with tab_snr:
            _panel_flag_notice("plot_snr", run_plot_flags)
            if merged_mode:
                _render_temp_plot("snr", run_info, run_plot_flags, run_plot_config)
            else:
                if grouped["SNR"]:
                    _show_images(grouped["SNR"])
                else:
                    _render_temp_plot("snr", run_info, run_plot_flags, run_plot_config)
        with tab_spec:
            _panel_flag_notice("plot_spectrogram", run_plot_flags)
            if merged_mode:
                _render_temp_plot("spec", run_info, run_plot_flags, run_plot_config)
            else:
                if grouped["时频"]:
                    _show_images(grouped["时频"])
                else:
                    _render_temp_plot("spec", run_info, run_plot_flags, run_plot_config)
        with tab_lofar:
            _panel_flag_notice("plot_lofar", run_plot_flags)
            if merged_mode:
                _render_temp_plot("lofar", run_info, run_plot_flags, run_plot_config)
            else:
                if grouped["LOFAR"]:
                    _show_images(grouped["LOFAR"])
                else:
                    _render_temp_plot("lofar", run_info, run_plot_flags, run_plot_config)
        with tab_mask:
            _panel_flag_notice("plot_azimuth_mask", run_plot_flags)
            if merged_mode:
                _render_temp_plot("mask", run_info, run_plot_flags, run_plot_config)
            else:
                if grouped["遮罩"]:
                    _show_images(grouped["遮罩"])
                else:
                    _render_temp_plot("mask", run_info, run_plot_flags, run_plot_config)
        with tab_azi:
            _panel_flag_notice("plot_azimuth", run_plot_flags)
            if merged_mode:
                _render_temp_plot("azi", run_info, run_plot_flags, run_plot_config)
            else:
                if grouped["方位角"]:
                    _show_images(grouped["方位角"])
                else:
                    _render_temp_plot("azi", run_info, run_plot_flags, run_plot_config)
        with tab_stab:
            _panel_flag_notice("plot_azimuth_stability", run_plot_flags)
            if merged_mode:
                _render_temp_plot("stab", run_info, run_plot_flags, run_plot_config)
            else:
                if grouped["稳定性"]:
                    _show_images(grouped["稳定性"])
                else:
                    _render_temp_plot("stab", run_info, run_plot_flags, run_plot_config)
        with tab_conf:
            _panel_flag_notice("plot_azimuth_confidence", run_plot_flags)
            if merged_mode:
                _render_temp_plot("conf", run_info, run_plot_flags, run_plot_config)
            else:
                if grouped["置信度"]:
                    _show_images(grouped["置信度"])
                else:
                    _render_temp_plot("conf", run_info, run_plot_flags, run_plot_config)
        with tab_log:
            _show_logs_and_downloads(run_info, output_paths)


if __name__ == "__main__":
    main()
