from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from src.pipeline import list_events, run_pipeline
from src.plotting import (
    PlotParams,
    SaveOptions,
    plot_azimuth_mask,
    plot_azimuth_spectrogram,
    plot_azimuth_stability,
    plot_confidence_map,
    plot_lofar,
    plot_spectrogram,
    plot_waveform,
)


PROJECT_ROOT = Path(__file__).resolve().parent

PLOT_OPTIONS = [
    "波形图",
    "时频谱图",
    "LOFAR图",
    "方位角遮罩谱",
    "方位角谱图",
    "方位角R值谱",
    "方位置信度图",
]

PRESET_CONFIGS = {
    "快速": {
        "window_length_s": 1.5,
        "overlap": 0.4,
        "stability_window": 10,
        "stability_step": 4,
        "confidence_threshold": 0.55,
        "plot_items": ["波形图", "时频谱图", "方位角遮罩谱"],
    },
    "标准": {
        "window_length_s": 2.0,
        "overlap": 0.5,
        "stability_window": 15,
        "stability_step": 5,
        "confidence_threshold": 0.6,
        "plot_items": ["波形图", "时频谱图", "方位角遮罩谱"],
    },
    "精细": {
        "window_length_s": 3.0,
        "overlap": 0.65,
        "stability_window": 24,
        "stability_step": 6,
        "confidence_threshold": 0.7,
        "plot_items": PLOT_OPTIONS,
    },
}

STATUS_STYLE = {
    "idle": ("未开始", "status-idle"),
    "running": ("进行中", "status-running"),
    "success": ("已完成", "status-success"),
    "error": ("失败", "status-error"),
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
        "方位角遮罩谱": "plot_azimuth_mask",
        "方位角谱图": "plot_azimuth",
        "方位角R值谱": "plot_azimuth_stability",
        "方位稳定性图": "plot_azimuth_stability",
        "方位置信度图": "plot_azimuth_confidence",
    }
    return {value: (label in selected_items) for label, value in mapping.items()}


def _apply_preset_to_state(preset_name: str) -> None:
    cfg = PRESET_CONFIGS[preset_name]
    st.session_state.window_length_s = float(cfg["window_length_s"])
    st.session_state.overlap = float(cfg["overlap"])
    st.session_state.stability_window = int(cfg["stability_window"])
    st.session_state.stability_step = int(cfg["stability_step"])
    st.session_state.confidence_threshold = float(cfg["confidence_threshold"])
    st.session_state.plot_items = list(cfg["plot_items"])


def _sync_preset_state() -> None:
    if "preset_name" not in st.session_state:
        st.session_state.preset_name = "标准"
    if "preset_applied" not in st.session_state:
        st.session_state.preset_applied = ""
    if st.session_state.preset_applied != st.session_state.preset_name:
        _apply_preset_to_state(st.session_state.preset_name)
        st.session_state.preset_applied = st.session_state.preset_name


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
    return f"{step_name} {_status_badge(status)} {continue_text}"


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
    )
    save_opts = SaveOptions(save=False)

    panel_to_flag = {
        "wave": "plot_waveform",
        "spec": "plot_spectrogram",
        "lofar": "plot_lofar",
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
    st.set_page_config(page_title="螺旋桨实验数据处理", layout="wide")
    st.markdown(
        """
        <style>
        :root {
            --ui-bg: #f4f8fb;
            --ui-card: #ffffff;
            --ui-border: #c8d6e5;
            --ui-strong: #0b3f5c;
            --ui-accent: #176087;
            --ui-ok: #0f766e;
            --ui-warn: #b45309;
            --ui-err: #b91c1c;
        }
        .stApp { background: linear-gradient(180deg, #f8fbfe 0%, var(--ui-bg) 100%); }
        .block-container { padding-top: 1rem; padding-bottom: 1.2rem; }
        .summary-card {
            background: #fbfdff;
            border: 1px solid var(--ui-border);
            border-radius: 10px;
            padding: 10px;
            min-height: 86px;
        }
        .summary-title {
            font-size: 0.82rem;
            font-weight: 700;
            color: #334155;
            margin-bottom: 5px;
        }
        .summary-value {
            font-size: 1.02rem;
            font-weight: 800;
            color: var(--ui-ok);
            overflow-wrap: anywhere;
        }
        .status-pill {
            display: inline-block;
            margin-left: 8px;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            border: 1px solid transparent;
        }
        .status-idle { background: #e2e8f0; color: #334155; border-color: #cbd5e1; }
        .status-running { background: #fff7ed; color: #b45309; border-color: #fdba74; }
        .status-success { background: #ecfdf5; color: #047857; border-color: #86efac; }
        .status-error { background: #fef2f2; color: #b91c1c; border-color: #fca5a5; }
        .step-next {
            display: inline-block;
            margin-left: 8px;
            color: #0b3f5c;
            font-size: 0.8rem;
            font-weight: 700;
        }
        .run-card {
            background: #ffffff;
            border: 1px solid var(--ui-border);
            border-left: 6px solid var(--ui-accent);
            border-radius: 10px;
            padding: 10px 12px;
            margin: 0.4rem 0 0.8rem 0;
        }
        .run-modal {
            border: 2px solid #fb923c;
            background: #fff7ed;
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 0.8rem;
            box-shadow: 0 6px 18px rgba(251,146,60,0.18);
        }
        .run-modal-title { font-weight: 800; color: #9a3412; margin-bottom: 4px; }
        .run-modal-body { color: #7c2d12; font-size: 0.9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _sync_preset_state()
    _sync_run_state()

    st.title("螺旋桨实验数据处理 UI")
    st.caption("手风琴分步：一次聚焦一步；运行阶段提供明显状态提示。")

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
            with st.expander("上传 SAC 文件", expanded=False):
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

    status2 = "success"
    step2_open = True

    with st.expander(_step_title("Step 2 参数配置", status2, can_continue=True), expanded=step2_open):
        pcol1, pcol2 = st.columns([1.0, 2.1], gap="small")
        with pcol1:
            st.selectbox(
                "参数预设",
                ["快速", "标准", "精细"],
                key="preset_name",
                help="切换预设会自动回填关键参数，但你仍可在下方继续调整。",
            )
            _sync_preset_state()

        with pcol2:
            tips = PRESET_CONFIGS[st.session_state.preset_name]
            st.info(
                "当前预设："
                f"时窗 {tips['window_length_s']}s，重叠 {tips['overlap']:.2f}，"
                f"稳定窗口 {tips['stability_window']}，步长 {tips['stability_step']}，"
                f"阈值 {tips['confidence_threshold']:.2f}。"
            )

        common1, common2, common3 = st.columns([1.2, 1.2, 1.0], gap="small")
        with common1:
            use_auto_band = st.checkbox("使用自动频段推荐", value=False)
            selected_band = None
            if not use_auto_band:
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

        with common3:
            use_time_slice = st.checkbox("启用时间裁切", value=False)
            time_slice_s = None
            if use_time_slice:
                slice_start = st.number_input("裁切起始时间 (秒)", min_value=0.0, value=0.0, step=1.0)
                slice_end = st.number_input("裁切结束时间 (秒)", min_value=1.0, value=300.0, step=1.0)
                if slice_end <= slice_start:
                    st.error("裁切结束时间必须大于起始时间。")
                    return
                time_slice_s = (float(slice_start), float(slice_end))

            st.number_input("稳定性窗口大小", min_value=2, step=1, key="stability_window")
            st.number_input("稳定性步长", min_value=1, step=1, key="stability_step")
            st.slider("置信度阈值", min_value=0.0, max_value=1.0, step=0.05, key="confidence_threshold")

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

        with st.expander("高级参数（绘图）", expanded=False):
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
            "<div class='run-card'>"
            f"<b>运行状态：</b>{status_label}<br>"
            f"<b>状态说明：</b>{st.session_state.run_message}<br>"
            f"<b>最近运行时间：</b>{st.session_state.last_run_at}"
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

        tab_all, tab_wave, tab_spec, tab_lofar, tab_mask, tab_azi, tab_stab, tab_conf, tab_log = st.tabs(
            ["合并图", "波形", "时频", "LOFAR", "遮罩", "方位角", "稳定性", "置信度", "日志与下载"]
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
