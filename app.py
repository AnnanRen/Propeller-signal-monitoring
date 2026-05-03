from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.pipeline import list_events, run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent


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


def _build_plot_flags(selected_items: list[str]) -> dict:
    mapping = {
        "波形图": "plot_waveform",
        "时频谱图": "plot_spectrogram",
        "LOFAR图": "plot_lofar",
        "方位角谱图": "plot_azimuth",
        "方位稳定性图": "plot_azimuth_stability",
        "方位置信度图": "plot_azimuth_confidence",
    }
    return {value: (label in selected_items) for label, value in mapping.items()}


def main() -> None:
    st.set_page_config(page_title="Data Processing Easy UI", layout="wide")
    st.title("Data Processing Easy UI")
    st.caption("选择数据、设置参数、点击运行、查看并下载结果。")

    st.subheader("数据输入")
    c0, c1 = st.columns(2)
    with c0:
        data_dir_text = st.text_input("数据目录路径", value="data")
    with c1:
        output_dir_text = st.text_input("结果目录路径", value="results")

    data_dir = _resolve_dir(data_dir_text, "data")
    output_dir = _resolve_dir(output_dir_text, "results")
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    with st.expander("可选：上传 SAC 文件到数据目录"):
        uploaded_files = st.file_uploader(
            "上传一个或多个 .sac 文件",
            type=["sac"],
            accept_multiple_files=True,
        )
        if st.button("保存上传文件", use_container_width=True):
            if not uploaded_files:
                st.warning("未选择文件。")
            else:
                saved = _save_uploaded_files(uploaded_files, data_dir)
                st.success(f"已保存 {len(saved)} 个文件：{', '.join(saved)}")
                st.rerun()

    events = list_events(data_dir)
    if not events:
        st.error("在数据目录中未找到完整事件（需同前缀的 BH1/BH2/BHZ/HYD 四个文件）。")
        return

    event_ids = [e.event_id for e in events]
    event_id = st.selectbox("事件选择", event_ids, index=0)
    manual_input = st.text_input("输入事件ID或文件名（可选，留空则使用上方选择）", value="")
    input_path = manual_input.strip() if manual_input.strip() else event_id

    st.subheader("预处理参数")
    p1, p2 = st.columns(2)
    with p1:
        enable_demean = st.checkbox("去均值", value=True)
    with p2:
        enable_detrend = st.checkbox("去趋势", value=True)

    o1, o2 = st.columns(2)
    with o1:
        apply_orientation = st.checkbox("方位角矫正", value=True)
    with o2:
        orientation_deg = st.number_input("方位矫正角度 (度，逆时针为正)", value=0.0, step=0.1)

    st.subheader("核心处理参数")
    k1, k2 = st.columns(2)
    with k1:
        window_length_s = st.number_input("时窗长度 (秒)", min_value=0.2, value=2.0, step=0.1)
    with k2:
        overlap = st.slider("重叠比例", min_value=0.0, max_value=0.95, value=0.5, step=0.05)

    use_auto_band = st.checkbox("使用自动频段推荐", value=False)
    selected_band = None
    if not use_auto_band:
        b1, b2 = st.columns(2)
        with b1:
            band_min = st.number_input("分析频段最小值 (Hz)", min_value=0.01, value=1.0, step=0.1)
        with b2:
            band_max = st.number_input("分析频段最大值 (Hz)", min_value=0.02, value=30.0, step=0.1)
        if band_max <= band_min:
            st.error("分析频段最大值必须大于最小值。")
            return
        selected_band = (float(band_min), float(band_max))

    use_time_slice = st.checkbox("启用时间裁切", value=False)
    time_slice_s = None
    if use_time_slice:
        t1, t2 = st.columns(2)
        with t1:
            slice_start = st.number_input("裁切起始时间 (秒)", min_value=0.0, value=0.0, step=1.0)
        with t2:
            slice_end = st.number_input("裁切结束时间 (秒)", min_value=1.0, value=300.0, step=1.0)
        if slice_end <= slice_start:
            st.error("裁切结束时间必须大于起始时间。")
            return
        time_slice_s = (float(slice_start), float(slice_end))

    cth1, cth2, cth3 = st.columns(3)
    with cth1:
        stability_window = st.number_input("稳定性窗口大小", min_value=2, value=15, step=1)
    with cth2:
        stability_step = st.number_input("稳定性步长", min_value=1, value=5, step=1)
    with cth3:
        confidence_threshold = st.slider("置信度阈值", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

    st.subheader("绘图参数")
    component = st.selectbox("主要分量（波形/时频/LOFAR）", ["BHZ", "BH1", "BH2", "HYD"], index=0)
    normalize_waveform = st.checkbox("波形标准化显示", value=True)

    plot_items = st.multiselect(
        "选择输出图类型",
        ["波形图", "时频谱图", "LOFAR图", "方位角谱图", "方位稳定性图", "方位置信度图"],
        default=["波形图", "时频谱图", "LOFAR图", "方位角谱图", "方位稳定性图", "方位置信度图"],
    )
    if not plot_items:
        st.error("至少选择一种输出图类型。")
        return
    plot_flags = _build_plot_flags(plot_items)

    format_items = st.multiselect("输出文件格式", ["png", "pdf"], default=["png", "pdf"])
    if not format_items:
        st.error("至少选择一种输出文件格式。")
        return
    formats = tuple(format_items)

    st.subheader("输出设置")
    save_plots = st.checkbox("保存结果文件到结果目录", value=True)

    st.subheader("高级参数")
    a1, a2, a3 = st.columns(3)
    with a1:
        plot_font_name = st.text_input("字体名称", value="Helvetica")
    with a2:
        plot_dpi = st.number_input("图像 DPI", min_value=72, value=300, step=1)
    with a3:
        plot_grid_alpha = st.slider("网格透明度", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    a4, a5, a6 = st.columns(3)
    with a4:
        plot_fig_width = st.number_input("图宽（英寸）", min_value=2.0, value=7.2, step=0.1)
    with a5:
        plot_fig_height = st.number_input("图高（英寸）", min_value=2.0, value=3.2, step=0.1)
    with a6:
        plot_linewidth_waveform = st.number_input("波形线宽", min_value=0.1, value=0.4, step=0.1)

    cmap_options = ["viridis", "plasma", "magma", "cividis", "inferno", "turbo", "hsv", "RdYlBu_r"]
    cma, cmb, cmc, cmd, cme = st.columns(5)
    with cma:
        plot_cmap_spec = st.selectbox("时频色图", cmap_options, index=cmap_options.index("viridis"))
    with cmb:
        plot_cmap_lofar = st.selectbox("LOFAR色图", cmap_options, index=cmap_options.index("plasma"))
    with cmc:
        plot_cmap_azi = st.selectbox("方位角色图", cmap_options, index=cmap_options.index("hsv"))
    with cmd:
        plot_cmap_stability = st.selectbox("稳定性色图", cmap_options, index=cmap_options.index("RdYlBu_r"))
    with cme:
        plot_cmap_confidence = st.selectbox("置信度色图", cmap_options, index=cmap_options.index("magma"))

    run_clicked = st.button("开始运行", type="primary", use_container_width=True)
    if not run_clicked:
        return

    with st.spinner("正在执行数据处理，请稍候..."):
        try:
            run_info = run_pipeline(
                input_path=input_path,
                output_dir=output_dir,
                data_dir=data_dir,
                component=component,
                selected_band=selected_band,
                time_slice_s=time_slice_s,
                window_length_s=float(window_length_s),
                overlap=float(overlap),
                enable_demean=bool(enable_demean),
                enable_detrend=bool(enable_detrend),
                apply_orientation=bool(apply_orientation),
                orientation_deg=float(orientation_deg),
                stability_window=int(stability_window),
                stability_step=int(stability_step),
                confidence_threshold=float(confidence_threshold),
                save_plots=bool(save_plots),
                formats=formats,
                plot_waveform=plot_flags["plot_waveform"],
                plot_spectrogram=plot_flags["plot_spectrogram"],
                plot_lofar=plot_flags["plot_lofar"],
                plot_azimuth=plot_flags["plot_azimuth"],
                plot_azimuth_stability=plot_flags["plot_azimuth_stability"],
                plot_azimuth_confidence=plot_flags["plot_azimuth_confidence"],
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
        except Exception as exc:
            st.exception(exc)
            return

    st.success("运行完成。")
    st.write(f"事件ID：{run_info['event_id']}")
    st.write(f"分量：{run_info['component']}")
    st.write(f"频段：{run_info['selected_band']}")
    st.write(f"UTC起始时间：{run_info['utc_start_iso']}")

    st.subheader("运行日志")
    result_payload = run_info["result"]
    preprocess_report = result_payload.get("preprocess_report", {})
    st.text(f"- 事件：{run_info['event_id']}")
    st.text(f"- 频段：{run_info['selected_band']}")
    st.text(f"- UTC起始时间：{run_info['utc_start_iso']}")
    st.text(f"- 分量：{run_info['component']}")
    st.text(f"- 时间裁切：{result_payload.get('time_slice_s')}")
    st.text(f"- 预处理：{preprocess_report}")
    st.text(f"- 输出文件数量：{len(run_info['output_files'])}")

    output_paths = [Path(p) for p in run_info["output_files"]]
    output_paths = sorted(output_paths, key=lambda p: p.name)
    image_paths = [p for p in output_paths if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]

    st.subheader("主要结果图")
    if image_paths:
        for img in image_paths:
            st.image(str(img), caption=img.name, use_container_width=True)
    else:
        st.info("本次没有可显示的图像文件。")

    st.subheader("输出文件列表与下载")
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


if __name__ == "__main__":
    main()
