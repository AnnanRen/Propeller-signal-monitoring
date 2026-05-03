# AGENTS.md

## 项目定位

本项目用于从四分量 OBS 事件（`BH1/BH2/BHZ/HYD`）中自动识别与评估船舶噪声，并输出可视化结果供人工判读。

## 处理流程与原理（简述）

1. 事件发现与四分量配对：按统一前缀 + 固定后缀（`.bh1/.bh2/.bhz/.hyd.sac`）匹配完整事件。
2. 预处理：去均值和去趋势以抑制直流与慢漂移；按设定角度旋转 `BH1/BH2` 到水平参考方向。
3. STFT 时频分析：对各分量分窗做短时傅里叶变换，得到频率-时间能量分布。
4. LOFAR 归一化：逐频标准化谱图，突出窄带线谱特征，便于识别船噪频率条纹。
5. 频带建议：依据中位谱与 MAD 阈值筛选候选高能频带并给出推荐频段。
6. 方位估计：利用声压（HYD）与水平分量互谱关系估计方位角谱（0-360°）。
7. 稳定性与置信度：滑窗圆统计评估方位稳定性，结合相干与强度构建置信度并做低置信度掩膜。

## 如何运行（简洁）

默认输入目录：`data/`  
默认输出目录：`results/`

```bash
pip install -r requirements.txt
streamlit run app.py
python -c "from src.pipeline import run_pipeline; info = run_pipeline('testdata', data_dir='data', output_dir='results_smoke'); print(info['event_id'], len(info['output_files']))"
```

## 项目路径结构

- `app.py`：Streamlit 界面入口（参数、运行、展示与下载）。
- `src/`：处理流程与算法模块。
- `data/`：输入 SAC 数据目录。
- `results/`：分析结果输出目录。
- `README.md`、`Easy_Example.ipynb`：使用说明与示例。

## src 模块作用（简洁）

- `src/pipeline.py`：主流程编排与结果组织。
- `src/data_io.py`：事件发现、四分量配对与 SAC 读取。
- `src/preprocess.py`：去均值、去趋势、水平分量旋转。
- `src/segment.py`：时间窗裁切。
- `src/spectral.py`：STFT、功率谱、LOFAR、频带建议、SNR 相关计算。
- `src/azimuth.py`：方位角谱、方位稳定性、置信度与掩膜。
- `src/plotting.py`：绘图与落盘保存。
- `src/utils.py`：通用辅助函数。

## 维护注意事项（高风险）

- 不要破坏四分量命名规则（`.bh1/.bh2/.bhz/.hyd.sac`）。
- 保持 `run_pipeline()` 返回结构兼容。
- 输出文件命名尽量保持稳定。
- 不覆盖 `data/` 原始数据。
- 不在无提示情况下清空 `results/`。

## 最低验证要求

```bash
python -m compileall app.py src
```

若改动读取、预处理、谱分析、方位或绘图逻辑，再补一次 `run_pipeline()` 烟测。
