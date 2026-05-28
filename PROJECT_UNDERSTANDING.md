# 项目全景分析：四分量 OBS 船舶噪声自动识别系统

## 一、项目定位

本项目用于从**四分量 OBS（Ocean Bottom Seismometer，海底地震仪）事件**中自动识别与评估**船舶噪声**，并输出可视化结果供人工判读。

四分量包括：
- **BH1 / BH2**：两路水平地震分量（正交安放）
- **BHZ**：垂直地震分量
- **HYD**：水听器（Hydrophone）声压分量

核心思路：利用声压（HYD）与水平分量（BH1/BH2）的互谱关系反演噪声源方位，结合时频分析（STFT/LOFAR）识别船舶窄带线谱特征，输出多维度可视化图表。

---

## 二、项目文件结构

```
project01/
├── app.py                  Streamlit Web UI（~1100行）
├── Easy_Example.ipynb      教学示例 Jupyter Notebook
├── requirements.txt        5个依赖
├── README.md               使用说明
├── AGENTS.md               项目定位与维护注意事项
├── data/                   输入数据目录
│   ├── testdata.bh1.sac    测试事件 — BH1 分量
│   ├── testdata.bh2.sac    测试事件 — BH2 分量
│   ├── testdata.bhz.sac    测试事件 — BHZ 分量
│   └── testdata.hyd.sac    测试事件 — HYD 分量
├── results/                输出结果目录
└── src/                    核心处理模块
    ├── __init__.py          公共 API 导出
    ├── pipeline.py          主流程编排
    ├── data_io.py           SAC 文件 I/O
    ├── preprocess.py        信号预处理
    ├── segment.py           时间窗裁切
    ├── spectral.py          频谱分析
    ├── azimuth.py           方位角估计
    ├── plotting.py          可视化绘图
    └── utils.py             通用工具函数
```

### 依赖项

```
streamlit    # Web UI 框架
numpy        # 数值计算
scipy        # 科学计算（信号处理、插值、统计）
matplotlib   # 绑图
obspy        # 地震学数据 I/O（SAC 格式读写）
```

---

## 三、处理流程详解

### 3.1 事件发现与四分量配对（`data_io.py`）

**入口函数**：`find_sac_bundles(data_dir) → List[SACBundle]`

- 扫描数据目录下所有文件
- 按命名规则解析 `event_id` 与分量类型
- 兼容两种命名格式（按优先级）：
  1. `xxx.bh1.sac` / `xxx.bh2.sac` / `xxx.bhz.sac` / `xxx.hyd.sac`（优先）
  2. `xxx.bh1` / `xxx.bh2` / `xxx.bhz` / `xxx.hyd`（备选）
- 过滤系统文件（`.` 开头、`._` 开头、`.DS_Store`）
- 冲突时优先 `.sac` 后缀（priority 0 < priority 1）
- 四分量齐全才视为完整事件，封装为 `SACBundle` 数据类

**SAC 读取**：通过 ObsPy 的 `read()` 读取 SAC 文件，提取：
- `t_sec`：时间轴（秒）
- `data`：信号采样值（float64）
- `meta`：SAC 头段信息（`nzyear`, `nzjday`, `nzhour`, `nzmin`, `nzsec`, `nzmsec`）
- `fs`：采样率（由 `delta` 计算，`fs = 1/delta`）

**事件概况**（`get_bundle_overview`）：读取四分量 SAC 头段（仅元数据），检查采样率与采样点数一致性，计算时长。

---

### 3.2 预处理（`preprocess.py`）

**函数**：`preprocess_signals(signals, fs, ...) → (processed_signals, report)`

处理步骤（顺序执行）：

1. **去均值（Demean）**：每个分量减去自身均值，抑制直流分量
2. **去趋势（Detrend）**：`scipy.signal.detrend(type="linear")`，消除慢漂移
3. **水平分量旋转（Orientation Correction）**：
   ```
   BH1_new = BH1 * cos(θ) - BH2 * sin(θ)    # 北分量
   BH2_new = BH1 * sin(θ) + BH2 * cos(θ)    # 东分量
   ```
   其中 θ 为用户指定的逆时针旋转角度（默认 0°）

**降采样**（`downsample_signals`）：
- 通过有理数重采样（`scipy.signal.resample_poly`）降低采样率
- 使用 `Fraction.limit_denominator` 保证精确有理数比率（最大分母 100000）
- 输出报告包含实际采样率与绝对误差

---

### 3.3 时间窗裁切（`segment.py`）

**函数**：`crop_signals_by_time(t_sec, signals, start_s, end_s)`

- 使用 `TimeSlice` 数据类，支持规范化（clamp 到有效范围）
- 按采样间隔 `dt` 将秒级时间范围映射到数组索引
- 裁切后时间轴归零（`t_crop = t_crop - t_crop[0]`）
- 支持三种裁切模式：
  - **不裁切**：使用全部数据
  - **手动裁切**：用户指定起止秒
  - **自动分段**：按固定时长切分为多个 segment，每个独立处理

---

### 3.4 时频分析（`spectral.py`）

#### STFT（Short-Time Fourier Transform）

**函数**：`compute_stft(x, fs, params) → (f_hz, t_spec, s_complex)`

- 窗函数：Hamming 窗
- 窗长：`nperseg = max(2, round(window_length_s * fs))`
- 重叠：`noverlap = round(nperseg * overlap)`，限制在 `[0, nperseg-1]`
- FFT 点数：`nfft = max(8, nperseg)`
- 返回复数谱（`mode="complex"`）
- 如果信号段短于窗长，自动缩减窗长并记录警告

#### 功率谱（dB）

**函数**：`power_db(s_complex) → s_db`

```
S_dB = 10 * log10(|S|² + ε)
```

#### LOFAR 归一化

**函数**：`lofar_from_spectrogram(s_db) → lofar_map`

逐频率行做 Z-score 标准化：`(row - mean(row)) / std(row)`

目的是突出窄带线谱特征，抑制宽带背景噪声，便于识别船舶频率条纹。

#### 频带推荐

**函数**：`suggest_frequency_bands(f_hz, s_db, fs, window_length_s)`

基于 HYD（水听器）功率谱自动推荐分析频段：

1. 约束：下限 ≥ max(5Hz, 2×频率分辨率)，上限 ≤ min(0.8×Nyquist, f_max)
2. 计算频率轴上的中位功率谱
3. MAD 阈值检测：`threshold = median + 1.5 × MAD`
4. 合并连续超阈值频率区间为候选频段
5. 过滤带宽 < min(30Hz, 8×分辨率) 的窄带
6. 按能量强度（mean - median）排序，取 top-3
7. 无候选时回退到 20-140 Hz 默认宽带

#### SNR 计算

**函数**：`compute_snr_from_spectrograms(spectrograms_sel, t_spec, noise_window_s)`

- 计算每个 STFT 时间窗的 RMS 幅度（频率维度的均方根）
- 噪声窗 RMS = 噪声窗内 RMS 的均方根
- `SNR(t) = 20 * log10(RMS(t) / noise_RMS)`
- 全段 SNR = SNR(t) 的均值（dB）

**自动噪声窗推荐**（`suggest_noise_window`）：
- 扫描 RMS 时间序列，滑窗计算局部均值与标准差
- 归一化后综合评分（低均值 + 低标准差 = 更可能是纯噪声段）
- 选取评分最低的窗口作为噪声参考

---

### 3.5 方位角估计（`azimuth.py`）

#### 方位角谱

**函数**：`compute_azimuth_spectrogram(s_p, s_vn, s_ve)`

利用声压（HYD）与水平速度分量（BH1/BH2）的互谱：

```
I_N = 0.5 * Re(conj(S_P) * S_VN)    # 北向声强
I_E = 0.5 * Re(conj(S_P) * S_VE)    # 东向声强
I   = sqrt(I_N² + I_E²)              # 总声强

Azimuth = arctan2(I_E, I_N) × 180/π  # 0-360°
```

低强度区域（intensity ≤ ε）标记为 NaN。

#### 方位稳定性（圆统计 R 值）

**函数**：`compute_azimuth_stability(azimuth_deg_tf, t_spec, window_size, step_size)`

- 对每个频率，滑窗计算圆形统计 R 值：
  ```
  R = sqrt((Σ cos(θ_i))² + (Σ sin(θ_i))²) / N
  ```
- R ∈ [0, 1]：R=1 表示方位完全一致，R=0 表示随机分布
- 窗口间线性插值回原始时间网格
- 窗口大小默认 15 个 STFT 帧，步长默认 5 帧

#### 置信度图

**函数**：`compute_confidence_map(s_p, s_vn, s_ve, intensity)`

```
coherence = |conj(S_P) * (S_VN + j*S_VE)| / (|S_P| * |V_H| + ε)   # 相干性
I_score   = clip(intensity / P95(intensity), 0, 1)                 # 强度分

confidence = 0.6 * coherence + 0.4 * I_score                        # 加权融合
```

#### 置信度掩膜

**函数**：`apply_confidence_mask(azimuth_deg_tf, confidence_tf, threshold)`

```python
azimuth[confidence < threshold] = NaN
```

默认阈值 0.6，屏蔽低置信度区域的方位角估计结果。

---

### 3.6 主流程编排（`pipeline.py`）

#### `run_pipeline()` — 顶层入口

```
输入 → 事件解析 → 时间裁切 → 降采样(可选) → 预处理 → STFT×4分量
→ 频带选择(mode: auto/manual) → 频带裁剪
→ 方位角谱 → 方位稳定性 → 置信度图 → 置信度掩膜 → SNR(可选)
→ 绑图输出(mode: merged/individual)
→ 返回结果字典
```

#### 自动分段模式

当指定 `auto_slice_length_s` 时，将全时段切分为多个定长 segment：
- 每个 segment 独立执行完整 pipeline
- 短到无法做 STFT 的段自动跳过并记录警告
- 所有 segment 结果汇总到统一输出

#### 输出结构

```python
{
    "event_id": str,           # 事件ID
    "component": str,          # 主分量名
    "selected_band": tuple,    # 选中的频段 (fmin, fmax)
    "utc_start_iso": str,      # UTC起始时间 ISO格式
    "output_files": [str],     # 输出文件路径列表
    "logs": [str],             # 运行日志
    "result": dict,            # 完整处理结果（含所有中间数据）
    "segments": [dict],        # 分段信息
}
```

---

## 四、可视化系统（`plotting.py`）

### 支持的可视化类型（9种 + 合并图）

| 图表类型 | 函数 | 内容 |
|---------|------|------|
| 波形图 | `plot_waveform()` | 时域信号 + 噪声窗标记 |
| 时频谱图 | `plot_spectrogram()` | STFT 功率谱（dB） |
| LOFAR 图 | `plot_lofar()` | 归一化谱图 |
| SNR 曲线 | `plot_snr_curve()` | HYD 分量 SNR(t) + 噪声窗 |
| 方位角谱图 | `plot_azimuth_spectrogram()` | 0-360° 方位角时频分布 |
| 方位置信度掩膜 | `plot_azimuth_mask()` | 低置信度 NaN 剔除后的方位角 |
| 方位置信度图 | `plot_confidence_map()` | 置信度（0-1）时频分布 |
| 方位稳定性图 | `plot_azimuth_stability()` | 圆统计 R 值时频分布 |
| 合并图 | `plot_merged_panels()` | 所选面板垂直堆叠 |

### 设计特点

- **双时间轴**：主轴显示本地时间（时区偏移可配置，默认 +8），副轴显示相对秒数
- **瑞士现代主义风格**：Helvetica 字体、白底、细网格线
- **自适应色阶**：百分位数法（2-98% 或 5-98%）自动确定动态范围
- **噪声窗可视化**：手动窗显示绿色，自动窗显示橙色
- **输出格式**：支持 PNG + PDF 同时输出

---

## 五、Streamlit UI（`app.py`）

### 设计系统

完整定义了一套 CSS Design Token 体系：

- **主色**：`#1E3A5F`（深蓝）
- **字体**：Inter（系统 UI）/ JetBrains Mono（等宽数据）
- **圆角**：8px（小）/ 12px（中）/ 16px（大）
- **阴影**：3级精细阴影（sm/md/lg）
- **交互状态**：悬浮过渡动画、聚焦光环

### 四步向导式布局

| Step | 名称 | 内容 |
|------|------|------|
| Step 1 | 数据与事件 | 目录配置、SAC文件上传、事件选择、事件概况预览 |
| Step 2 | 参数配置 | 频段、时窗、预处理开关、裁切模式、绘图参数（含高级色图选项） |
| Step 3 | 运行控制 | 运行按钮、状态指示徽章、运行中横幅（脉冲动画）、异常捕获 |
| Step 4 | 结果查看 | 10个Tab页 + 概要卡片 + 文件下载 |

### Step 2 参数全景

**数据输入**：
- 分析频段（Hz）— 手动或自动推荐
- 时窗长度（秒）、重叠比例
- 时区偏移（小时）

**预处理**：
- 去均值 / 去趋势 / 方位角矫正
- 矫正角度（度，逆时针为正）
- 降采样目标采样率

**核心处理**：
- 时间裁切模式（不裁切 / 手动 / 自动分段）
- 稳定性窗口大小、步长
- 置信度阈值
- SNR 噪声窗（自动 / 手动）

**绘图**：
- 输出图类型选择（8种）
- 输出文件格式（PNG / PDF）
- 合并图开关
- 高级参数：字体、DPI、图尺寸、线宽、网格透明度、5组色图

### 状态管理

使用 `st.session_state` 管理运行状态：
- `run_status`：idle → running → success / error
- 运行结果持久化到 session_state，支持多次查看
- 合并图模式下临时渲染图中面板到独立 Tab

---

## 六、关键约束与注意事项

1. **不破坏四分量命名规则**：兼容 `.bh1/.bh2/.bhz/.hyd` 与对应 `.sac` 形式，冲突时优先 `.sac`
2. **保持 `run_pipeline()` 返回结构兼容**：下游（UI、脚本）依赖其返回字典结构
3. **输出文件命名保持稳定**：`{event_id}_{module}_{component}.{fmt}`
4. **不覆盖 `data/` 原始数据**：只读取 SAC 文件
5. **不在无提示情况下清空 `results/`**
6. **STFT 窗长自动适配**：信号段短于窗长时缩减窗长并记录警告
7. **频段必须 ≤ Nyquist 频率**：`selected_band_max ≤ fs/2`

---

## 七、模块依赖关系

```
app.py
  └── src/pipeline.py
        ├── src/data_io.py          (find_sac_bundles, load_bundle, parse_event_component_from_filename)
        ├── src/preprocess.py       (preprocess_signals, downsample_signals)
        ├── src/segment.py          (crop_signals_by_time)
        ├── src/spectral.py         (compute_stft, power_db, lofar_from_spectrogram,
        │                            suggest_frequency_bands, compute_snr_from_spectrograms,
        │                            suggest_noise_window)
        ├── src/azimuth.py          (compute_azimuth_spectrogram, compute_azimuth_stability,
        │                            compute_confidence_map, apply_confidence_mask)
        └── src/plotting.py         (plot_waveform, plot_spectrogram, plot_lofar,
                                     plot_azimuth_spectrogram, plot_azimuth_stability,
                                     plot_azimuth_mask, plot_confidence_map, plot_snr_curve,
                                     plot_merged_panels, PlotParams, SaveOptions)
              └── src/utils.py      (ensure_dir, zscore_safe)
```

---

## 八、快速验证

```bash
# 安装依赖
pip install -r requirements.txt

# 编译检查
python -m compileall app.py src

# 启动 UI
streamlit run app.py

# 命令行烟测
python -c "from src.pipeline import run_pipeline; info = run_pipeline('testdata', data_dir='data', output_dir='results_smoke'); print(info['event_id'], len(info['output_files']))"
```
