# Propeller Signal Monitoring — 四分量 OBS 船舶噪声自动识别系统

基于声压-水平分量互谱的船舶噪声方位反演与时频分析系统，用于从四分量海底地震仪（Ocean Bottom Seismometer, OBS）数据中自动识别、评估和可视化船舶噪声信号。

---

## 科学背景

海洋环境噪声中，船舶辐射噪声是最主要的人为噪声源之一。船舶噪声在频域上表现为窄带线谱（由螺旋桨、引擎等旋转机械产生），在空间上具有稳定的入射方位。利用 OBS 的四分量记录——两路水平地震分量（BH1/BH2）、垂直分量（BHZ）和水听器声压分量（HYD），可通过**声压-速度互谱（p-v cross-spectrum）**反演噪声源的水平方位角，结合 **STFT 时频分析**和 **LOFAR 归一化**识别窄带特征线谱，实现船舶噪声的自动检测与评估。

### 四分量说明

| 分量 | 含义 | 用途 |
|------|------|------|
| BH1 | 水平地震分量 1 | 经旋转后作为北向（N）分量 |
| BH2 | 水平地震分量 2 | 经旋转后作为东向（E）分量 |
| BHZ | 垂直地震分量 | 辅助分析 |
| HYD | 水听器声压 | 声压参考，互谱核心输入 |

---

## 处理流程

```
SAC 文件导入 → 四分量配对 → 预处理（去均值/去趋势/旋转）
    → 时间窗裁切 → STFT 时频分析（4分量）
    → 频带选择（自动推荐/手动指定）
    → 方位角谱估计（p-v 互谱）→ 方位稳定性评估
    → 置信度计算与掩膜 → SNR 评估
    → 多维度可视化输出
```

### 核心算法

1. **STFT 时频分析**：Hamming 窗短时傅里叶变换，功率谱 dB 转换
2. **LOFAR 归一化**：逐频率 Z-score 标准化，突出窄带线谱
3. **频带自动推荐**：MAD（中位数绝对偏差）阈值检测 × 能量排序，返回 top-3 候选频段
4. **方位角反演**：声压-水平速度互谱 → 声强矢量 → 水平方位角 arctan2(IE, IN)
5. **方位稳定性**：滑窗圆统计 R 值（0-1），量化方位一致性
6. **置信度图**：相干性（0.6）+ 归一化强度（0.4）加权融合
7. **SNR 评估**：噪声窗 RMS 参考 × 时变 RMS → dB 信噪比

---

## 项目结构

```
├── app.py                     Streamlit Web UI 入口
├── Easy_Example.ipynb         Jupyter 教学示例
├── requirements.txt           依赖清单
├── README.md                  项目说明
├── AGENTS.md                  项目维护注意事项
├── data/                      输入 SAC 数据目录
│   ├── .gitkeep
│   ├── testdata.bh1.sac       示例 — BH1 分量
│   ├── testdata.bh2.sac       示例 — BH2 分量
│   ├── testdata.bhz.sac       示例 — BHZ 分量
│   └── testdata.hyd.sac       示例 — HYD 分量
├── results/                   输出结果目录
│   └── .gitkeep
└── src/                       核心处理模块
    ├── __init__.py             公共 API 导出
    ├── pipeline.py             主流程编排
    ├── data_io.py              SAC 文件 I/O 与事件发现
    ├── preprocess.py           信号预处理
    ├── segment.py              时间窗裁切
    ├── spectral.py             频谱分析（STFT/LOFAR/SNR/频带推荐）
    ├── azimuth.py              方位角估计（互谱/稳定性/置信度）
    ├── plotting.py             可视化绘图（9 种图表类型）
    └── utils.py                通用工具函数
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

依赖项：

| 包 | 用途 |
|---|---|
| `streamlit` | Web UI 框架 |
| `numpy` | 数值计算 |
| `scipy` | 信号处理、统计 |
| `matplotlib` | 科学绘图 |
| `obspy` | 地震学 SAC 格式读写 |

### 2. 准备数据

将四分量 SAC 文件放入 `data/` 目录，每个事件需四个文件：

```
<event_prefix>.bh1.sac
<event_prefix>.bh2.sac
<event_prefix>.bhz.sac
<event_prefix>.hyd.sac
```

支持两种命名格式（按优先级）：`.<comp>.sac` > `.<comp>`（如 `.bh1.sac` 优先于 `.bh1`）。

也可在 UI 中直接上传 SAC 文件，系统会自动保存至 `data/`。

### 3. 启动 Web 界面

```bash
streamlit run app.py
```

### 4. 命令行烟测

```bash
python -c "from src.pipeline import run_pipeline; info = run_pipeline('testdata', data_dir='data', output_dir='results_smoke'); print(info['event_id'], len(info['output_files']))"
```

---

## Web UI 使用指南

界面采用**四步向导式布局**：

| 步骤 | 功能 | 操作 |
|------|------|------|
| Step 1 | 数据与事件 | 配置数据目录、上传 SAC 文件、选择事件、预览事件概况 |
| Step 2 | 参数配置 | 设置频段、时窗、预处理、裁切模式、绘图参数 |
| Step 3 | 运行控制 | 点击"开始运行"，查看实时日志与状态 |
| Step 4 | 结果查看 | 浏览 10 个 Tab 页，查看图表、下载结果文件 |

### 可配置参数

#### 数据输入
- **分析频段**（Hz）：手动指定或启用自动推荐（基于 HYD 功率谱 MAD 检测）
- **时窗长度**（秒）：STFT 窗长，控制频率分辨率与时间分辨率的权衡
- **重叠比例**：相邻窗之间的重叠率
- **时区偏移**（小时）：本地时间显示偏移（默认 +8）

#### 预处理
- **去均值 / 去趋势**：抑制直流分量与线性漂移
- **水平分量旋转**：按逆时针角度旋转 BH1/BH2 到地理方向
- **降采样**：重采样至指定目标采样率

#### 核心处理
- **时间裁切模式**：不裁切 / 手动指定起止秒 / 自动定长分段
- **稳定性窗口**：圆统计 R 值的滑窗大小与步长
- **置信度阈值**：0-1，低置信度区域将被 NaN 掩膜
- **SNR 噪声窗**：自动推荐（RMS 稳定性评分）或手动指定

#### 绘图输出
- **图类型选择**：波形图、时频谱图、LOFAR 图、SNR 曲线、方位角谱图、方位稳定性图、置信度图、置信度掩膜图
- **输出格式**：PNG / PDF
- **合并图模式**：多面板垂直堆叠
- **高级参数**：字体、DPI、图尺寸、线宽、网格透明度、5 组独立色图

---

## 可视化图表说明

| 图表 | 内容 | 判读要点 |
|------|------|---------|
| 波形图 | 时域信号 + 噪声窗标记 | 信号形态、事件起止、噪声段 |
| 时频谱图 | STFT 功率谱（dB），双时间轴 | 频率-时间能量分布 |
| LOFAR 图 | Z-score 归一化谱图 | 窄带线谱条纹（船舶特征） |
| SNR 曲线 | HYD 信噪比时间序列 | 信号相对噪声强度变化 |
| 方位角谱图 | 0-360° 方位角时频分布 | 噪声源水平方位随时间变化 |
| 方位稳定性 | 圆统计 R 值（0-1） | 方位估计的可靠性 |
| 置信度图 | 相干性 + 强度加权 | 高置信度 = 可靠方位 |
| 置信度掩膜 | 低置信度 NaN 剔除后的方位角 | 过滤后的可信方位 |

所有图表均采用**瑞士现代主义风格**（Helvetica、白底、细网格线），支持双时间轴（本地时间 + 相对秒数），通过百分位数法自适应确定色阶动态范围。

---

## API 参考

### 命令行

```python
from src.pipeline import run_pipeline, list_events, preview_auto_band

# 列出可用事件
events = list_events(data_dir="data")

# 预览自动频带推荐
band = preview_auto_band(event_id="testdata", data_dir="data")

# 执行完整处理流程
info = run_pipeline(
    event_id="testdata",
    data_dir="data",
    output_dir="results",
    selected_band=(20, 140),      # Hz
    window_length_s=2.0,          # STFT 窗长
    overlap=0.5,                  # 窗重叠率
    demean=True,                  # 去均值
    detrend=True,                 # 去趋势
    rotation_deg=0.0,             # 水平分量旋转角
    target_fs=None,               # 降采样目标（None=不降采样）
    crop_mode="none",             # 裁切模式
    crop_start_s=0,               # 手动裁切起始秒
    crop_end_s=None,              # 手动裁切结束秒
    auto_slice_length_s=None,     # 自动分段时长
    azimuth_window_size=15,       # 稳定性滑窗大小
    azimuth_step_size=5,          # 稳定性步长
    confidence_threshold=0.6,     # 置信度阈值
    noise_window_s=None,          # SNR 噪声窗
    no_snr=False,                 # 跳过 SNR
    plot_types=None,              # 图表类型列表
    save_options=None,            # 保存参数
    plot_params=None,             # 绘图样式
    verbose=True,
)
```

### 模块 API

| 模块 | 主要导出函数 |
|------|-------------|
| `src.data_io` | `find_sac_bundles()`, `load_bundle()`, `get_bundle_overview()` |
| `src.preprocess` | `preprocess_signals()`, `downsample_signals()` |
| `src.segment` | `crop_signals_by_time()` |
| `src.spectral` | `compute_stft()`, `power_db()`, `lofar_from_spectrogram()`, `suggest_frequency_bands()`, `compute_snr_from_spectrograms()`, `suggest_noise_window()` |
| `src.azimuth` | `compute_azimuth_spectrogram()`, `compute_azimuth_stability()`, `compute_confidence_map()`, `apply_confidence_mask()` |
| `src.plotting` | `plot_waveform()`, `plot_spectrogram()`, `plot_lofar()`, `plot_azimuth_spectrogram()`, `plot_azimuth_stability()`, `plot_azimuth_mask()`, `plot_confidence_map()`, `plot_snr_curve()`, `plot_merged_panels()` |

---

## 维护注意事项

- **不破坏命名规则**：兼容 `.<comp>` 与 `.<comp>.sac` 两种命名，冲突时优先 `.sac`
- **保持 `run_pipeline()` 返回结构兼容**：下游 UI 与脚本依赖其返回字典结构
- **不覆盖原始数据**：`data/` 目录仅供读取
- **不在无提示下清空 `results/`**

---

## License

MIT License
