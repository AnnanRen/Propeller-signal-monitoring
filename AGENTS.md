# AGENTS.md

## 项目概述

这是一个用于螺旋桨实验/水声地震数据处理的 Python 项目。项目读取同一事件的四分量 SAC 文件（`BH1`、`BH2`、`BHZ`、`HYD`），通过 Streamlit 图形界面完成数据选择、预处理、时频分析、LOFAR 分析、方位角谱计算、方位稳定性评估和置信度掩膜绘图，并将结果保存为图片/PDF。

普通用户入口是 `app.py`，教学入口是 `Easy_Example.ipynb`。核心处理流程由 `src/pipeline.py` 编排，底层算法按数据读取、预处理、分段、谱分析、方位角计算、绘图拆分到多个模块。

## 目录组织

- `app.py`：Streamlit UI 入口。负责参数表单、文件上传、调用处理流程、展示日志、展示结果图和下载按钮。
- `src/`：核心代码包。
  - `src/data_io.py`：查找、读取和组装 SAC 四分量事件。
  - `src/preprocess.py`：去均值、去趋势、带通滤波、水平分量方位旋转。
  - `src/segment.py`：按时间裁切信号。
  - `src/spectral.py`：STFT、功率谱、LOFAR、自动推荐频段。
  - `src/azimuth.py`：方位角谱、方位稳定性、置信度图和掩膜。
  - `src/plotting.py`：Matplotlib 绘图和输出保存。
  - `src/pipeline.py`：事件级流程编排、参数对象、绘图保存和返回运行信息。
  - `src/utils.py`：通用小工具。
- `data/`：输入 SAC 数据目录。完整事件必须包含同前缀的四个文件：`*.bh1.sac`、`*.bh2.sac`、`*.bhz.sac`、`*.hyd.sac`。
- `results/`：默认输出目录，保存运行生成的 `.png` 和 `.pdf` 结果图。
- `Easy_Example.ipynb`：教学/演示 notebook，保留给用户交互探索。
- `README.md`：面向用户的安装、启动和最小操作流程说明。
- `requirements.txt`：运行依赖列表。

## 不要随意改动的文件和数据

- 不要删除或覆盖 `data/` 中的 SAC 数据。它们是处理入口和示例数据；新增测试数据时保持四分量同前缀命名。
- 不要随意清空 `results/`。这里保存用户运行结果；如需验证输出，优先写入新的临时输出目录或明确说明会覆盖哪些文件。
- 不要无理由重写 `Easy_Example.ipynb`。Notebook 容易产生大 diff，除非任务明确要求更新教学流程。
- 不要随意改变输出文件命名规则：当前格式是 `{event_id}_{module}_{component}.{fmt}`，其中方位角类图使用 `ALL` 作为 component。
- 不要随意改变 SAC 文件匹配规则。`find_sac_bundles()` 依赖小写后缀 `.bh1.sac/.bh2.sac/.bhz.sac/.hyd.sac` 查找完整事件。
- 不要提交 `src/__pycache__/`、`.pyc`、临时缓存或本地环境文件。

## 代码风格

- 使用 Python 3 风格代码，保留 `from __future__ import annotations`。
- 优先使用 `pathlib.Path` 处理路径，不要拼接字符串路径。
- 保持模块边界清晰：UI 留在 `app.py`，流程编排留在 `src/pipeline.py`，算法细节留在对应模块。
- 新增参数时尽量同时更新 Streamlit UI、`run_pipeline()`、`PipelineParams` 和必要的绘图/算法调用，避免 UI 与后端参数脱节。
- 数值计算使用 NumPy/SciPy 的向量化能力；不要用低效循环重写现有数组运算，除非逻辑需要。
- 绘图继续使用 Matplotlib，并通过 `PlotParams`、`SaveOptions` 传递样式和保存设置。
- 错误信息应说明具体输入、参数或事件名，便于用户在 UI 中定位问题。
- 保持中文用户文案和 README 风格一致；代码内部函数名、变量名使用英文。

## 运行命令

安装依赖：

```bash
pip install -r requirements.txt
```

启动 Streamlit 界面：

```bash
streamlit run app.py
```

在 Python 中直接调用流程（示例会写入输出目录）：

```bash
python -c "from src.pipeline import run_pipeline; print(run_pipeline('testdata', output_dir='results_smoke', data_dir='data')['output_files'])"
```

仅检查可发现的完整事件：

```bash
python -c "from src.pipeline import list_events; print([e.event_id for e in list_events('data')])"
```

## 测试方式

当前仓库没有独立的自动化测试套件。修改代码后至少执行以下检查：

```bash
python -m compileall app.py src
```

如果修改了数据读取、流程参数、预处理、谱分析、方位角或绘图逻辑，还应做一次烟测：

```bash
python -c "from src.pipeline import run_pipeline; info = run_pipeline('testdata', output_dir='results_smoke', data_dir='data'); print(info['event_id'], len(info['output_files']))"
```

如果修改了 UI，运行：

```bash
streamlit run app.py
```

然后在浏览器中确认能发现事件、能运行、能显示结果图、能列出下载文件。

## 输出要求

- 默认输出目录是 `results/`，除非用户或调用参数指定其他目录。
- 支持的输出格式来自 UI/参数中的 `formats`，当前常用为 `png` 和 `pdf`。
- 输出图类型包括波形图、时频谱图、LOFAR 图、方位角谱图、方位稳定性图、方位置信度图。
- 输出文件命名必须稳定，便于 UI 根据事件、模块和分量重新定位文件。
- `run_pipeline()` 返回值应至少包含 `event_id`、`component`、`selected_band`、`utc_start_iso`、`output_files`、`logs`、`result`，不要破坏现有 UI 对这些键的依赖。
- 处理完成后，UI 应显示事件 ID、分量、频段、UTC 起始时间、运行日志、主要结果图和输出文件下载按钮。
