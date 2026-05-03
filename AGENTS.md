# AGENTS.md

## 先理解这个项目

这是一个面向螺旋桨实验/水声地震数据处理的 Python 项目。它处理的是同一事件的四分量 SAC 数据：

- `BH1`
- `BH2`
- `BHZ`
- `HYD`

项目的核心目标不是做通用文件管理，而是把一组完整的四分量事件数据，经过预处理和频谱/方位分析，产出一组可视化结果图，供用户判断频带、能量分布、方位角特征和方位稳定性。

它更像一个“小型分析工作台”：

- `app.py` 提供 Streamlit 图形界面，方便普通用户点选参数并运行
- `src/pipeline.py` 把整个处理流程串起来
- `src/` 下其他模块分别负责读数据、预处理、分段、谱分析、方位角分析和绘图
- `results/` 保存运行后的图像/PDF 结果

如果你刚接手这个仓库，先把它理解成：

1. 读入一个完整事件的四分量 SAC 文件
2. 做预处理，例如去均值、去趋势、带通、水平分量方位矫正
3. 做时频分析和 LOFAR 分析
4. 在选定频段上估计方位角，并评估方位稳定性与置信度
5. 输出一组图，供用户在界面中查看和下载

## 这个项目适合怎么读

推荐按下面顺序理解，而不是一开始就钻算法细节：

1. 看 `README.md`
   先理解项目怎么启动、数据放哪、结果放哪。
2. 看 `app.py`
   理解用户在界面里能调哪些参数，界面最终如何调用主流程。
3. 看 `src/pipeline.py`
   这是最重要的总控文件，能看清一次运行到底经历了哪些步骤。
4. 再按需看 `src/` 子模块
   当你要修改某一块逻辑时，再进入对应模块。

如果只是想快速回答“项目是干嘛的”，通常读完 `README.md`、`app.py` 和 `src/pipeline.py` 就够了。

## 一次运行实际发生了什么

从代码结构看，一次典型运行大致是这样：

1. 在 `data/` 中找到完整事件，或者由 UI 上传 `.sac` 文件后写入 `data/`
2. 按事件 ID 组装出该事件的四个分量文件
3. 读取波形与采样信息
4. 视参数决定是否做时间裁切
5. 做预处理
6. 对每个分量做 STFT，得到时频谱
7. 从谱图生成 LOFAR 结果
8. 在用户指定或自动推荐的频段内做方位角谱计算
9. 计算方位稳定性和置信度，并生成掩膜结果
10. 保存图像到输出目录，并把结果信息返回给 UI

因此，这个项目的主线是：

`四分量事件数据 -> 预处理 -> 频谱分析 -> 方位角分析 -> 绘图输出`

## 主要输入和输出

### 输入

- 默认输入目录是 `data/`
- 一个完整事件必须包含同前缀的四个文件：
  - `*.bh1.sac`
  - `*.bh2.sac`
  - `*.bhz.sac`
  - `*.hyd.sac`

例如 `testdata.bh1.sac`、`testdata.bh2.sac`、`testdata.bhz.sac`、`testdata.hyd.sac` 会被识别为同一个事件。

### 输出

- 默认输出目录是 `results/`
- 常见输出格式是 `png` 和 `pdf`
- 常见输出图包括：
  - 波形图
  - 时频谱图
  - LOFAR 图
  - 方位角谱图
  - 方位稳定性图
  - 方位置信度图

输出的价值在于：用户不是只拿到一个数值结论，而是拿到一组可以直观看到时频特征和方位特征的分析图。

## 目录职责

- `app.py`
  Streamlit 入口。负责收集参数、触发运行、展示日志、展示结果图和下载文件。
- `src/pipeline.py`
  主流程编排入口。把读取、预处理、谱分析、方位分析和绘图保存串成一次完整运行。
- `src/data_io.py`
  负责查找和读取四分量 SAC 事件。
- `src/preprocess.py`
  负责去均值、去趋势、带通滤波、水平分量方位旋转等预处理。
- `src/segment.py`
  负责按时间裁切信号。
- `src/spectral.py`
  负责 STFT、功率谱、LOFAR 和频段推荐。
- `src/azimuth.py`
  负责方位角谱、方位稳定性、置信度和掩膜计算。
- `src/plotting.py`
  负责 Matplotlib 绘图和落盘保存。
- `Easy_Example.ipynb`
  教学/演示入口，不是主业务入口。
- `README.md`
  面向普通用户的运行说明。

## 哪些地方最容易改坏

下面这些约束比“代码风格”更重要，因为它们直接关系到项目是否还能正常跑：

- 不要随意改 SAC 文件匹配规则
  事件发现依赖 `.bh1.sac/.bh2.sac/.bhz.sac/.hyd.sac` 这一套命名。
- 不要破坏 `run_pipeline()` 返回结构
  UI 依赖返回结果中的事件信息、日志、输出文件列表和结果对象。
- 不要随意改变输出命名规则
  UI 和用户都会依赖稳定文件名重新定位结果。
- 不要删除或覆盖 `data/` 里的原始数据
  这是处理入口，也是示例数据。
- 不要随意清空 `results/`
  里面可能有用户已经生成的分析结果。

## 改代码时的实用原则

- 新增参数时，优先检查这四处是否要一起更新：
  - `app.py`
  - `run_pipeline()`
  - `PipelineParams`
  - 对应算法/绘图调用
- 保持职责边界清晰：
  - UI 逻辑放在 `app.py`
  - 流程调度放在 `src/pipeline.py`
  - 算法细节放在各自模块
- 路径优先用 `pathlib.Path`
- 数值计算优先延续 NumPy/SciPy 向量化写法
- 用户可见文案保持中文风格一致
- 错误信息尽量带上事件名、输入路径或参数名，方便在 UI 中定位问题

## 常用运行方式

安装依赖：

```bash
pip install -r requirements.txt
```

启动界面：

```bash
streamlit run app.py
```

列出可发现事件：

```bash
python -c "from src.pipeline import list_events; print([e.event_id for e in list_events('data')])"
```

直接做一次烟测运行：

```bash
python -c "from src.pipeline import run_pipeline; info = run_pipeline('testdata', output_dir='results_smoke', data_dir='data'); print(info['event_id'], len(info['output_files']))"
```

## 最低验证要求

当前仓库没有独立测试套件。改完代码后，至少做：

```bash
python -m compileall app.py src
```

如果修改了读取、预处理、谱分析、方位角或绘图逻辑，再做一次 `run_pipeline()` 烟测。

如果修改了 UI，再启动一次 Streamlit，确认：

- 能发现事件
- 能成功运行
- 能显示结果图
- 能列出可下载文件
