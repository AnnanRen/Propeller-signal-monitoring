# 数据处理项目（Streamlit UI）

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 启动界面

```bash
streamlit run app.py
```

## 3. 数据放置位置

- 输入数据目录：`data/`
- 每个事件需要同前缀四个文件：
  - `xxx.bh1.sac`
  - `xxx.bh2.sac`
  - `xxx.bhz.sac`
  - `xxx.hyd.sac`

你也可以在 UI 中直接上传 `.sac` 文件，系统会自动保存到 `data/`。

## 4. 结果保存位置

- 输出目录：`results/`
- 处理后生成的图像和文件统一保存在该目录。

## 5. 最小操作流程

1. 把 `.sac` 数据放到 `data/`（或在 UI 上传）。
2. 运行 `streamlit run app.py`。
3. 打开浏览器后：
   - 选择事件
   - 调整参数（数据输入、预处理、核心处理、绘图、输出、高级参数）
   - 点击“开始运行”
4. 查看页面中的运行日志、结果图和输出文件列表。
5. 点击下载按钮获取结果文件。

## 6. 入口说明

- 普通用户入口：`app.py`（Streamlit UI）
- 教学示例入口：`Easy_Example.ipynb`（保留）
