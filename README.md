# 实验一：新闻文本分类

本项目使用 TF-IDF 和经典机器学习模型完成英文新闻文本 10 分类。

## 1. 安装依赖

建议使用 Python 3.10 或更高版本：

```bash
conda create -n project1 python=3.10
conda activate project1
pip install -r requirements.txt
```

## 2. 运行

在包含 `main.py` 的目录中执行：

```bash
python main.py
```

程序会完成数据读取、分层划分验证集、模型训练比较、最佳模型选择、最终重训、测试集预测和图表生成。

## 3. 当前最佳结果

```text
配置：svm_tfidf_50k_min_df1_c_10.0
Accuracy：0.919267
Macro-F1：0.920120
```

该配置使用线性 SVM `C=10.0`，TF-IDF 参数为 `max_features=50000`、`min_df=1`。

## 4. 输出文件

| 文件 | 作用 |
|---|---|
| `validation_results.csv` | 各配置的验证集指标 |
| `predictions.csv` | 测试集预测结果，2457 行、单列、无表头 |
| `macro_f1_comparison.png` | 不同配置的 Macro-F1 对比图 |
| `confusion_matrix.png` | 最佳模型验证集混淆矩阵 |
| `mlp_loss_curve.png` | MLP 训练 loss 曲线 |

## 5. 代码结构

| 文件 | 作用 |
|---|---|
| `main.py` | 主流程和结果可视化 |
| `config.py` | 路径、随机种子、TF-IDF 参数和实验配置 |
| `data_loader.py` | 读取数据和划分验证集 |
| `preprocessing.py` | 文本清洗和 TF-IDF 构建 |
| `models.py` | 创建分类器和 Pipeline |
| `evaluation.py` | 训练、评估和最佳模型选择 |
| `output_utils.py` | 保存 CSV 结果和预测文件 |
