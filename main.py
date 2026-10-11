from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 只保存图片文件，不弹窗显示
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from config import (
    TRAIN_FILE,
    TEST_FILE,
    PREDICTION_FILE,
    VALIDATION_RESULT_FILE,
    RANDOM_STATE,
    VALIDATION_SIZE,
    EXPERIMENT_CONFIGS,
    MACRO_F1_CHART_FILE,
    CONFUSION_MATRIX_FILE,
    LOSS_CURVE_FILE,
)
from data_loader import load_data, split_data
from models import build_pipeline
from evaluation import run_one_experiment, choose_best_result, print_report
from output_utils import save_predictions, save_validation_results

# Windows 下显示中文需要指定中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_macro_f1_comparison(results_df: pd.DataFrame, output_file: Path) -> None:
    """按 macro_f1 从高到低画出所有配置的验证集对比柱状图。"""
    ranked = results_df.sort_values("macro_f1", ascending=False)
    colors = ["#d62728" if i == 0 else "#1f77b4" for i in range(len(ranked))]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(ranked["config"], ranked["macro_f1"], color=colors)
    ax.bar_label(bars, fmt="%.4f", fontsize=9)
    ax.set_ylabel("验证集 macro F1")
    ax.set_ylim(0, 1)
    ax.set_title("各配置验证集 macro F1 对比（红色为最佳配置）")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, output_file: Path) -> None:
    """画出最佳模型在验证集上的混淆矩阵热力图。"""
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        ax=ax,
        cmap="Blues",
        colorbar=True,
    )
    ax.set_title("最佳模型验证集混淆矩阵")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def plot_mlp_loss_curve(fitted_models: dict, output_file: Path) -> None:
    """Plot the MLP training loss for each completed epoch."""
    mlp_model = fitted_models.get("mlp_hidden_100")
    if mlp_model is None:
        print("未找到 MLP 实验，跳过 loss 曲线。")
        return

    classifier = mlp_model.named_steps["classifier"]
    loss_curve = getattr(classifier, "loss_curve_", None)
    if not loss_curve:
        print("MLP 没有可用的 loss_curve_，跳过 loss 曲线。")
        return

    epochs = range(1, len(loss_curve) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, loss_curve, marker="o", markersize=3, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("MLP 训练损失随 Epoch 的变化")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    print("已保存 MLP loss 曲线：", output_file.name)


def main():
    print("=" * 60)
    print("实验一：文本分类实验开始")
    print("=" * 60)

    # 第 1 步：读取数据
    print("\n[1/6] 正在读取数据...")
    texts, labels, test_texts = load_data(TRAIN_FILE, TEST_FILE)
    class_labels = sorted(int(value) for value in labels.unique())
    print("训练文本数量：", len(texts))
    print("训练标签数量：", len(labels))
    print("测试文本数量：", len(test_texts))
    print("类别数量：", labels.nunique())
    print("类别标签：", class_labels)
    if len(texts) != len(labels):
        raise ValueError("训练文本数量和训练标签数量不一致")
    if not set(class_labels).issubset(set(range(10))):
        raise ValueError("发现不在 0 到 9 范围内的标签")
    print("[PASS] 数据读取和基本检查完成")

    # 第 3 步：划分训练集和验证集
    print("\n[2/6] 正在划分训练集和验证集...")
    x_train, x_valid, y_train, y_valid = split_data(
        texts,
        labels,
        validation_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )
    print("训练部分：", len(x_train))
    print("验证部分：", len(x_valid))
    if len(x_train) != len(y_train) or len(x_valid) != len(y_valid):
        raise ValueError("划分后的文本和标签数量不一致")
    print("[PASS] 验证集划分完成，随机种子：", RANDOM_STATE)

    # 第 4-6 步：遍历配置，训练并比较模型
    print("\n[3/6] 正在训练并比较候选模型...")
    results = []
    fitted_models = {}
    for config in EXPERIMENT_CONFIGS:
        feature_mode = config.get("feature_mode", "word")
        print(
            f"正在训练：{config['name']} ({config['model']}, "
            f"features={feature_mode})"
        )
        pipeline = build_pipeline(
            config["model"],
            config["params"],
            feature_mode=feature_mode,
        )
        result, fitted_pipeline = run_one_experiment(
            config,
            pipeline,
            x_train,
            y_train,
            x_valid,
            y_valid,
        )
        print(
            f"[RESULT] {result['config']}: "
            f"accuracy={result['accuracy']:.4f}, "
            f"macro_f1={result['macro_f1']:.4f}, "
            f"seconds={result['seconds']:.2f}"
        )
        results.append(result)
        fitted_models[config["name"]] = fitted_pipeline

    if len(results) != len(EXPERIMENT_CONFIGS):
        raise ValueError("并非所有候选配置都完成了训练")
    results_df = pd.DataFrame(results)
    save_validation_results(results_df, VALIDATION_RESULT_FILE)
    model_names = {result["model"] for result in results}
    print(f"[PASS] 已完成 {len(results)} 组实验，比较算法：{sorted(model_names)}")
    print("[PASS] 已保存验证结果：", VALIDATION_RESULT_FILE.name)

    # 图 1：所有配置的验证集 macro F1 对比柱状图
    plot_macro_f1_comparison(results_df, MACRO_F1_CHART_FILE)
    print("已保存对比图：", MACRO_F1_CHART_FILE.name)
    plot_mlp_loss_curve(fitted_models, LOSS_CURVE_FILE)

    # 第 7 步：选择最佳模型
    print("\n[4/6] 正在根据验证集 macro F1 选择最佳模型...")
    best_name = choose_best_result(results_df, metric="macro_f1")
    print("最佳配置：", best_name)
    best_config = next(c for c in EXPERIMENT_CONFIGS if c["name"] == best_name)
    print("[PASS] 最佳模型只根据验证集结果选择，没有使用测试集标签")

    # 对最佳模型打印验证集分类报告（供实验报告使用），并画出混淆矩阵
    best_predictions = fitted_models[best_name].predict(x_valid)
    print("\n[5/6] 最佳模型验证集分类报告：")
    print_report(y_valid, best_predictions)
    plot_confusion_matrix(y_valid, best_predictions, CONFUSION_MATRIX_FILE)
    print("已保存混淆矩阵：", CONFUSION_MATRIX_FILE.name)

    # 第 8 步：使用全部有标签数据重新训练
    print("\n[6/6] 正在使用全部有标签数据训练最终模型...")
    final_model = build_pipeline(
        best_config["model"],
        best_config["params"],
        feature_mode=best_config.get("feature_mode", "word"),
    )
    final_model.fit(texts, labels)
    print("最终模型训练样本数：", len(texts))
    print("[PASS] 最终模型已使用全部有标签数据重新训练")

    # 第 9 步：生成提交文件
    print("正在预测测试集并保存 predictions.csv...")
    predictions = final_model.predict(test_texts)
    prediction_labels = sorted(int(value) for value in set(predictions))
    if len(predictions) != len(test_texts):
        raise ValueError("预测数量与测试样本数量不一致")
    if not set(prediction_labels).issubset(set(range(10))):
        raise ValueError("预测结果中出现不在 0 到 9 范围内的标签")
    save_predictions(predictions, PREDICTION_FILE)
    print("预测数量：", len(predictions))
    print("预测标签范围：", prediction_labels)
    print("[PASS] predictions.csv 已生成，格式为单列、无表头、无索引")
    print("\n实验一全部流程执行完成。")


if __name__ == "__main__":
    main()
