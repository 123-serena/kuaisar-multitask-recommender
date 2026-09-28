"""
把多个模型组合的 metrics_curve + loss_curve 拼到一张大图里(一行一个组合,
左边metrics、右边loss),不用再一个个打开单独的png对比。

用法:
    先把下面 RUNS 列表里的 label 和 json 文件名改成你实际存的文件名
    (对应之前批量训练脚本里存的 final_results_{model_name}_{weighting}.json),
    然后:
        python plot_dashboard.py
        python plot_dashboard.py --outdir .
"""

import argparse
import json
import matplotlib
matplotlib.use("Agg")  # 远程服务器没有显示器,必须用无界面后端
import matplotlib.pyplot as plt
from matplotlib import font_manager


def _setup_chinese_font():
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "PingFang SC",
        "sans-serif",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Arial Unicode MS", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


_setup_chinese_font()

# ---- 在这里改成你实际的文件名和展示用的标签 ----
RUNS = [
    # ("MMoE + Fixed",        "final_results_MMoE_fixed.json"),
    # ("MMoE + Uncertainty",  "final_results_MMoE_uncertainty.json"),
    # ("PLE + Fixed",         "final_results_PLE_fixed.json"),
    # ("PLE + Uncertainty",   "final_results_PLE_uncertainty.json"),
    
    ("PLE_multilayer + Uncertainty + Click Task Focal Loss", "final_results_PLE_multilayer_uncertainty_click_early_frozen.json"),
]

# ---- 配色:固定顺序,跟之前 plot_results.py 保持一致 ----
COLOR_CLICK = "#2a78d6"    # blue
COLOR_LIKE = "#eb6834"     # orange
COLOR_FOLLOW = "#1baf7a"   # aqua
COLOR_MEAN = "#eda100"     # yellow
COLOR_MAE = "#e87ba4"      # magenta
COLOR_RMSE = "#008300"     # green
COLOR_TOTAL = "#4a3aa7"    # violet

GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#898781"
TEXT_COLOR = "#0b0b0b"
SURFACE = "#fcfcfb"


def _style_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(colors=AXIS_COLOR, labelsize=8)


def load_series(final_results, epochs, key):
    return [final_results[str(e)][key] for e in epochs]


def draw_metrics(ax, epochs, fr, title):
    click_auc = load_series(fr, epochs, "click_AUC")
    like_auc = load_series(fr, epochs, "like_AUC")
    follow_auc = load_series(fr, epochs, "follow_AUC")
    mean_auc = [(c + l + f) / 3 for c, l, f in zip(click_auc, like_auc, follow_auc)]
    mae = load_series(fr, epochs, "watch_ratio_MAE")
    rmse = load_series(fr, epochs, "watch_ratio_RMSE")

    _style_ax(ax)
    series = [
        ("click_AUC", click_auc, COLOR_CLICK),
        ("like_AUC", like_auc, COLOR_LIKE),
        ("follow_AUC", follow_auc, COLOR_FOLLOW),
        ("mean_AUC", mean_auc, COLOR_MEAN),
        ("watch_ratio_MAE", mae, COLOR_MAE),
        ("watch_ratio_RMSE", rmse, COLOR_RMSE),
    ]
    for label, values, color in series:
        ax.plot(epochs, values, color=color, linewidth=1.5, marker="o", markersize=3, label=label)
    ax.set_title(f"{title} — Metrics", color=TEXT_COLOR, fontsize=10, loc="left")
    ax.set_xticks(epochs)



def draw_loss(ax1, epochs, fr, title):
    total_loss = load_series(fr, epochs, "total_loss")
    click_loss = load_series(fr, epochs, "click_loss")
    like_loss = load_series(fr, epochs, "like_loss")
    follow_loss = load_series(fr, epochs, "follow_loss")
    watch_loss = load_series(fr, epochs, "watch_ratio_loss")

    _style_ax(ax1)
    l1, = ax1.plot(epochs, total_loss, color=COLOR_TOTAL, linewidth=1.5, marker="o", markersize=3, label="total_loss")
    l2, = ax1.plot(epochs, click_loss, color=COLOR_CLICK, linewidth=1.5, marker="o", markersize=3, label="click_loss")
    l3, = ax1.plot(epochs, watch_loss, color=COLOR_MAE, linewidth=1.5, marker="o", markersize=3, label="watch_ratio_loss")
    ax1.set_xticks(epochs)

    ax2 = ax1.twinx()
    ax2.set_facecolor(SURFACE)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(colors=AXIS_COLOR, labelsize=8)
    l4, = ax2.plot(epochs, like_loss, color=COLOR_LIKE, linewidth=1.5, linestyle="--", marker="s", markersize=3, label="like_loss")
    l5, = ax2.plot(epochs, follow_loss, color=COLOR_FOLLOW, linewidth=1.5, linestyle="--", marker="s", markersize=3, label="follow_loss")

    ax1.set_title(f"{title} — Loss", color=TEXT_COLOR, fontsize=10, loc="left")
    return [l1, l2, l3, l4, l5]


def main(outdir):
    n = len(RUNS)
    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n), facecolor=SURFACE)
    if n == 1:
        axes = axes.reshape(1, 2)

    metrics_handles_labels = None
    loss_handles_labels = None

    for row, (label, json_path) in enumerate(RUNS):
        with open(json_path, "r") as f:
            fr = json.load(f)
        epochs = sorted(int(e) for e in fr.keys())

        draw_metrics(axes[row, 0], epochs, fr, label)
        if metrics_handles_labels is None:
            h, l = axes[row, 0].get_legend_handles_labels()
            metrics_handles_labels = (h, l)

        lines = draw_loss(axes[row, 1], epochs, fr, label)
        if loss_handles_labels is None:
            loss_handles_labels = ([ln for ln in lines], [ln.get_label() for ln in lines])

    # 统一在图的最下面放两个图例,一个对应metrics列,一个对应loss列
    fig.legend(
        metrics_handles_labels[0], metrics_handles_labels[1],
        loc="lower center", bbox_to_anchor=(0.27, -0.01 / n), ncol=3, frameon=False, fontsize=9,
    )
    fig.legend(
        loss_handles_labels[0], loss_handles_labels[1],
        loc="lower center", bbox_to_anchor=(0.73, -0.01 / n), ncol=3, frameon=False, fontsize=9,
    )

    fig.suptitle("Model Comparison Dashboard", color=TEXT_COLOR, fontsize=15, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(f"{outdir}/特征私有.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {outdir}/特征私有.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()
    main(args.outdir)