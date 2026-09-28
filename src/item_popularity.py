"""
物料(视频/item)热度分布图 —— 用于报告里的数据集EDA部分,
展示 KuaiSAR 数据里 item 交互次数的长尾分布(少数热门item占了大部分交互)。

数据来源支持两种(自动识别):
    1. 原始 rec_inter.csv (或任何有 item id 列的交互表)
    2. 训练时保存的 processor.pt (torch.load 出来的对象里读 .train_df)

用法:
    # 直接指定原始交互表(推荐,最准):
    python plot_item_popularity.py --csv /path/to/rec_inter.csv --id-col video_id

    # 或者用你训练脚本里已经存好的 processor.pt:
    python plot_item_popularity.py --processor processor.pt --id-col video_id

    如果不确定 id 列到底叫什么,不传 --id-col,脚本会在常见候选名里自动找
    (video_id / item_id / photo_id / material_id ...),找不到会报错并列出
    实际的列名,你照着改一下参数重跑就行。

输出:
    item_popularity_distribution.png (两张图: 长尾直方图 + 热度集中度曲线)
    并在终端打印几个可以直接抄进报告里的统计量。
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


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

# ---- 配色 ----
COLOR_CLICK = "#2a78d6"
COLOR_MEAN = "#eda100"
COLOR_TOTAL = "#4a3aa7"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#898781"
TEXT_COLOR = "#0b0b0b"
SURFACE = "#fcfcfb"

CANDIDATE_ID_COLS = ["video_id", "item_id", "photo_id", "material_id", "article_id"]


def _style_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(colors=AXIS_COLOR, labelsize=9)


def load_dataframe(args):
    if args.csv:
        print(f"[load] 读取原始交互表: {args.csv}")
        return pd.read_csv(args.csv)
    if args.processor:
        print(f"[load] 读取 processor: {args.processor}")
        import torch
        processor = torch.load(args.processor, map_location="cpu", weights_only=False)
        if hasattr(processor, "train_df"):
            return processor.train_df
        raise RuntimeError(
            "processor 对象里没有 .train_df,看一下 Kuaiprocessor.py 里实际存的属性名,"
            "换成 --csv 直接指向原始 rec_inter.csv 更省事。"
        )
    raise RuntimeError("请用 --csv 或 --processor 指定数据来源。")


def resolve_id_col(df, id_col):
    if id_col:
        if id_col not in df.columns:
            raise RuntimeError(f"列 '{id_col}' 不存在。实际列名有: {list(df.columns)}")
        return id_col
    for c in CANDIDATE_ID_COLS:
        if c in df.columns:
            print(f"[id-col] 自动识别到物料id列: {c}")
            return c
    raise RuntimeError(
        f"没能自动识别出物料id列,请用 --id-col 手动指定。实际列名有: {list(df.columns)}"
    )


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="原始交互表路径,如 rec_inter.csv")
    parser.add_argument("--processor", default=None, help="processor.pt 路径(备选数据来源)")
    parser.add_argument("--id-col", default=None, help="物料id列名,不传则自动识别")
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    if not args.csv and not args.processor:
        print("请至少指定 --csv 或 --processor 其中一个数据来源,例如:\n"
              "  python plot_item_popularity.py --csv /root/autodl-tmp/kuairec/rec_inter.csv",
              file=sys.stderr)
        sys.exit(1)

    df = load_dataframe(args)
    id_col = resolve_id_col(df, args.id_col)

    counts = df[id_col].value_counts().sort_values(ascending=False)
    n_items = len(counts)
    n_inter = int(counts.sum())

    # ---- 统计量(可以直接抄进报告) ----
    def top_share(pct):
        k = max(1, int(n_items * pct))
        return counts.iloc[:k].sum() / n_inter * 100

    print("\n=== 物料热度分布统计 ===")
    print(f"物料(item)总数: {n_items}")
    print(f"交互总数: {n_inter}")
    print(f"人均/物料均值交互次数: {counts.mean():.2f}, 中位数: {counts.median():.0f}, 最大值: {counts.max()}")
    print(f"热度Top 1%  的物料贡献了 {top_share(0.01):.1f}% 的交互")
    print(f"热度Top 5%  的物料贡献了 {top_share(0.05):.1f}% 的交互")
    print(f"热度Top 10% 的物料贡献了 {top_share(0.10):.1f}% 的交互")
    only_once = int((counts == 1).sum())
    print(f"只被交互过1次的物料占比: {only_once / n_items * 100:.1f}% ({only_once}/{n_items})")

   # ---- 画图 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE)

    # 左图: 长尾直方图(x轴log刻度,交互次数分箱)
    _style_ax(ax1)
    bins = np.logspace(0, np.log10(counts.max()), 40)
    ax1.hist(counts.values, bins=bins, color=COLOR_CLICK, edgecolor="white", linewidth=0.4)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("单个物料的交互次数(log)", color=TEXT_COLOR, fontsize=9)
    ax1.set_ylabel("物料数量(log)", color=TEXT_COLOR, fontsize=9)
    ax1.set_title("物料热度分布(长尾直方图)", color=TEXT_COLOR, fontsize=11, loc="left")

    # 右图: 累积热度集中度曲线(Lorenz-style)
    _style_ax(ax2)
    cum_items = np.arange(1, n_items + 1) / n_items * 100
    cum_inter = np.cumsum(counts.values) / n_inter * 100
    ax2.plot(cum_items, cum_inter, color=COLOR_TOTAL, linewidth=2)
    ax2.plot([0, 100], [0, 100], color=AXIS_COLOR, linewidth=1, linestyle="--", label="完全均匀(理想线)")
    ax2.set_xlabel("物料累积占比(按热度从高到低排序,%)", color=TEXT_COLOR, fontsize=9)
    ax2.set_ylabel("交互累积占比(%)", color=TEXT_COLOR, fontsize=9)
    ax2.set_title("热度集中度曲线", color=TEXT_COLOR, fontsize=11, loc="left")
    ax2.annotate(
        f"Top10%物料 → {top_share(0.10):.0f}%交互",
        xy=(10, top_share(0.10)), xytext=(25, max(top_share(0.10) - 15, 5)),
        arrowprops=dict(arrowstyle="->", color=COLOR_MEAN), color=TEXT_COLOR, fontsize=8,
    )
    ax2.legend(loc="lower right", frameon=False, fontsize=8)

    fig.suptitle("物料热度分布 (Item Popularity Distribution)", color=TEXT_COLOR, fontsize=14, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = f"{args.outdir}/item_popularity_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()