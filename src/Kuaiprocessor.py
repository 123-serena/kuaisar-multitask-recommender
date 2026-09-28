
import pandas as pd
import numpy as np
DATADIR = '/root/autodl-tmp/kuairec/KuaiSAR_final'
from sklearn.preprocessing import StandardScaler


import pandas as pd

class CategoryEncoder:
    """自己实现的LabelEncoder：0留给UNK（没见过的值），从1开始编号。"""

    def __init__(self):
        self.mapping = {}
        self.n_classes = 1  # 0是UNK，所以至少有1个类别

    def fit(self, values):
        uniques = pd.unique(values)
        self.mapping = {v: i + 1 for i, v in enumerate(uniques)}  # 从1开始编号
        self.n_classes = len(uniques) + 1
        return self

    def transform(self, values):
        # 在mapping里查得到就用查到的编号；查不到（没见过）就填0
        return pd.Series(values).map(self.mapping).fillna(0).astype(int).values
import numpy as np
import pandas as pd


def hist_agg(df, group_cols, target_col, method, new_col):
    """
    通用的"历史统计"函数,point-in-time安全(不含当前行,只用当前行之前的历史)。
    全程用原生groupby方法(cumsum/cumcount),不套自定义lambda的transform,
    避免逐组调用Python函数导致的严重性能问题。
    """
    if method == "count":
        df[new_col] = df.groupby(group_cols).cumcount()
        return df

    g = df.groupby(group_cols)[target_col]
    cum_incl = g.cumsum()                       # 含当前行的组内累加,原生向量化,不会跨组
    hist_sum = cum_incl - df[target_col]         # 减掉当前行自己,等价于"shift(1)后再cumsum"但快得多

    if method == "sum":
        df[new_col] = hist_sum.fillna(0)
    elif method == "mean":
        hist_cnt = df.groupby(group_cols).cumcount()   # 组内当前行之前出现的次数,原生向量化
        df[new_col] = (hist_sum / hist_cnt.replace(0, np.nan)).fillna(0)
    else:
        raise ValueError(f"未知method: {method}")
    return df


def hist_rolling(df, group_cols, target_col, window, new_col):
    """"最近N次交互"版本(而不是全部历史),抓用户近期兴趣漂移,同样point-in-time安全。"""
    def _roll(s):
        return s.shift(1).rolling(window, min_periods=1).mean().fillna(0)

    df[new_col] = df.groupby(group_cols)[target_col].transform(_roll)
    return df


# ---- 按"6维度交叉"整理的特征清单:(分组列, 目标列, 统计方法, 新特征名) ----
FEATURE_SPECS = [
    # A. 用户 x 一级类目:兴趣分布
    (["user_id", "first_level_category_id"], "click", "mean", "user_hist_click_rate_by_cat1"),
    (["user_id", "first_level_category_id"], "like", "mean", "user_hist_like_rate_by_cat1"),
    (["user_id", "first_level_category_id"], "follow", "mean", "user_hist_follow_rate_by_cat1"),
    (["user_id", "first_level_category_id"], "watch_ratio", "mean", "user_hist_avg_watch_ratio_by_cat1"),

    # B. 用户 x 作者:对特定创作者的偏好(次数类)
    (["user_id", "author_id"], "click", "sum", "user_hist_click_cnt_by_author"),
    (["user_id", "author_id"], "like", "sum", "user_hist_like_cnt_by_author"),

    # C. 用户 x 时段:个人作息偏好
    (["user_id", "hour_of_day"], "click", "mean", "user_hist_click_rate_by_hour"),
    (["user_id", "hour_of_day"], "watch_ratio", "mean", "user_hist_watch_ratio_by_hour"),

    # D. 用户整体活跃度(不分类目,兜底特征)
    (["user_id"], "click", "mean", "user_hist_click_rate_overall"),
    (["user_id"], "click", "count", "user_hist_interaction_cnt"),
    (["user_id"], "watch_ratio", "mean", "user_hist_watch_ratio_overall"),

    # E. 物料(item)自身热度 —— point-in-time版的"物料热度"
    (["item_id"], "click", "mean", "item_hist_click_rate"),
    (["item_id"], "like", "mean", "item_hist_like_rate"),
    (["item_id"], "watch_ratio", "mean", "item_hist_avg_watch_ratio"),

    # F. 作者(author)号召力
    (["author_id"], "click", "mean", "author_hist_click_rate"),
    (["author_id"], "watch_ratio", "mean", "author_hist_avg_watch_ratio"),
]

# ---- 滑动窗口版:(分组列, 目标列, 窗口大小, 新特征名) ----
ROLLING_SPECS = [
    (["user_id"], "click", 20, "user_recent20_click_rate"),
    (["user_id", "first_level_category_id"], "watch_ratio", 10, "user_recent10_avg_watch_ratio_by_cat1"),
]

COUNT_LIKE_COLS = [
    "user_hist_click_cnt_by_author",
    "user_hist_like_cnt_by_author",
    "user_hist_interaction_cnt",
]

def build_features(df, feature_specs=FEATURE_SPECS, rolling_specs=ROLLING_SPECS, add_cross=True):
    df = df.sort_values("timestamp").reset_index(drop=True)

    for group_cols, target_col, method, new_col in feature_specs:
        df = hist_agg(df, group_cols, target_col, method, new_col)

    for group_cols, target_col, window, new_col in rolling_specs:
        df = hist_rolling(df, group_cols, target_col, window, new_col)

    if add_cross:
        # G. 交叉特征:用户对该类目的兴趣 x 物料自身热度
        df["cross_user_interest_x_item_hotness"] = (
            df["user_hist_click_rate_by_cat1"] * df["item_hist_click_rate"]
        )

    # count/sum类特征长尾严重，log1p压一下，标准化放到processor里(只在train上fit)
    for c in COUNT_LIKE_COLS:
        df[c] = np.log1p(df[c])

    return df



NUM_FEATURES = [
    "log_duration_ms","log_video_age",
    "hour_of_day", "day_of_week", "is_weekend",
    "user_hist_click_rate_by_cat1", "user_hist_like_rate_by_cat1",
    "user_hist_follow_rate_by_cat1", "user_hist_avg_watch_ratio_by_cat1",
    "user_hist_click_cnt_by_author", "user_hist_like_cnt_by_author",
    "user_hist_click_rate_by_hour",
    "user_hist_click_rate_overall", "user_hist_interaction_cnt",
    "item_hist_click_rate", "item_hist_like_rate", "item_hist_avg_watch_ratio",
    "author_hist_click_rate", "author_hist_avg_watch_ratio",
    "user_recent20_click_rate", "user_recent10_avg_watch_ratio_by_cat1",
    "cross_user_interest_x_item_hotness","user_hist_watch_ratio_by_hour","user_hist_watch_ratio_overall"
]  # 必须和 KuaiSARTorchDataset 里的 numeric_cols 完全一致(顺序也要一致!)

CAT_FEATURES = [
    "author_id", "item_type", "upload_type",
    
    "first_level_category_id", "second_level_category_id",
    "third_level_category_id", "fourth_level_category_id"
]  # 必须和 KuaiSARTorchDataset 里的 categorical_cols 完全一致

NEED_SCALE = [
    "log_duration_ms",
    "user_hist_click_cnt_by_author", "user_hist_like_cnt_by_author",
    "user_hist_interaction_cnt",
    "user_hist_avg_watch_ratio_by_cat1", "item_hist_avg_watch_ratio",
    "author_hist_avg_watch_ratio", "user_recent10_avg_watch_ratio_by_cat1",
]


class processor():
    def __init__(self):
        item_features = pd.read_csv(f'{DATADIR}/item_features.csv')
        rec_inter = pd.read_csv(f'{DATADIR}/rec_inter.csv')
        df = pd.merge(rec_inter, item_features, on='item_id', how='left')
        df = df[df["duration_ms"] > 0]
        df["watch_ratio"] = (df["playing_time"] / df["duration_ms"]).clip(0, 5)

        df["log_duration_ms"] = np.log(df["duration_ms"] + 1)
        df["log_playing_time"] = np.log(df["playing_time"] + 1)
        df["upload_time"] = pd.to_datetime(df["upload_time"], format="%Y-%m-%d", errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        age_ms = (df["timestamp"] - df["upload_time"]).dt.total_seconds().fillna(0) * 1000
        age_ms = age_ms.clip(lower=0)
        df["log_video_age"] = np.log(age_ms + 1)

        n_bad_age = df["log_video_age"].isna().sum()
        if n_bad_age > 0:
            print(f"[warn] log_video_age 出现 {n_bad_age} 个 NaN，已用0填充")
            df["log_video_age"] = df["log_video_age"].fillna(0)

        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        df = df.sort_values("timestamp").reset_index(drop=True)
        df = build_features(df)
        self.df = df
        self.num_features = NUM_FEATURES
        self.cat_features = CAT_FEATURES
        self.n_num_feature = len(NUM_FEATURES)

        split_idx = int(len(df) * 0.6)
        split_idx1 = int(len(df) * 0.8)
        self.train_df = df.iloc[:split_idx].reset_index(drop=True).copy()
        self.val_df = df.iloc[split_idx:split_idx1].reset_index(drop=True).copy()
        self.test_df = df.iloc[split_idx1:].reset_index(drop=True).copy()

        # 连续特征标准化：只在train上fit(sklearn要求2D输入，所以要双中括号)
        for cf in NEED_SCALE:
            scaler = StandardScaler()
            self.train_df[[cf]] = scaler.fit_transform(self.train_df[[cf]])
            self.val_df[[cf]] = scaler.transform(self.val_df[[cf]])
            self.test_df[[cf]] = scaler.transform(self.test_df[[cf]])

        # 类别特征编码：CategoryEncoder是自己写的类,要分开调用fit()/transform()，
        # 而且要传1维Series(单中括号)，不是DataFrame(双中括号)
        self.n_cat_feature = []
        for cf in CAT_FEATURES:
            encoder = CategoryEncoder()
            encoder.fit(self.train_df[cf])
            self.train_df[cf] = encoder.transform(self.train_df[cf])
            self.n_cat_feature.append(encoder.n_classes)
            self.val_df[cf] = encoder.transform(self.val_df[cf])
            self.test_df[cf] = encoder.transform(self.test_df[cf])