import torch
from torch.utils.data import Dataset

class KuaiSARTorchDataset(Dataset):
    def __init__(self, df):
        categorical_cols = [
                        "author_id", "item_type", "upload_type",
                        
                            "first_level_category_id", "second_level_category_id",
                            "third_level_category_id", "fourth_level_category_id"
        ]
        numeric_cols  = [
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
        ]
        self.cat = df[categorical_cols].values.astype("int64")
        self.num = df[numeric_cols].values.astype("float32")
        self.click = df["click"].values.astype("float32")
        self.like = df["like"].values.astype("float32")
        self.follow = df["follow"].values.astype("float32")
        self.watch_ratio = df["watch_ratio"].values.astype("float32")

    def __len__(self):
        return len(self.cat)

    def __getitem__(self, idx):
        return {
            "cat": torch.tensor(self.cat[idx]),
            "num": torch.tensor(self.num[idx]),
            "click": torch.tensor(self.click[idx]),
            "like": torch.tensor(self.like[idx]),
            "follow": torch.tensor(self.follow[idx]),
            "watch_ratio": torch.tensor(self.watch_ratio[idx]),
        }
