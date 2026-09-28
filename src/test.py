import torch

model = torch.load("/root/autodl-tmp/kuairec/grid_search_results_m2/best_model_grid_s4_c2_l2_f2_w2.pt")
ed = torch.load("/root/autodl-tmp/kuairec/eval_dataset.pt")

cat = torch.tensor(ed.cat)
num = torch.tensor(ed.num,torch.float32)
outputs,_ = model(cat,num,return_gate_probs = False)
pred_wr = outputs["watch_ratio"]
for i,feature in enumerate([
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
        ]):
    f = feature[:,i,i+1]
    
    