import numpy as np

from Kuaiprocessor import processor


def test_video_age_has_no_nan_after_feature_processing():
    p = processor()
    for split in ["train_df", "val_df", "test_df"]:
        df = getattr(p, split)
        assert df["log_video_age"].notna().all()
        assert np.isfinite(df["log_video_age"]).all()
        assert (df["log_video_age"] >= 0).all()
