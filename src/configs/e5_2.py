
from default_cv_config import basic_cv_cfg as cfg
import os
import albumentations as A


# paths
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.train_df = cfg.data_dir + "tree_train_df.csv"
cfg.test_df = cfg.data_dir + "tree_test_df.csv"
cfg.output_dir = "../output/models/" + cfg.name


# model
cfg.model = "model"
cfg.backbone = "efficientnetv2_rw_t"
cfg.batch_size = 24
cfg.grad_accumulation = 1
cfg.drop_out = 0.5
cfg.loss = "mae"

# DATASET
cfg.dataset = "dataset"
cfg.classes = ["Target"]
cfg.image_width = 768
cfg.image_height = 768

# OPTIMIZATION & SCHEDULE
cfg.epochs = 50
cfg.lr = 0.0003
cfg.optimizer = "AdamW"
cfg.warmup = 0

# AUGMENTATION
cfg.train_aug = A.Compose([
    A.Resize(height=cfg.image_height, width=cfg.image_width, always_apply=True, p=1),
    A.HorizontalFlip(always_apply=False, p=0.5),
    A.VerticalFlip(always_apply=False, p=0.5),
    A.RandomRotate90(always_apply=False, p=0.5),
    A.Downscale(scale_min=0.2, scale_max=0.2, always_apply=False, p=0.5),
    A.ShiftScaleRotate(
        p=0.5,
        shift_limit = 0.06,
        scale_limit = 0.06,
        rotate_limit = 10,
        interpolation=1,
        border_mode=0,
    ),
    A.HueSaturationValue(
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        always_apply=False,
        p=0.5,
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5,
    ),
    A.CoarseDropout(
        min_holes=5,
        max_holes=12,
        max_height=0.09,
        max_width=0.09,
        min_height=0.04,
        min_width=0.04,
        p=0.5,
    ),
    A.OneOf([
        A.OpticalDistortion(distort_limit=0.05, border_mode=0,),
        A.GridDistortion(num_steps=5, distort_limit=0.05, border_mode=0,),
    ], p=0.5),
])

cfg.val_aug = A.Compose([
    A.Resize(height=cfg.image_height, width=cfg.image_width, always_apply=True, p=1),
])
