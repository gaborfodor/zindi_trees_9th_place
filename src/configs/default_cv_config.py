from default_config import basic_cfg

cfg = basic_cfg

# img model
cfg.backbone = "tf_efficientnet_b0_ns"
cfg.pretrained = True
cfg.pool = "avg"
cfg.in_chans = 3
cfg.warmup = 0.0
cfg.drop_out = 0.0

cfg.normalization = "no"
cfg.lg_rounds = 2
cfg.loss = 'mse'

basic_cv_cfg = cfg
