from types import SimpleNamespace


cfg = SimpleNamespace(**{})

# stages
cfg.train = True
cfg.test = True

# dataset
cfg.dataset = "ds_dummy"
cfg.batch_size = 16
cfg.normalization = None
cfg.train_aug = None
cfg.val_aug = None

# training routine
cfg.fold = 0
cfg.lr = 1e-4
cfg.schedule = "cosine"
cfg.num_cycles = 0.5
cfg.weight_decay = 0
cfg.optimizer = "AdamW"
cfg.epochs = 10
cfg.seed = -1
cfg.do_test = True
cfg.eval_ddp = True
cfg.save_val_data = True

# resources
cfg.mixed_precision = True
cfg.grad_accumulation = 1
cfg.gpu = 0
cfg.num_workers = 8
cfg.drop_last = True    
cfg.save_checkpoint = True
cfg.save_only_last_ckpt = False
cfg.save_weights_only = True
cfg.pin_memory = False


cfg.tags = None
cfg.loss = "mse"

# You need to change
cfg.data_dir = "/home/gabor/kaggle/zindi_trees_9th_place/"
cfg.wandb_project = "palm"
cfg.wandb_user = "WANDB_USER"

basic_cfg = cfg
