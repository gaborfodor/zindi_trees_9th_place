import numpy as np
import pandas as pd
import importlib
import sys
import random
from tqdm import tqdm
import gc
import argparse
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from copy import copy
import os
from collections import defaultdict

from sklearn import metrics

import wandb

from utils import (
    sync_across_gpus,
    set_seed,
    get_model,
    create_checkpoint,
    get_dataloader,
    get_optimizer,
    get_scheduler,
)

def garbage_collection_cuda():
    """Garbage collection Torch (CUDA) memory."""
    gc.collect()
    torch.cuda.empty_cache()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    import cv2

    cv2.setNumThreads(0)
except:
    print("no cv2 installed, running without")

sys.path.append("configs")
sys.path.append("models")
sys.path.append("datasets")


def run_predict(model, test_dataloader, valid_df, cfg):

    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    test_data = defaultdict(list)

    for data in tqdm(test_dataloader, disable=cfg.local_rank != 0):

        batch = cfg.batch_to_device(data, cfg.device)
        output = model(batch)
        output["preds"] = output["logits"].to(dtype=torch.float32)
        
        for key, test in output.items():
            test_data[key] += [output[key]]

    for key, val in output.items():
        value = test_data[key]
        if isinstance(value[0], list):
            test_data[key] = [item for sublist in value for item in sublist]
        else:
            if len(value[0].shape) == 0:
                test_data[key] = torch.stack(value)
            else:
                test_data[key] = torch.cat(value, dim=0)

    if cfg.distributed and cfg.eval_ddp:
        for key, test in output.items():
            test_data[key] = sync_across_gpus(test_data[key], cfg.world_size)

    if cfg.local_rank == 0:
        if cfg.save_val_data:
            if cfg.distributed:
                for k, v in test_data.items():
                    test_data[k] = v[: len(test_dataloader.dataset)]
            for i, c in enumerate(cfg.classes):
                valid_df[f'pred_{c}'] = test_data["preds"].cpu()[:, i]
            
            print(valid_df.head())
            valid_df.pred_Target = valid_df.pred_Target.clip(0, None) 
            mse = metrics.mean_squared_error(valid_df.Target, valid_df.pred_Target)
            row = {
                "mse": mse,
                "rmse": np.sqrt(mse),
                "mae": metrics.mean_absolute_error(valid_df.Target, valid_df.pred_Target),
            }
            row["epoch"] = cfg.curr_epoch
            print(row)
            wandb.log(row)
            valid_df.to_csv(f"{cfg.output_dir}/fold{cfg.fold}/valid_pred_df.csv", index=False)
            
    if cfg.distributed: 
        torch.distributed.barrier()


def save_test_predictions(model, test_dataloader, test_df, cfg):

    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    test_data = defaultdict(list)

    for data in tqdm(test_dataloader, disable=cfg.local_rank != 0):

        batch = cfg.batch_to_device(data, cfg.device)
        output = model(batch)
        output["preds"] = output["logits"].to(dtype=torch.float32)
        
        for key, test in output.items():
            test_data[key] += [output[key]]

    for key, val in output.items():
        value = test_data[key]
        if isinstance(value[0], list):
            test_data[key] = [item for sublist in value for item in sublist]
        else:
            if len(value[0].shape) == 0:
                test_data[key] = torch.stack(value)
            else:
                test_data[key] = torch.cat(value, dim=0)

    if cfg.distributed and cfg.eval_ddp:
        for key, test in output.items():
            test_data[key] = sync_across_gpus(test_data[key], cfg.world_size)

    if cfg.local_rank == 0:
        if cfg.save_val_data:
            if cfg.distributed:
                for k, v in test_data.items():
                    test_data[k] = v[: len(test_dataloader.dataset)]
            for i, c in enumerate(cfg.classes):
                test_df[f'pred_{c}'] = test_data["preds"].cpu()[:, i]
            
            print(test_df.head())
            test_df.pred_Target = test_df.pred_Target.clip(0, None) 
            test_df.to_csv(f"{cfg.output_dir}/fold{cfg.fold}/test_pred_df.csv", index=False)
    if cfg.distributed: 
        torch.distributed.barrier()


def train(cfg, train_df, valid_df, test_df):
    # set seed
    if cfg.seed < 0:
        cfg.seed = np.random.randint(1_000_000)
    print("seed", cfg.seed)

    cfg.distributed = False
    if "WORLD_SIZE" in os.environ:
        cfg.distributed = int(os.environ["WORLD_SIZE"]) > 1

    if cfg.distributed:
        cfg.local_rank = int(os.environ["LOCAL_RANK"])
        print("RANK", cfg.local_rank)
        device = "cuda:%d" % cfg.local_rank
        cfg.device = device
        print("device", device)

        torch.cuda.set_device(cfg.local_rank)

        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        cfg.world_size = torch.distributed.get_world_size()
        cfg.rank = torch.distributed.get_rank()
        print(
            "Training in distributed mode with multiple processes, 1 GPU per process."
        )
        print(
            f"Process {cfg.rank}, total {cfg.world_size}, local rank {cfg.local_rank}."
        )
        garbage_collection_cuda()
        s=32
        torch.nn.functional.conv2d(
            torch.zeros(s, s, s, s, device=device),
            torch.zeros(s, s, s, s, device=device)
        )
        cfg.group = torch.distributed.new_group(np.arange(cfg.world_size))
        print("Group", cfg.group)
        # syncing the random seed
        cfg.seed = int(
            sync_across_gpus(torch.Tensor([cfg.seed]).to(device), cfg.world_size)
            .detach()
            .cpu()
            .numpy()[0]
        )  #
        print("seed", cfg.local_rank, cfg.seed)

    else:
        cfg.local_rank = 0
        cfg.world_size = 1
        cfg.rank = 0

        device = "cuda:%d" % cfg.gpu
        cfg.device = device

    set_seed(cfg.seed)

    val_dataset = cfg.CustomDataset(valid_df, mode="test", aug=cfg.val_aug, cfg=cfg)
    val_dataloader = get_dataloader(val_dataset, cfg, mode="test")

    test_dataset = cfg.CustomDataset(test_df, mode="test", aug=cfg.val_aug, cfg=cfg)
    test_dataloader = get_dataloader(test_dataset, cfg, mode="test")

    model = get_model(cfg)
    model.to(device)

    if cfg.distributed:
        model = NativeDDP(
            model, device_ids=[cfg.local_rank], find_unused_parameters=False
        )

    train_dataset = cfg.CustomDataset(train_df, mode="train", aug=cfg.train_aug, cfg=cfg)
    train_dataloader = get_dataloader(train_dataset, cfg, mode="train")
    total_steps = len(train_dataset)

    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(cfg, optimizer, total_steps)

    if cfg.mixed_precision:
        scaler = GradScaler()
    else:
        scaler = None

    cfg.curr_step = 0
    i = 0
    optimizer.zero_grad()
    val_score = 0
    if cfg.local_rank == 0:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_user,
            config = vars(cfg)
        )

    for epoch in range(cfg.epochs):

        set_seed(cfg.seed + epoch + cfg.local_rank)

        cfg.curr_epoch = epoch
        if cfg.local_rank == 0:
            print("EPOCH:", epoch)
        if cfg.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        progress_bar = tqdm(range(len(train_dataloader)))
        tr_it = iter(train_dataloader)
        losses = []
        gc.collect()

        if cfg.train:
            # ==== TRAIN LOOP
            for itr in progress_bar:
                i += 1
                cfg.curr_step += cfg.batch_size * cfg.world_size
                try:
                    data = next(tr_it)
                except Exception as e:
                    print(e)
                    print("DATA FETCH ERROR")
                model.train()
                torch.set_grad_enabled(True)

                if (cfg.local_rank == 0) and (i == 1):
                    fb = f"{cfg.output_dir}/fold{cfg.fold}/first_batch_e{epoch}.npy"
                    np.save(fb, data['input'].numpy())

                batch = cfg.batch_to_device(data, device)
                if cfg.mixed_precision:
                    with autocast():
                        output_dict = model(batch)
                else:
                    output_dict = model(batch)
                loss = output_dict["loss"]
                losses.append(loss.item())
                if cfg.grad_accumulation != 0:
                    loss /= cfg.grad_accumulation
                # Backward pass
                if cfg.mixed_precision:
                    scaler.scale(loss).backward()

                    if i % cfg.grad_accumulation == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if i % cfg.grad_accumulation == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                if cfg.distributed:
                    torch.cuda.synchronize()
                if scheduler is not None:
                    scheduler.step()
                if cfg.local_rank == 0 and cfg.curr_step % (cfg.batch_size) == 0:
                    wandb.log({"train_loss": np.mean(losses[-10:])})
                    wandb.log({"lr": optimizer.param_groups[0]["lr"]})
                    progress_bar.set_description(f"loss: {np.mean(losses[-10:]):.4f}")

            print(f"Mean train_loss {np.mean(losses):.4f}")

        if cfg.distributed:
            torch.cuda.synchronize()
            torch.distributed.barrier()

        if (cfg.local_rank == 0) and (cfg.epochs > 0) and (cfg.save_checkpoint):
            if not cfg.save_only_last_ckpt:
                checkpoint = create_checkpoint(
                    cfg, model, optimizer, epoch, scheduler=scheduler, scaler=scaler
                )
                torch.save(
                    checkpoint,
                    f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pth",
                )
        
        if cfg.test:
            run_predict(model, val_dataloader, valid_df, cfg)
            save_test_predictions(model, test_dataloader, test_df, cfg)

    if (cfg.local_rank == 0) and (cfg.epochs > 0) and (cfg.save_checkpoint):
        # print(f'SAVING LAST EPOCH: val_loss {val_loss:.5}')
        checkpoint = create_checkpoint(
            cfg, model, optimizer, epoch, scheduler=scheduler, scaler=scaler
        )
        torch.save(
            checkpoint,
            f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pth",
        )
    return val_score

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-C", "--config", help="config filename")
    parser.add_argument("-f", "--fold", type=int, default=0)
    parser_args, other_args = parser.parse_known_args(sys.argv)

    fold = parser_args.fold
    cfg = copy(importlib.import_module(parser_args.config).cfg)
    cfg.CustomDataset = importlib.import_module(cfg.dataset).CustomDataset
    cfg.batch_to_device = importlib.import_module(cfg.dataset).batch_to_device
    
    cfg.fold = fold
    if fold == -1:
        cfg.test = False

    os.makedirs(str(cfg.output_dir + f"/fold{cfg.fold}/"), exist_ok=True)

    df = pd.read_csv(cfg.train_df)

    valid_df = df[df.fold == fold].copy()
    train_df = df[df.fold != fold].copy()
    test_df = pd.read_csv(cfg.test_df)

    result = train(cfg, train_df, valid_df, test_df)
