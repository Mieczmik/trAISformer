# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Boilerplate for training a neural network.

References:
    https://github.com/karpathy/minGPT
"""

import os
import math
import logging
import time

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
import utils
import mlflow

from trAISformer import TB_LOG, MLFLOW_LOG, bgdf
from config_trAISformer import Config

cf = Config()

logger = logging.getLogger(__name__)


@torch.no_grad()
def sample(model,
           seqs,
           steps,
           temperature=1.0,
           sample=False,
           sample_mode="pos_vicinity",
           r_vicinity=20,
           top_k=None):
    """
    Take a conditoning sequence of AIS observations seq and predict the next observation,
    feed the predictions back into the model each time. 
    """
    max_seqlen = model.get_max_seqlen()
    model.eval()
    for k in range(steps):
        seqs_cond = seqs if seqs.size(1) <= max_seqlen else seqs[:, -max_seqlen:]  # crop context if needed

        # logits.shape: (batch_size, seq_len, data_size)
        logits, _ = model(seqs_cond)
        d2inf_pred = torch.zeros((logits.shape[0], 6)).to(seqs.device) + 0.5

        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature  # (batch_size, data_size)

        lat_logits, lon_logits, sog_logits, cog_logits, ctp_logits, dtp_logits = \
            torch.split(logits, (model.lat_size, model.lon_size, model.sog_size, model.cog_size, model.ctp_size, model.dtp_size), dim=-1)

        # optionally crop probabilities to only the top k options
        if sample_mode in ("pos_vicinity",):
            idxs, idxs_uniform = model.to_indexes(seqs_cond[:, -1:, :])
            lat_idxs, lon_idxs = idxs_uniform[:, 0, 0:1], idxs_uniform[:, 0, 1:2]
            lat_logits = utils.top_k_nearest_idx(lat_logits, lat_idxs, r_vicinity)
            lon_logits = utils.top_k_nearest_idx(lon_logits, lon_idxs, r_vicinity)

        if top_k is not None:
            lat_logits = utils.top_k_logits(lat_logits, top_k)
            lon_logits = utils.top_k_logits(lon_logits, top_k)
            sog_logits = utils.top_k_logits(sog_logits, top_k)
            cog_logits = utils.top_k_logits(cog_logits, top_k)
            ctp_logits = utils.top_k_logits(ctp_logits, top_k)
            dtp_logits = utils.top_k_logits(dtp_logits, top_k)

        # apply softmax to convert to probabilities
        lat_probs = F.softmax(lat_logits, dim=-1)
        lon_probs = F.softmax(lon_logits, dim=-1)
        sog_probs = F.softmax(sog_logits, dim=-1)
        cog_probs = F.softmax(cog_logits, dim=-1)
        ctp_probs = F.softmax(ctp_logits, dim=-1)
        dtp_probs = F.softmax(dtp_logits, dim=-1)

        # sample from the distribution or take the most likely
        if sample:
            lat_ix = torch.multinomial(lat_probs, num_samples=1)  # (batch_size, 1)
            lon_ix = torch.multinomial(lon_probs, num_samples=1)
            sog_ix = torch.multinomial(sog_probs, num_samples=1)
            cog_ix = torch.multinomial(cog_probs, num_samples=1)
            ctp_ix = torch.multinomial(ctp_probs, num_samples=1)
            dtp_ix = torch.multinomial(dtp_probs, num_samples=1)
        else:
            _, lat_ix = torch.topk(lat_probs, k=1, dim=-1)
            _, lon_ix = torch.topk(lon_probs, k=1, dim=-1)
            _, sog_ix = torch.topk(sog_probs, k=1, dim=-1)
            _, cog_ix = torch.topk(cog_probs, k=1, dim=-1)
            _, ctp_ix = torch.topk(ctp_probs, k=1, dim=-1)
            _, dtp_ix = torch.topk(dtp_probs, k=1, dim=-1)

        ix = torch.cat((lat_ix, lon_ix, sog_ix, cog_ix, ctp_ix, dtp_ix), dim=-1)
        # convert to x (range: [0,1))
        x_sample = (ix.float() + d2inf_pred) / model.att_sizes

        # append to the sequence and continue
        seqs = torch.cat((seqs, x_sample.unsqueeze(1)), dim=1)

    return seqs


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    # ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, savedir=None, device=torch.device("cpu"), aisdls={},
                 INIT_SEQLEN=0):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.savedir = savedir

        self.device = device
        self.model = model.to(device)
        self.aisdls = aisdls
        self.INIT_SEQLEN = INIT_SEQLEN

        self.best_epoch = 0

    def save_checkpoint(self, epoch, best=False, final=False):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        #         logging.info("saving %s", self.config.ckpt_path)
        local_path = os.path.join(self.savedir, f"{'best' if best else 'last' if final else str(epoch).zfill(3)}_model.pt")
        logging.info(f"Best epoch: {epoch:03d}, saving model to {local_path}")
        torch.save(raw_model.state_dict(), local_path)
        tag = 'best' if best else 'last' if final else f"epoch_{epoch}"
        mlflow.log_artifact(local_path, artifact_path=f"checkpoints/{tag}")

    def train(self):
        model, config, aisdls, INIT_SEQLEN, = self.model, self.config, self.aisdls, self.INIT_SEQLEN
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        if model.mode in ("gridcont_gridsin", "gridcont_gridsigmoid", "gridcont2_gridsigmoid",):
            return_loss_tuple = True
        else:
            return_loss_tuple = False

        def run_epoch(split, epoch=0):
            is_train = split == 'Training'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            n_batches = len(loader)
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            d_loss, d_reg_loss, d_n = 0, 0, 0
            for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:

                # place data on the correct device
                seqs = seqs.to(self.device)
                masks = masks[:, :-1].to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    if return_loss_tuple:
                        logits, loss, loss_tuple = model(seqs,
                                                         masks=masks,
                                                         with_targets=True,
                                                         return_loss_tuple=return_loss_tuple)
                    else:
                        logits, loss = model(seqs, masks=masks, with_targets=True)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                d_loss += loss.item() * seqs.shape[0]
                if return_loss_tuple:
                    reg_loss = loss_tuple[-1]
                    reg_loss = reg_loss.mean()
                    d_reg_loss += reg_loss.item() * seqs.shape[0]
                d_n += seqs.shape[0]
                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (
                                seqs >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch + 1} iter {it}: loss {loss.item():.5f}. lr {lr:e}")

                    # tb logging
                    if TB_LOG:
                        tb.add_scalar("loss",
                                      loss.item(),
                                      epoch * n_batches + it)
                        tb.add_scalar("lr",
                                      lr,
                                      epoch * n_batches + it)

                        for name, params in model.head.named_parameters():
                            tb.add_histogram(f"head.{name}", params, epoch * n_batches + it)
                            tb.add_histogram(f"head.{name}.grad", params.grad, epoch * n_batches + it)
                        if model.mode in ("gridcont_real",):
                            for name, params in model.res_pred.named_parameters():
                                tb.add_histogram(f"res_pred.{name}", params, epoch * n_batches + it)
                                tb.add_histogram(f"res_pred.{name}.grad", params.grad, epoch * n_batches + it)

            epoch_loss     = d_loss / d_n
            epoch_reg_loss = d_reg_loss / d_n if return_loss_tuple else None
            
            if is_train:
                if return_loss_tuple:
                    logging.info(
                        f"{split}, epoch {epoch + 1}, loss {epoch_reg_loss:.5f}, {epoch_reg_loss:.5f}, lr {lr:e}.")
                else:
                    logging.info(f"{split}, epoch {epoch + 1}, loss {epoch_loss:.5f}, lr {lr:e}.")
            else:
                if return_loss_tuple:
                    logging.info(f"{split}, epoch {epoch + 1}, loss {epoch_loss:.5f}.")
                else:
                    logging.info(f"{split}, epoch {epoch + 1}, loss {epoch_loss:.5f}.")

            if MLFLOW_LOG:
                if is_train:
                    mlflow.log_metric("train_loss", epoch_loss, step=epoch)
                    if epoch_reg_loss is not None:
                        mlflow.log_metric("train_reg_loss", epoch_reg_loss, step=epoch)
                    mlflow.log_metric("lr", lr, step=epoch)
                else:
                    mlflow.log_metric("val_loss", epoch_loss, step=epoch)
            
            return epoch_loss

        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        no_improve_count = 0
        patience = self.config.patience

        for epoch in range(config.max_epochs):

            run_epoch('Training', epoch=epoch)
            if self.test_dataset is not None:
                test_loss = run_epoch('Valid', epoch=epoch)


            # supports early stopping based on the test loss, or just save always if no test set is provided
            is_best = self.test_dataset is not None and test_loss < best_loss
            if self.config.savedir is not None and is_best:
                best_loss = test_loss
                self.save_checkpoint(epoch + 1, best=True)

            if self.config.savedir is not None:
                self.save_checkpoint(epoch + 1, final=True)

            if (epoch + 1) % self.config.save_every == 0:
                ## SAMPLE AND PLOT
                # ==========================================================================================
                # ==========================================================================================
                raw_model = model.module if hasattr(self.model, "module") else model
                img_path = os.path.join(self.savedir, f'epoch_{epoch + 1:03d}.jpg')
                fig = Trainer.sample_and_plot(raw_model, cf, self.aisdls, 'valid', bgdf)
                fig.savefig(img_path)


                if MLFLOW_LOG:
                    mlflow.log_artifact(img_path, artifact_path="plots")

                ## Log Heversine metrics
                # ==========================================================================================
                # ==========================================================================================
                pred_errors = Trainer.compute_haversine_val(raw_model, cf, self.aisdls, 'valid')
                time_interval = cf.time_interval
                hours = np.array([1, 3, 5, 10, 20])
                steps = hours * 3600 / time_interval
                for idx, step in enumerate(steps, start=1):
                    step = int(step)
                    if step >= len(pred_errors):
                        break
                    err = pred_errors[step]
                    if TB_LOG:
                        tb.add_scalar(f"haversine/val_{hours[idx-1]}h", err, epoch)
                    if MLFLOW_LOG:
                        mlflow.log_metric(f"val_haversine_{hours[idx-1]}h", err, step=epoch)
                

            if test_loss is not None:
                if test_loss < best_loss:
                    best_loss = test_loss
                    self.best_epoch = epoch
                    no_improve_count = 0
                    if self.config.savedir is not None:
                        self.save_checkpoint(epoch + 1, best=True)
                else:
                    no_improve_count += 1
                    logger.info(f"No improvement for {no_improve_count} epoch(s) (patience={patience})")
                if no_improve_count >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
                    break

    @staticmethod
    @torch.no_grad()
    def compute_haversine_val(model, cf, dataloaders, case):
        dataloader = dataloaders[case]
        model.eval()
        device = cf.device
        init_seqlen = cf.init_seqlen

        v_ranges  = torch.tensor([model.lat_max-model.lat_min, model.lon_max-model.lon_min, 0, 0, 0, 0], device=device)
        v_roi_min = torch.tensor([model.lat_min, model.lon_min, 0, 0, 0, 0], device=device)
        if case == 'test':
            max_seqlen = init_seqlen + 6 * 5 + 1
        else:
            max_seqlen = init_seqlen + 6 * 3 + 1
        horizon = max_seqlen - init_seqlen

        all_mean, all_mask = [], []
        for seqs, masks, seqlens, mmsis, time_starts in tqdm(dataloader, desc=f"Haversine_{case}"):
            seqs = seqs.to(device)
            masks_ = masks[:, :max_seqlen].to(device)
            B = seqs.size(0)

            error_ens = torch.zeros(B, horizon, cf.n_samples, device=device)
            seqs_init = seqs[:, :init_seqlen, :]

            for i in range(cf.n_samples):
                preds = sample(model,
                               seqs_init,
                               horizon,
                               temperature=1.0,
                               sample=True,
                               sample_mode=cf.sample_mode,
                               r_vicinity=cf.r_vicinity,
                               top_k=cf.top_k)
                inputs = seqs[:, :max_seqlen, :]
                inp_rad = (inputs * v_ranges + v_roi_min) * torch.pi/180
                pred_rad= (preds  * v_ranges + v_roi_min) * torch.pi/180
                d = utils.haversine(inp_rad, pred_rad) * masks_
                error_ens[:, :, i] = d[:, init_seqlen:]

            mean_err = error_ens.mean(dim=-1)    # [B, horizon]
            all_mean.append(mean_err)
            all_mask.append(masks_[:, init_seqlen:])

        all_mean  = torch.cat(all_mean, dim=0)   # [N, horizon]
        all_mask = torch.cat(all_mask, dim=0)  # [N, horizon]
        summed = (all_mean * all_mask).sum(dim=0)  
        counts = all_mask.sum(dim=0)
        pred_errors = (summed / counts).cpu().numpy()
        return pred_errors
    
    @staticmethod
    @torch.no_grad()
    def sample_and_plot(model, cf, dataloaders, case, bgdf=None, n_plots=7, figsize=(9, 6), dpi=300):
        model.eval()
        device = cf.device
        init_seqlen = cf.init_seqlen
        # fetch one batch
        loader = dataloaders[case]
        seqs, masks, seqlens, mmsis, time_starts = next(iter(loader))
        seqs = seqs.to(device)
        # prepare conditioning
        seqs_init = seqs[:n_plots, :init_seqlen, :]
        # sample
        horizon = 96 - init_seqlen
        
        preds = sample(
            model,
            seqs_init,
            horizon,
            temperature=1.0,
            sample=True,
            sample_mode=cf.sample_mode,
            r_vicinity=cf.r_vicinity,
            top_k=cf.top_k
        )

        # prepare for plotting
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        if bgdf is not None:
            if bgdf.crs is not None and bgdf.crs.to_epsg() != 4326:
                bgdf = bgdf.to_crs(epsg=4326)
            bgdf.plot(ax=ax, color="lightblue", edgecolor="blue", alpha=0.5)
            lat_range = cf.lat_max - cf.lat_min
            lon_range = cf.lon_max - cf.lon_min
            def to_plot(coords):
                lat = cf.lat_min + coords[..., 0] * lat_range
                lon = cf.lon_min + coords[..., 1] * lon_range
                return lon, lat
            ax.set_xlim(cf.lon_min, cf.lon_max)
            ax.set_ylim(cf.lat_min, cf.lat_max)
        else:
            def to_plot(coords):
                return coords[..., 1], coords[..., 0]
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)

        cmap = plt.cm.get_cmap("jet")
        preds_np = preds.detach().cpu().numpy()
        inputs_np = seqs.detach().cpu().numpy()

        for idx in range(n_plots):
            c = cmap(idx / n_plots)
            try:
                seqlen = seqlens[idx].item()
            except:
                continue

            # historical trajectory
            lon_h, lat_h = to_plot(inputs_np[idx, :init_seqlen, :2])
            ax.plot(lon_h, lat_h,      color=c)
            ax.plot(lon_h, lat_h, "o", markersize=2, color=c)

            # full ground‑truth
            lon_gt, lat_gt = to_plot(inputs_np[idx, :seqlen, :2])
            ax.plot(lon_gt, lat_gt, linestyle="-.", color=c)

            # sampled future
            lon_p, lat_p = to_plot(preds_np[idx, init_seqlen:, :2])
            ax.plot(lon_p, lat_p, "x", markersize=3, color=c)

        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")

        return fig
    

    @staticmethod
    @torch.no_grad()
    def log_batch_inference_time(
        model,
        cf,
        dataloader,
    ):
        model.eval()
        device       = cf.device
        init_seqlen  = cf.init_seqlen
        horizon      = 96 - init_seqlen

        # ------------------------ pierwszy batch --------------------------
        seqs, *_ = next(iter(dataloader))
        seqs = seqs.to(device)
        batch_size = seqs.size(0)          # liczba próbek w batchu
        seqs_init  = seqs[:, :init_seqlen, :]

        # ------------------------ pomiar czasu ----------------------------
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        _ = sample(
            model,
            seqs_init,
            horizon,
            temperature = 1.0,
            sample      = True,
            sample_mode = cf.sample_mode,
            r_vicinity  = cf.r_vicinity,
            top_k       = cf.top_k,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_batch = time.perf_counter() - t0 
        elapsed_per_sample = elapsed_batch / batch_size

        return elapsed_batch, elapsed_per_sample

        # # Final state
        # raw_model = self.model.module if hasattr(self.model, "module") else self.model
        # #         logging.info("saving %s", self.config.ckpt_path)
        # logging.info(f"Last epoch: {epoch:03d}, saving model to {self.config.ckpt_path}")
        # save_path = self.config.ckpt_path.replace("model.pt", f"model_{epoch + 1:03d}.pt")
        # if MLFLOW_LOG:
        #     final_path = self.config.ckpt_path.replace("model.pt", f"final_{config.max_epochs:03d}.pt")
        #     mlflow.log_artifact(final_path, artifact_path="models")
        # torch.save(raw_model.state_dict(), save_path)
