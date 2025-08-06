#!/usr/bin/env python
# coding: utf-8
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

"""Pytorch implementation of TrAISformer---A generative transformer for
AIS trajectory prediction

https://arxiv.org/abs/2109.03958

"""
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import os
import sys
import pickle
from tqdm import tqdm
import math
import logging
import pdb
import random
import contextlib

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

import models, trainers, datasets, utils
from config_trAISformer import Config

import mlflow
import mlflow.pytorch
import inspect
import geopandas as gpd


cf = Config()
TB_LOG = cf.tb_log
if TB_LOG:
    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter()

MLFLOW_LOG = cf.mlflow_log

## Background to plots:
bgdf = gpd.read_file(cf.bgdf_path)

# make deterministic
utils.set_seed(42)
torch.pi = torch.acos(torch.zeros(1)).item() * 2

if MLFLOW_LOG:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://10.90.90.95:5000"))
    exp_name = f"traisformer_{cf.dataset_name}"
    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(
            name=exp_name,
            artifact_location=f"mlflow-artifacts:/{exp_name}"
        )
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", exp_name))
    run_name = cf.filename

if __name__ == "__main__":

    @contextlib.contextmanager
    def mlflow_run_if_enabled(enabled, **kw):
        if enabled:
            with mlflow.start_run(**kw) as run:
                yield run
        else:
            yield None
    with mlflow_run_if_enabled(MLFLOW_LOG, run_name=run_name) as run:
        run_id = run.info.run_id if run else None
        device = cf.device
        init_seqlen = cf.init_seqlen

        ## Logging
        # ===============================
        if not os.path.isdir(cf.savedir):
            os.makedirs(cf.savedir)
            print('======= Create directory to store trained models: ' + cf.savedir)
        else:
            print('======= Directory to store trained models: ' + cf.savedir)
        utils.new_log(cf.savedir, "log")

        ## Model
        # ===============================
        if cf.ctp_size == 0 and cf.dtp_size == 0 and cf.n_ctp_embd == 0 and cf.n_dtp_embd == 0:
            model = models.BaseTrAISformer(cf, partition_model=None)
            model_type = 'base'
        else:
            model = models.TrAISformer(cf, partition_model=None)
            model_type = None
        
        ## Data
        # ===============================
        moving_threshold = 0.05
        l_pkl_filenames = [cf.trainset_name, cf.validset_name, cf.testset_name]
        Data, aisdatasets, aisdls = {}, {}, {}
        for phase, filename in zip(("train", "valid", "test"), l_pkl_filenames):
            datapath = os.path.join(cf.datadir, filename)
            print(f"Loading {datapath}...")
            with open(datapath, "rb") as f:
                l_pred_errors = pickle.load(f)
            for V in l_pred_errors:
                try:
                    moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]
                except:
                    moving_idx = len(V["traj"]) - 1  # This track will be removed
                V["traj"] = V["traj"][moving_idx:, :]
            Data[phase] = [x for x in l_pred_errors if not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen]
            print(len(l_pred_errors), len(Data[phase]))
            print(f"Length: {len(Data[phase])}")
            print("Creating pytorch dataset...")
            # Latter in this scipt, we will use inputs = x[:-1], targets = x[1:], hence
            # max_seqlen = cf.max_seqlen + 1.
            if cf.mode in ("pos_grad", "grad"):
                aisdatasets[phase] = datasets.AISDataset_grad(Data[phase],
                                                            max_seqlen=cf.max_seqlen + 1,
                                                            device=cf.device,
                                                            model_type=model_type)
            else:
                aisdatasets[phase] = datasets.AISDataset(Data[phase],
                                                        max_seqlen=cf.max_seqlen + 1,
                                                        device=cf.device,
                                                        model_type=model_type)
            if phase == "test":
                shuffle = False
            else:
                shuffle = True
            aisdls[phase] = DataLoader(aisdatasets[phase],
                                    batch_size=cf.batch_size,
                                    shuffle=shuffle)
        cf.final_tokens = 2 * len(aisdatasets["train"]) * cf.max_seqlen

        ## Trainer
        # ===============================
        trainer = trainers.Trainer(
            model, aisdatasets["train"], aisdatasets["valid"], cf, savedir=cf.savedir, device=cf.device, aisdls=aisdls, INIT_SEQLEN=init_seqlen)

        ## Training
        # ===============================
        if MLFLOW_LOG:
            params = {}
            for name, val in inspect.getmembers(cf):
                if name.startswith("_") or inspect.ismethod(val) or inspect.isfunction(val):
                    continue
                if isinstance(val, (str, int, float, bool)):
                    params[name] = val

            mlflow.log_params(params)
            total_params = sum(p.numel() for p in model.parameters())
            mlflow.log_param("total_parameters", total_params)
        if cf.retrain:
            trainer.train()


        ## Evaluation
        # ===============================
        # Load the best model
        model.load_state_dict(torch.load(cf.ckpt_path))
        model.to(cf.device)


        pred_errors    = trainer.compute_haversine_val(model, cf, aisdls, 'test')
        hours          = np.array([1, 3, 5, 10, 15])
        steps          = (hours * 3600 // cf.time_interval).astype(int)

        for k, step in enumerate(steps, start=1):
            if step >= len(pred_errors):
                break
            err = float(pred_errors[step])
            if TB_LOG:
                tb.add_scalar(f"haversine/test_{hours[k-1]}h", err, 0)
            if MLFLOW_LOG:
                mlflow.log_metric(f"test_haversine_{hours[k-1]}h", err)
        elapsed_batch, elapsed_per_sample = trainer.log_batch_inference_time(
            model      = model,
            cf         = cf,
            dataloader = aisdls["test"]
            )
        if MLFLOW_LOG:
            mlflow.log_metric("batch_inference_time_ms",  elapsed_batch  * 1e3)
            mlflow.log_metric("sample_inference_time_ms", elapsed_per_sample * 1e3) 


        ## Plot
        # ===============================
        img_path = os.path.join(cf.savedir, f'trajectories_best.jpg')
        fig = trainer.sample_and_plot(model, cf, aisdls, 'test', bgdf)
        fig.savefig(img_path)
        if MLFLOW_LOG:
            mlflow.log_artifact(img_path, artifact_path="plots")

        plt.figure(figsize=(9, 6), dpi=150)
        v_times = np.arange(len(pred_errors)) / 6
        plt.plot(v_times, pred_errors)

        timestep = 6
        plt.plot(1, pred_errors[timestep], "o")
        plt.plot([1, 1], [0, pred_errors[timestep]], "r")
        plt.plot([0, 1], [pred_errors[timestep], pred_errors[timestep]], "r")
        plt.text(1.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

        timestep = 12
        plt.plot(2, pred_errors[timestep], "o")
        plt.plot([2, 2], [0, pred_errors[timestep]], "r")
        plt.plot([0, 2], [pred_errors[timestep], pred_errors[timestep]], "r")
        plt.text(2.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

        timestep = 18
        plt.plot(3, pred_errors[timestep], "o")
        plt.plot([3, 3], [0, pred_errors[timestep]], "r")
        plt.plot([0, 3], [pred_errors[timestep], pred_errors[timestep]], "r")
        plt.text(3.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)
        plt.xlabel("Time (hours)")
        plt.ylabel("Prediction errors (km)")
        plt.xlim([0, 12])
        plt.ylim([0, 20])
        # plt.ylim([0,pred_errors.max()+0.5])
        plt.savefig(cf.savedir + "prediction_error.png")

        # Yeah, done!!!
