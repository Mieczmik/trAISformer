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

"""Configuration flags to run the main script.
"""

import os
import pickle
import torch
import re
import random

def get_next_exp_number(results_dir='./results', width=3):
    dirs = [
        name for name in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, name))
    ]
    nums = []
    # wyciągnij z każdej nazwy początkowy ciąg cyfr
    for name in dirs:
        m = re.match(r'^(\d+)', name)
        if m:
            nums.append(int(m.group(1)))
    # oblicz kolejny numer
    next_num = max(nums, default=0) + 1
    # zwróć jako zero‑padowany string
    return str(next_num).zfill(width)

class Config():
    retrain = True
    tb_log = False
    mlflow_log = True
    device = torch.device("cuda:0")
#     device = torch.device("cpu")
    
    save_every = 10
    
    max_epochs = 50
    batch_size = 128
    n_samples = 16
    
    init_seqlen = 12
    max_seqlen = 120
    min_seqlen = 24
    
    dataset_name = "baltic_small"

    bgdf_path = "/home/machineblue/repositories/kalinaintelligence/data/datasets/common/global_oceans_and_seas/goas_v02/baltic/baltic.geojson"

    if dataset_name == "baltic": #==============================
   
        # When mode == "grad" or "pos_grad", sog and cog are actually dlat and 
        # dlon    
        lat_size = 725
        lon_size = 1200
        sog_size = 30
        cog_size = 72
        ctp_size = 0
        dtp_size = 0 # bins = [0, 5, 10, 20, 40, 80, 160, 320, 640, 1280, float("inf")]

        
        n_lat_embd = 512
        n_lon_embd = 512
        n_sog_embd = 128
        n_cog_embd = 128
        n_ctp_embd = 0
        n_dtp_embd = 0
    
        lat_min = 52.6
        lat_max = 67.1
        lon_min = 9.0
        lon_max = 33.0

    elif dataset_name == "baltic_small" or dataset_name == "test_baltic_small": #==============================
   
        # When mode == "grad" or "pos_grad", sog and cog are actually dlat and 
        # dlon    
        lat_size = 720
        lon_size = 1180
        sog_size = 30
        cog_size = 72
        ctp_size = 0
        dtp_size = 0 # bins = [0, 5, 10, 20, 40, 80, 160, 320, 640, 1280, float("inf")]

        
        n_lat_embd = 512
        n_lon_embd = 512
        n_sog_embd = 128
        n_cog_embd = 128
        n_ctp_embd = 0
        n_dtp_embd = 0
    
        lat_min = 53.1
        lat_max = 60.3
        lon_min = 13.6
        lon_max = 25.4

    elif dataset_name == "ct_dma": #==============================
   
        # When mode == "grad" or "pos_grad", sog and cog are actually dlat and 
        # dlon    
        lat_size = 250
        lon_size = 270
        sog_size = 30
        cog_size = 72

        
        n_lat_embd = 256
        n_lon_embd = 256
        n_sog_embd = 128
        n_cog_embd = 128
    
        lat_min = 55.5
        lat_max = 58.0
        lon_min = 10.3
        lon_max = 13

    
    #===========================================================================
    # Model and sampling flags
    mode = "pos"  #"pos", "pos_grad", "mlp_pos", "mlpgrid_pos", "velo", "grid_l2", "grid_l1", 
                            # "ce_vicinity", "gridcont_grid", "gridcont_real", "gridcont_gridsin", "gridcont_gridsigmoid"
    sample_mode =  "pos_vicinity" # "pos", "pos_vicinity" or "velo"
    top_k = 10 # int or None 
    r_vicinity = 40 # int
    
    # Blur flags
    #===================================================
    blur = True
    blur_learnable = False
    blur_loss_w = 1.0
    blur_n = 2
    if not blur:
        blur_n = 0
        blur_loss_w = 0
    
    # Data flags
    #===================================================
    datadir = f"./data/{dataset_name}/"
    trainset_name = f"{dataset_name}_train.pkl"
    validset_name = f"{dataset_name}_valid.pkl"
    testset_name = f"{dataset_name}_test.pkl"
    
    
    # model parameters
    #===================================================
    n_head = 8
    n_layer = 8
    full_size = lat_size + lon_size + sog_size + cog_size + ctp_size + dtp_size
    n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd + n_ctp_embd + n_dtp_embd
    # base GPT config, params common to all GPT versions
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    
    # optimization parameters
    #===================================================
    learning_rate = 6e-4 # 6e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # early stopping
    patience = 10
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = True
    warmup_tokens = 512*20 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    num_workers = 4 # for DataLoader
    
    exp_number = get_next_exp_number('./results', width=3)
    filename = f"{exp_number}"\
        + f"-{dataset_name}"\
        + f"-{mode}-{sample_mode}-{top_k}-{r_vicinity}"\
        + f"-blur-{blur}-{blur_learnable}-{blur_n}-{blur_loss_w}"\
        + f"-data_size-{lat_size}-{lon_size}-{sog_size}-{cog_size}"\
        + f"-embd_size-{n_lat_embd}-{n_lon_embd}-{n_sog_embd}-{n_cog_embd}"\
        + f"-head-{n_head}-{n_layer}"\
        + f"-bs-{batch_size}"\
        + f"-lr-{learning_rate}"\
        + f"-seqlen-{init_seqlen}-{max_seqlen}"
    savedir = "./results/"+filename+"/"

    
    ckpt_path = os.path.join(savedir,"best_model.pt")

    @classmethod
    def get_time_interval(cls):
        train_path = os.path.join(cls.datadir, cls.trainset_name)
        with open(train_path, 'rb') as f:
            l_pred_errors = pickle.load(f)
        traj1 = random.choice(l_pred_errors)["traj"]
        traj2 = random.choice(l_pred_errors)["traj"]
        dt1 = traj1[1, -2] - traj1[0, -2]
        dt2 = traj2[1, -2] - traj2[0, -2]
        if dt1 != dt2:
            raise ValueError(f"Differing time intervals: dt1={dt1}, dt2={dt2}! Check your dataset.")
        cls.time_interval = dt1
        print(f"[Config] inferred time_interval = {cls.time_interval}")

Config.get_time_interval()