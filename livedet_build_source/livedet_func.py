import os
os.environ["PYTORCH_JIT"] = "0" #DO NOT DEL THIS LINE

import torch # cuda check, torch pt load

import sys

from models import Effnet_MMC
from dataset1 import get_df_stone, get_transforms, MMC_ClassificationDataset
from utils.util import *
from tqdm import tqdm
import pandas as pd # <- for ensemble df

### liveness score func

def calc_liveness_score(test_loader, model_mode, model_dir='weights', fold=10, device=torch.device('cpu')) :
    folds = range(fold) # fold 개수만큼 반복
    df_result = pd.DataFrame() # each folds output
    df_avg = pd.DataFrame() # ensemble avg_df

    #Loop 10 fold
    for fold_idx in folds:

        # model mode 체크  1 = (Dermalog) 3 = (Greenbit) 
        # get model file (.pth) 
        if model_mode == '1' :
            model_file = os.path.join(model_dir, f'Dermalog_best_fold{fold_idx}.pth')
        elif model_mode == '3' :
            model_file = os.path.join(model_dir, f'GreenBit_best_fold{fold_idx}.pth')

        # Get model -> hard coding
        model = Effnet_MMC(
            enet_type='tf_efficientnet_b3_ns',
            out_dim=2,
            n_meta_features=0,
            n_meta_dim=[512, 128]
        )

        # model to cpu or gpu
        model = model.to(device)

        #load .pth file
        try:  # single GPU model_file
            model.load_state_dict(torch.load(model_file), strict=True)
            print(f'\n LIVENESS MODEL_PT_LOADED!!! -> {model_file}')
        except:  # multi GPU model_file
            state_dict = torch.load(model_file)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=True)

        #Start model eval()
        model.eval()

        ############################ Predict ###########################################
        PROBS = [] # total prob data
        with torch.no_grad():
            for (data) in tqdm(test_loader):
                data = data.to(device)
                probs = torch.zeros((data.shape[0], 2)).to(device) # (fake, live) output 틀
                l = model(data) # (fake, live) output value
                probs = l.softmax(1) # softmax prob value, dim=1
            
                PROBS.append(probs.detach().cpu()) # total prob data 

        PROBS = torch.cat(PROBS).numpy() # concat (because fold avg ensemble) # dim=0
        #################################################################################

        ### Ensemble ###

        df_result[str(fold_idx)] = PROBS[:, 0] # target_idx = 0 이므로 label이 0으로 부여된 확률만 가져옴

    df_avg = df_result.sum(axis=1) / fold # 결과확률에 대한 avg ensemble

    return df_avg


### matcher score func

