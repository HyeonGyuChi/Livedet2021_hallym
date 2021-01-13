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
def get_fp_matchingScore(test_loader):
    ###############################Required parameters##########################
    args_k_fold = 5
    args_model_mode = '3'
    args_model_dir = 'weights'
    args_enet_type = 'tf_efficientnet_b3_ns'
    args_n_meta_dim = '512,128'
    args_out_dim = 1000
    args_n_test = 1

    n_meta_features = 0
    #n_meta_dim = [int(nd) for nd in args_n_meta_dim(',')]
    target_idx = 0

    if torch.cuda.is_available() :
        print(' === USING CUDA === ')
        device = torch.device('cuda')
    else :
        print('=== USING CPU ===')
        device = torch.device('cpu')
    #############################################################################
    

    
    folds = range(args_k_fold)  # For fold loop
    df_avg = pd.DataFrame()     # For ensemble
    df_result = pd.DataFrame()  # For collect result

    #Loop ? fold 
    for fold in folds:
        
        #I think it is better, get model mode as func parameter
        # model mode check  1 = (Dermalog) 3 = (Greenbit) 
        if args_model_mode == '1' :
            model_file = os.path.join(args_model_dir, f'Dermalog_best_fold{fold}.pth')
        elif args_model_mode == '3' :
            model_file = os.path.join(args_model_dir, f'Greenbit_tl_{fold}.pth')

        
        #Get model (ModelClass -> Effnet_MMC)
        model = Effnet_MMC(
            args_enet_type,
            n_meta_features=n_meta_features,
            out_dim = args_out_dim
        )


        #Load model on CPU or GPU
        model = model.to(device)


        #Set model weights
        try:  # single GPU model_file
            model.load_state_dict(torch.load(model_file), strict=True)
            print(f'\n MODEL_PT_LOADED!!! -> {model_file}')
        except:  # multi GPU model_file
            state_dict = torch.load(model_file)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=True)


        #Start model eval()
        model.eval()


        #######################################################################################
        SCORE = []
        with torch.no_grad():
            for (data) in tqdm(test_loader):
                
                #No needs meta...
                
                template_data = data[0].to(device)
                probe_data = data[1].to(device)

                #matched_score = torch.zeros((data[0].shape[0])).to(device)

                for I in range(args_n_test):# I think this line is useless(cause args_n_test == 1)
                    pdist = nn.PairwiseDistance(p=2)
                    input1 = model(template_data)# Is it right?
                    input2 = model(probe_data)
                    output = pdist(input1, input2)

                    #matched_score += output

                    #Thresholding
                    # if output > 1.0:    return False
                    # else:               return True

                #score /= args_n_test#Useless too....

                SCORE.append(output.detach().cpu())

            SCORE = torch.cat(SCORE).numpy()
        #######################################################################################

        ### Ensemble ###
        df_result[str(fold)] = SCORE

    df_avg = df_result.sum(axis=1)/args_k_fold
    
    print(type(df_avg))

    return df_avg








        

    
