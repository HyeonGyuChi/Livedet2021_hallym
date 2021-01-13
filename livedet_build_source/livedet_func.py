import os

import torch
from torch import nn
import pandas as pd
from tqdm import tqdm
from models import Effnet_MMC

### liveness score func


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








        

    
