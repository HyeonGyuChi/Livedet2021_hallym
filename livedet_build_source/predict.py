import os
os.environ["PYTORCH_JIT"] = "0" #DO NOT DEL THIS LINE

import sys

import torch

from dataset1 import get_df, get_transforms, MMC_FPDataset
import livedet_func
#from utils.util import *
from tqdm import tqdm
import pandas as pd # <- for ensemble df

import matplotlib.pylab as plt

#Argument len check
if (len(sys.argv)-1) != 5 :
    print('\n ====== EXE ARGUMENT ERROR =====')
    print('WRONG .EXE ARGUMENT, PLEASE CHECK YOUR ARGUMENT (INPUT OF ARGUMENT LENGTH : {})'.format(len(sys.argv)-1))
    sys.exit(1)

class Arguments:
    def __init__(self, ndataset, templateimagesfile, probeimagesfile, livenessoutputfile, IMSoutputfile):
        self.ndataset           = ndataset
        self.templateimagesfile = templateimagesfile
        self.probeimagesfile    = probeimagesfile
        self.livenessoutputfile = livenessoutputfile
        self.IMSoutputfile      = IMSoutputfile

def main():
    # device 설정 -- CUDA 사용시 out of memory error 발생가능
    if torch.cuda.is_available() :
        print(' === USING CUDA === ')
        device = torch.device('cuda')
    else :
        print('=== USING CPU ===')
        device = torch.device('cpu')

    #Get dataframe template & probe
    df_template, template_targetind = get_df(args.templateimagesfile)
    df_probe, probe_target_ind = get_df(args.probeimagesfile)

    # DataFrame length Error
    if df_template.shape[0] != df_probe.shape[0]:
        print('\n ====== IMAGE NUM NOT MATCHED ERROR =====')
        print('Number of images does not matched [templateimagesfile] and [probeimagesfile]')
        print('Match the number of images in [templateimagesfile] and [probeimagesfile]')
        sys.exit(1)


    #Get Albumentaion transforms template & probe (imagesize : 256)
    transforms_template, transforms_probe = get_transforms(256)

    print(df_template)
    print(df_probe)

    #HUN : I Skipped 'if args.DEBUG:'

    #Get Dataset
    dataset_fp = MMC_FPDataset(df_template, df_probe, transforms_template, transforms_probe)

    #Get test_loader
    test_loader = torch.utils.data.DataLoader(dataset_fp, batch_size=4, num_workers=0, shuffle=False)

    #### liveness ####
    liveness_score_df = livedet_func.get_fp_livenessScore(test_loader, model_mode=args.ndataset, fold=10, device=device)
    print(liveness_score_df)
    print(type(liveness_score_df))

    ###### matcher #####
    #Get Maching score dataframe
    matching_score_df = livedet_func.get_fp_matchingScore(test_loader, model_mode=args.ndataset, fold=5, device=device)
    print(matching_score_df)
    print(type(matching_score_df))

    #Plz add here, livedet_func.get_fp_integratedScore()

    #Write livenessoutputfile on file(.txt)
    change_txt = open(args.livenessoutputfile,'w')
    for i in liveness_score_df:
        change_txt.write(str(int(float(i) * 100)) + '\n')
    change_txt.close()

    #Write IMSoutputfile on file(.txt)
    change_txt = open(args.IMSoutputfile,'w')
    for i in matching_score_df:
        change_txt.write(str(float(i)) + '\n')
    change_txt.close()



if __name__ == '__main__':
    #IMPORTANT : Get ModelInfo HERE!!!!!!
    args = Arguments(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

    # model_mode error처리
    if args.ndataset not in ('1', '3') :
        print('\n ====== MODEL_MODE ERROR =====')
        print('You input wrong ndataset (INPUT OF MODEL_MODE : {})'.format(args.ndataset))
        print('Please input ndataset only [1,3] == [Greenbit, Dermalog]')
        sys.exit(1)

    # templateimagesfile error리턴
    if not os.path.exists(args.templateimagesfile) :
        print('\n ====== SOURCE_DIRECTORY ERROR =====')
        print('SOURCE FILE is not exist')
        print('Please check your SOURCE DIRECTORY (INPUT OF SOURCE DIR : {})'.format(args.templateimagesfile))
        sys.exit(1)

    #  probeimagesfile error리턴
    if not os.path.exists(args.probeimagesfile) :
        print('\n ====== SOURCE_DIRECTORY ERROR =====')
        print('SOURCE FILE is not exist')
        print('Please check your SOURCE DIRECTORY (INPUT OF SOURCE DIR : {})'.format(args.probeimagesfile))
        sys.exit(1)

    #HUN : I Skipped processing 'args.enent_type'

    #Print log for ModelInfo
    print('==== args.dict ====')
    print(args.__dict__)

    ''' Imsoutput demo
    live_df = pd.DataFrame([0.02523,0.322,0.61,0.9999]) # fake fake live live
    matcher_df = pd.DataFrame([0.2,2.3,0.2,1.3]) # correct wrong correct wrong

    live_df = live_df.apply(lambda x : (round(x * 100))).astype(int)
    # live_df.apply(lambda x : round(x * 100))
    # live_df[0].map(lambda x : round(x * 100))
    print('===live_df=== \n', live_df)

    print('===matcher_df=== \n', matcher_df)

    # 3 32 100 0
    ims_df = livedet_func.get_fp_IMSoutputScore(live_df, matcher_df) 
    print('===ims_df=== \n', ims_df)
    
    sys.exit(1)
    '''
    
    main()

    

