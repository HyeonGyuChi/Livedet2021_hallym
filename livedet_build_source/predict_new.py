import os
os.environ["PYTORCH_JIT"] = "0" #DO NOT DEL THIS LINE

import torch

import sys

# from models import Effnet_MMC, Resnest_MMC, Seresnext_MMC
from models import Effnet_MMC
from dataset1 import get_df_stone, get_transforms, MMC_ClassificationDataset
from utils.util import *
from tqdm import tqdm
import pandas as pd # <- for ensemble df

import livedet_func

# def parse_args -> class
# Can be used as dictionary (ClassInstance.__dict__)

# Argument check Error
if (len(sys.argv)-1) != 3 :
    print('\n ====== EXE ARGUMENT ERROR =====')
    print('WRONG .EXE ARGUMENT, PLEASE CHECK YOUR ARGUMENT (INPUT OF ARGUMENT LENGTH : {})'.format(len(sys.argv)-1))
    sys.exit(1)


class ModelInfo:
    def __init__(self):
        self.kernel_type = '10fold_b3_10ep'

        #self.data_dir = './data/'
        #self.data_folder = './data/'

        self.image_size = 256 # -> training시, 256
        self.enet_type = 'tf_efficientnet_b3_ns' #tf_efficientnet_b3_ns
        self.batch_size = 16
        self.num_workers = 0
        self.out_dim = 2
        self.use_amp = True
        self.use_ext = False # <- True
        self.k_fold = 10

        self.use_meta = False #True
        self.DEBUG = False # <- False (smaple 개수에 따라 잘 설정) (df_test)를 shffle하는 용도 (IMPORTANT args.DEBUG == False ALWAYS)
        self.model_dir = 'weights'
        self.log_dir = './logs'
        self.sub_dir = './subs'
        self.eval = "best" #choices=['best', 'best_no_ext', 'final'] <- 사용 안될것.
        self.n_test = 1 # TTA 횟수 1일경우 실행 x
        self.CUDA_VISIBLE_DEVICES = '0'
        self.n_meta_dim = '512,128'

        
        #Get argv[1,2,3]
        self.model_mode = sys.argv[1]
        self.source_dir = sys.argv[2]
        self.output_dir = sys.argv[3]

def main(device):

    #Get test dataframe & target_idx
    df_test, meta_features, n_meta_features, target_idx = get_df_stone(
        k_fold = args.k_fold,
        source_path = args.source_dir,
        out_dim = 2,
        use_meta = args.use_meta,
        use_ext = args.use_ext
    )


    #Get validation transforms 
    transforms_val = get_transforms(args.image_size)


    #IMPORTANT : args.DEBUG == False ALWAYS!!!
    if args.DEBUG: 
        # batch_size 배수만큼 (testdatset 개수에 따라 주의) -> shuffle # 추후 앙상블 할 때 조심, 같은 데이터가 아닌 서로 다른 데이터 결과끼리 앙상블 될 가능성
        df_test = df_test.sample(args.batch_size * 2)


    #Print logs
    print('--Get Model Info--')
    print('==== df_test ====')
    print(df_test)
    print('==== meta_features ====')
    print(meta_features)
    print('==== n_meta_features ====')
    print(n_meta_features)
    print('==== target_idx ====')
    print(target_idx)
    print('==== transforms_val ====')
    print(transforms_val)

    # liveness_dataset
    dataset_test = MMC_ClassificationDataset(df_test, 'test', meta_features, transform=transforms_val)
    # shuffle = false되도록 변경
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    
    ############
    
    # from livedet_func.py 
    df_liveness_score = livedet_func.calc_liveness_score(test_loader, args.model_mode, args.model_dir, args.k_fold, device)
    ############

    change_txt = open(args.output_dir,'w')

    #Write result on file(.txt)
    for i in df_liveness_score:
        change_txt.write(str(int(float(i) * 100)) + '\n')

    change_txt.close()


if __name__ == '__main__':
    ## Need some try-catch to argv len

    # os.makedirs(args.sub_dir, exist_ok=True)
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
    
    #IMPORTANT : Get ModelInfo HERE!!!!!!
    args = ModelInfo()

    # model_mode error처리
    if args.model_mode not in ('1', '3') :
        print('\n ====== MODEL_MODE ERROR =====')
        print('You input wrong MODEL_MODE (INPUT OF MODEL_MODE : {})'.format(args.model_mode))
        print('Please input MODEL_MODE only [1,3] == [Greenbit, Dermalog]')
        sys.exit(1)


    # source_dir error리턴
    if not os.path.exists(args.source_dir) :
        print('\n ====== SOURCE_DIRECTORY ERROR =====')
        print('SOURCE FILE is not exist')
        print('Please check your SOURCE DIRECTORY (INPUT OF SOURCE DIR : {})'.format(args.source_dir))
        sys.exit(1)

    
    #Get model type
    if 'efficientnet' in args.enet_type:
        ModelClass = Effnet_MMC
    # elif args.enet_type == 'resnest101':
    #     ModelClass = Resnest_MMC
    #     print('resnest101')
    # elif args.enet_type == 'seresnext101':
    #     ModelClass = Seresnext_MMC
    #     print('seresnext101')
    else:
        raise NotImplementedError()

    #Print log for ModelInfo
    print('==== args.dict ====')
    print(args.__dict__)


    # device 설정 -- CUDA 사용시 out of memory error 발생가능
    if torch.cuda.is_available() :
        print(' === USING CUDA === ')
        device = torch.device('cuda')
    else :
        print('=== USING CPU ===')
        device = torch.device('cpu')

    
    main(device)

