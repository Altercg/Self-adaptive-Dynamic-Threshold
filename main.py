import argparse
import warnings
import torch
import os
from configs import cfg, update_config

from dataset import generate_loader, load_train_dataset
from train import FixMatch, FlexMatch, FldtMatch, test
# (BasicTrain, PiModel, SAM_FixMatch, SAM_BasicTrain, Temporal, MeanTeacher)
from utils import create_logger

warnings.filterwarnings('ignore')

# 参数设置
def parse_args():
    parser = argparse.ArgumentParser(description='dfu')
    parser.add_argument(
        '--cfg',                                                        
        required=False,
        default="./configs/DFU.FldtMatch.EfficientNet.yaml"
    )   # EfficientNet FixMatch DenseNet Hierarchy Temporal PiModel mocoMatch BiT Basic
    args = parser.parse_args()
    return args

def main(cfg, logger, model_save_dir):
    labelImgs, unlabelImgs = load_train_dataset(cfg.DATASET.TRAIN_IMG_ROOT)
    logger.info(
        "Finish, total number:{}, label number:{}, unlabel number:{}".format(
            len(labelImgs)+len(unlabelImgs), len(labelImgs), len(unlabelImgs)))

    labelLoader = generate_loader(labelImgs, cfg.DATASET.LABEL_BATCHSIZE, True)
    unlabelLoader = generate_loader(unlabelImgs, cfg.DATASET.UNLABEL_BATCHSIZE, False)
    loader = [labelLoader, unlabelLoader]

    logger.info("Start training")
    t = eval(cfg.TRAIN.SSL_TYPE)(cfg, loader, logger, model_save_dir)
    t.train()
    logger.info("Start testing")
    test(cfg, logger, model_save_dir)

    torch.cuda.empty_cache()

if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args.cfg)                    
    logger, log_dir= create_logger(cfg)
    main(cfg, logger, log_dir)
    # file = os.listdir('/root/autodl-tmp/results/ISIC/NOAUC')
    # for f in file:
    #     logger.info("{}".format(f))
    #     test(cfg, logger, '/root/autodl-tmp/results/ISIC/NOAUC/'+f+'/')
'''
ps aux|grep root|grep python
'''
