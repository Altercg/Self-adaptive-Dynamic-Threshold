import datetime
import numpy as np
import logging
import os
import csv
import matplotlib.pyplot as plt
import torchvision.models as models
from sklearn.metrics import classification_report
from torch import nn
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from utils.SAM import *
from glob import glob
from utils.efficientnet_pytorch import EfficientNet
from utils.bit_pytorch import models as BiT

def create_logger(cfg):
    config_name = cfg.BASIC.NAME
    result_dir = cfg.BASIC.RESULT_DIR
    experiment_index = len(glob(f"{result_dir}/*"))
    experiment_dir = result_dir+f'{experiment_index:03d}-'+config_name+'/'  # 文件夹路径
    os.mkdir(experiment_dir)                           # 创建文件夹
    
    time_str = datetime.datetime.now().strftime('%Y-%m-%d')
    log_name = "{}_{}.log".format(config_name, time_str)
    log_path = str(os.path.join(experiment_dir, log_name)) # log文件
    
    # set up logger
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger('DFU')    # 实例一个日志器
    console = logging.FileHandler(log_path)   # 设置处理器日志输出
    logging.getLogger('DFU').addHandler(console)
    logger.info("-----------------Cfg is set as follow---------------")
    logger.info(cfg)
    logger.info("----------------------------------------------------")
    return logger, experiment_dir

def get_model(class_num, net_type):
    if net_type.split('-')[0] == 'efficientnet':
        model = EfficientNet.from_name(net_type)
        model.load_state_dict(torch.load("/root/autodl-tmp/models/"+net_type+".pth"))
        in_f = model._fc.in_features  # 改变类别
        model._fc = nn.Linear(in_f, class_num, bias=True)
    elif net_type == 'DenseNet-121':
        model = models.densenet121(pretrained=True)
        in_f = model.classifier.in_features  # 改变类别
        model.classifier = nn.Linear(in_f, class_num, bias=True)
    elif net_type == 'DenseNet-201':
        model = models.densenet201(pretrained=True)
        in_f = model.classifier.in_features  # 改变类别
        model.classifier = nn.Linear(in_f, class_num, bias=True)
    elif net_type.split('-')[0] == 'BiT':# BiT-S-R50x1 BiT-S-R101x1
        model = BiT.KNOWN_MODELS[net_type](head_size=class_num, zero_head=True)
        model.load_from(np.load("/root/autodl-tmp/models/"+net_type+".pth"))

    return model

def get_optimizer(model, lr, momentum, weight_decay, opti_type):
    if opti_type == 'Adam': # pimodel 和 temporal 用
        optimizer = Adam(model.parameters(), lr)# self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, 
    elif opti_type == 'SAM':
        base_optimizer = SGD
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
            lr=lr,
            momentum=momentum,
            adaptive=True,
            rho=2
            )
    else:
        optimizer = eval(opti_type)(
            model.parameters(),
            momentum=momentum, 
            lr=lr,
            weight_decay=weight_decay
        )
    return optimizer

def get_scheduler(optimizer, sche_type, paramenter_1, paramenter_2):
    scheduler = eval(sche_type)(optimizer, paramenter_1, paramenter_2)
    return scheduler

def evaluating_indicator(y_true, y_pred, labels, target_name):
    """
        return: 'per_class_indicator' include per class precision, recall,
                    f1-score, support(在真值里面出现的次数),
                    per indicator macro-avg
                micro-average f1-score
                micro-average auc
    """
    labels = [i for i in range(len(target_name))]
    all = classification_report(
        y_true=y_true, y_pred=y_pred,
        labels=labels,
        target_names=target_name,
        output_dict=True)    # 可以转化为字典，使用指标数值
    
    return all

def loss_drawing(all_loss, all_f1, k, model_save_path, step):
    epoch = range(1, len(all_loss)+1)
    epoch2= range(step, len(all_f1)*step+1, step)
    plt.plot(epoch, all_loss, 'b', label='Train_loss')
    plt.plot(epoch2, all_f1, 'r', label='F1-score')
    plt.legend()
    plt.savefig(model_save_path + "{}_{}_loss_f1-score.png".format(k, step))
    plt.clf()

def save_all(logger, all):
    for i in all.keys():
        logger.info("{}:{}".format(i, all[i]))
