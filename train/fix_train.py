import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.utils import get_model, get_optimizer, get_scheduler, save_all, loss_drawing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FixMatch:
    def __init__(self, cfg, loader, logger, model_save_file):
        super(FixMatch, self).__init__()
        self.train_loader, self.un_loader = loader
        self.logger = logger
        self.model_save_file = model_save_file
        
        self.num_classes = cfg.BASIC.CLASS_NUM
        self.net_type = cfg.TRAIN.NET_TYPE
        
        self.lr = cfg.TRAIN.OPTIMIZER.LR_SCHEDULER
        self.momentum = cfg.TRAIN.OPTIMIZER.MOMENTUM
        self.weight_decay = cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY
        self.opti_type = cfg.TRAIN.OPTIMIZER.TYPE
    
        self.sche_type = cfg.TRAIN.SCHEDULER.TYPE
        self.paramenter_1 = cfg.TRAIN.SCHEDULER.PARAMETER_1
        self.paramenter_2 = cfg.TRAIN.SCHEDULER.PARAMETER_2
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        self.threshold = cfg.TRAIN.THRESHOLD
        self.epoch = cfg.TRAIN.EPOCH
        self.thresh_warmup = cfg.TRAIN.THRESH_WARMUP

        self.get_model_and_optimizer()

    def get_model_and_optimizer(self):
        self.model = get_model(self.num_classes, self.net_type)
        self.optimizer = get_optimizer(self.model, self.lr, self.momentum, 
                                       self.weight_decay, self.opti_type)
        self.scheduler = get_scheduler(self.optimizer, self.sche_type, 
                                       self.paramenter_1, self.paramenter_2)
        self.model.to(device)

    def train(self):
        epoch_iter = len(self.train_loader)
        num_train_iter = self.epoch * epoch_iter
        
        num_iter = 0
        unsup_warmup = 1

        for e in range(0, self.epoch):
            loss_e= 0
            num_use = 0
            self.model.train()
            for i, (input_l, input_u) in enumerate(zip(self.train_loader, self.un_loader)):
                if self.thresh_warmup == True:
                    unsup_warmup = np.clip(num_iter/(0.3 * num_train_iter), a_min=0.0, a_max=1.0)
                    unsup_warmup = torch.tensor(unsup_warmup)
                    
                # 提取有标签的训练数据 torch.Size([b, 3, 224, 224]) torch.Size([b])
                _, x, y = input_l
                y = y.type(torch.LongTensor)
                y = y.to(device)
                label_batchsize = y.shape[0]
                # 无标签 torch.Size([ub, 3, 224, 224]) torch.Size([ub, 3, 224, 224])
                _, x_u, _ = input_u
                w, s = x_u
                # 数据合并 torch.Size([b+ub+ub, 3, 224, 224])
                inputs = torch.cat((x, w, s), dim=0).to(device)
                outputs = self.model(inputs)
                # 数据拆解
                # torch.Size([b, 4]) torch.Size([ub, 4]) torch.Size([ub, 4])
                out_x = outputs[:label_batchsize]
                out_w, out_s = outputs[label_batchsize:].chunk(2)

                loss_x = nn.CrossEntropyLoss()(out_x, y)
                # 无标签损失
                # 先获得弱标签的伪标签
                pseudo_label = torch.softmax(out_w.detach(), dim=1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(self.threshold).float()
                num_use += torch.count_nonzero(mask)
                # 比较损失
                loss_w = (F.cross_entropy(out_s, targets_u, reduction='none')*mask).mean()

                # 参数更新
                loss = unsup_warmup * loss_w + loss_x
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_e += loss.detach().cpu()
                num_iter += 1

            info = "E|L:{}|{} lr:{} num_use:{}".format(
                e+1, loss_e, self.optimizer.state_dict()['param_groups'][0]['lr'], num_use)
            
            if (e+1) % 20 == 0 :
                torch.save({
                    'model_state_dict': self.model.state_dict()
                }, self.model_save_file+'model.pt')
            
            self.logger.info(info)
            self.scheduler.step()
