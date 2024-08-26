import numpy as np
import torch
import torch.nn as nn
from utils import evaluating_indicator
from utils.utils import get_model, get_optimizer, get_scheduler, save_all

from train.valid import validate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicTrain():
    def __init__(self, cfg, loader, logger, model_save_file, k, step):
        super(BasicTrain, self).__init__()
        self.train_loader,  self.valide_loader, self.un_loader = loader
        self.model_save_file = model_save_file
        self.logger = logger
        self.step = step
        self.cfg = cfg
        self.k = k
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.start_epoch = 0
        self.num_classes = cfg.BASIC.CLASS_NUM
        self.resume = False
        if cfg.BASIC.RESUME == True:
            self.resume = True
            self.checkpoint = torch.load(cfg.BASIC.RESUME_PATH)
        self.get_model_and_optimizer()

    def get_model_and_optimizer(self):
        self.model = get_model(self.cfg, self.step)
        self.optimizer = get_optimizer(self.cfg, self.model, self.step)
        self.scheduler = get_scheduler(self.cfg, self.optimizer, self.step)
        self.start_epoch = 0
        if self.resume == True:
            self.model = self.model.load_state_dict(self.checkpoint['model_state_dict'])
            self.optimizer = self.optimizer.load_state_dict(self.checkpoint['optim_state_dict'])
            self.start_epoch = self.checkpoint['epoch']
        self.model.to(device)

    def train(self):
        # all_loss = []
        # all_f1 = []
        best_f1 = (0, 0)

        epoch = eval('self.cfg.TRAIN_'+ str(self.step) + '.EPOCH')
        iters = len(self.train_loader)
        for e in range(self.start_epoch, epoch):
            loss_e = 0
            self.model.train()
            for i, input_l in enumerate(self.train_loader):
                _, _, x, label = input_l
                if self.step == 1:  # 1代表第一阶段
                    y = label[0].type(torch.LongTensor)
                elif self.step == 2:
                    y = label[1].type(torch.LongTensor)
                elif self.step == 3:
                    y = label[1].type(torch.LongTensor)
                    y -= 2
                else:
                    y = label[1].type(torch.LongTensor)
                y = y.to(device)
                inputs = x.to(device)
                outputs = self.model(inputs)
                # 有标签损失
                loss = nn.CrossEntropyLoss()(outputs, y)
                # 反馈
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # 损失存储
                loss_e += loss
                # self.scheduler.step(e+i/iters)
            
            # 存储该epoch的损失
            # all_loss.append(loss_e/count)
            # save best model
            if (e+1) % self.cfg.BASIC.VALID_STEP == 0 :
                all = self.validate()
                f1_score = all['macro avg']['f1-score']
                # all_f1.append(f1_score)
                # 保存中间模型   
                if best_f1[1] < f1_score: # 存储该折最好模型
                    best_f1 = (e+1, f1_score)
                    torch.save({
                        'model_state_dict': self.model.state_dict()
                    }, self.model_save_file+'{}-{}-best-model.pt'.format(self.k, self.step))
            
            info = "E|L:{}|{} lr:{} best-f1:{}".format(
                e+1, loss_e, self.optimizer.state_dict()['param_groups'][0]['lr'], best_f1
            )
            self.logger.info(info)

            if (e+1) % self.cfg.BASIC.VALID_STEP == 0 :
                save_all(self.logger, all)
            self.scheduler.step()

    def cal_unlabel_img(self):    
        # 修改dic unlabel, 第一阶段的值
        threshold = eval('self.cfg.TRAIN_'+ str(self.step) + '.THRESHOLD')
        self.model.eval()
        unlabel_imgs = []
        with torch.no_grad():   
            for i, j in enumerate(self.un_loader):
                name, _,  x, _ = j
                w,_ = x
                inputs = w.to(device)   # 这里是用增强的弱图分类，如果用原图呢？其实也不用，弱增强只是进行旋转而已
                outputs = self.model(inputs)
                out = torch.softmax(outputs, dim=1)
                max_probs, targets_u = torch.max(out, dim=-1)
                mask = max_probs.ge(threshold).float()
                for i, m in enumerate(mask):
                    if m==1:
                        unlabel_imgs.append([name[i], [targets_u[i], -1]])
            return unlabel_imgs      # 存在可能, 把所有的unlabel都属于了一个类别

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            y_true = None
            y_pred = None
            for i, j in enumerate(self.valide_loader):
                _, _, x, label = j
                inputs = x.to(device)
                if self.step == 1:  # 1代表第一阶段
                    y = label[0].type(torch.LongTensor)
                elif self.step == 2:
                    y = label[1].type(torch.LongTensor)
                elif self.step == 3:
                    y = label[1].type(torch.LongTensor)
                    y -= 2
                else:
                    y = label[1].type(torch.LongTensor)
                outputs = self.model(inputs)
                out = torch.softmax(outputs, dim=1)
                pred = out.max(1, keepdim=False)[1].cpu()
                if i == 0:
                    y_true = y
                    y_pred = pred
                else:
                    y_true = torch.cat((y_true, y), dim=-1)
                    y_pred = torch.cat((y_pred, pred), dim=-1)
            all = evaluating_indicator(y_true, y_pred, self.step)
        return all
