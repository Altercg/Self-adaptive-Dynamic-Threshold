import torch
import csv
from dataset.load_data import load_test
from train.valid import validate
from utils.utils import get_model, save_all
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(cfg, logger, model_save_file):
    # 导入数据
    test_loader = load_test(cfg.DATASET.TEST_IMG_ROOT, cfg.DATASET.TEST_BATCHSIZE)
    logger.info("Finish, total number:{}".format(len(test_loader.dataset)))
    # 导入模型
    logger.info("-"*30+"Start test"+"-"*30)
    # ----- BEGIN MODEL BUILDER -----
    model = get_model(cfg.BASIC.CLASS_NUM, cfg.TRAIN.NET_TYPE)
    model.load_state_dict(torch.load(model_save_file +'model.pt')['model_state_dict'])
    # ----- END MODEL BUILDER -----
    model.to(device)
    dataname = cfg.DATASET.TEST_IMG_ROOT.split('/')[-3]
    if dataname == 'DFU':
        test_dfu(model, test_loader, model_save_file, ['image', 'none', 'infection', 'ischaemia', 'both'])
    elif dataname == 'ISIC':
        all, auc =  validate(model, test_loader, ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'])
        save_all(logger, all)
        logger.info("auc:{}".format(auc))
    else:
        ('Error!')
    logger.info("Finish!")

def test_dfu(model, test_loader, model_save_file, target_name):
    model.eval()
    # 开始测试
    with torch.no_grad():
        result_filename =  model_save_file +'result.csv'
        # 输出结果的格式为cvs
        with open(result_filename, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(target_name)
            number = 0
            for img in test_loader:
                x, _ = img
                inputs = x.to(device)
                outputs = model(inputs)
                outputs = torch.softmax(outputs, dim=1).cpu().numpy()
                np.set_printoptions(
                    precision=15, floatmode='fixed', suppress=True)
                outs = outputs.astype("float32").tolist()
                for o in outs:
                    name =test_loader.dataset.imgs[number][0].split('/')[-1]
                    o = ['{:.15f}'.format(round(i, 15)) for i in o]
                    o.insert(0, name)
                    writer.writerow(o)
                    number+=1
        csvfile.close()

    
