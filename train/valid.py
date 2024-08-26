import torch
from utils import analysis, evaluating_indicator
from sklearn import metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate(model, val_loader, target_name):
    model.eval()
    with torch.no_grad():
        y_true = None
        y_pred = None
        y_pred2 = None
        for i, j in enumerate(val_loader):
            _, x, y = j
            inputs = x.to(device)
            outputs = model(inputs)
            out = torch.softmax(outputs, dim=1).cpu()
            pred = out.max(1, keepdim=False)[1]
            if i == 0:
                y_true = y
                y_pred = pred
                y_pred2 = out
            else:
                y_true = torch.cat((y_true, y), dim=-1)
                y_pred = torch.cat((y_pred, pred), dim=-1)
                y_pred2 = torch.cat((y_pred2, out), dim=0)
        labels = torch.unique(y_true).numpy().tolist()
        all = evaluating_indicator(y_true, y_pred, labels, target_name)
        auc = metrics.roc_auc_score(y_true.numpy(), y_pred2.numpy(), average='macro', multi_class='ovr')
        return all, auc
