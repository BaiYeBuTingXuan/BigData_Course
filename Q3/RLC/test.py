import torch
import numpy
from torch.utils.data import DataLoader
from Dataset import ModelDataset
from model import LSTMBiClassifier, LSTMResBiClassifier

model_path = './model.pth'
data_path = "./data.npy"
label_path = "./label.npy"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    model = LSTMResBiClassifier().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_data_loader = DataLoader(ModelDataset(data_file=data_path, label_file=label_path),
                                  batch_size=1, shuffle=True, num_workers=1)
    # criterion = torch.nn.CrossEntropyLoss()
    ground_true = []
    guess_list = []
    with torch.no_grad():
        for epoch, batch in enumerate(test_data_loader):
            data, label = batch
            data, label = data.to(device), label.to(device)
            result = model(data)
            # loss = criterion(data, label)
            result = result.cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            # print(result, label)
            

            for (p1, p2), (c1, c2) in zip(result, label):
                if p1 > p2:
                    guess_list.append(1)

                else:
                    guess_list.append(0)

                if c1 > c2:
                    ground_true.append(1)
                else:
                    ground_true.append(0)

            # print(ground_true)
            # print(guess_list)
    TP, TN, FP, FN = 0, 0, 0, 0
    for gt, gl in zip(ground_true, guess_list):
        if gt == 1 and gl == 1:
            TP += 1
        elif gt == 1 and gl == 0:
            FN += 1
        elif gt == 0 and gl == 1:
            FP += 1
        elif gt == 0 and gl == 0:
            TN += 1
    precisionP, precisionN = TP/(TP+FP), TN/(TN+FN)
    recallP,recallN = TP/(TP+FN), TN/(TN+FP)
    try:
        F1P = 2*(precisionP*recallP)/(precisionP+recallP)
    except ZeroDivisionError:
        F1P = 0

    try:
        F1N = 2*(precisionN*recallN)/(precisionN+recallN)
    except ZeroDivisionError:
        F1N = 0
    print('TP=%d,TN=%d,FP=%d,FN=%d' % (TP, TN, FP, FN))
    print('precisionP=%f,recallP=%f' % (precisionP, recallP))
    print('precisionN=%f,recallN=%f' % (precisionN, recallN))
    print('F1P=%f,F1N=%f' % (F1P,F1N))

