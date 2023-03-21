import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import random
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import LSTMBiClassifier, TransformerBiClassifier, LSTMResBiClassifier,weights_init
from Dataset import ModelDataset
import numpy as np

random.seed(datetime.now())
torch.manual_seed(666)
torch.cuda.manual_seed(666)
torch.set_num_threads(16)


def eval_model(model, criterion, dataloader):
    model.eval()
    test_loss_list = []
    for _, batch in enumerate(dataloader):
        r, label = batch
        r, label = r.to(device), label.to(device)
        c = model(r)
        loss = criterion(c.float(), label.float())
        test_loss_list.append(loss.cpu().detach().numpy())
    avg_loss = np.mean(test_loss_list)
    model.train()
    return avg_loss


def get_filename(path, prename, filetype):
    """
    根据路径、前缀名、时间生成文件名 文件类型为filetype
    input:
    path：文件路径 后缀要包含\\
    prename:文件前缀名
    filetype：文件类型 如 "txt"
    Output: 文件名
    例如：path=".\\train\\" ,prename = example, filetype = ".txt"
        当前时间 2022年5月14日12点21分
        Out:= .\\train\\example_20220514_1221.txt
    """
    now = datetime.now()
    time = now.strftime("_%Y%m%d_%H%M")
    name = path + prename + time +filetype
    return name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./data.npy", help='path of the data')
    parser.add_argument('--label_path', type=str, default="./label.npy", help='path of the label')
    parser.add_argument('--n_cpu', type=int, default=20, help='number of CPU threads to use during batches generating')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--batch_size', type=float, default=8, help='batch size')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
    parser.add_argument('--train_time', type=int, default=1000, help='total training time')
    parser.add_argument('--checkpoints_interval', type=int, default=1000, help='interval between model checkpoints')
    parser.add_argument('--clip_value', type=float, default=1.0, help='Clip value if gradient vanishing')
    parser.add_argument('--test_split', type=float, default=0.1, help='Spilt rate to divide test dataset')
    parser.add_argument('--epoches', type=int, default=10000, help='training epoches')
    parser.add_argument('--model', type=str, default='LSTMResBiClassifier' ,help='model to train')
    opt = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    path = get_filename('./', '/result'+opt.model, '/')
    print(path)
    
    os.makedirs(path+'/model/', exist_ok=True)
    os.makedirs(path+'/log/', exist_ok=True)
    logger = SummaryWriter(log_dir=path+'/log/')

    if opt.model == 'LSTMBiClassifier':
        model = LSTMBiClassifier().to(device)
    elif opt.model == 'TransformerBiClassifier':
        model = TransformerBiClassifier().to(device)
    elif opt.model == 'LSTMResBiClassifier':
        model = LSTMResBiClassifier().to(device)
    else:
        print('Unkown model')
    model.apply(weights_init)
    model.to(device)

    full_dataset = ModelDataset(data_file=opt.data_path, label_file=opt.label_path)
    test_size = int(opt.test_split*len(full_dataset))
    train_size = len(full_dataset) - test_size
    print("train_size:%d,test_size:%d" % (train_size ,test_size))
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset,
                              batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

    test_loader = DataLoader(test_dataset,
                             batch_size=1, shuffle=True, num_workers=1)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    total_steps = 0
    print("Start Training ...")
    for epoch in range(opt.epoches):
        # --- train ----#
        for _, batch in enumerate(train_loader):
            model.train()
            
            r, label = batch
            r, label = r.type(torch.FloatTensor).to(device), label.to(device)
            c = model(r)
            loss = criterion(c.float(), label.float())
            logger.add_scalar('train_loss', loss, total_steps)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=opt.clip_value)
            optimizer.step()
        
            # --- test ----#
            if total_steps % opt.checkpoints_interval == 0:
                test_loss = eval_model(model, criterion, dataloader=test_loader)
                print('test_loss = %f, total_step = %d'% (test_loss,total_steps))
                logger.add_scalar('test_loss', loss, total_steps)
                torch.save(model.state_dict(), path+'//model//'+opt.model+'_%d.pth' % epoch)
            total_steps += 1
