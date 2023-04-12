# coding=UTF-8
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import math

import torch.optim as optim
import datetime
from env import *
from test import *


def get_lambda(cur_epoch, num_epoch, coarse_train_ep, fine_train_ep):
    if cur_epoch > num_epoch - fine_train_ep - 1:
        my_lambda = 0
    elif cur_epoch < coarse_train_ep:
        my_lambda = 1
    else:
        my_lambda = 1 - ((cur_epoch + 1 - coarse_train_ep) / (num_epoch - coarse_train_ep - fine_train_ep + 1))

    return my_lambda


def get_scalingfac(num1, num2):
    s1 = int(math.floor(math.log10(num1)))
    s2 = int(math.floor(math.log10(num2)))
    scale = 10 ** (s1 - s2)
    return scale


def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')  # 可以改为sum试试结果

    return loss


class MyMSLEoss(torch.nn.Module):
    # 对数损失，mse基础上加了以二为底的对数
    def __init__(self):
        super(MyMSLEoss, self).__init__()

    def forward(self, output, label):
        return torch.mean(torch.pow(output - torch.log(label + 1) / np.log(2.0), 2))


class MyMAPEoss(torch.nn.Module):
    def __init__(self):
        super(MyMAPEoss, self).__init__()

    def forward(self, output, label):
        # mape = (abs(y_predict - y_test) / y_test).mean()
        # return torch.mean(abs(torch.pow(2, output) - (label + 1)) / (label + 1))
        return torch.mean(abs(output - label) / label)


class MyNLLLoss(nn.Module):
    def forward(self, pre, y_true, weight=1):
        '''
        对于 利好 判断为 利空的，或 利空判断为 利好的，设置惩罚权重大一点
        :param pre: 经过 logsoftmax之后的输出值
        :param y_true: 真实的标签值
        :param weight: 利好利空的惩罚权重
        :return:
        '''
        size = pre.size(0)
        size2 = pre.size(1)
        all_loss = torch.tensor(0.).cuda()
        for i in range(size):
            for j in range(size2):
                if abs(pre[i][j].argmax(-1) - y_true[i][j]) == 1:
                    all_loss.add_(weight * pre[i][j][y_true[i][j]])
                else:
                    all_loss.add_(pre[i][j][y_true[i][j]])

        return all_loss / size  # 不要漏掉 负号


class Focal_Loss(nn.Module):
    def __init__(self, weight, gamma=2):
        super(Focal_Loss,self).__init__()
        self.device = get_device()
        self.gamma = gamma
        self.weight = weight.to(self.device)     # 是tensor数据格式的列表

    def forward(self, preds, labels):
        """
        preds:logist输出值
        labels:标签
        """
        preds = F.softmax(preds, dim=1)
        # print(preds)
        eps = 1e-7

        target = self.one_hot(preds.size(1), labels).to(self.device)
        # print(target)
        # preds =preds.view((preds.size()[0],preds.size()[1],-1)) #B*C*H*W->B*C*(H*W)
        # target=labels.view(y_pred.size()) #B*C*H*W->B*C*(H*W)
        ce = -1 * torch.log(preds+eps) * target
        # print(ce)
        floss = torch.pow((1-preds), self.gamma) * ce
        # print(floss)
        floss = torch.mul(floss, self.weight)
        # print(floss)
        floss = torch.sum(floss, dim=1)
        # print(floss)
        return torch.mean(floss)

    def one_hot(self, num, labels):
        one = torch.zeros((labels.size(0), num))
        one[range(labels.size(0)), labels] = 1
        return one

def computecc(outputs, targets):
    """Computes and stores the average and current value"""
    # print(outputs, targets, outputs.shape, targets.shape)   # torch.Size([31744])
    xBar = targets.mean()
    yBar = outputs.mean()
    # print("train xBar, yBar", xBar, yBar)
    SSR = 0
    varX = 0  # 公式中分子部分
    varY = 0  # 公式中分母部分
    for i in range(0, targets.shape[0]):
        diffXXBar = targets[i] - xBar
        diffYYBar = outputs[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2
    SST = torch.sqrt(varX * varY)
    xxx = SSR / SST
    # print("xxxxxxxxx", xxx)
    return torch.mean(xxx)


def rmse(preds, labels):
    # print(preds, labels, preds.shape, labels.shape)
    loss = (preds - labels) ** 2
    # print("train_rmse_loss", loss)
    loss = torch.mean(loss)
    # print("train_rmse_loss", loss)
    return torch.sqrt(loss)


def mae(preds, labels):
    # print(preds, labels, preds.shape, labels.shape)
    loss = torch.abs(preds - labels)
    # print("train_mae_loss", loss)
    # print("train_mae_torch.mean loss", torch.mean(loss))
    return torch.mean(loss)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(model=None, save_path='', args=None, train_dataloader=None, val_dataloader=None, train_scale_y=None,
          val_scale_y=None, train_group=None, val_group=None):   # 粗粒度指的是分类任务，细粒度指的是回归任务
    device = get_device()
    criterion = nn.CrossEntropyLoss().to(device)
    lf = Focal_Loss(torch.tensor([0.48, 0.1, 0.42])).to(device)

    nll = nn.NLLLoss().cuda()
    mse = nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.decay)

    epoch = args.epoch
    coarse_ep = args.coarse_ep
    fine_ep = args.fine_ep
    early_stop_win = 15
    min_loss = 1e+8

    train_loss_list = []
    stop_improve_count = 0
    for i_epoch in range(epoch):
        loss_total = 0.0    # loss_total用来保存批量误差
        loss_1 = 0.0    # 分类loss
        loss_2 = 0.0    # 回归loss
        loss_3 = 0.0    # TCPloss
        running_acc_1 = 0.0     # 分类准确率

        # my_lambda = get_lambda(i_epoch, epoch, coarse_train_ep=coarse_ep, fine_train_ep=fine_ep)
        # print(i_epoch, my_lambda)
        lambda_conf = args.lambda_conf
        lambda_cls = args.lambda_cls
        lambda_reg = args.lambda_reg
        true_scores = []
        pred_scores = []
        # print(len(train_dataloader))
        model.train()

        for h, solar_data in enumerate(train_dataloader, 0):
            loss = 0.0
            # print("solar_data", solar_data)
            speed, label_reg, label_cls = solar_data
            # speed = torch.as_tensor(speed, dtype=torch.float32).cuda()
            true_scores.extend(label_reg.numpy())
            label_reg = torch.as_tensor(label_reg, dtype=torch.float32).cuda()
            label_cls = torch.as_tensor(label_cls, dtype=torch.long).cuda()
            # print(speed.shape, label_reg.shape, label_cls.shape)

            # predictions
            regression, TCPConfidence, TCPLogit, classification = model(speed)  # 回归结果，分类结果
            # print(logits.shape, logits2.shape)  # torch.Size([32, 8]) torch.Size([32, 8])

            # tree-level label
            cls, glabel, rlabel = train_group.produce_label(label_reg)
            # print(cls, glabel)

            # loss
            pred = F.softmax(TCPLogit, dim=1)
            p_target = torch.gather(input=pred, dim=1, index=cls.unsqueeze(dim=1)).view(
                -1)  # 根据维度dim按照索引列表index从input中选取指定元素
            # print("TCP:", TCPConfidence.squeeze(dim=1))
            # print("P_target:", p_target)
            loss_conf = torch.mean(F.mse_loss(TCPConfidence.squeeze(dim=1).view(-1), p_target) + criterion(TCPLogit, cls))
            loss_cls = criterion(classification, cls)
            # 回归loss
            for i in range(train_group.number_leaf()):
                mask = rlabel[i] >= 0
                # print('mask:', mask)
                # print('logits, logits2:', i, rlabel[i], mask, regression, regression[:, i][mask], rlabel[i][mask])
                if mask.sum() != 0:
                    loss += mse(regression[:, i][mask].reshape(-1, 1).float(),
                                rlabel[i][mask].reshape(-1, 1).float())  # 正确
            # loss_reg = F.mse_loss(regression, label_reg)
            loss_reg = loss
            loss = lambda_conf * loss_conf + lambda_cls * loss_cls + lambda_reg * loss_reg   # loss_reg 权重更大

            loss_1 += loss_cls  # 分类loss
            loss_2 += loss_reg  # 回归loss
            loss_3 += loss_conf  # TCPloss
            loss_total += loss  # loss_total用来保存批量误差
            # print('loss, loss_total: ', loss, loss_total)

            # Back propagate
            optimizer.zero_grad()
            loss.backward()  # 反向传播计算参数的梯度
            optimizer.step()  # 使用优化方法进行梯度更新

            _, pred_c = classification.max(dim=1)  # 分类
            acc_batch_1 = (pred_c == glabel.argmax(0)).sum()   # 分类的准确率
            # print(pred_c, glabel.argmax(0), acc_batch_1)
            running_acc_1 += acc_batch_1.item()  # 分类准确率

            relative_scores = train_group.inference(classification.detach().cpu().numpy(), regression.detach().cpu().numpy())

            pred_scores.extend([i.item() for i in relative_scores])
            # pred_scores.extend(regression.detach().cpu().numpy())

        # analysis on results
        pred_scores = torch.tensor(np.array(pred_scores)).reshape(-1, 1)
        true_scores = torch.tensor(np.array(true_scores).reshape(-1, 1))
        pred_scores = train_scale_y.inverse_transform(pred_scores)
        true_scores = train_scale_y.inverse_transform(true_scores)

        rmse_all = rmse(pred_scores, true_scores)
        mae_all = mae(pred_scores, true_scores)
        train_cc = computecc(pred_scores, true_scores)

        # each epoch
        num_train = len(train_dataloader)
        acc = running_acc_1 / (num_train * args.batch)  # all label: 31744
        # print(running_acc_1, num_train * args.batc)  # 56 * 32 (batch) = 1792    496

        print('epoch({} / {}) Loss:{:.3f}, Loss_cls:{:.3f}, Loss_reg:{:.3f}, Loss_Conf:{:.3f}, Acc_cls:{:.3f}, '
              'RMSE:{:.3f}, MAE:{:.3f}, CC:{:.3f}, lr:{:.6f}'
              .format(i_epoch, epoch, loss_total / num_train, loss_1 / num_train, loss_2 / num_train,
                      loss_3 / num_train, acc, rmse_all, mae_all, train_cc, get_lr(optimizer)), flush=True)

        # use val dataset to judge
        if val_dataloader is not None:

            val_loss, val_loss_cls, val_loss_reg, val_loss_conf, val_acc_cls, val_result, val_rmse, val_mae, val_cc, val_mape = \
                val(model, i_epoch, args, val_dataloader, val_scale_y, train_group)

            # val_loss, val_loss_cls, val_acc_cls, val_loss_reg, val_result, val_rmse, val_mae, val_cc, val_mape = \
                # val(model, i_epoch, args, val_dataloader, val_scale_y)

            print('epoch({} / {}), Loss: {:.3f}, Loss_cls: {:.3f}, Loss_reg: {:.3f}, Loss_Conf:{:.3f}, Acc1: {:.3f},'
                  'RMSE:{:.3f}, MAE:{:.3f}, CC:{:.3f}, MAPE:{:.3f}'.
                  format(i_epoch, epoch, val_loss, val_loss_cls, val_loss_reg, val_loss_conf, val_acc_cls, val_rmse,
                         val_mae, val_cc, val_mape))

            if val_loss < min_loss:
                # torch.save(model.state_dict(), save_path)
                torch.save(model, save_path)
                print("save best model at ", save_path)

                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1

            if stop_improve_count >= early_stop_win:
                break

        else:
            if loss_reg < min_loss:
                torch.save(model.state_dict(), save_path)
                min_loss = loss_reg

        # scheduler.step(val_loss)

    return loss
