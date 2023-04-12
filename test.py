import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import math
import config_GT
import torch.optim as optim
import datetime
from env import *
import pickle


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
        # print(output.shape, label.shape)    # torch.Size([64, 1]) torch.Size([64, 1])
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


def computecc(outputs, targets):
    """Computes and stores the average and current value"""
    # print("***************train_computecc", targets.shape, outputs.shape)  # torch.Size([64, 1, 4]) torch.Size([64, 1, 4])
    # print("train_computecc, outputs, targets", outputs, targets)
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
    loss = (preds - labels) ** 2
    #print("train_rmse_loss", loss)
    loss = torch.mean(loss)
    # print("train_rmse_loss", loss)
    return torch.sqrt(loss)


def mae(preds, labels):
    loss = torch.abs(preds - labels)
    # print("train_mae_loss", loss)
    # print("train_mae_torch.mean loss", torch.mean(loss))
    return torch.mean(loss)


def val(model, i_epoch, args, eval_loader, val_scale=None, group=None):
    epoch = args.epoch
    coarse_ep = args.coarse_ep
    fine_ep = args.fine_ep
    # print(epoch, coarse_ep, fine_ep)
    # my_lambda = get_lambda(i_epoch, epoch, coarse_train_ep=coarse_ep, fine_train_ep=fine_ep)
    lambda_conf = args.lambda_conf
    lambda_cls = args.lambda_cls
    lambda_reg = args.lambda_reg
    with torch.no_grad():
        loss_total = 0.0    # loss_total用来保存批量误差
        loss_1 = 0.0    # 分类loss
        loss_2 = 0.0    # 回归loss
        loss_3 = 0.0
        eval_acc_1 = 0.0
        model.eval()

        true_scores = []
        pred_scores = []

        device = get_device()
        criterion = nn.CrossEntropyLoss().to(device)
        nll = nn.NLLLoss().to(device)
        mse = nn.MSELoss().cuda()
        lossmape = MyMAPEoss().to(device)
        for h, solar_wind in enumerate(eval_loader, 0):
            loss = 0.0
            loss_reg = 0.0
            speed, label_reg, label_cls = solar_wind    # [item.to(device).float() for item in [speed, label_reg, label_cls]]
            true_scores.extend(label_reg.numpy())

            label_reg = torch.as_tensor(label_reg, dtype=torch.float32).cuda()
            label_cls = torch.as_tensor(label_cls, dtype=torch.long).cuda()

            # tree-level label
            cls, glabel, rlabel = group.produce_label(label_reg)
            # predictions
            regression, TCPConfidence, TCPLogit, classification = model(speed)  # 回归结果，分类结果
            # print(logits, allatt)
            # loss
            pred = F.softmax(TCPLogit, dim=1)
            p_target = torch.gather(input=pred, dim=1, index=cls.unsqueeze(dim=1)).view(
                -1)  # 根据维度dim按照索引列表index从input中选取指定元素
            # print(TCPConfidence.shape, p_target.shape, TCPLogit.shape, cls.shape, cls, glabel)
            loss_conf = torch.mean(F.mse_loss(TCPConfidence.squeeze(dim=1).view(-1), p_target) + criterion(TCPLogit, cls))
            loss_cls = criterion(classification, cls)

            # 回归loss
            for i in range(group.number_leaf()):
                mask = rlabel[i] >= 0
                if mask.sum() != 0:
                    loss_reg += F.mse_loss(regression[:, i][mask].reshape(-1, 1).float(),
                                           rlabel[i][mask].reshape(-1, 1).float())

            # loss_reg = F.mse_loss(regression, label_reg)

            loss = lambda_conf * loss_conf + lambda_cls * loss_cls + lambda_reg * loss_reg   # loss_reg 权重更大
            # loss = loss_reg + loss_cls
            loss_1 += loss_cls  # 分类loss
            loss_2 += loss_reg  # 回归loss
            loss_3 += loss_conf
            loss_total += loss  # loss_total用来保存批量误差

            _, pred_c = classification.max(dim=1)
            acc_batch_1 = (pred_c == glabel.argmax(0)).sum()   # 分类的准确率
            eval_acc_1 += acc_batch_1.item()

            # evaluate result of training phase
            relative_scores = group.inference(classification.detach().cpu().numpy(), regression.detach().cpu().numpy())

            pred_scores.extend([i.item() for i in relative_scores])

    # analysis on results
    pred_scores = torch.tensor(np.array(pred_scores)).reshape(-1, 1)
    true_scores = torch.tensor(np.array(true_scores).reshape(-1, 1))
    pred_scores = val_scale.inverse_transform(pred_scores)
    true_scores = val_scale.inverse_transform(true_scores)

    val_rmse = rmse(pred_scores, true_scores)
    val_mae = mae(pred_scores, true_scores)
    val_cc = computecc(pred_scores, true_scores)
    val_mape = lossmape(pred_scores, true_scores)   # 回归loss
    test_predicted_list = pred_scores.tolist()
    test_ground_list = true_scores.tolist()

    num_eval = len(eval_loader)   # 106
    # print(len(eval_loader))  # 56 * 32 (batch) = 1792
    acc = eval_acc_1 / (num_eval * args.batch)

    return loss_total / num_eval, loss_1 / num_eval, loss_2 / num_eval, loss_3 / num_eval, acc, \
           [test_predicted_list, test_ground_list], val_rmse, val_mae, val_cc, val_mape


def test(model, args, test_loader, test_scale=None, path='', group=None):
    epoch = args.epoch
    with torch.no_grad():
        loss_total = 0.0    # loss_total用来保存批量误差
        test_acc_1 = 0.0    # 分类
        model.eval()

        testcls_loss = []
        cls_label = []
        # cls_label = torch.tensor(cls_label)
        true_scores = []
        pred_scores = []

        device = get_device()
        lossmape = MyMAPEoss().to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        nll = nn.NLLLoss().to(device)
        mse = nn.MSELoss().cuda()

        for h, solar_wind in enumerate(test_loader, 0):
            loss = 0.0
            loss_reg = 0.0
            speed, label_reg, label_cls = solar_wind # [item.to(device).float() for item in [speed, label_reg, label_cls]]
            true_scores.extend(label_reg.numpy())

            # label, ipt15, label_plus, label_cls = weibodata
            label_reg = torch.as_tensor(label_reg, dtype=torch.float32).cuda()
            label_cls = torch.as_tensor(label_cls, dtype=torch.long).cuda()

            # tree-level label
            cls, glabel, rlabel = group.produce_label(label_reg)
            # predictions
            regression, TCPConfidence, TCPLogit, classification = model(speed)  # 回归结果，分类结果

            _, pred_c = classification.max(dim=1)
            acc_batch_1 = (pred_c == glabel.argmax(0)).sum()   # 分类的准确率
            # print(type(glabel.argmax(0)), cls_label)
            if len(cls_label):
                cls_label = torch.cat([cls_label, glabel.argmax(0)], -1)
            else:
                cls_label = glabel.argmax(0)
            # cls_label.append(glabel.argmax(0).tolist())
            # print('cls_label:', cls_label)
            test_acc_1 += acc_batch_1.item()

            testcls_loss.append(pred_c)
            # evaluate result of training phase
            relative_scores = group.inference(classification.detach().cpu().numpy(), regression.detach().cpu().numpy())

            pred_scores.extend([i.item() for i in relative_scores])
            # pred_scores.extend(regression.detach().cpu().numpy())
            '''
            loss_GT = criterion(logits, label_tensor)  # 回归loss
            loss_reg.append(float(loss_GT))

            prediction = torch.max(F.softmax(allatt, dim=1), 1)[1]  # torch.max(,1)因为经过softmax之后的值都小于等于1, [1]指输出index
            # print(prediction)
            testcls_loss.append(prediction)
            '''
    # analysis on results
    pred_scores = torch.tensor(np.array(pred_scores)).reshape(-1, 1)
    true_scores = torch.tensor(np.array(true_scores).reshape(-1, 1))
    pred_scores = test_scale.inverse_transform(pred_scores)
    true_scores = test_scale.inverse_transform(true_scores)

    test_rmse = rmse(pred_scores, true_scores)
    test_mae = mae(pred_scores, true_scores)
    test_cc = computecc(pred_scores, true_scores)
    loss_mape = lossmape(pred_scores, true_scores)   # 回归loss
    test_reg_ground_list = true_scores.tolist()
    test_reg_predicted_list = pred_scores.tolist()

    with open(path, 'wb') as f1:
        pickle.dump(testcls_loss, f1, pickle.HIGHEST_PROTOCOL)  # 将对象obj保存到文件file中去

    num_test = len(test_loader)
    acc = test_acc_1 / (num_test * args.batch)

    print('Test...Finish {} epoch, Acc1: {:.3f}, Rmse: {:.3f}, Mae: {:.3f}, CC: {:.3f}, Mape: {:.3f}'.format(
        epoch + 1, acc, test_rmse, test_mae, test_cc, loss_mape))

    return test_reg_ground_list, test_reg_predicted_list, test_rmse, test_mae, test_cc, loss_mape, acc, cls_label
    # return [test_predicted_list, test_ground_list], val_rmse, val_mae, val_cc



    '''
            test_result.setdefault(1, []).append(logits)
            test_result.setdefault(2, []).append(label_tensor)

            loss_GT = criterion(logits, label_tensor)   # 回归loss
            loss_mape = lossmape(logits, label_tensor)   # 回归loss
            test_rmse = rmse(logits, label_tensor)
            test_mae = mae(logits, label_tensor)
            test_corr = computecc(logits, label_tensor)
            loss_cls = criterion_ipt(allatt, label_cls.cuda())  # 分类loss
            prediction = torch.max(F.softmax(allatt), 1)[1]
            # print(prediction, loss_cls)

            tmp1.append([logits, label_tensor])
            testcls_loss.append(prediction)
            test_loss.append(float(loss_GT))
            test_loss_mape.append(float(loss_mape))
            att.setdefault(0, []).append(allatt)

            return1.setdefault(0, []).append(ipt5)
            return1.setdefault(1, []).append(ipt10)
            return1.setdefault(2, []).append(ipt15)

    test_lossvalue = np.mean(test_loss)
    test_mapelossvalue = np.mean(test_loss_mape)
    if eval_lossvalue < best_evalloss:
        test_pt = tmp1
        bestcls = testcls_loss
        finalatt = att
        finalre = return1
        best_evalloss = eval_lossvalue
        best_testloss = test_lossvalue
        best_mapeloss = test_mapelossvalue
        patience = maxtry
        with open('/mnt/syanru/Multi-Task/result_save/result_testfushion_gate.pkl',
                  'wb') as f1:
            pickle.dump(bestcls, f1, pickle.HIGHEST_PROTOCOL)   # 将对象obj保存到文件file中去

    patience -= 1
    if not patience:
        break
    print("epoch:", epoch)
    print('eval_loss:', eval_lossvalue, 'test_loss', test_lossvalue, 'test_mape', test_mapelossvalue)
    print("best_evalloss:", best_evalloss, "best_testloss:", best_testloss, 'best_mapeloss', best_mapeloss,
          'test_rmse:', test_rmse, 'test_mae:', test_mae, 'test_corr:', test_corr)

    print(epochloss / config_GT.train_lteration)
    return best_testloss, best_mapeloss, test_rmse, test_mae, test_corr
    #
    # with open('result/test{}_multitask_gate.pkl'.format(3), 'wb') as f13:
    #    pickle.dump(test_pt, f13, pickle.HIGHEST_PROTOCOL)
    '''