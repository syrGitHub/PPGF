import pickle
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, r2_score
from data import DataGenerator
from Group_helper import Group_helper

import argparse
import re

# https://blog.csdn.net/zhangyuexiang123/article/details/107952241

def getLabelData(predict_file):
    '''
    模型的预测生成相应的label文件，以及真实类标文件，根据文件读取并加载所有label
    1、参数说明：
        file_dir：加载的文件地址。
        文件内数据格式：每行包含两列，第一列为编号1,2，...，第二列为预测或实际的类标签名称。两列以空格为分隔符。
        需要生成两个文件，一个是预测，一个是实际类标，必须保证一一对应，个数一致
    2、返回值：
        返回文件中每一行的label列表，例如['true','false','false',...,'true']

    '''
    with open(predict_file, 'r') as in_file:
        txt = in_file.readlines()  # read()返回字符串 readline()读一行，返回字符串  readlines()
        # 全部读，返回字符串列表，含有\n
        for i in range(len(txt)):
            txt[i] = list(map(float, filter(None, re.split('[\t \n]', txt[i].strip()))))  # 这一行和后面注释的两行等价
        # print(txt)

    return txt

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


def evaluation(y_predict, y_test):
    mape = torch.mean(abs(y_predict - y_test) / y_test)
    r_2 = r2_score(y_test, y_predict)
    return mape, r_2


def regression_error(predict_file, true_file, group, args, scale):
    '''
    该为主函数，可将该函数导入自己项目模块中
    打印精度、召回率、F1值的格式可自行设计
    '''
    predict = getLabelData(predict_file)
    true = getLabelData(true_file)
    # print('Prediction')
    # print(predict)
    # print('True')
    # print(true)

    true = true[0: len(predict)]  # test的时候 drop_last=True
    # print(len(y_pred), len(y_true))  # 8704， 8704
    predict_list = []
    true_list = []
    for j in range(args.num_groups):
        region_left = scale.inverse_transform(group.Group[j][0]).float()
        region_right = scale.inverse_transform(group.Group[j][1]).float()
        print(region_left, region_right)
        leaf_pred = []
        leaf_true = []
        for i in range(len(true)):
            if true[i][0] < region_right and true[i][0] >= region_left:
                leaf_pred.append(predict[i][0])
                leaf_true.append(true[i][0])

        leaf_pred = torch.tensor(leaf_pred)
        leaf_true = torch.tensor(leaf_true)

        rmse_result = rmse(leaf_pred, leaf_true)
        mae_result = mae(leaf_pred, leaf_true)
        cc_result = computecc(leaf_pred, leaf_true)
        mape, r_2 = evaluation(leaf_pred, leaf_true)

        print("RMSE, MAE, CCOR, mape, r_2:", rmse_result.item(), mae_result.item(), cc_result.item(), mape.item(), r_2.item())


def build_group(dataset_train, args, scale_y):
    delta_list = dataset_train.tolist()
    print(dataset_train.shape, type(delta_list))
    group = Group_helper(delta_list, args.num_groups, scale_y, Symmetrical=False) # , Max=args.score_range, Min=0)
    return group

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', help='batch size', type=int, default=256)
    parser.add_argument('--epoch', help='train epoch', type=int, default=25)
    parser.add_argument('--coarse_ep', help='classification epoch', type=int, default=10)
    parser.add_argument('--fine_ep', help='regression epoch', type=int, default=10)
    parser.add_argument('--save_path_pattern', help='save path pattern', type=str, default='lr_0.001')
    parser.add_argument('--load_model_path', help='trained model path', type=str, default='')
    parser.add_argument('--device', help='cuda / cpu', type=str, default='cuda')
    parser.add_argument('--random_seed', help='random seed', type=int, default=5)
    parser.add_argument('--report', help='best / val', type=str, default='best')
    parser.add_argument('--interval', help='EUV数据采样频率（间隔几张采一次）', type=int, default=1)
    parser.add_argument('--input_length', help='input_length', type=int, default=24)  # 输入数据长度
    parser.add_argument('--predict_length', help='predict length', type=int, default=23)  # 输出数据长度
    parser.add_argument('--norm', help='normalize type', type=int, default=3)  # 正则化方式

    parser.add_argument('--input_dim', type=int, default=1, help='the dimension of the input data')  # 卷积核大小
    parser.add_argument('--num_classes', type=int, default=5, help='the number of classes')
    parser.add_argument('--hidden_dim', type=int, default=32, help='the dimension of the hidden layer')
    parser.add_argument('--decay', help='decay', type=float, default=0.0001)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--muti1', type=int, default=5, help='multi scale 1')
    parser.add_argument('--muti2', type=int, default=10, help='multi scale 2')
    parser.add_argument('--muti3', type=int, default=15, help='multi scale 3')
    parser.add_argument("--num_groups", default=4, type=int, help="depth of regression tree")

    args = parser.parse_args()

    test_dataset = DataGenerator(args, 'test', args.interval)
    true_label = test_dataset.label_cls
    print(type(true_label), true_label)

    print('train:')
    group = build_group(test_dataset.label_reg, args, test_dataset.scale_y)

    regression_error(
        '/mnt/syanru/Multi-Task/b-Multi-task-solar-regression-classification_output_all_regression_TCP/result_save/classification-regression/epoch_60_bs_32_lambda_4/num_classification_2/lr_0.0001/pred.txt',
        '/mnt/syanru/Multi-Task/b-Multi-task-solar-regression-classification_output_all_regression_TCP/result_save/classification-regression/epoch_60_bs_32_lambda_4/num_classification_2/lr_0.0001/true.txt',
        group, args, test_dataset.scale_y)
# '/mnt/syanru/Multi-Task/Multi-task-solar/data/Series_data_classification/2017.csv',
# confusion_matrix('/mnt/syanru/Multi-Task/3-Multi-Task-Lambda/result_save/test/result_fushion_gate_test.pkl',
#                  '/mnt/syanru/Weibo_data/deephawkes_cmp_data/G_G_classifylabel_test_timewindow{}.pkl'.format(5))
#
