import pickle

import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from data import DataGenerator

import argparse
# https://blog.csdn.net/zhangyuexiang123/article/details/107952241
'''
with open('/mnt/syanru/Multi-Task/result_save/result_testfushion_gate.pkl', 'rb') as fw2:
    predict_file = pickle.load(fw2)
with open('/mnt/syanru/Weibo_data/deephawkes_cmp_data/G_G_classifylabel_test_timewindow{}.pkl'.format(5),
          'rb') as fw2:
    true_file = pickle.load(fw2)
        
ipt = {}
index = 0
for i in predict_file:
    tmp = list(i)
    for j in tmp:
        ipt[index] = int(j)
        index += 1

y_true = []
y_pred = []
for k, v in ipt.items():
    l = true_file[k]

    y_pred.append(v)
    y_true.append(l)

print(y_pred, y_true)
res = confusion_matrix(y_true, y_pred)
print(res)
'''

def getLabelData(predict_file, true_label):
    '''
    模型的预测生成相应的label文件，以及真实类标文件，根据文件读取并加载所有label
    1、参数说明：
        file_dir：加载的文件地址。
        文件内数据格式：每行包含两列，第一列为编号1,2，...，第二列为预测或实际的类标签名称。两列以空格为分隔符。
        需要生成两个文件，一个是预测，一个是实际类标，必须保证一一对应，个数一致
    2、返回值：
        返回文件中每一行的label列表，例如['true','false','false',...,'true']

    labels = []
    with open(file_dir, 'r', encoding="utf-8") as f:
        for i in f.readlines():
            labels.append(i.strip().split(' ')[1])
    return labels
    '''
    with open(predict_file, 'rb') as fw2:
        predict_file = pickle.load(fw2)

    y_true = true_label.tolist()
    y_true = list(map(int, y_true))

    ipt = {}
    index = 0
    for i in predict_file:
        tmp = list(i)
        for j in tmp:
            ipt[index] = int(j)
            index += 1
    y_pred = []
    for k, v in ipt.items():
        y_pred.append(v)

    y_true = y_true[0: len(y_pred)]  # test的时候 drop_last=True
    # print(len(y_pred), len(y_true))  # 8704， 8704
    return y_pred, y_true

def getLabel2idx(labels):
    '''
    获取所有类标
    返回值：label2idx字典，key表示类名称，value表示编号0,1,2...
    '''
    label2idx = dict()
    for i in labels:
        if i not in label2idx:
            label2idx[i] = len(label2idx)
    return label2idx


def buildConfusionMatrix(predict_file, true_label):
    '''
    针对实际类标和预测类标，生成对应的矩阵。
    矩阵横坐标表示实际的类标，纵坐标表示预测的类标
    矩阵的元素(m1,m2)表示类标m1被预测为m2的个数。
    所有元素的数字的和即为测试集样本数，对角线元素和为被预测正确的个数，其余则为预测错误。
    返回值：返回这个矩阵numpy
    '''
    predict_labels, true_labels = getLabelData(predict_file, true_label)
    label2idx = getLabel2idx(true_labels)
    confMatrix = np.zeros([len(label2idx), len(label2idx)], dtype=np.int32)
    for i in range(len(true_labels)):
        true_labels_idx = label2idx[true_labels[i]]
        predict_labels_idx = label2idx[predict_labels[i]]
        confMatrix[true_labels_idx][predict_labels_idx] += 1
    return confMatrix, label2idx


def calculate_all_prediction(confMatrix):
    '''
    计算总精度：对角线上所有值除以总数
    '''
    total_sum = confMatrix.sum()
    correct_sum = (np.diag(confMatrix)).sum()
    prediction = round(100 * float(correct_sum) / float(total_sum), 2)
    return prediction

def calculate_all_prediction_sklearn(predict_labels, true_labels, labels):
    precision = precision_score(predict_labels, true_labels, labels=labels, average="micro")  # average 指定为 micro
    precision = round(100 * precision, 2)
    return precision

def calculate_all_recall(predict_labels, true_labels, labels):
    recall = recall_score(predict_labels, true_labels, labels=labels, average="micro")
    recall = round(100 * recall, 2)
    return recall

def calculate_all_F1(predict_labels, true_labels, labels):
    f1 = f1_score(predict_labels, true_labels, labels=labels, average="micro")
    f1 = round(100 * f1, 2)
    return f1

def calculate_label_prediction(confMatrix, labelidx):
    '''
    计算某一个类标预测精度：该类被预测正确的数除以该类的总数
    '''
    label_total_sum = confMatrix.sum(axis=0)[labelidx]
    label_correct_sum = confMatrix[labelidx][labelidx]
    prediction = 0
    if label_total_sum != 0:
        prediction = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
    return prediction


def calculate_label_recall(confMatrix, labelidx):
    '''
    计算某一个类标的召回率：
    '''
    label_total_sum = confMatrix.sum(axis=1)[labelidx]
    label_correct_sum = confMatrix[labelidx][labelidx]
    recall = 0
    if label_total_sum != 0:
        recall = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
    return recall


def calculate_f1(prediction, recall):
    if (prediction + recall) == 0:
        return 0
    return round(2 * prediction * recall / (prediction + recall), 2)


def confusion_matrix(predict_file, true_label):
    '''
    该为主函数，可将该函数导入自己项目模块中
    打印精度、召回率、F1值的格式可自行设计
    '''
    # 读取文件并转化为混淆矩阵,并返回label2idx
    confMatrix, label2idx = buildConfusionMatrix(predict_file, true_label)
    total_sum = confMatrix.sum()
    predict_labels, true_labels = getLabelData(predict_file, true_label)
    labels = [0, 1, 2]

    all_prediction = calculate_all_prediction(confMatrix)
    all_prediction_sklearn = calculate_all_prediction_sklearn(predict_labels, true_labels, labels)
    all_recall = calculate_all_recall(predict_labels, true_labels, labels)
    all_F1 = calculate_all_F1(predict_labels, true_labels, labels)

    label_prediction = []
    label_recall = []
    print('total_sum=', total_sum, ',label_num=', len(label2idx), '\n')
    for i in label2idx:
        print('  ', i)
    print('  ')
    for i in label2idx:
        print(i, end=' ')
        label_prediction.append(calculate_label_prediction(confMatrix, label2idx[i]))
        label_recall.append(calculate_label_recall(confMatrix, label2idx[i]))
        for j in label2idx:
            labelidx_i = label2idx[i]
            label2idx_j = label2idx[j]
            print('  ', confMatrix[labelidx_i][label2idx_j], end=' ')
        print('\n')

    print('MICRO-prediction(accuracy)=', all_prediction, '%,recall=', all_recall, '%,f1=', all_F1)  # micro 模式下的acc, recall, f1结果相同
    print('individual result\n')
    for ei, i in enumerate(label2idx):
        print(ei, '\t', i, '\t', 'prediction=', label_prediction[ei], '%,\trecall=', label_recall[ei], '%,\tf1=',
              calculate_f1(label_prediction[ei], label_recall[ei]))
    p = round(np.array(label_prediction).sum() / len(label_prediction), 2)
    r = round(np.array(label_recall).sum() / len(label_prediction), 2)
    print('MACRO-averaged:\nprediction=', p, '%,recall=', r, '%,f1=', calculate_f1(p, r))

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

    args = parser.parse_args()

    test_dataset = DataGenerator(args, 'test', args.interval)
    true_label = test_dataset.label_cls
    print(type(true_label), true_label)

    confusion_matrix('/mnt/syanru/Multi-Task/5.1-Multi-task-solar-ar-shapelets-clf-reg/result_save/classification-regression/lr_0.0001/result_fushion_gate_test.pkl',
                     true_label)
# '/mnt/syanru/Multi-Task/Multi-task-solar/data/Series_data_classification/2017.csv',
    # confusion_matrix('/mnt/syanru/Multi-Task/3-Multi-Task-Lambda/result_save/test/result_fushion_gate_test.pkl',
    #                  '/mnt/syanru/Weibo_data/deephawkes_cmp_data/G_G_classifylabel_test_timewindow{}.pkl'.format(5))
    #
