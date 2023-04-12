# -*- coding: utf-8 -*-
import os
import torch
import pandas as pd
import numpy as np
import argparse


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class MinMaxScaler():
    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return (data * (self.max - self.min)) + self.min


def normalized(x, args):
    normalize = args.norm
    print("preprocess_m,n:", x.shape)  # torch.Size([1826, 6, 24]) torch.Size([1826, 1])   torch.Size([43799, 24]) torch.Size([43799, 1])
    # normlized by the maximum value of each row(sensor).
    if normalize == 2:  # 最大最小归一化
        # dataset = np.array(dataset)
        scale = []
        scaler1 = MinMaxScaler(min=x[:, 0].min(), max=x[:,  0].max())
        # scaler2 = MinMaxScaler(min=x[:, :, 1].min(), max=x[:, :, 1].max())
        # scaler3 = MinMaxScaler(min=x[:, :, 2].min(), max=x[:, :, 2].max())
        # scaler4 = MinMaxScaler(min=x[:, :, 3].min(), max=x[:, :, 3].max())
        # scaler5 = MinMaxScaler(min=x[:, :, 4].min(), max=x[:, :, 4].max())
        # scaler6 = MinMaxScaler(min=x[:, :, 5].min(), max=x[:, :, 5].max())
        # scaler_y = MinMaxScaler(min=y.min(), max=y.max())
        x[:, 0] = scaler1.transform(x[:, 0])
        # x[:, :, 1] = scaler2.transform(x[:, :, 1])
        # x[:, :, 2] = scaler3.transform(x[:, :, 2])
        # x[:, :, 3] = scaler4.transform(x[:, :, 3])
        # x[:, :, 4] = scaler5.transform(x[:, :, 4])
        # x[:, :, 5] = scaler6.transform(x[:, :, 5])
        # y = scaler_y.transform(y)
        print("TimeDataset_tansform_x.shape, x", x, x.shape)  # torch.Size([43799, 24]) torch.Size([43799, 1])
        scale.append(scaler1)
        # scale.append(scaler2)
        # scale.append(scaler3)
        # scale.append(scaler4)
        # scale.append(scaler5)
        # scale.append(scaler6)

    if normalize == 3:  # 标准差方差归一化
        # dataset = np.array(dataset)
        scale = []
        scaler1 = StandardScaler(mean=x[:, 0].mean(), std=x[:, 0].std())
        # scaler2 = StandardScaler(mean=x[:, :, 1].mean(), std=x[:, :, 1].std())
        # scaler3 = StandardScaler(mean=x[:, :, 2].mean(), std=x[:, :, 2].std())
        # scaler4 = StandardScaler(mean=x[:, :, 3].mean(), std=x[:, :, 3].std())
        # scaler5 = StandardScaler(mean=x[:, :, 4].mean(), std=x[:, :, 4].std())
        # scaler6 = StandardScaler(mean=x[:, :, 5].mean(), std=x[:, :, 5].std())
        # scaler_y = StandardScaler(mean=y.mean(), std=y.std())
        x[:, 0] = scaler1.transform(x[:, 0])
        # x[:, :, 1] = scaler2.transform(x[:, :, 1])
        # x[:, :, 2] = scaler3.transform(x[:, :, 2])
        # x[:, :, 3] = scaler4.transform(x[:, :, 3])
        # x[:, :, 4] = scaler5.transform(x[:, :, 4])
        # x[:, :, 5] = scaler6.transform(x[:, :, 5])
        # y = scaler_y.transform(y)
        print("TimeDataset_tansform_x.shape, x", x.shape)
        scale.append(scaler1)
        # scale.append(scaler2)
        # scale.append(scaler3)
        # scale.append(scaler4)
        # scale.append(scaler5)
        # scale.append(scaler6)
    return x, scaler1


class DataGenerator():
    def __init__(self, args, phase_gen, interval):
        self.args = args
        self.phase_gen = phase_gen
        self.interval = interval
        print(self.phase_gen, self.interval)

        self.speed, self.label_reg, self.label_cls, self.scale_y = self.generator()

    def __len__(self):
        return len(self.label_reg)

    def generator(self):
        print("aaaaaaaaaa")
        speed, label_reg, label_cls, scale_y = self.construct_data()

        print(self.phase_gen, speed.shape, label_reg.shape, label_cls.shape)  # torch.Size([1824, 6, 24]) torch.Size([1825, 1, 256, 256]) torch.Size([1824, 1])
        return speed, label_reg, label_cls, scale_y

    def construct_data(self):
        if self.phase_gen == 'train':
            f = pd.read_csv(f'./data/Series_data_classification/train.csv', sep=',', index_col=0)
        elif self.phase_gen == 'val':
            f = pd.read_csv(f'./data/Series_data_classification/test.csv', sep=',', index_col=0)
        else:
            f = pd.read_csv(f'./data/Series_data_classification/test.csv', sep=',', index_col=0)

        feature_file = open(f'./data/Series_data_classification/list.txt', 'r')
        feature_map = []
        res = []
        for ft in feature_file:
            feature_map.append(ft.strip())

        for feature in feature_map:
            if feature in f.columns:
                res.append(f.loc[:, feature].values.tolist())
            else:
                print(feature, 'not exist in data')

        # sample_n = len(res[0])
        # print("preprocess_sample_n", sample_n, len(res), len(res[0]))  # sample_n是所有sample的总数，res是所有特征组合的list，大小为[6x43824] 43824 2 43824

        x_arr, y_arr_reg, y_arr_cls, data = [], [], [], []

        res = torch.tensor(res).double()
        node_num, total_time_len = res.shape
        print("TimeDataset_data.shape: ", res.shape)  # torch.Size([2, 43824])

        # rang = range(self.args.input_length, total_time_len - self.args.predict_length + 1, self.interval)
        rang_all = range(0, total_time_len)
        print("rang_all: ", rang_all)  # range(0, 43824)
        rang_ft = range(0, total_time_len - (self.args.input_length + self.args.predict_length) * self.interval, self.interval)
        print("rang_ft:", rang_ft)  # range(0, 43799)
        rang_tar = range((self.args.input_length + self.args.predict_length) * self.interval, total_time_len, self.interval)
        print("rang_tar:", rang_tar)  # range(25, 43824)    range(25, 43824)

        for i in rang_all:
            # print(i)  # 0, 1, 2, 3, .... 43823
            data_1 = res[:, i]
            data.append(data_1)  # 所有数据

        data = torch.stack(data).contiguous()   # 所有数据
        print("all_data.shape:", data.shape)  # torch.Size([43824, 2])

        data, scale_y = normalized(data, self.args)
        # print("TimeDataset_normalize_x,y:", data,  data.shape)  # torch.Size([1826, 24, 6]) torch.Size([1826, 1])

        # data = torch.tensor(data).double()
        input_length = self.args.input_length * self.interval
        print(input_length)     # 24
        for i in rang_ft:
            ft = data[i: i + input_length : self.interval, 0]
            x_arr.append(ft)

        for j in rang_tar:
            # print("j:", j)  # 48, 72, .... 43800(43800/24=1825)
            tar_reg = data[j, 0]
            tar_class = data[j, 1]

            y_arr_reg.append(tar_reg)
            y_arr_cls.append(tar_class)

        x = torch.stack(x_arr).contiguous()
        # x = x.reshape(x.shape[0], 1, x.shape[1])
        y_reg = torch.stack(y_arr_reg).contiguous()
        y_cls = torch.stack(y_arr_cls).contiguous()

        # result1 = np.array(y_cls)
        # np.savetxt("/mnt/syanru/Multi-Task/5-Multi-task-solar-modify-lambda-ar-focal-shapelets/data/train.txt", result1)

        # y_reg = y_reg.reshape(y_reg.shape[0], 1)
        # y_cls = y_cls.reshape(y_cls.shape[0], 1)
        print("TimeDataset_x, y_reg, y_cls, x.shape, y_reg.shape, y_cls.shape:", x.shape, y_reg.shape, y_cls.shape)    # torch.Size([43799, 24]) torch.Size([43799, 1]) torch.Size([43799, 1])

        return x, y_reg, y_cls, scale_y

    def __getitem__(self, idx):
        speed = self.speed[idx].double()
        label_reg = self.label_reg[idx].double()
        label_cls = self.label_cls[idx].double()

        # print("TimeDataset_feature, y:", feature, y)    # feature 是 x, y是预测标签

        return speed, label_reg, label_cls


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', help='batch size', type=int, default=32)
    parser.add_argument('--input_length', help='input_length', type=int, default=4)  # 输入数据长度
    parser.add_argument('--predict_length', help='predict length', type=int, default=0)  # 输出数据长度  # 预测一天后的数据
    parser.add_argument('--interval', help='EUV数据采样频率（间隔几张采一次）', type=int, default=24)
    parser.add_argument('--norm', help='正则化方式', type=int, default=2)

    args = parser.parse_args()
    # main = DataGenerator(args, 'train', args.interval)
    # main = DataGenerator(args, 'val', args.interval)
    main = DataGenerator(args, 'test', args.interval)
    # main.generator()
    # gen = generator(phase_gen='train')
    # print(gen)
    # print("start")
    # generator(phase_gen='val')
    # generator(phase_gen='test')
# python -u data.py |tee ./test
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type=int, default=32)
    parser.add_argument('-epoch', help='train epoch', type=int, default=30)
    parser.add_argument('-save_path_pattern', help='save path pattern', type=str, default='')
    parser.add_argument('-dataset', help='wadi / swat', type=str, default='Solar_hour')
    parser.add_argument('-device', help='cuda / cpu', type=str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type=int, default=5)
    parser.add_argument('-decay', help='decay', type=float, default=0)
    parser.add_argument('-report', help='best / val', type=str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type=str, default='')
    parser.add_argument('-slide_win', help='input_length', type=int, default=24)  # 输入数据长度
    parser.add_argument('-predict_length', help='predict length', type=int, default=96)  # 输出数据长度

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    # main = Main(train_config, env_config, debug=False)
    # main.run()
# im = process(input_train + 'image629.png', 'image629.png')
'''
