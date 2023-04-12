import os
import torch
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np
import argparse
from torch.utils.data import DataLoader

import random
from pathlib import Path
from data import DataGenerator
from model_multi_task import ModelB
from train import train
from test import val, test

from datetime import datetime
from env import *
from confusion_matrix import confusion_matrix
from regression_error import regression_error
from Group_helper import Group_helper


class Main():
    def __init__(self, args):
        self.args = args
        self.datestr = None
        set_device(self.args.device)
        self.device = get_device()

        self.train_dataset = DataGenerator(self.args, 'train', self.args.interval)
        self.val_dataset = DataGenerator(self.args, 'val', self.args.interval)
        self.test_dataset = DataGenerator(self.args, 'test', self.args.interval)

        self.train_scale_y = self.train_dataset.scale_y
        self.val_scale_y = self.val_dataset.scale_y
        self.test_scale_y = self.test_dataset.scale_y

        train_data = self.train_dataset.speed
        train_data = train_data.numpy()
        train_label = self.train_dataset.label_cls
        train_label = train_label.numpy()


        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch, shuffle=False, num_workers=4,
                                           drop_last=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=args.batch, shuffle=False, num_workers=4,
                                         drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=args.batch, shuffle=False, num_workers=4,
                                          drop_last=True)

        print('train:')
        self.train_group = self.build_group(self.train_dataset.label_reg, args, self.train_scale_y)
        print('val:')
        self.val_group = self.build_group(self.val_dataset.label_reg, args, self.train_scale_y)
        print('test:')
        self.test_group = self.build_group(self.test_dataset.label_reg, args, self.train_scale_y)

        self.model = ModelB(self.args).to(self.device)

    def run(self, rang):
        if len(self.args.load_model_path) > 0:
            model_save_path = self.args.load_model_path
            print("main_model_save_path_load_model_path:", model_save_path)
        else:
            model_save_path = self.get_save_path(rang)[0]
            print("main_model_save_path:", model_save_path)
            print(self.model)
            nParams = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            # nParams = sum([p.nelement() for p in self.model.parameters()])
            print('Number of model parameters is', nParams)
            self.train_log = train(self.model, model_save_path,
                                   args=self.args,
                                   train_dataloader=self.train_dataloader,
                                   val_dataloader=self.val_dataloader,
                                   train_scale_y=self.train_scale_y,
                                   val_scale_y=self.val_scale_y,
                                   train_group=self.train_group,
                                   val_group=self.val_group
                                   )
            # print("main_self.train_log:", self.train_log)

        # test
        # self.model.load_state_dict(torch.load(model_save_path))
        self.model = torch.load(model_save_path)
        print("load ok")
        best_model = self.model.to(self.device)
        nParams = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # nParams = sum([p.nelement() for p in self.model.parameters()])
        print('Number of test model parameters is', nParams)

        test_true_result, test_result, test_rmse, test_mae, test_cc, test_mape, acc, cls_label = test(
            best_model, self.args, self.test_dataloader, self.test_scale_y,
            path=f'./result_save/{self.args.save_path_pattern}/result_fushion_gate_test.pkl', group=self.train_group)

        # print(type(cls_label), cls_label)
        confusion_matrix(f'./result_save/{self.args.save_path_pattern}/result_fushion_gate_test.pkl', cls_label)

        self.save_outputs(test_result, test_true_result, self.args)

        return test_rmse, test_mae, test_cc, test_mape, test_msle, acc

    def build_group(self, dataset_train, args, scale_y):
        delta_list = dataset_train.tolist()
        print(dataset_train.shape, type(delta_list))
        group = Group_helper(delta_list, args.num_groups, scale_y, Symmetrical=False) # , Max=args.score_range, Min=0)
        return group

    def save_outputs(self, pred_scores, true_scores, args):
        save_path_pred = os.path.join(f'./result_save/{args.save_path_pattern}', 'pred.txt')
        save_path_true = os.path.join(f'./result_save/{args.save_path_pattern}', 'true.txt')
        np.savetxt(save_path_pred, pred_scores)
        np.savetxt(save_path_true, true_scores)
        regression_error(save_path_pred, save_path_true, self.train_group, args, self.train_scale_y)

    def get_save_path(self, rang):

        dir_path = self.args.save_path_pattern

        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr

        paths = [
            f'./result_save/{dir_path}/best_{datestr}_{rang}.pt',
            f'./result_save/{dir_path}/{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths


if __name__ == '__main__':
    print("start")
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', help='batch size', type=int, default=32)  # 256
    parser.add_argument('--epoch', help='train epoch', type=int, default=60)
    parser.add_argument("--sge", default=1, type=int, help="stop-gradient epochs")
    parser.add_argument('--coarse_ep', help='classification epoch', type=int, default=10)
    parser.add_argument('--fine_ep', help='regression epoch', type=int, default=10)
    parser.add_argument('--save_path_pattern', help='save path pattern', type=str, default='lr_0.001')
    parser.add_argument('--load_model_path', help='trained model path', type=str, default='')
    parser.add_argument('--device', help='cuda / cpu', type=str, default='cuda')
    parser.add_argument('--random_seed', help='random seed', type=int, default=5)
    parser.add_argument('--report', help='best / val', type=str, default='best')
    parser.add_argument('--interval', help='EUV数据样频率（间隔几张采一次）', type=int, default=24)   # 24
    parser.add_argument('--input_length', help='input_length', type=int, default=27)  # 输入数据长度
    parser.add_argument('--predict_length', help='predict length', type=int, default=0)  # 输出数据长度
    parser.add_argument('--norm', help='normalize type', type=int, default=3)  # 正则化方式

    parser.add_argument('--input_dim', type=int, default=1, help='the dimension of the input data')
    parser.add_argument('--hidden_dim', type=int, default=300, help='the dimension of the hidden layer')
    parser.add_argument('--decay', help='decay', type=float, default=0.0001)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

    parser.add_argument("--num_groups", default=4, type=int, help="depth of regression tree")
    parser.add_argument("--score_range", default=10, type=int, help="max of data")
    parser.add_argument("--lambda_conf", default=1, type=float, help="proportion of confidence")
    parser.add_argument("--lambda_cls", default=1, type=float, help="proportion of classification")
    parser.add_argument("--lambda_reg", default=1, type=float, help="proportion of regression")

    args = parser.parse_args()
    print("训练参数： ", args)
    # setup_seed(args.random_seed)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    test_rmse, test_mae, test_cc, test_mape, test_msle = [], [], [], [], []

    if len(args.load_model_path) > 0:
        main = Main(args)
        test_rmse_1, test_mae_1, test_cc_1, test_mape_1, test_msle_1, acc = main.run(0)
        test_rmse.append(test_rmse_1)
        test_mae.append(test_mae_1)
        test_cc.append(test_cc_1)
        test_mape.append(test_mape_1)
        test_msle.append(test_msle_1)
        print(test_rmse, test_mae, test_cc, test_mape, test_msle)
    else:
        for i in range(1):
            print("第几轮：", i)
            main = Main(args)
            test_rmse_1, test_mae_1, test_cc_1, test_mape_1, test_msle_1, acc = main.run(i)
            test_rmse.append(test_rmse_1)
            test_mae.append(test_mae_1)
            test_cc.append(test_cc_1)
            test_mape.append(test_mape_1)
            test_msle.append(test_msle_1)
