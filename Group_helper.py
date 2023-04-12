import torch

class Group_helper(object):
    def __init__(self, dataset, num_groups, scale_y=None, Symmetrical = True, Max = None, Min = None):
        '''
            dataset : list of deltas (CoRe method) or list of scores (RT method)
            depth : depth of the tree
            Symmetrical: (bool) Whether the group is symmetrical about 0.
                        if symmetrical, dataset only contains th delta bigger than zero.
            Max : maximum score or delta for a certain sports.
        '''
        self.dataset = sorted(dataset)
        self.length = len(dataset)
        self.num_leaf = num_groups
        print(self.length, self.num_leaf)
        self.scale = scale_y
        self.symmetrical = Symmetrical
        self.max = self.dataset[-1]
        self.min = self.dataset[0]
        print("self.max, self.min:", self.max, self.min)
        self.Group = [[] for _ in range(self.num_leaf)]
        self.build()

    def build(self):
        '''
            separate region of each leaf
        '''
        if self.symmetrical:
            # delta in dataset is the part bigger than zero.
            for i in range(self.num_leaf // 2): # 4
                # bulid positive half first
                Region_left = self.dataset[int( (i / (self.num_leaf//2)) * (self.length-1) )]
                if i == 0:
                    if self.min != None:
                        Region_left = self.min
                    else:
                        Region_left = self.dataset[0]
                print('Region_left:', Region_left)

                Region_right = self.dataset[int( ( (i + 1) /(self.num_leaf//2)) * (self.length-1) )]
                if i == self.num_leaf//2 - 1:
                    if self.max != None:
                        Region_right = self.max
                    else:
                        Region_right = self.dataset[-1]
                print('Region_right:', Region_right)

                self.Group[self.num_leaf // 2 + i] = [Region_left, Region_right]
            for i in range(self.num_leaf // 2):
                self.Group[i] = [-i for i in self.Group[self.num_leaf - 1 - i]]  # 对称到负数区间
            for group in self.Group:
                group.sort()
            for i in range(self.num_leaf):
                print('symmetrical_Group:', self.Group[i], self.scale.inverse_transform(Region_left), self.scale.inverse_transform(Region_right))
        else:
            for i in range(self.num_leaf):
                Region_left = self.dataset[int( (i / self.num_leaf) * (self.length-1) )]
                if i == 0:
                    if self.min != None:
                        Region_left = self.min
                    else:
                        Region_left = self.dataset[0]

                Region_right = self.dataset[int( ( (i + 1) / self.num_leaf) * (self.length-1) )]
                if i == self.num_leaf - 1:
                    if self.max != None:
                        Region_right = self.max
                    else:
                        Region_right = self.dataset[-1]
                self.Group[i] = [Region_left, Region_right]
                print('no_symmetrical_Group:', self.Group[i], self.scale.inverse_transform(Region_left), self.scale.inverse_transform(Region_right))

    def produce_label(self, scores):
        # print(scores)
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy().reshape(-1,)
        glabel = []
        rlabel = []
        cls = []
        for i in range(self.num_leaf):
            # if in one leaf : left == right
            # we should treat this leaf differently
            leaf_cls = []
            laef_reg = []
            for score in scores:
                if i == 0 and score < self.Group[0][1]:
                    leaf_cls.append(1)
                elif i == self.num_leaf-1 and score >= self.Group[self.num_leaf-1][0]:
                    leaf_cls.append(1)
                elif score < self.Group[i][1] and score >= self.Group[i][0]:
                    leaf_cls.append(1)
                # elif score < 0 and (score <= self.Group[i][1] and score > self.Group[i][0]):
                    # leaf_cls.append(1)
                else:
                    leaf_cls.append(0)

                if leaf_cls[-1] == 1:
                    if self.Group[i][1] == self.Group[i][0]:
                        rposition = score - self.Group[i][0]
                    else:
                        rposition = (score - self.Group[i][0])/(self.Group[i][1] - self.Group[i][0])
                else:
                    rposition = -1
                laef_reg.append(rposition)
            glabel.append(leaf_cls)
            rlabel.append(laef_reg)
        glabel = torch.tensor(glabel).cuda()
        rlabel = torch.tensor(rlabel).cuda()
        for i in range(glabel.shape[1]):
            for j in range(self.num_leaf):
                if glabel[j][i] == 1:
                    cls.append(j)
        cls = torch.tensor(cls).cuda()
        return cls, glabel, rlabel   # rlabel是经过归一化之后的结果了，减去了右边界，除以整个区间差

    def inference(self, probs, deltas):
        '''
            probs: bs * leaf
            delta: bs * leaf
        '''
        predictions = []
        for n in range(probs.shape[0]):
            # print(probs.shape, deltas.shape)    # (32, 8) (32, 8)
            prob = probs[n]
            delta = deltas[n]
            leaf_id = prob.argmax()
            # print(leaf_id, n, delta)
            # print(self.Group[leaf_id][0], self.Group[leaf_id][1])
            if self.Group[leaf_id][0] == self.Group[leaf_id][1]:
                prediction = self.Group[leaf_id][0] + delta[leaf_id]
            else:
                prediction = self.Group[leaf_id][0] + (self.Group[leaf_id][1] - self.Group[leaf_id][0]) * delta[leaf_id]
            # print(prediction)
            predictions.append(prediction)  # 计算真实的回归值
            # print(delta[leaf_id], self.scale.inverse_transform(prediction))
        return torch.tensor(predictions).reshape(-1, 1)

    def get_Group(self):
        return self.Group

    def number_leaf(self):
        return self.num_leaf


def main():
    RT_depth = 4
    # normalize(dataset[i][2], class_idx, score_range)
    score_range = 100
    delta_list = [20.3, 23.4, 10.5, 21.5, 36.7]
    num_leaf = 2 ** (RT_depth - 1)
    print(num_leaf)
    group = Group_helper(delta_list, RT_depth, Symmetrical=True, Max=score_range, Min=0)
    glabel_1, rlabel_1 = group.produce_label([20, 34, 35])
    print(glabel_1, rlabel_1)
    '''
    tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 1]], device='cuda:0') tensor([[-1.0000, -1.0000, -1.0000],
        [-1.0000, -1.0000, -1.0000],
        [-1.0000, -1.0000, -1.0000],
        [-1.0000, -1.0000, -1.0000],
        [ 0.9852, -1.0000, -1.0000],
        [-1.0000, -1.0000, -1.0000],
        [-1.0000, -1.0000, -1.0000],
        [-1.0000,  0.1384,  0.1514]], device='cuda:0')
    竖着看是一个数字的label
    '''
    leaf_probs_2 = [[2.3, 3.6], [6.3, 3.6], [6.6, 3.6], [2.6, 3.6], [3.6, 3.6], [3.2, 3.6], [6.2, 3.6], [7.1, 3.6]]
    leaf_probs_2 = torch.tensor(leaf_probs_2)
    delta_2 = [[2.2, 3.6], [3.5, 3.6], [6.7, 3.6], [2.4, 3.6], [1.3, 3.6], [2.6, 3.6], [1.4, 3.6], [7.4, 3.6]]  # [batch_size]，每个b_s里面表示叶子节点的个数，区间的个数
    delta_2 = torch.tensor(delta_2)
    print(leaf_probs_2.shape, delta_2.shape)    # torch.Size([8, 2]) torch.Size([8, 2])
    relative_scores = group.inference(leaf_probs_2, delta_2)
    print(relative_scores)

if __name__ == '__main__':
    main()