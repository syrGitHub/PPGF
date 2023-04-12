file_name = "deephawkes_cmp_data"


cascades  = "../data/dataset_weibo.txt"


length=70
batchsize=60
train_size= 31780
test_size= 6800
train_lteration=train_size//batchsize
test_lteration=test_size//batchsize
Timewindow=5
pre_times = [24 * 3600]
muti1=5
muti2=16
muti3=17