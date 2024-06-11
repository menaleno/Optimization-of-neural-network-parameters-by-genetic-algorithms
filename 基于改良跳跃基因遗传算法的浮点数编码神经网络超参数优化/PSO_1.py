import numpy as np
import time
import random
from random import getrandbits
import csv
import net_work

# 变量维度
num = net_work.net_*2 + 2

# 速度最大比例 ， 占相应变量取值区间的比例  0111
maxer = 0.2

# 种群规模
group_num = 20

# 迭代次数
step = 30
# 是否求最小值，若是则为1
min_tap = True

# 取值范围
minx = []
maxx = []

# 惯量权重
w = 0.8
w_min = 0.4

# 加速系数
c1 = 1
c1_min = 0.5
c2 = 1
c2_min = 0.5


filename = "E:\\study\\计算智能\\big work\\论文\\PSO_1.csv"
filename_2 = "E:\\study\\计算智能\\big work\\论文\\PSO_1_MAE.csv"
steps = 2


input_set = None
output_set = None

# def f(x):  # 香蕉函数   求极小值 0
#     x = np.array(x)
#     y = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
#     return np.sum(y)


# def f(x):       # Ackley 函数  极小值为0
#     x = np.array(x)
#     y = -20 * np.exp(-0.2 * np.sqrt(1 / num * np.sum(x ** 2))) - np.exp(
#         1 / num * np.sum(np.cos(2 * np.pi * x))) + 20 + np.e
#     return y

# def f(x):    # 极大值
#     x = np.array(x)
#     y = 20 + x + 10 * np.sin(4*x) + 8
#     return np.sum(x)

# def f(x):     # 1维 x 取值0~4 极大值  1.6
#     x = np.array(x)
#     if 0 <= x[0] <= 0.5:
#         y = 3.2 * x[0]
#     elif 0.5 < x[0] <= 1:
#         y = -3.2 * x[0] +3.2
#     else:
#         y = np.sqrt(2.25 - x[0]**2)
#     return y

# def f(x):   # 1维 x 取值 -12 ~ 12  最大值6
#     x = np.array(x)
#     if -9.4 <= x[0] <= -9:
#         y = 12 * x[0] + 114
#     elif -9 < x[0] <= -8.6:
#         y = -12 * x[0] - 102
#     elif x[0] >= 0:
#         y = -0.176 * x[0]**2 + 1.64 * x[0] +1.2
#     else:
#         y = 1.2
#     return y

# 函数6   shekel函数 变量维度为4 ，求极小值
# B = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
# D = [[4, 1, 8, 6, 3, 2, 5, 8, 6, 7], [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6], [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
#      [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]]
#
#
# def f(x):
#     x = np.array(x)
#     y = 0
#     for j in range(10):
#         y1 = 0
#         for i in range(num):
#             y1 += (x[i] - D[i][j]) ** 2
#         y1 = 1 / (y1 + B[j])
#         y += y1
#     y = -y
#     return y

# def f(x):  # Rastrigin 函数 -10 到 10 ，极小值在 变量值为 0 处
#     x = np.array(x)
#     A = 10
#     y = A * num + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))
#     return y


# def f(x):   # Griewank函数 极小值f(0)=0
#     x = np.array(x)
#     y = 1
#
#     for i in range(1,num+1):
#         y *= np.cos(x[i-1]/np.sqrt(i))
#
#     y = np.sum(x**2)/4000 - y + 1
#     return y


def copy_u(x):
    xx = uni()
    xx.best_v = x.best_v
    xx.best = x.best.copy()
    xx.v = x.v.copy()
    xx.place = x.place.copy()
    return xx


# 计算惯量
def get_w(nn):
    return w - (w - w_min) * nn / step


def get_w_tanh(nn):  # tanh 区间 [-4,4]
    # return np.tanh(8 * (step - nn) / step) * (w - w_min)
    return (w + w_min) / 2 + np.tanh(8 * (step - nn) / step) * (w - w_min) / 2


def get_w_exp(nn):  # exp 区间
    return w - np.exp(- 8 * (step - nn) / step) * (w - w_min)


# 计算c1,c2
def get_c1(nn):
    return c1 - (c1 - c1_min) * nn / step


def get_c2(nn):
    return c2_min - (c2_min - c2) * nn / step


# 粒子类，表示单个粒子
class uni():

    def __init__(self):
        # 位置
        self.place = None
        # 速度
        self.v = None
        # 历史最优位置的适应值
        self.best_v = None
        # 历史最优位置
        self.best = None

        self.MAE = None

        self.init_uni()

    def init_uni(self):
        self.place = []
        self.v = []

        for i in range(num):
            self.place.append(np.random.uniform(minx[i], maxx[i]))  # 生成初始位置
            self.v.append(np.random.uniform(maxer * minx[i], maxer * maxx[i]))  # 生成初始速度

        self.place = np.array(self.place)
        self.v = np.array(self.v)
        self.best = self.place

        xx = self.place.astype(int).tolist()
        xx.insert(0, input_set)
        xx.insert(net_work.net_ + 1, output_set)
        self.best_v,self.MAE = net_work.work(xx)

    # 适应度计算
    def counts(self):
        xx = self.place.astype(int).tolist()
        xx.insert(0,input_set)
        xx.insert(net_work.net_ + 1, output_set)

        values_now, MAE_now = net_work.work(xx)

        # 若新的适应度比自身历史最优还高，则替换
        if min_tap:
            if values_now < self.best_v:
                self.best = self.place
                self.best_v = values_now
                self.MAE = MAE_now
        else:
            if values_now > self.best_v:
                self.best = self.place
                self.best_v = values_now
                self.MAE = MAE_now


class PSO:
    def __init__(self):
        self.group = None
        self.his_best = None
        self.his_best_v = None
        self.MAE = None

    def inited(self):  # 初始化函数

        # 初始化种群
        self.group = [uni() for i in range(group_num)]

        # 最优粒子所在索引
        best_value = 0

        for i in range(1, group_num):
            if min_tap:
                if self.group[best_value].best_v > self.group[i].best_v:
                    best_value = i
            else:
                if self.group[best_value].best_v < self.group[i].best_v:
                    best_value = i

        # 更新种群历史最优
        self.his_best = self.group[best_value].best.copy()
        self.his_best_v = self.group[best_value].best_v
        self.MAE = self.group[best_value].MAE

    def renew(self, u, nn):  # 更新函数，传入要更新的粒子u，以及当前迭代次数nn,用于计算自适应惯量
        # 生成随机权，重用于计算粒子新速度，其分别代表自身历史最优位置与种群历史最优位置对新速度的影响力
        rand1 = np.random.random(num)
        rand2 = np.random.random(num)

        # 惯量为get_w  或 get_w_tanh，随着迭代次数增加，惯量越来越小，用于逼近最优值
        u.v = u.v * get_w(nn) + get_c1(nn) * rand1 * (np.array(u.best) - np.array(u.place)) + get_c2(nn) * rand2 * (
                np.array(self.his_best) - np.array(u.place))

        # 超速限制
        for i in range(num):
            if u.v[i] > 0:
                if u.v[i] > (maxx[i] - minx[i]) * maxer:
                    u.v[i] = (maxx[i] - minx[i]) * maxer
            elif u.v[i] < 0:
                if u.v[i] < (minx[i] - maxx[i]) * maxer:
                    u.v[i] = (minx[i] - maxx[i]) * maxer

        # 更新地址
        u.place = u.place + u.v
        for i in range(num):
            if u.place[i] > maxx[i]:
                u.place[i] = maxx[i]-1
            elif u.place[i] < minx[i]:
                u.place[i] = minx[i]

        # 重新计算适应值
        u.counts()

        # 更新种群最优解
        if min_tap:
            if u.best_v < self.his_best_v:
                self.his_best = u.best.copy()
                self.his_best_v = u.best_v
                self.MAE = u.MAE
        else:
            if u.best_v > self.his_best_v:
                self.his_best = u.best.copy()
                self.his_best_v = u.best_v
                self.MAE = u.MAE

    def work(self):

        self.inited()
        yy = []
        yy1 = []
        for i in range(1, step + 1):
            for j in self.group:
                self.renew(j, i)
            print("第", i, "次迭代，历史最优为---", self.his_best, "---最优值为---", self.his_best_v)
            yy.append(self.his_best_v)
            yy1.append(self.MAE)

        print("最终结果，历史最优为---", self.his_best, "---最优值为---%.10f" % self.his_best_v)
        return yy,yy1


# 设置函数，进行一些必要的初始化工作
def setting_work():
    # 进行神经网络部分的初始化，处理好训练集与测试集
    net_work.init_work()

    global minx, maxx,input_set,output_set
    input_set = net_work.train_fea.size()[1]
    output_set = net_work.train_tar.size()[1]

    # d 取 8
    r = int(np.sqrt(input_set+output_set)+5)
    # 神经元个数
    for i in range(net_work.net_):
        minx.append(2)
        maxx.append(2*r-2)

    # 每层的激活函数
    for i in range(net_work.net_):
        minx.append(0)
        maxx.append(3)

    # batch
    minx.append(0)
    maxx.append(4)

    # 学习率
    minx.append(0)
    maxx.append(3)

    minx = np.array(minx)
    maxx = np.array(maxx)


def runs(filenames,filenames_2,st):
    # 计时
    time_start = time.time()

    for i in range(st):
        aaa = PSO()
        yy,yy1= aaa.work()
        with open(filenames, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(yy)
        with open(filenames_2, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(yy1)

    time_end = time.time()
    print("共耗时:", time_end - time_start, "秒")


setting_work()
runs(filename,filename_2, steps)
