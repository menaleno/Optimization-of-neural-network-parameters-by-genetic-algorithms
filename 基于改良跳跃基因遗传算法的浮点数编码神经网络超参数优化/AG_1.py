import numpy as np
import time
import random
from random import getrandbits
import csv
import net_work


# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
# plt.rcParams['axes.unicode_minus'] = False

# 迭代次数
step = 30
# 基因长度
max_v = net_work.net_*2 + 2
# 种群规模
max_n = 20
# 交叉概率
jc = 0.8
# 变异概率
by = 0.1


# 最小开关
min_tap = True
# 改进算法设置
op_set = 2
# 区间左端点
left = 0
# 区间右端点
right = 1

# 取值区间，基于左右端点的进一步计算
ll = []
rr = []


# 转位基因个数
T = int(max_v/2)
# 转位基因长度
L = 1
# 跳跃方式为复制的概率
p_c = 0.5
# 基因跳跃概率的调整权重，该值越大跳跃基因概率越高
apha = 0.2
# 密度权重，该值越大密度越高的个体适应度越低
beta = 10000

filename = "E:\\study\\计算智能\\big work\\论文\\AG_1_22.csv"
filename_2 = "E:\\study\\计算智能\\big work\\论文\\AG_1_22_MAE.csv"
steps = 2


input_set = None
output_set = None

def lim(x):
    if x <= 0:
        return 1/(1+np.exp(-x))
    else:
        return np.exp(-x)/(1+np.exp(-x))


# 随机洗牌函数，用于产生一定范围内的不重复随机数序列
def rand_get(l,r):
    temp = [i for i in range(l,r)]
    random.shuffle(temp)
    return temp

class dna:
    def __init__(self):
        # 基因数组，这里为浮点数
        self.peace = np.array([np.random.uniform(left,right) for i in range(max_v)])
        # 适应度
        self.grade = None
        # MAE
        self.MAE = None
    def caculate(self):

        # y = (1-self.peace[0])**2 + 100*(self.peace[1]-self.peace[0]**2)**2
        # A = 10
        # y = A * max_v + np.sum(self.peace ** 2 - A * np.cos(2 * np.pi * self.peace))

        xx = [input_set]
        for i in range(max_v):
            xx.append(ll[i] + int(rr[i]*self.peace[i]))
        xx.insert(net_work.net_+1,output_set)
        self.grade,self.MAE = net_work.work(xx)

    def __copy__(self,p):
        for i in range(max_v):
            self.peace[i] = p.peace[i]
        self.grade = p.grade

    def __lt__(self, other):
        return self.grade < other.grade

    def __gt__(self, other):
        return self.grade > other.grade

    def __le__(self, other):
        return self.grade <= other.grade

    def __ge__(self, other):
        return self.grade >= other.grade

    def __eq__(self, other):
        return self.grade == other.grade

    def __ne__(self, other):
        return self.grade != other.grade

    # 下面为几个基因的交叉函数

    # 基因的单点交叉函数，传入交叉对象p,以及交叉地址add
    def one_point_change(self,p):
        for i in range(np.random.randint(0,max_v),max_v):
            self.peace[i],p.peace[i] = p.peace[i],self.peace[i]

    # 阿法交叉，使用0`1的随机数阿法来进行交叉
    def afa_chagne(self,p):
        afa = np.random.random()

        for i in range(max_v):
            self.peace[i], p.peace[i] = afa * self.peace[i] + (1 - afa) * p.peace[i], afa * p.peace[i] + (1 - afa) * self.peace[i]

    # 变异函数一，由于需要生成随机值，故同样分无参和有参两种

    # 单点变异
    def mutate(self):
        ids = np.random.randint(0,max_v)

        self.peace[ids] = np.random.uniform(left,right)

    # 变异函数二，采用非均匀突变，g为当前迭代数与最大迭代次数比值，b为形状因子，起到调节函数曲线非均匀变化的作用
    def mutate_2(self,g,b = 3):
        # 获取变异位置
        add = np.random.randint(0,max_v)

        # 获取增减判断依据
        sign = not not getrandbits(1)

        if sign:
            self.peace[add] = self.peace[add] + (right - self.peace[add]) * (np.random.random()*(1-g))**b
        else:
            self.peace[add] = self.peace[add] - (self.peace[add]-left) * (np.random.random()*(1-g))**b

    # 下面为跳跃基因相关操作函数
    def jump_copy(self, a1, b1, c1):
        for i in range(L):
            self.peace[a1+i],b1.peace[c1+i] = b1.peace[c1+i],self.peace[a1+i]

# 遗传算法类
class yc:
    def __init__(self,generation=3,change_p=0.8,mutate_p=0.0002):
        # 参数的设置
        self.C = [dna() for i in range(max_n+1)]
        self.P = [None for i in range(max_n)]
        self.gen = generation
        self.change_set = change_p
        self.mutate_set = mutate_p

        if min_tap:
            for i in range(1,max_n+1):
                self.P[i-1] = apha * (1-i/max_n)
        else:
            for i in range(max_n):
                self.P[i] = apha * i / max_n

        best = 0 # best存放最优适应值基因的数组索引
        for i in range(max_n):
            self.C[i].caculate()

            # 下面为最小优化开关判断，
            if min_tap:
                # 若为最小优化，则best存放适应值更小的基因位置
                if self.C[best] > self.C[i]:
                    best = i
            else:
                # 若为最大优化，则best存放适应值更大的基因位置
                if self.C[best] < self.C[i]:
                    best = i

        # 获取最优基因,存放于数组末端
        self.C[max_n].__copy__(self.C[best])

        # 输出初始化后初代种群的最优个体基因及其适应值
        self.C[max_n].caculate()
        print("初始种群最优个体的基因：",self.C[max_n].peace," 适应值为：",self.C[max_n].grade)

    # 计算差值
    def counter(self,ii):
        sums = 0

        for j in range(max_n):
            if ii == j:
                continue
            for i in range(max_v):
                sums += (self.C[ii].peace[i] - self.C[j].peace[i]) ** 2

        if min_tap:
            return (1+sums)/beta
        else:
            return beta/(1+sums)

    # 跳跃基因操作
    def jump(self):
        # 加权适应度计算
        for i in range(max_n):
            self.C[i].grade = self.C[i].grade * np.exp(-self.counter(i))

        self.C = sorted(self.C)

        u1 = max_v-L
        ch = [None for i in range(L)]

        for i in range(max_n):
            if np.random.random() < self.P[i]:  # 若满足跳跃发生条件，则进入基因跳跃
                if np.random.random() < p_c: # 若满足复制粘贴方式，则使用复制粘贴方式
                    self.C[i].jump_copy(np.random.randint(u1),self.C[np.random.randint(max_n)],np.random.randint(u1))
                else:  # 否则使用剪贴方式
                    a1,a2 = np.random.randint(u1),np.random.randint(u1) # 跳跃基因位置
                    b1 = np.random.randint(max_n)
                    c1,c2 = np.random.randint(u1),np.random.randint(u1)

                    for j in range(L):
                        ch[j] = self.C[i].peace[a1+j]

                    if c2 >= a1:
                        for j in range(a1+L,c2+L):
                            self.C[i].peace[j-L] = self.C[i].peace[j]
                    else:
                        for j in range(c2,a1):
                            self.C[i].peace[j+1] = self.C[i].peace[j]

                    for j in range(L):
                        self.C[i].peace[c2+j] = self.C[b1].peace[a2+j]

                    if c1 >= a2:
                        for j in range(a2+L,c1+L):
                            self.C[b1].peace[j-L] = self.C[b1].peace[j]
                    else:
                        for j in range(c1,a2):
                            self.C[b1].peace[j+1] = self.C[b1].peace[j]

                    for j in range(L):
                        self.C[b1].peace[c1+j] = ch[j]

    # 下面为几个选择算法：
    def lun(self): # 轮盘法选择，每次根据轮盘赌选出一个基因，返回该基因的数组索引
        sums = 0
        r = np.random.random()
        for j in range(max_n):
            sums += self.C[j].grade
            if r <= sums:
                return j

        return max_n-1

    def lunpan_select(self): # 轮盘法选择函数,这里提供选择前的一些初始化工作，同时调用轮盘选择函数来进行
        if min_tap:
            for i in range(max_n):
                self.C[i].grade = 1/self.C[i].grade
                if self.C[i].grade < 0 :
                    self.C[i].grade = -self.C[i].grade

        # 计算总适应值大小
        sums = 0
        for i in range(max_n):
            sums += self.C[i].grade

        # 将各个适应值除以总适应值，得出其占比
        for i in range(max_n):
            self.C[i].grade = self.C[i].grade / sum

        # 下面选择新的基因,使用轮盘赌算法选出下一个种群
        new_c = [dna() for i in range(max_n)]
        for i in new_c:
            i.__copy__(self.C[self.lun()])

        new_c.append(self.C[max_n])
        self.C = new_c

    # 锦标赛算法，有两个可选参数，s（锦标压力），winer（锦标小组的胜出者）
    def jinbiao_select(self,s = int(max_n/10),winer = 1):
        temp = None
        new_gen = [dna() for i in range(max_n)]
        gr = None
        i = 0
        while i < max_n:
            gr = []
            temp = rand_get(0,max_n)
            for j in range(s):
                gr.append(self.C[temp[j]])

            if min_tap:
                gr = sorted(gr)
            else:
                gr = sorted(gr,reverse=True)

            for j in range(winer):
                new_gen[i].__copy__(gr[j])
                i += 1
                if i == max_n:
                    break

        new_gen.append(self.C[max_n])
        # 新族群替换旧族群
        self.C = new_gen

    # 交叉调度算法，这里会调用 dna 类的多种交叉算法,根据传入的参数 set 来判断调用哪个交叉算法
    def crossover(self,set = 2):
        sums = 0
        g = [False for i in range(max_n)]
        chang = dna()
        j_1 = ch_add = None
        j_2 = -1

        # 生成数组，记录每个个体是否参与交配
        for i in range(max_n):
            if np.random.random() <= self.change_set:
                g[i] = True
                sums += 1

        # 若参与交配的个体为奇数，可以则这些个体无法两两交配，将随机剔除一个参与交配的个体
        if sums % 2 == 1:
            r = np.random.randint(1,sums)  # 随机产生剔除的个体索引

            i = 1
            while i < sums:  # i 表示已参与交配的个体数，每当找到一个参与交配的个体，i+1
                j_1 = j_2 + 1

                while not g[j_1]:  # 找到参与交配的基因
                    j_1 += 1

                if i == r:  # 若为剔除的基因则继续寻找下一个参与交配个体
                    j_1 += 1
                    while not g[j_1]:
                        j_1 += 1

                # 找到参与交配的个体j_1，i加一
                i += 1

                j_2 = j_1 + 1
                while not g[j_2]:
                    j_2 += 1

                if i == r:
                    j_2 += 1
                    while not g[j_2]:
                        j_2 += 1

                i += 1

                if set == 1:
                    self.C[j_1].one_point_change(self.C[j_2])
                elif set == 2:
                    self.C[j_1].afa_chagne(self.C[j_2])
                else:
                    print("交叉出错！")
                    exit(1)
        else:  # 若参与交配个体为偶数，则可两两交配，无需剔除
            i = 1
            while i < sums:
                j_1 = j_2 + 1

                # 找到参与交配的基因
                while not g[j_1]:
                    j_1 += 1
                i += 1
                while not g[j_2]:
                    j_2 += 1
                i += 1

                # 根据传入的参数来判断使用哪一种交叉算法
                if set == 1:
                    self.C[j_1].one_point_change(self.C[j_2])
                elif set == 2:
                    self.C[j_1].afa_chagne(self.C[j_2])
                else:
                    print("交叉出错！")
                    exit(1)

    # 变异函数
    def mutation(self,sets=2,g=0):
        for i in range(max_n):
            # 若变异概率符合要求，则变异
            if np.random.random() <= self.mutate_set:
                # 根据传入的参数来判断使用哪一种变异算法
                if sets == 1:
                    self.C[i].mutate()
                elif sets == 2:
                    self.C[i].mutate_2(g/self.gen)
                else:
                    print("变异出错！")
                    exit(1)

    # 迭代函数 , 参数b1为1 则使用传统交叉变异算子，为2则使用改进算子
    def begin(self,bl):
        ends = []
        ends_1 = []
        # 开始迭代
        for i in range(1,self.gen+1):
            self.jinbiao_select() # 种群选择
            self.jump() # 基因跳跃
            self.crossover(bl) # 种群交配
            self.mutation(bl,i) # 种群变异

            best = 0
            for j in range(max_n):
                # 计算新种群的适应度
                self.C[j].caculate()

                # 判断是否最小优化
                if min_tap:
                    if self.C[best] > self.C[j]:
                        best = j
                else:
                    if self.C[best] < self.C[j]:
                        best = j

            print("第",i,"次迭代，最优个体基因",self.C[best].peace,"适应值为:",self.C[best].grade)

            # 记录本次迭代的最优值
            ends.append(self.C[best].grade)
            ends_1.append(self.C[best].MAE)

            # 对这一代的最优与之前的最优作比较，若之前最优比当代最优更好，则替换，否则不作替换
            if min_tap:
                if self.C[max_n] > self.C[best]:
                    self.C[max_n].__copy__(self.C[best])
            else:
                if self.C[max_n] < self.C[best]:
                    self.C[max_n].__copy__(self.C[best])



            # 输出至今最优基因以及其适应值
            print("\n最终的个体基因为：",self.C[max_n].peace,"适应值为:",self.C[max_n].grade)

        return ends,ends_1


# def dept_draw():
#     # 计时
#     time_start = time.time()
#     aaa = yc(step,jc,by)
#
#     xx = [i for i in range(step)]
#     yy = aaa.begin(op_set)
#
#     # print("最终种群：")
#     # for i in range(max_n):
#     #     print("第", i + 1, "个个体-----", aaa.C[i].peace, "---适应度---", aaa.C[i].grade)
#
#     time_end = time.time()
#     print("共耗时:", time_end - time_start, "秒")
#
#     plt.plot(xx, yy)
#     if min_tap:
#         s = "迭代次数   最小值为%f" % np.min(yy)
#     else:
#         s = "迭代次数   最大值为%f" % np.max(yy)
#     plt.xlabel(s)
#     plt.ylabel("最佳值")
#     # plt.ylim(0, 0.8)
#     plt.show()


# 设置函数，进行一些必要的初始化工作
def setting_work():
    # 进行神经网络部分的初始化，处理好训练集与测试集
    net_work.init_work()

    global ll, rr,input_set,output_set
    input_set = net_work.train_fea.size()[1]
    output_set = net_work.train_tar.size()[1]

    # d 取 8
    r = int(np.sqrt(input_set+output_set)+5)
    # 神经元个数
    for i in range(net_work.net_):
        ll.append(2)
        rr.append(2*r-2)

    # 每层的激活函数
    for i in range(net_work.net_):
        ll.append(0)
        rr.append(3)

    # batch
    ll.append(0)
    rr.append(4)

    # 学习率
    ll.append(0)
    rr.append(3)


def runs(filenames,filenames_2, st):
    # 计时
    time_start = time.time()

    for i in range(st):
        aaa = yc(step,jc,by)
        yy,yy1 = aaa.begin(op_set)
        with open(filenames, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(yy)

        with open(filenames_2, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(yy1)

    time_end = time.time()
    print("共耗时:", time_end - time_start, "秒")


setting_work()
runs(filename,filename_2,steps)



