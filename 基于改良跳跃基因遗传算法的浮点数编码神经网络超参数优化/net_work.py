import numpy as np
import torch as tr
import csv

# 网络层数
net_= 4
# 是否为分类问题
is_class = False
# 是否为多分类
is_more = False
# 训练次数
steps = 450

F = [tr.nn.ReLU(),tr.nn.Sigmoid(),tr.nn.Tanh()]
batchs = [256,512,1024,2048]
learn_rate = [0.0001,0.001,0.01]

filename = ["data/train_features.csv","data/train_targets.csv","data/test_features.csv","data/test_targets.csv"]
train_fea = []
text_fea = []
train_tar = []
text_tar = []
loss_f = None
data_num = None
device = None

MAE_loss = tr.nn.L1Loss()

class Net(tr.nn.Module):
    def __init__(self,Ns,Fs):
        super(Net, self).__init__()

        self.pars = []
        self.fun_set = Fs

        # 存入网络模型
        for i in range(len(Ns)-1):
            self.pars.append(tr.nn.Linear(Ns[i],Ns[i+1]))

        # self.pars = tr.nn.Linear(9,100)
        # self.pars1 = tr.nn.Linear(100,10)
        # self.pars2 = tr.nn.Linear(10,1)

    def forward(self,data):
        # 计算图
        for i in range(net_):
            data = self.fun_set[i](self.pars[i](data))

        data = self.pars[-1](data)
        if is_class:
            data = self.fun_set[-1](data)

        return data

    # def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
    #     for i in self.pars:
    #         for name,parm in i.named_parameters():
    #             yield parm


def train(ns, fun_set, w_set):
    # 生成网络，获取其参数
    if device is None:
        net = Net(ns, fun_set)
    else:
        net = Net(ns, fun_set).to(device)
    para = []

    # print(ns,fun_set,w_set)
    batch = batchs[w_set[0]]

    for i in net.pars:
        for j,k in i.named_parameters():
            para.append(k)

    opm = tr.optim.Adam(para,lr=learn_rate[w_set[1]])

    # 开始迭代
    for i in range(steps):
        for j in range(int(data_num/batch)):
            y_hat = net(train_fea[j*batch:(j+1)*batch])
            loss = loss_f(y_hat, train_tar[j*batch:(j+1)*batch])

            # 梯度清零
            opm.zero_grad()
            # 反向传播
            loss.backward()
            # 优化
            opm.step()

        if train_fea.size()[0]%batch > 0:
            y_hat = net(train_fea[int(data_num/batch) * batch:])
            loss = loss_f(y_hat, train_tar[int(train_fea.size()[0]/batch) * batch:])

            # 梯度清零
            opm.zero_grad()
            # 反向传播
            loss.backward()
            # 优化
            opm.step()

        y_hat = net(train_fea)
        loss = loss_f(y_hat, train_tar)
        # 输出信息
        if i % 100 == 0:
            print("迭代次数--", i, "--当前损失值--", loss.item())

    y_hat = net(train_fea)
    loss = loss_f(y_hat, train_tar)
    print("最终结果,损失值：", loss.item())

    test_y_hat = net(text_fea)
    text_loss = loss_f(test_y_hat,text_tar)
    text_loss_MAE = tr.mean(tr.abs(test_y_hat-text_tar))
    print("预测结果,损失值：", text_loss.item())
    return text_loss.item(),text_loss_MAE.item()


def init_work():
    global train_fea,train_tar,text_fea,text_tar,loss_f,data_num,device
    if tr.cuda.is_available():
        device = tr.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = tr.device("cpu")
        print("Running on the CPU")

    if is_class:
        if is_more:
            loss_f = tr.nn.NLLLoss
        else:
            loss_f = tr.nn.MSELoss
    else:
        loss_f = tr.nn.functional.mse_loss

    with open(filename[0], "r") as f:
        reader = csv.reader(f)
        header_fea = next(reader)
        for row in reader:
            train_fea.append(row)

    with open(filename[1], "r") as f:
        reader = csv.reader(f)
        header_tar = next(reader)
        for row in reader:
            train_tar.append(row)

    with open(filename[2], "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            text_fea.append(row)

    with open(filename[3], "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            text_tar.append(row)

    train_fea,train_tar,text_fea,text_tar = np.array(train_fea).astype(np.float32),np.array(train_tar).astype(np.float32),np.array(text_fea).astype(np.float32),np.array(text_tar).astype(np.float32)

    train_fea, train_tar, text_fea, text_tar = tr.from_numpy(train_fea), tr.from_numpy(train_tar), tr.from_numpy(text_fea), tr.from_numpy(text_tar)

    data_num = train_fea.size()[0]


def work(xx):
    ns = xx[:net_+2]
    f_s = []
    for i in xx[net_+2:2*net_+2]:
        f_s.append(F[i])
    if is_class:
        if is_more:
            f_s.append(tr.nn.Softmax)
        else:
            f_s.append(tr.nn.Sigmoid)

    w_s = xx[2*net_+2:]
    return train(ns, f_s, w_s)


# init_work()
# work([9, 10, 4, 6, 4,1,0,0,0, 0, 3, 1])






