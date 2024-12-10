import torch  #导入torch包
import torch.nn as nn   #导入torch常用的一个库
import matplotlib.pyplot as plt  #导入可视化

# x = torch.linspace(0,20,500) #表示在0-20这个区间取500个点
# k = 3
# b = 10
# y =k*x+b
#这里的xy都是Tensor，需要data改numpy，也就是把x变为x.data.numpy()
# plt.scatter(x.data.numpy(),y.data.numpy())  #表示x,y里面的离散点全部给画出来,plt.scatter()绘制散点图形式的数据样本点
# plt.show() #运行完图像呈现一条向上的蓝色直线


#1.data
x = torch.rand(512)  #创建一个均匀分布在[0-1)的随机Tensor数，里面有512个点
#加入一个高斯白噪声，就是一个标准的正态分布的均值为0方差为1
noise = 0.2 * torch.randn(x.size())
k = 3
b = 10
y =k*x+b+noise
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()


#建立线性模型
class LinearModel(nn.Module):  #继承库
    def __init__(self,in_fea,out_fea):
        super(LinearModel,self).__init__()
        self.out = nn.Linear(in_features=in_fea,out_features=out_fea)
    def forward(self,x):
        x = self.out(x)
        return x

#对xy进行加一层维度
input_x = torch.unsqueeze(x,dim=1)
input_y = torch.unsqueeze(y,dim=1)

# print(x.size(),y.size())
# print(input_x.size())#打印一下看看加了一层维度后x的大小


#前面的x是512，如果不进行维度转圜输出到模型里就是(1,512),就相当于1个样本，他的特征维度是512，其实应该是输入512个样本，每个样本有一个数
#需要对xy进行加一层维度
model = LinearModel(1,1) #模型实例一下，括号里表示线性模型输入输出的特征是多少
#（1，1）表示每个样本一个数，相当于一个特征


loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.02)#model.parameters()表示model里的所有参数
#lr是learning_rate的缩写，表示学习率
#optimizer表示优化器，主要在torch.optim这个库里，SGD表示随机梯度下降，给整体这个模型的参数做一个优化



#开始训练
plt.ion()#打开交互模式：直接在画板上动态的变
for step in range(200):  #给它200步
    pred = model(input_x)   #表示用这个模型做一个前向传播
    loss = loss_func(pred,input_y)  #用loss去计算对应的损失函数
    optimizer.zero_grad() #梯度清零，不清零就相当于累加了
    loss.backward()  #对loss进行反传
    optimizer.step()   #用优化器的梯度进行更新一下

    #进行可视化
    if step % 10 == 0:  #每10次变一下
        plt.cla() #把上次的清掉
        plt.scatter(input_x.data.numpy(),input_y.data.numpy())  #把输入的数据展示一下
        plt.plot(input_x.data.numpy(),pred.data.numpy(),'y-',lw=5)#‘r-’表示红色的直线，lw表示线的粗度
        [w,b] = model.parameters()  #取对应模型的值w，b
        plt.xlim(0,1)
        plt.ylim(0,20)#定一下x和y的范围
        plt.text(0,0.5,'loss=%.4f,k=%.2f,b=%.2f'%(loss.item(),w.item(),b.item()))
        #plt.text()表示打印loss值，进行一个动态的显示，4f表示4位小数点，loss原本还是一个Tensor，可以用item把他的值取出来，就是一个标量了
        plt.pause(1)  #停隔一秒

plt.ioff()#关闭交互模式
plt.show()