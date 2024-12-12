import numpy as np
import matplotlib.pyplot as plt

#prepare the train set
x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

#define the model
def forward(x):   #forward前馈
    return x * w   #forward(x)就等于x * w

#定义损失函数
def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)  #这个就相当于损失函数公式loss= (y_pred-y)的平方， 也=(x * w -y)的平方

w_list= []
mse_list = []  #需要把一些权重和损失值保存到列表里，所以要先准备两个空列表

for w in np.arange(0.0,4.1,0.1):  #w取一个范围，是0-4，间隔为0.1
    print("w=",w)
    l_sum = 0
    for x_val,y_val in zip(x_data,y_data):#b表示把x_data,y_data里的数据拿出来用zip拼成这次我们要的x值和y值
        y_pred_val = forward(x_val)
        loss_val = loss(x_val,y_val)
        l_sum += loss_val
        print('/t',x_val,y_val,y_pred_val,loss_val)
    print('MSE=',l_sum/3)
    w_list.append(w)
    mse_list.append(l_sum/3)

    #绘制图形
    plt.plot(w_list,mse_list)
    plt.ylabel('loss')  #表示图上y轴的标签
    plt.xlabel('w')    #表示图上x轴的标签
    plt.show()