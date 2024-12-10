import torch
import matplotlib.pyplot as plt

def Produce_X(x):   #Produce_X()函数用来生成矩阵X
    x0 = torch.ones(x.numpy().size) #x0是根据输入的x的维度生成的各元素为1的向量
    X = torch.stack((x,x0),dim = 1)#torch.stack()函数将向量x与x0合并成一个矩阵X，X的数据类型为Tensor
    return X

x = torch.Tensor([1.4,5,11,16,21])
y = torch.Tensor([14.4,29.6,62,85.5,5,113.4])
X = Produce_X(x)
print(X)

inputs = X
target = y
w =torch.rand(2,requires_grad=True)



def train(epochs=1,learning_rate=0.01):

    for epoch in range(epochs):
        output = inputs.mv(w)
        loss = (output - target).pow(2).sum()

        loss.backward()
        w.data -= learning_rate * w.grad
        w.grad.zero_()
        if epochs % 80 == 0:
            draw(output,loss)

    return w,loss

def draw(output,loss):
    plt.cla()
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),output.data.numpy(),'r-',lw=5)
    plt.text(0.5,0,'loss=%' % (loss.item()),predict={'size':20,'color':'red'})
    plt.pause(0.005)


    w,loss = train(10000,learning_rate= 1e-4)

    print("final loss:",loss.item())
    print("weight:",w.data)
    plt.show()