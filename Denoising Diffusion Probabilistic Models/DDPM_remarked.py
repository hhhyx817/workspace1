import torchvision
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose, Lambda, ToTensor

def download_dataset():
    # torchvision 中自带 MNIST, root = 下载路径(directory),将在这个路径下下载 MNIST 数据集
    mnist = torchvision.datasets.MNIST(root = "/data/home/huangyx/workspace/diffusion_learn/Diffusion", download = True)
    id = 4
    img, label = mnist[id] # 这里就是随便取了一张 mnist 的图片打印出来看一下
    print(img)
    print(label) # 9：这张图代表的是 9
    
    img.save("/data/home/huangyx/workspace/diffusion_learn/Diffusion/mnist_tmp0.jpg")
    tensor = ToTensor()(img)
    print(tensor.shape) # torch.Size([1, 28, 28])
    print(tensor.max()) # tensor(1.)
    print(tensor.min()) # tensor(0.)
    
'''
pytorch 中的函数不用一开始就把参数和对象传完， 例如这个 transform 是没有具体对象的
'''    
def get_dataloader(batch_size: int):
    # Compose 用于将多个变换组合到一起
    # ToTensor 将 PIL 图像或 NumPy 数组转换为 PyTorch 的张量，并将像素值缩放到 [0, 1] 范围。
    # lambda 将像素值从 [0, 1] 转换到 [-1,1]
    transform_defined = Compose([ToTensor(), Lambda(lambda x : 2 * x - 1 )])
    # torchvision.datasets.MNIST: 这是 PyTorch 的 MNIST 数据集类。
    # root 参数指定数据集的存储路径。
    # transform 参数将之前定义的变换应用于数据集中的每个图像。这意味着下载的图像在加载时会被转换。
    dataset = torchvision.dataset.MNIST(root = "/data/home/huangyx/workspace/diffusion_learn/Diffusion",
                                        transform = transform_defined)
    # 这里两个 batch_size 和上面的 transform 一样，前一个是函数or类的参数，后一个是传入的值
    return DataLoader(dataset, batch_size = batch_size, shuffle = True)
    
def get_img_shape():
    return (1, 28, 28)  

if __name__ == "__main__":
    download_dataset()      
    
    
    
    
    
    
    
    
import torch

class DDPM():
    # n_steps 指的是noise 通过 T 步 变成图片，每一步都随机采样一个 beta
    # self 在这个class中是核心变量，在 class 内的所有 def 都需要靠 self 来传递参数和性质
    def __init__(self, device, n_steps:int, min_beta: float = 0.0001, max_beta: float = 0.02):
        '''
        正向加噪过程中，
        X_t ~ N(sqrt(1 - beta_t) * X_t-1, beta_t·I)
        x_t = sqrt(1 - beta_t) * x_t-1 + sqrt(beta_t) * epsilon_t 重参数化  
        ==> x_t = sqrt(alpha_t ^) * x_0 + sqrt(1 - alpha_t^) * epsilon
            beta_i 为较小的随机数, 
            alpha_i = 1 - beta_i, alpha_i^ = alpha_1 * alpha_2 * ... alpha_i
        '''
        # torch.linspace(min, max, number of steps), linsapce 生成数组
        # device: 张量存在 device 上并在这里操作，如 cuda 或者 CPU
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1 - betas # 这个是根据数学推导来的，自己去看原文
        alpha_bars = torch.empty_like(alphas) # 初值 empty， 格式和 alphas 一样
        product = 1
        # enumerate 是一个内置函数，用于从可迭代对象（如列表、元组、张量等）中获取 index 和 value
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product # [alpha_i^, i from 1 to t], recall:torch.arange(start=0, end=None, step=1, dtype=None, device=None)
        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars
    
    '''
    下面这一大段加起来才叫做前向传播! 这个forward 意思是正向过程而不是前向传播
    前向过程是把整个流程走完一遍(这里指:加噪,去噪)
    例：在 VAE 中 forward 函数中走完了 encoder(正向过程) 和 decoder(反向过程) 的全过程
    '''
    # 正向过程，加噪  
    def sample_forward(self, x_0, t, epsilon = None):
        # .reshape(-1, 1, 1, 1)：将提取的张量重塑为四维张量。-1 表示自动计算维度，以确保总元素数量不变
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        # epsilon 根据文章定义应该服从正态分布 randn，一般是随机取的
        if epsilon is None:
            # _like() 表示和()具有相同格式， 如上面的 alphas_bars = torch.empty_like(alphas)
            epsilon = torch.randn_like(x)
        x_t = epsilon * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x_0
        return x_t
    
     # 反向过程，去噪   
     
    def sample_backward(self, img_shape, net, device, simple_var=True):
        x = torch.randn(img_shape).to(device)
        net = net.to(device)
        # 这是一段递归
        # 即从 t-1 到 0
        for t in range(self.n_steps - 1, -1, -1): 
            x = self.sample_backward_step(x, t, net, simple_var)
        return x
    
    # 求的是 x_t-1
    def sample_backward_step(self, x_t, t, net, simple_var=True):
    # net：一个神经网络，用于预测噪声或其他相关信息。
    # simple_var（可选）：一个布尔值，指示是否使用简单的方差计算
        # x_t 是代表整个图像 dataset 的特征向量，例如可能分为(N, C, H, W) shape[0]一般代表样本总量
        n = x_t.shape[0]
        # [t]*n = [t, t, t, ..t] n个t
        # unsqueeze(1)：在第一个维度插入一个新维度，变成形状为 (n, 1) 的张量，用于与net的输入形状匹配
        t_tensor = torch.tensor([t] * n, dtype=torch.long).to(x_t.device).unsqueeze(1) 
        # 弄成[t]*n是因为 x_t对应 n 个样本量, eps 中每一个 slot 都不一样，每一个t_tensor 中的量对应一个x_t中的样本
        eps = net(x_t, t_tensor)
        # var 和 mean 的计算方法 blog 里面也写了，用到贝叶斯条件概率
        if t == 0:
            noise = 0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alpha_bars[t - 1]) / (
                    1 - self.alpha_bars[t]) * self.betas[t]
            noise = torch.randn_like(x_t)
            noise *= torch.sqrt(var)

        mean = (x_t -(1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) *eps) / torch.sqrt(self.alphas[t])
        x_t_1 = mean + noise

        return x_t_1