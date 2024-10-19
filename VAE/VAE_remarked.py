# standard library module to perform operating-system-related tasks such as managing files, directories, environment variables, path operations
import os 
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from time import time
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from dldemos.VAE.load_celebA import get_dataloader
from dldemos.VAE.model import VAE

# 区分清楚：class 子类(父类)，def 函数(参数) ，这里是继承pytorch中的父类 Dataset 并进行改写
class CelebADataset(Dataset):
    # __init__是类的默认构造函数，self必写，root 和 img_shape 是子类的默认必有的参数，之后传参的时候要用的；
    def __init__(self, root, img_shape=(64, 64)) -> None:
        super().__init__()
        self.root = root
        self.img_shape = img_shape
        # 获取根目录 root 下的所有 directory
        self.filenames = sorted(os.listdir(root))

    # 返回文件名的数量，即数据集的大小
    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        # 使用 os.path.join 将根目录和其中获取的 directory name 组合成完整的文件路径。
        path = os.path.join(self.root, self.filenames[index])
        img = Image.open(path).convert('RGB')
        # 图片处理流水线，裁剪、放缩、torch张量
        pipeline = transforms.Compose([
            transforms.CenterCrop(168),
            transforms.Resize(self.img_shape),
            transforms.ToTensor()
        ])
        return pipeline(img)
    
# DataLoader 批量加载数据，设置 epoch 大小为 16，并随机打乱数据（shuffle=True）
# 如果你想表示一个目录，通常在路径末尾加上斜杠是一个好习惯; 如果想表示文件，路径末尾不应加斜杠：    
def get_dataloader(root="/data/home/huangyx/workspace/diffusion_learn/img_align_celeba/", **kwargs):
    # root 和 img_shape 是定义的默认参数，一定要有的；img_shape已经固定了这里可以不传，root 是一定要传的    
    # **kwargs 说明允许传入其他参数
    # crop, resize 等操作已经在定义类时一同定义了；所以传参传入后，子类定义的过程就会进行一遍；这里输出的 dataset 也是经过子类加工的对象
    dataset = CelebADataset(root, **kwargs)
    return DataLoader(dataset, 16, shuffle=True)


# 为了验证Dataloader的正确性，我们可以写一些脚本来查看Dataloader里的一个batch的图片。(和主任务没关系)
# if 表示以下内容的代码块将仅在脚本作为主程序运行时执行，而在被导入为模块时不会执行。
if __name__ == '__main__':
    dataloader = get_dataloader()
    img = next(iter(dataloader))
    print(img.shape)
    # Concat 4x4 images, number, channel, height, width
    N, C, H, W = img.shape
    # assert 是一个检验程序，检查 N == 16 的正确性。否则程序将抛出 AssertionError，确保数据加载器正确返回了预期的批量大小。
    assert N == 16
    
    '''
    [集中报错情况解析]:
    
    img0 = transforms.ToPILImage()(img)
    img0.save('/data/home/huangyx/workspace/diffusion_learn/tmp.jpg')
    ToPILImage 只接受的形状是 2D(例如：(H, W))或 3D 如:((C, H, W)）的张量。
    如果想看单张图片,可以: img0 = transforms.ToPILImage()(img[0])  
    
    img = torch.reshape(img, (N/16, C, 4 * H, 4 * W)) 不对的
    N 是批量大小,N/16 实际上并不能表示出每个新组的数量; 把图片分割成4 * 4就是分了4组; 如果分成 3*4 可以认为是3组or 4组, 看你一开始是对W or H操作了
    '''
    
    '''
    img = torch.reshape(img, (C, 4 * H, 4 * W)) # N 不写出来就是直接不要了; 这步将每组的宽度合并到一起，形成一个新的宽度。
    # 这个permute不可缺少! 
    
    reshape从高维变成低维的时候, 会自动执行合并
    例如: 当对 (C, N/4, 4 * H, W) 执行 reshape(C, 4 * H, 4 * W) 时，它会按照现有顺序，从最右边的维度开始依次合并维度。
    且: 如果最右边的一组无法合并, 就直接报错, 不会考虑向左顺延是否会合并
    
    所以reshape 的第一步是把 最右边的两个维度（即 N/4 和 W)合并, 生成新维度 4 * W。
    这个操作的原因在于,内存中的数据是按照批次维度 N/4 和每个图像的宽度 W 来排列的    
      
    如果没有 permute:
    4*H 会和 4* W合并, 几何意义上解释不通，直接报错
    
    '''
    
    # 使用 torch.permute 重新排列张量的维度，使张量的形状变为 (C, N, H, W)。是为了更方便(即不是必须的)后续拼接、伸缩等操作
    img = torch.permute(img, (1, 0, 2, 3))
    img = torch.reshape(img, (C, N / 4, 4 * H, W)) # "img内容即16张图不变", N / 4 表示将N分成4组，4*H 表示将分组后的图片在 height 上堆叠
    img = torch.permute(img, (0, 2, 1, 3))  #（C,4*H,N/4,W）
    img = torch.reshape(img, (C, 4 * H, 4 * W)) # N 不写出来就是直接不要了; 这步将每组的宽度合并到一起，形成一个新的宽度。
    # 最终得到的 img 形状为 (C, 4 * H, 4 * W)，这是一个包含 16 张图像的拼接图像：高度变为 4 * H，表示 4 张图像的高度叠加; 宽度变为 4 * W，表示 4 张图像的宽度并排。
    img = transforms.ToPILImage()(img) # PIL类图像 才有save方法
    img.save('/data/home/huangyx/workspace/diffusion_learn/tmp.jpg')


'''
VAE for 64x64 face generation. The hidden dimensions can be tuned.
这段代码定义了一个用于生成 64x64 人脸图像的变分自编码器(VAE)。它包含一个编码器和一个解码器,
编码器将输入图像映射到潜在空间,解码器则从潜在空间重构出图像. 每个部分都采用多个卷积(in encoder)和反卷积层(in decoder)来逐步提取特征和生成图像。
'''
# Python 的 class和 java不同，创建的时候只体现 父类关系，定义部分参数，但不传入任何对象！
class VAE(nn.Module):
    # hiddens：一个列表，定义隐藏层的通道数。latent_dim：潜在空间的维度，默认为 128。
    # n.mudule中，分为input、hidden(用于加工处理，获得特征等；如 全连接层，卷积层，循环层，池化层）、activate function、output。 CNN指的是 hidden layer采用是卷积层（池化层）
    # 潜在空间（latent space）就是变分自编码器（VAE）编码器的输出空间, 即将图片 encode 成 vector 的空间
    # hiddens=[16, 32, 64, 128, 526] 表示在编码器中使用五个卷积层，每个层的输出特征数逐渐增加。这种设计通常有助于模型更有效地学习和提取输入数据中的特征。
    # 最终hidden layer输出 256 维特征，经过压缩后压缩成 128 维向量，作为 encoder 的结果
    def __init__(self, hiddens=[16, 32, 64, 128, 256], latent_dim=128) -> None:
        super().__init__()

        # encoder
        prev_channels = 3 # 输入图像为 RGB 图像（3 个通道）。
        modules = [] # 存储 Encoder 的所有卷积模块。
        img_length = 64 # 初始化为 64，表示输入图像的尺寸（64x64）
        
        # 循环创建 Encoder 的卷积层
        # 整个卷积操作分5次进行，每次提取的特征数翻倍
        for cur_channels in hiddens:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels, # prev_channels 是输入通道数（来自上一层或输入)
                              cur_channels, # cur_channels 是输出通道数（当前层的通道数）
                              kernel_size=3,
                              stride=2,
                              padding=1), 
                    nn.BatchNorm2d(cur_channels),
                    nn.ReLU()))
            prev_channels = cur_channels
            img_length //= 2
        # 将所有创建的模块组合成一个顺序容器 self.encoder。
        # * 符号用于解包（unpacking），它可以将列表或元组中的元素拆分并作为单独的参数传递
        self.encoder = nn.Sequential(*modules) # 即 hidden layer 的输出
        # prev_channels * img_length * img_length。这是编码器输出的特征向量，经过展平后得到的。
        '''
        每个线性层的参数（权重和偏置）是独立的
        self.mean_linear是需要用 nn.linear计算的, 对象是 encoder得到的特征向量, 参数未知，之后训练模型的时候会有
        
    for data in dataloader:
        optimizer.zero_grad()
        reconstructed, mean, log_var = model(data)
        loss = compute_loss(reconstructed, data, mean, log_var)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

    print("Mean Linear Weights:", model.mean_linear.weight.grad)
    print("Variance Linear Weights:", model.var_linear.weight.grad)
        '''
        self.mean_linear = nn.Linear(prev_channels * img_length * img_length, latent_dim)
        self.var_linear = nn.Linear(prev_channels * img_length * img_length, latent_dim)
        self.latent_dim = latent_dim
        
        # decoder
        modules = []
        self.decoder_projection = nn.Linear(
            latent_dim, prev_channels * img_length * img_length)
        self.decoder_input_chw = (prev_channels, img_length, img_length)
        
        for i in range(len(hiddens) - 1, 0, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hiddens[i],
                                       hiddens[i - 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hiddens[i - 1]), nn.ReLU()))
            
        # 尽管这一步将通道数保持不变，但通过增加卷积层，你可以引入非线性特征。即使通道数不变，网络仍然可以学习到更复杂的特征表示。    
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hiddens[0],
                                   hiddens[0],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hiddens[0]), nn.ReLU(),
                nn.Conv2d(hiddens[0], 3, kernel_size=3, stride=1, padding=1),
                nn.ReLU()))
        # 依次储存所有的神经网络层
        self.decoder = nn.Sequential(*modules) 
        


'''
实现了变分自编码器的前向传播过程，
通过编码器将输入数据编码为潜在空间的均值和方差，然后使用重参数化技巧生成潜在变量，
并将其解码为重构的输出。最终返回重构结果以及潜在变量的统计信息

前向传播：
输入数据通过网络进行处理，直到生成输出的过程；(即依次经过input layer, hidden layer, activate function, 得到output layer 的过程)
目的：   计算预测：前向传播的主要目的是计算神经网络的输出，以便与真实标签进行比较，从而计算损失。
        为损失计算做准备：前向传播的结果用于后续的学习过程（如反向传播），以更新网络的权重和偏置。
        
pytorch中输入是通过 forward 步骤传入的,面定义class的时候都没有表现输入
'''
def forward(self, x):
    encoded = self.encoder(x)
    encoded = torch.flatten(encoded, 1)
    mean = self.mean_linear(encoded)
    logvar = self.var_linear(encoded)
    eps = torch.randn_like(logvar) # 随机生成图：此处产生噪声
    std = torch.exp(logvar / 2)
    z = eps * std + mean
    z_1 = self.decoder_projection(z) # 维度对应，之前向量伸缩过
    x_1 = torch.reshape(z_1, (-1, *self.decoder_input_chw))
    decoded = self.decoder(x_1)

    return decoded, mean, logvar


def sample(self, device='cuda'):
    z = torch.randn(1, self.latent_dim).to(device)
    x = self.decoder_projection(z)
    x = torch.reshape(x, (-1, *self.decoder_input_chw))
    decoded = self.decoder(x)
    return decoded        


# Hyperparameters
n_epochs = 10
kl_weight = 0.00025
lr = 0.005


def loss_fn(y, y_hat, mean, logvar):
    recons_loss = F.mse_loss(y_hat, y)
    kl_loss = torch.mean(
        -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), 1), 0)
    loss = recons_loss + kl_loss * kl_weight
    return loss


def train(device, dataloader, model):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    dataset_len = len(dataloader.dataset)

    begin_time = time()
    # train
    for i in range(n_epochs):
        loss_sum = 0
        for x in dataloader:
            x = x.to(device)
            y_hat, mean, logvar = model(x)
            loss = loss_fn(x, y_hat, mean, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss
        loss_sum /= dataset_len
        training_time = time() - begin_time
        minute = int(training_time // 60)
        second = int(training_time % 60)
        print(f'epoch {i}: loss {loss_sum} {minute}:{second}')
        torch.save(model.state_dict(), 'dldemos/VAE/model.pth')
        

def reconstruct(device, dataloader, model):
    model.eval()
    batch = next(iter(dataloader))
    x = batch[0:1, ...].to(device)
    output = model(x)[0]
    output = output[0].detach().cpu()
    input = batch[0].detach().cpu()
    combined = torch.cat((output, input), 1)
    img = ToPILImage()(combined)
    img.save('work_dirs/tmp.jpg')
    
    
def generate(device, model):
    model.eval()
    output = model.sample(device)
    output = output[0].detach().cpu()
    img = ToPILImage()(output)
    img.save('work_dirs/tmp.jpg')
