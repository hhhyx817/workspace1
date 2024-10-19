import torch

# 这是个化简版本， 基于 Denoising Diffusion Probabilistic Models 论文的复现
class DDPM():

    def __init__(self, device, n_steps: int, min_beta: float = 0.0001, max_beta: float = 0.02):
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def sample_forward(self, x_0, t, epsilon = None):
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        if epsilon is None:
            epsilon = torch.randn_like(x)
        x_t = epsilon * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x_0
        return x_t

    def sample_backward(self, img_shape, net, device, simple_var=True):
        x = torch.randn(img_shape).to(device)
        net = net.to(device)
        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_step(x, t, net, simple_var)
        return x

    def sample_backward_step(self, x_t, t, net, simple_var=True):
        n = x_t.shape[0]
        t_tensor = torch.tensor([t] * n, dtype=torch.long).to(x_t.device).unsqueeze(1)
        eps = net(x_t, t_tensor)
        # var 和 mean 的计算方法 blog 里面也写了具体表达式，用到贝叶斯条件概率
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

