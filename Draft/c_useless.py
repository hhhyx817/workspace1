import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ------------------------------------------------------------------------------------------------------------------
# 通过 cal_loader 确定阈值 c, (建立在没有distribution shift 或者 concept drift 的基础上)
# class LossOptimizer(nn.Module) def forward(self): return abs(0.9 - precision)
# loss_optimizer: return gamma
# c: return c
# ------------------------------------------------------------------------------------------------------------------

class LossOptimizer(nn.Module):
    def __init__(self, X_cal, Y_cal, gamma, aimed_precision):
        super().__init__()
        '''
        如果 X_cal, Y_cal 本来就是 Tensor, 它们本身可能有 .requires_grad=True
        torch.tensor() 会把它当作 数据值 拷贝进去 --> 但这会阻断梯度
        wrong: self.X_cal = torch.tensor(X_cal, dtype=torch.float32)  # (size, H, W)  probability of positivity
               self.Y_cal = torch.tensor(Y_cal, dtype=torch.float32)  # (size, H, W)  groundtruth
        '''
        
        self.X_cal = X_cal.clone().detach() # clone: 像 java 的 object 一样, 即使一开始阻断了, 之后还可能被 in-place 操作影响
        self.Y_cal = Y_cal.clone().detach()
        self.aimed_precision = aimed_precision
        self.gamma = nn.Parameter(torch.tensor(gamma))    

    # __call__
    def forward(self):
        eps = 1e-8
        total_FP = 0.
        total_TP = 0.
        total_FN = 0.

        for x, y in zip(self.X_cal, self.Y_cal):  
            pred_soft = torch.sigmoid((x - self.gamma) * 100)  # x - self.gamma >0 近似1 (为了求导所以要连续化)

            true_pos_mask = y == 1
            true_neg_mask = y == 0

            # ==== Ground truth masks ====
            true_pos_mask = (y == 1).float()  # GT=1, 正类 mask
            true_neg_mask = (y == 0).float()  # GT=0, 负类 mask

            TP = torch.sum(pred_soft * true_pos_mask)           # 预测为正 & GT=1（True Positive）用矩阵乘积的方式
            FP = torch.sum(pred_soft * true_neg_mask)           # 预测为正 & GT=0（False Positive）
            FN = torch.sum((1 - pred_soft) * true_pos_mask)     # 预测为负 & GT=1（False Negative）

            total_TP += TP
            total_FP += FP
            total_FN += FN

        # ---- Precision & recall
        precision = total_TP / (total_TP + total_FP + eps)
        print(f"precison: {precision:.2f}")
        L_precision = (self.aimed_precision - precision)**2  # abs 没法求导, 0 附近不合适
    
        return L_precision
        
    
 
#  LossOptimize gamma and lambda(via log_lambda), Gradient-based End-to-End Optimization
def loss_optimizer(X_cal, Y_cal, aimed_precision, init_gamma=0.8, epochs=500, lr=0.01):
    
    model = LossOptimizer(X_cal, Y_cal, init_gamma, aimed_precision)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()  # delete the past gradient
        loss = model()         # automatically use forward(), but not other functions  (the __call__ method heritated in nn.module)
        loss.backward()        # calculate partial derivative
        optimizer.step()       
        
        with torch.no_grad():
            model.gamma.clamp_(0.01, 0.999)
           
        if epoch % 10 == 0 or epoch == 99:
            print(f"[Epoch {epoch}] Loss = {loss.item():.4f} | gamma = {model.gamma.item():.4f} ")
    
    return model.gamma.item()
   

def c(cal_loader, aimed_precision = 0.9):
    init_c = 0.8
    X_all = []
    Y_all = []
    for batch in cal_loader:                  
        X_all.append(batch["mask_binary"])     # probability of positivity
        Y_all.append(batch["mask"])            # groudtruth 0/1
        
    X_tensor = torch.cat(X_all, dim=0)  # X_tensor: size_of_full_set(sum of batch_size) *H*W
    Y_tensor = torch.cat(Y_all, dim=0)  # X_all contains matrices with size [batch_size,H,W], concat along dim[0] = batch
    
    optimized_c \
            = loss_optimizer(X_tensor, Y_tensor, aimed_precision, init_c, epochs=500, lr=0.01)
            
    return optimized_c
         
    
