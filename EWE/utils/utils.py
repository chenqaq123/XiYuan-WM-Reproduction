import torch
import numpy as np
import random
import torch.nn.functional as F

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # 用于保证CUDA 卷积运算的结果确定
    torch.backends.cudnn.benchmark = False      # 用于保证数据变化的情况下，减少网络效率的变化

def save_model(model, saved_name, save_dir):
    torch.save(model.state_dict(), f'{save_dir}/{saved_name}.pth')

def load_model(model, model_name, save_dir):
    model.load_state_dict(torch.load(f'{save_dir}/{model_name}.pth'))

def pairwise_euclid_distance(A):
    # 计算每个样本的平方范数
    sqr_norm_A = torch.sum(A**2, dim=1, keepdim=True)
    # 计算内积
    inner_prod = torch.mm(A, A.t())
    # 计算平方范数的矩阵形式
    tile_1 = sqr_norm_A.expand(A.size(0), A.size(0))
    tile_2 = sqr_norm_A.expand(A.size(0), A.size(0)).t()
    # 计算欧几里得距离矩阵
    return tile_1 + tile_2 - 2 * inner_prod

def pairwise_cos_distance(A):
    # 进行 L2 归一化
    normalized_A = torch.nn.functional.normalize(A, p=2, dim=1)
    # 计算余弦相似度矩阵
    cosine_sim = torch.mm(normalized_A, normalized_A.t())
    # 计算余弦距离矩阵
    return 1 - cosine_sim

def snnl_single(x, y, t, metric='euclidean'):
    # ReLU 激活
    x = F.relu(x)
    # 计算相同标签的掩码
    same_label_mask = (y.unsqueeze(0) == y.unsqueeze(1)).float()
    # 计算距离矩阵
    x = x.view(x.size(0), -1)  # 确保 x 是二维的
    if metric == 'euclidean':
        dist = pairwise_euclid_distance(x)
    elif metric == 'cosine':
        dist = pairwise_cos_distance(x)
    else:
        raise NotImplementedError()
    # 计算指数
    exp = torch.clamp(torch.exp(-dist / t) - torch.eye(x.size(0), device=x.device), 0, 1)
    # 计算概率
    prob = exp / (0.00001 + exp.sum(dim=1, keepdim=True))
    prob = prob.to('cpu')
    same_label_mask = same_label_mask.to('cpu')
    prob = prob * same_label_mask
    # 计算损失
    loss = -torch.mean(torch.log(0.00001 + prob.sum(dim=1)))
    return loss