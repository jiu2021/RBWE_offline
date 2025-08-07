from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np
import torch
def negative_gmm_density(a, pi, mu, sigma):
    """
    目标函数：负的混合高斯密度
    """
    density = np.sum(pi * norm.pdf(a, loc=mu, scale=sigma))
    return -density
    

def find_gmm_mode(pi, mu, sigma, initial_guess=None):
    """
    利用数值优化方法求解一维混合高斯分布的众数（mode）。
    
    参数：
        pi: 混合分量权重数组，形状 (K,)
        mu: 各分量的均值数组，形状 (K,)
        sigma: 各分量的标准差数组，形状 (K,)
        initial_guess: 初始猜测值（可选），默认为各分量均值的加权平均
        
    返回：
        mode: 求得的混合分布众数（标量）
    """
    if initial_guess is None:
        initial_guess = np.sum(pi * mu)
    res = minimize(negative_gmm_density, x0=initial_guess, args=(pi, mu, sigma), method='L-BFGS-B', tol=1e-3)
    if res.success:
        return res.x[0]
    else:
        # fallback: 返回占比最大的分支的均值
        idx = np.argmax(pi)
        return mu[idx]
    
def find_gmm_modes_batch(mu_batch, sigma_batch, pi_batch):
    """
    批量求解
    pi_batch:   (batch, K)
    mu_batch:   (batch, K)
    sigma_batch:(batch, K)
    返回: modes (batch,)
    """
    mu_batch, sigma_batch, pi_batch =  mu_batch.detach().cpu().numpy(), sigma_batch.detach().cpu().numpy(), pi_batch.detach().cpu().numpy()
    batch_size = pi_batch.shape[0]
    modes = np.zeros(batch_size)
    for i in range(batch_size):
        pi = pi_batch[i]
        mu = np.squeeze(mu_batch[i])
        sigma = np.squeeze(sigma_batch[i])
        modes[i] = find_gmm_mode(pi, mu, sigma)
    return torch.tensor(modes, dtype=torch.float32).unsqueeze(1)