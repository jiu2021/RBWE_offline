# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import math
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
import pickle
import json
import onnxruntime as ort
from module import MultiOutputDecoder, StateEncoder

TensorBatch = List[torch.Tensor]
pickle_path = '/data2/kj/Schaferct/training_dataset_pickle/training_dataset_pickle_K_1800_wo_v5.pickle'
# pickle_path = '/data2/kj/Schaferct/training_dataset_pickle/new_training_dataset_pickle.pickle'
# evaluation_dataset_path = '/data2/kj/Schaferct/ALLdatasets/evaluate'
evaluation_dataset_path = '/data2/kj/Schaferct/ALLdatasets/emulated_dataset_policy'
ENUM = 20  # every 5 evaluation set
small_evaluation_datasets = []
policy_dir_names = os.listdir(evaluation_dataset_path)
for p_t in policy_dir_names:
    policy_type_dir = os.path.join(evaluation_dataset_path, p_t)
    if os.path.isdir(policy_type_dir):
        for e_f_name in os.listdir(policy_type_dir)[:ENUM]:
            e_f_path = os.path.join(policy_type_dir, e_f_name)
            small_evaluation_datasets.append(e_f_path)

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
USE_WANDB = 1
b_in_Mb = 1e6

MAX_ACTION = 20  # Mbps
STATE_DIM = 150
ACTION_DIM = 1

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0
DEVICE = "cuda:3"
ln_001 = torch.log(torch.tensor(0.01, dtype=torch.float64))
ln_100 = torch.log(torch.tensor(100.0, dtype=torch.float64))
ln_800 = torch.log(torch.tensor(800.0, dtype=torch.float64))

@dataclass
class TrainConfig:
    # Experiment
    device: str = DEVICE
    env: str = "v14"
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = '/data2/kj/Schaferct/code/checkpoints_iql'  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # IQL
    buffer_size: int = 6535343  # Replay buffer size 6_538_000
    batch_size: int = 512  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    iql_deterministic: bool = False  # Use deterministic actor
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate 3e-4
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    # RIQL
    num_critics: int = 10
    quantile: float = 0
    sigma: float = 0.5
    # cvar
    cvar_alpha = 0
    lambda_risk = 0
    # params
    slope: str = ""
    # dataset
    dataset: str = "v0_v1_v3_v4_v2_v5 1/1 "
    # Wandb logging
    project: str = "BWE"
    group: str = "IQL"
    name: str = f"riql-new_act-beta_{beta}-quantile_{quantile}-sigma_{sigma}-K1800-few_state-act_80-r-delay_0.5-jitter_0.3-gmm_4-argmax-emulated"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        # self._states = torch.zeros(
        #     (buffer_size, state_dim), dtype=torch.float32, device="cpu"
        # )
        # self._actions = torch.zeros(
        #     (buffer_size, action_dim), dtype=torch.float32, device="cpu"
        # )
        # self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device="cpu")
        # self._next_states = torch.zeros(
        #     (buffer_size, state_dim), dtype=torch.float32, device="cpu"
        # )
        # self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device="cpu")
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device="cpu")

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_dataset(self, data: Dict[str, np.ndarray], sample_num: int = 2000000, policy_id: int = 0):
        # if self._size != 0:
        #     raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        print(f"Dataset size: {n_transitions}")
        # if n_transitions > self._buffer_size:
        #     raise ValueError(
        #         "Replay buffer is smaller than the dataset you are trying to load!"
        #     )
        # sample_num = 2000000
        new_act = torch.clamp(self._to_tensor(data["actions"][:sample_num]), min=-6.0, max=6.0)
        # new_act = torch.log(new_act)
        # new_act = (torch.log(new_act) + ln_100) * 2.0 / ln_800 - 1.0
        if self._size == 0:
            self._states = self._to_tensor(data["observations"][:sample_num])
            self._actions = new_act
            self._rewards = self._to_tensor(data["rewards"][:sample_num][..., None])
            self._next_states = self._to_tensor(data["next_observations"][:sample_num])
            self._dones = self._to_tensor(data["terminals"][:sample_num][..., None])
            # self._size += n_transitions
            self._size += sample_num
        else:
            self._states = torch.cat((self._states, self._to_tensor(data["observations"][:sample_num])), 0)
            self._actions = torch.cat((self._actions, new_act), 0)
            self._rewards = torch.cat((self._rewards, self._to_tensor(data["rewards"][:sample_num][..., None])), 0)
            self._next_states = torch.cat((self._next_states, self._to_tensor(data["next_observations"][:sample_num])), 0)
            self._dones = torch.cat((self._dones, self._to_tensor(data["terminals"][:sample_num][..., None])), 0)
            self._size += sample_num
        # if policy_id in [2, 3, 4]:
        #     self._actions = torch.where(self._actions > 0, self._actions * 0.8, self._actions)
        # self._pointer = min(self._size, n_transitions)
        self._pointer = self._size

    def sample(self, batch_size: int) -> TensorBatch:
        # 从0到self._size和self._pointer之间的最小值中随机生成batch_size个整数
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        # 将indices转换为列表
        # indices = indices.tolist()
        # 从self._states中获取indices对应的元素
        states = self._states[indices]
        # 从self._actions中获取indices对应的元素
        actions = self._actions[indices]
        # 从self._rewards中获取indices对应的元素
        rewards = self._rewards[indices]
        # 从self._next_states中获取indices对应的元素
        next_states = self._next_states[indices]
        # 从self._dones中获取indices对应的元素
        dones = self._dones[indices]

        last_loss = states[:, 105:106]
        last_jitter = states[:, 85:86]
        last_jitter = torch.clamp(last_jitter - 5.0, min=0, max=30.0)
        jitter_penalty = torch.exp(last_jitter / 30.0 * torch.log(torch.tensor(5.0))) / 5.0

        last_delay = states[:, 35:36]
        last_delay = torch.clamp(last_delay, min=0, max=467.5)
        # 定义分段函数的系数
        delay_coeff = torch.zeros_like(last_delay)  # 初始化系数为0
        delay_coeff = torch.where(last_delay < 30, last_delay / 200.0, delay_coeff)
        delay_coeff = torch.where((last_delay >= 30) & (last_delay < 80), last_delay / 100.0 + 0.15, delay_coeff)
        delay_coeff = torch.where(last_delay >= 80, last_delay / 50.0 + 0.65, delay_coeff)
        delay_penalty = delay_coeff / 10.0

        rewards = (rewards / 10.0 - last_loss - 0.3 * jitter_penalty - 0.5 * delay_penalty) * 10.0

        # 修改动作
        actions = torch.where(actions > 0, actions * 0.8, actions)
        # actions = torch.where(actions < 0, actions * 0.95, actions)

        # 将states增加一个维度
        # states = torch.unsqueeze(states, 0)
        # 返回states, actions, rewards, next_states, dones
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError
    

def compute_slope(sample):
    # 时间序列：[0, 1, 2, 3, 4]
    x = torch.arange(5, dtype=torch.float32)
    
    # 使用最小二乘法计算斜率
    # y = sample (样本值)
    # 斜率计算公式：slope = (n * Σ(xy) - Σx * Σy) / (n * Σ(x^2) - (Σx)^2)
    n = sample.size(0)
    xy_sum = torch.sum(x * sample)
    x_sum = torch.sum(x)
    y_sum = torch.sum(sample)
    x_square_sum = torch.sum(x**2)
    
    slope = - (n * xy_sum - x_sum * y_sum) / (n * x_square_sum - x_sum**2)
    
    return slope


def set_seed(
    seed: int, deterministic_torch: bool = False
):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()

def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

def huber_loss(diff, sigma=1):
    beta = 1. / (sigma ** 2)
    diff = torch.abs(diff)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss

def inverse_yeo_johnson_transform(y):
    """
    使用指定的λ参数对数据y进行Yeo-Johnson逆变换。
    
    参数:
      y: 转换后的数据（numpy数组）
      lam: 指定的λ参数
    返回:
      原始数据
    """
    y = np.array(y)
    inverse_transformed = np.empty_like(y)
    
    # 对于y >= 0
    pos_mask = y >= 0
    inverse_transformed[pos_mask] = np.exp(y[pos_mask]) - 1
    
    # 对于y < 0
    neg_mask = y < 0
    inverse_transformed[neg_mask] = -np.exp(-y[neg_mask]) + 1
    
    return inverse_transformed

class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)

class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResidualBlock, self).__init__()
        self.output_size = output_size
        self.block = nn.Sequential(
            nn.Linear(input_size, self.output_size),
            nn.LayerNorm(self.output_size),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out
    

class GaussianPolicy_NewAct(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action
        self.encoder0 = nn.Parameter(torch.tensor([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 
                                #  1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                #  1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                #  1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                #  1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1
                                 ], dtype=torch.float32))
        self.encoder0.requires_grad_(False)

        # encoder 1
        self.encoder1 = nn.Sequential(
            # encoder 1
            nn.Linear(150, 256),
            # nn.LayerNorm(256),
            nn.ReLU()
        )
        # GRU
        self.gru = nn.GRU(256, 256, 2)
        # FC
        self.fc_mid = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        # Recisual Block 1(rb1)
        self.rb1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU()
        )
        # Recisual Block 2(rb2)
        self.rb2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU()
        )
        # final 'gmm'
        self.final = nn.Sequential(
            nn.Linear(256, 1),
            nn.Tanh()
        )

        self.rb1_std = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU()
        )
        self.rb2_std = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU()
        )
        self.final2 = nn.Sequential(
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, obs: torch.Tensor, h, c):
        obs_ = torch.squeeze(obs, 0)
        obs_ = obs_ * self.encoder0

        x = self.encoder1(obs_)
        x, _ = self.gru(x)
        x = self.fc_mid(x)
        
        # 均值分支
        mean_branch = x
        mem1_mean = mean_branch
        mean_branch = self.rb1(mean_branch) + mem1_mean
        mem2_mean = mean_branch
        mean_branch = self.rb2(mean_branch) + mem2_mean
        mean = self.final(mean_branch)
        mean = mean * 6.0  # Mbps -> bps
        # mean = mean.clamp(min=0)
        
        # 标准差分支（独立残差层）
        std_branch = x
        mem1_std = std_branch
        std_branch = self.rb1_std(std_branch) + mem1_std
        mem2_std = std_branch
        std_branch = self.rb2_std(std_branch) + mem2_std
        std = self.final2(std_branch)
        std = std * 5.0
        std = torch.exp(std.clamp(min=LOG_STD_MIN, max=LOG_STD_MAX))

        # std = std.expand(mean.shape[0], 1)
        ret = torch.cat((mean, std), 1)
        ret = torch.unsqueeze(ret, 0)  # (1, bs, 2)
        return ret, h, c
    

class GMMPolicy_NewAct(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        num_components: int = 4,  # 混合分量数
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: float = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_components = num_components
        self.act_dim = act_dim
        self.max_action = max_action

        # 固定的 encoder0 参数（与你原来一样）
        self.encoder0 = nn.Parameter(torch.tensor([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1
                                 ], dtype=torch.float32))
        self.encoder0.requires_grad_(False)

        # 公共编码部分
        self.encoder1 = nn.Sequential(
            nn.Linear(150, hidden_dim),
            nn.ReLU()
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, 2)
        self.fc_mid = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # --------------------------
        # 均值分支：输出维度为 act_dim * num_components
        self.rb1_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.rb2_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.final_mean = nn.Sequential(
            nn.Linear(hidden_dim, act_dim * num_components),
            nn.Tanh()
        )
        
        # 标准差分支：输出维度为 act_dim * num_components
        self.rb1_std = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.rb2_std = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.final_std = nn.Sequential(
            nn.Linear(hidden_dim, act_dim * num_components),
            nn.Tanh()
        )
        
        # 混合权重分支：输出 num_components 个分量的权重
        self.pi_layer = nn.Sequential(
            nn.Linear(hidden_dim, num_components)
        )
        
    def forward(self, obs: torch.Tensor, h, c):
        # 对输入做处理：squeeze、乘以 encoder0
        obs_ = torch.squeeze(obs, 0)
        obs_ = obs_ * self.encoder0

        # 公共编码部分
        x = self.encoder1(obs_)
        x, _ = self.gru(x)
        x = self.fc_mid(x)
        
        batch_size = x.shape[0]
        
        # 均值分支
        mean_branch = x
        mem1_mean = mean_branch
        mean_branch = self.rb1_mean(mean_branch) + mem1_mean
        mem2_mean = mean_branch
        mean_branch = self.rb2_mean(mean_branch) + mem2_mean
        mean_out = self.final_mean(mean_branch)
        mean_out = mean_out * 6.0  # 调整尺度（例如 Mbps -> bps）
        # 重塑为 (batch_size, num_components, act_dim)
        mean = mean_out.view(batch_size, self.num_components, self.act_dim)
        
        # 标准差分支
        std_branch = x
        mem1_std = std_branch
        std_branch = self.rb1_std(std_branch) + mem1_std
        mem2_std = std_branch
        std_branch = self.rb2_std(std_branch) + mem2_std
        std_out = self.final_std(std_branch)
        std_out = std_out * 5.0
        std_out = torch.exp(std_out.clamp(min=LOG_STD_MIN, max=LOG_STD_MAX))
        # 重塑为 (batch_size, num_components, act_dim)
        std = std_out.view(batch_size, self.num_components, self.act_dim)
        
        # 混合权重：对每个分量计算权重，并用 softmax 归一化
        pi = self.pi_layer(x)  # (batch_size, num_components)
        pi = F.softmax(pi, dim=-1)

        # 返回一个元组：(mean, std, pi)
        # 如果后续需要构造混合分布，可以根据这三个输出构造相应分布对象
        return (mean, std, pi), h, c


class GMMDistribution:
    def __init__(self, mean, std, pi):
        """
        mean: (batch, K, act_dim) 每个分量的均值
        std:  (batch, K, act_dim) 每个分量的标准差
        pi:   (batch, K) 混合权重（已归一化）
        """
        self.mean = mean
        self.std = std
        self.pi = pi
        self.components = torch.distributions.Normal(mean, std)

    def log_prob(self, x):
        # x: (batch, act_dim)，扩展成 (batch, 1, act_dim)
        x = x.unsqueeze(1)
        # 计算每个分量的 log_prob，形状 (batch, K, act_dim)
        log_probs = self.components.log_prob(x)
        # 对动作维度求和，得到每个分量整体的 log_prob，形状 (batch, K)
        log_probs = log_probs.sum(-1)
        # 加上混合权重（取对数），再利用 logsumexp 合并各分量
        log_pi = torch.log(self.pi + 1e-8)  # 防止 log(0)
        return torch.logsumexp(log_pi + log_probs, dim=1)  # 输出 (batch,)
    
    def sample(self):
        # 先根据 pi 采样分量索引
        mix_idx = torch.distributions.Categorical(self.pi).sample().unsqueeze(-1)  # (batch, 1)
        # 从对应分量采样
        selected_mean = torch.gather(self.mean, 1, mix_idx.unsqueeze(-1).expand(-1, -1, self.mean.size(-1))).squeeze(1)
        selected_std = torch.gather(self.std, 1, mix_idx.unsqueeze(-1).expand(-1, -1, self.std.size(-1))).squeeze(1)
        return torch.distributions.Normal(selected_mean, selected_std).rsample()
    

class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)
        self.encoder = nn.Parameter(torch.tensor([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 
                                 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1
                                 ], dtype=torch.float32))
        self.encoder.requires_grad_(False)

    def both(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state_ = state * self.encoder
        sa = torch.cat([state_, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)
        self.encoder = nn.Parameter(torch.tensor([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 
                                #  1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                #  1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                #  1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                #  1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1
                                 ], dtype=torch.float32))
        self.encoder.requires_grad_(False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state_ = state * self.encoder
        return self.v(state_)
    

class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias


class VectorizedQ(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        num_critics: int = 5,
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden)]
        model = []
        for i in range(len(dims) - 1):
            model.append(VectorizedLinear(dims[i], dims[i + 1], num_critics))
            model.append(nn.ReLU())
        model.append(VectorizedLinear(dims[-1], 1, num_critics))
        self.critic = nn.Sequential(*model)

        # init as in the EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)
        self.num_critics = num_critics
        self.encoder = nn.Parameter(torch.tensor([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 
                                #  1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                #  1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                #  1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                #  1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1
                                 ], dtype=torch.float32))
        self.encoder.requires_grad_(False)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # [batch_size, state_dim + action_dim]
        state = state * self.encoder
        state_action = torch.cat([state, action], dim=-1)
        # [num_critics, batch_size, state_dim + action_dim]
        state_action = state_action.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)
        # [num_critics, batch_size]
        q_values = self.critic(state_action).squeeze(-1)
        return q_values


class ImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            # target_q = self.q_target(observations, actions)
            # RIQL
            target_q_all = self.q_target(observations, actions)
            target_q = torch.quantile(target_q_all.detach(), TrainConfig.quantile, dim=0)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        #  IQL
        # qs = self.qf.both(observations, actions)
        # q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        # log_dict["q_rewards"] = torch.mean(rewards).item()
        # log_dict["q_targets"] = torch.mean(targets).item()
        # log_dict["q_score"] = (torch.mean(qs[0]).item() + torch.mean(qs[1]).item()) / 2
        # log_dict["q_loss"] = q_loss.item()

        #### Huber loss for Q functions
        # target clipping
        qs = self.qf(observations, actions)
        # targets = torch.clamp(targets, -100, 1000).view(1, targets.shape[0])
        q_loss = huber_loss(targets.detach() - qs, sigma=TrainConfig.sigma).mean()

        log_dict["q_targets"] = torch.mean(targets).item()
        log_dict["q_diff"] = torch.mean(targets.detach() - qs).item()
        log_dict["q_loss"] = q_loss.item()
        log_dict['q_score'] = torch.mean(qs).item()

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        out_, _, _ = self.actor(observations, torch.zeros((1, 1)), torch.zeros((1, 1)))
        out_ = torch.squeeze(out_, 0)
        mean = out_[:, :1]
        # mean = mean / 1e6 
        std = out_[:, 1:]
        policy_out = Normal(mean, std)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_all_loss"] = policy_loss.item()
        log_dict["actor_bc_loss"] = torch.mean(bc_losses).item()
        log_dict["actor_expadv"] = torch.mean(exp_adv).item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def _update_policy_GMM(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: dict,
    ):
        # 计算优势加权系数
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        
        # 调用策略网络，返回混合模型的参数
        # 假设 self.actor 返回 ((mean, std, pi), h, c)
        (mean, std, pi), h, c = self.actor(
            observations, 
            torch.zeros((1, 1)), 
            torch.zeros((1, 1))
        )
        
        # 如果需要 squeeze 掉第一个 batch 维度（视你的网络实现而定）
        # 例如：mean = mean.squeeze(0), std = std.squeeze(0), pi = pi.squeeze(0)
        
        # 构造混合高斯分布
        policy_out = GMMDistribution(mean, std, pi)
        
        # 计算动作的 log 概率，注意这里返回的是 (batch,) 的向量
        bc_losses = -policy_out.log_prob(actions)
        
        # 优势加权损失
        policy_loss = torch.mean(exp_adv * bc_losses)
        
        # 记录日志
        log_dict["actor_all_loss"] = policy_loss.item()
        log_dict["actor_bc_loss"] = torch.mean(bc_losses).item()
        log_dict["actor_expadv"] = torch.mean(exp_adv).item()
        
        # 优化更新
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()


    def _update_policy_cvar(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        # 1. 计算加权优势项
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        
        # 2. 前向传播获得策略网络输出，提取均值和标准差
        out_, _, _ = self.actor(observations, torch.zeros((1, 1)), torch.zeros((1, 1)))
        out_ = torch.squeeze(out_, 0)
        mean = out_[:, :1]
        std = out_[:, 1:]
        
        # 3. 构造高斯分布，计算行为克隆损失（BC loss）
        policy_out = Normal(mean, std)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape mismatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError

        # 4. 计算加权损失：乘以 exp_adv 权重
        weighted_losses = exp_adv * bc_losses
        
        # 5. 标准均值损失
        mean_loss = torch.mean(weighted_losses)
        
        cvar_alpha = 0.1
        lambda_risk = 0.5
        # 6. CVaR计算：选取加权损失中最糟糕的样本
        # 计算1-cvar_alpha分位数作为VaR阈值
        threshold = torch.quantile(weighted_losses, 1 - cvar_alpha)
        # 计算CVaR：平均所有大于等于该阈值的损失
        cvar_loss = torch.mean(weighted_losses[weighted_losses >= threshold])
        
        # 7. 综合损失：结合均值损失和CVaR损失
        # lambda_risk取值为0时完全使用均值损失，取值为1时完全使用CVaR损失
        policy_loss = (1 - lambda_risk) * mean_loss + lambda_risk * cvar_loss
        
        # 记录损失日志
        log_dict["actor_all_loss"] = policy_loss.item()
        # 可选记录均值损失和CVaR损失，便于调试与分析
        log_dict["actor_mean_loss"] = mean_loss.item()
        log_dict["actor_cvar_loss"] = cvar_loss.item()
        
        # 8. 反向传播和参数更新
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()


    def _update_policy_cvar1(
            self,
            adv: torch.Tensor,
            observations: torch.Tensor,
            actions: torch.Tensor,
            log_dict: Dict,
        ):
            # 原始IQL损失计算
            exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
            out_, _, _ = self.actor(observations, torch.zeros((1, 1)), torch.zeros((1, 1)))
            out_ = torch.squeeze(out_, 0)
            mean = out_[:, :1]
            std = out_[:, 1:]
            policy_out = Normal(mean, std)
            
            # 计算行为克隆损失
            if isinstance(policy_out, torch.distributions.Distribution):
                bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
            elif torch.is_tensor(policy_out):
                bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
            else:
                raise NotImplementedError
            
            # 原始IQL损失
            loss_iql = exp_adv * bc_losses
            policy_loss_iql = torch.mean(loss_iql)

            # CVaR损失计算 --------------------------------------------
            alpha = 0.1  # 需在类初始化中添加为超参数
            lambda_ = 0.9  # 混合系数，需在类初始化中添加
            
            # 计算风险价值 VaR_alpha (alpha分位数)
            var_alpha = torch.quantile(adv.detach(), alpha)
            
            # 构建掩码选择低优势样本
            # mask = (adv.detach() <= var_alpha).float()

            cvar_loss = torch.mean(loss_iql[adv.detach() <= var_alpha])
            # # CVaR损失：仅优化最差alpha比例样本
            # if torch.sum(mask) > 0:
            #     cvar_loss = torch.sum(mask * bc_losses * exp_adv) / torch.sum(mask)
            # else:
            #     cvar_loss = 0.0  # 无低优势样本时退化为0
            
            # 混合损失目标
            total_loss = lambda_ * cvar_loss + (1 - lambda_) * policy_loss_iql
            # ----------------------------------------------------------
            
            # 记录日志
            log_dict["actor_all_loss"] = total_loss.item()
            log_dict["actor_cvar_loss"] = cvar_loss.item()
            log_dict["actor_mean_loss"] = policy_loss_iql.item()

            # 反向传播
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.actor_lr_schedule.step()


    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        # next state's score
        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        # self._update_policy(adv, observations, actions, log_dict)
        # self._update_policy_cvar1(adv, observations, actions, log_dict)
        # self._update_policy_quantile(adv, observations, actions, log_dict, quantile_tau=0.25)
        self._update_policy_GMM(adv, observations, actions, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]


def get_input_from_file():
    # dummy -> real input
    # evaluation_file = '/data2/kj/Schaferct/evaluation/data/02560.json'
    evaluation_file = '/data2/kj/Schaferct/data/02560.json'
    with open(evaluation_file, "r") as file:
        call_data = json.load(file)
    observations = np.asarray(call_data['observations'], dtype=np.float32)
    observations = observations.reshape(1, -1, STATE_DIM)
    return observations


def export2onnx(pt_path, onnx_path):
    """
    trans pt to onnx
    """
    BS = 1  # batch size
    hidden_size = 1  # number of hidden units in the LSTM

    # instantiate the ML BW estimator
    # torchBwModel = GaussianPolicy(STATE_DIM, ACTION_DIM, MAX_ACTION)
    # torchBwModel = MultiPolicy(STATE_DIM, ACTION_DIM, MAX_ACTION)
    # torchBwModel = GaussianPolicy_NewAct(STATE_DIM, ACTION_DIM, MAX_ACTION)
    torchBwModel = GMMPolicy_NewAct(STATE_DIM, ACTION_DIM, MAX_ACTION)
    torchBwModel.load_state_dict(torch.load(pt_path))
    # create inputs: 1 episode x T timesteps x obs_dim features
    dummy_inputs = get_input_from_file()
    torch_dummy_inputs = torch.as_tensor(dummy_inputs)
    torch_initial_hidden_state = torch.zeros((BS, hidden_size))
    torch_initial_cell_state = torch.zeros((BS, hidden_size))
    # predict dummy outputs: 1 episode x T timesteps x 2 (mean and std)
    torchBwModel.to("cpu")
    # for policy in torchBwModel.policy_list:
    #     policy.to("cpu")
    dummy_outputs, final_hidden_state, final_cell_state = torchBwModel(torch_dummy_inputs, torch_initial_hidden_state, torch_initial_cell_state)
    # save onnx model
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torchBwModel.eval()
    torch.onnx.export(
        torchBwModel,
        (torch_dummy_inputs[0:1, 0:1, :], torch_initial_hidden_state, torch_initial_cell_state),
        onnx_path,
        opset_version=11,
        input_names=['obs', 'hidden_states', 'cell_states'], # the model's input names
        output_names=['output', 'state_out', 'cell_out'], # the model's output names
    )
    
    # verify torch and onnx models outputs
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    onnx_hidden_state, onnx_cell_state = (np.zeros((1, hidden_size), dtype=np.float32), np.zeros((1, hidden_size), dtype=np.float32))
    torch_hidden_state, torch_cell_state = (torch.as_tensor(onnx_hidden_state), torch.as_tensor(onnx_cell_state))
    # online interaction: step through the environment 1 time step at a time
    with torch.no_grad():
        for i in tqdm(range(dummy_inputs.shape[1]), desc="Verifing  "):
            torch_estimate, torch_hidden_state, torch_cell_state = torchBwModel(torch_dummy_inputs[0:1, i:i+1, :], torch_hidden_state, torch_cell_state)
            feed_dict= {'obs': dummy_inputs[0:1, i:i+1, :], 'hidden_states': onnx_hidden_state, 'cell_states': onnx_cell_state}
            # (mean, std, pi), onnx_hidden_state, onnx_cell_state = ort_session.run(None, feed_dict)
            mean, std, pi, onnx_hidden_state, onnx_cell_state = ort_session.run(None, feed_dict)
        #     assert np.allclose(torch_estimate.numpy(), onnx_estimate, atol=100), 'Failed to match model outputs!, {}, {}'.format(torch_estimate.numpy(), onnx_estimate)
        #     assert np.allclose(torch_hidden_state, onnx_hidden_state, atol=1e-7), 'Failed to match hidden state1'
        #     assert np.allclose(torch_cell_state, onnx_cell_state, atol=1e-7), 'Failed to match cell state!'
        
        # assert np.allclose(torch_hidden_state, final_hidden_state, atol=1e-7), 'Failed to match final hidden state!'
        # assert np.allclose(torch_cell_state, final_cell_state, atol=1e-7), 'Failed to match final cell state!'
        # print("Torch and Onnx models outputs have been verified successfully!")


def evaluate(onnx_path):
    ort_session = ort.InferenceSession(onnx_path)

    every_call_mse = []
    every_call_accuracy = []
    every_call_over_estimated_rate = []
    every_call_under_estimated_rate = []
    for f_path in tqdm(small_evaluation_datasets, desc="Evaluating"):
        with open(f_path, 'r') as file:
            call_data = json.load(file)
        
        observations = np.asarray(call_data['observations'], dtype=np.float32)
        true_capacity = np.asarray(call_data['true_capacity'], dtype=np.float32)

        model_predictions = []
        hidden_state, cell_state = np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)    
        for t in range(observations.shape[0]):
            obss = observations[t:t+1,:].reshape(1,1,-1)
            feed_dict = {'obs': obss,
                        'hidden_states': hidden_state,
                        'cell_states': cell_state
                        }
            # mean, std, pi, hidden_state, cell_state = ort_session.run(None, feed_dict)
            mean, std, pi, _, _ = ort_session.run(None, feed_dict)
            # 对于每个样本，找到权重最大的分支索引
            max_component_indices = np.argmax(pi, axis=1)
            # 利用高级索引选取对应分支的均值
            batch_indices = np.arange(mean.shape[0])
            selected_actions = mean[batch_indices, max_component_indices, :][0,0]
            model_predictions.append(obss[0][0][5] * np.exp(selected_actions))
            # model_predictions.append(obss[0][0][5] + inverse_yeo_johnson_transform(bw_prediction[0,0,0]))
            # model_predictions.append(np.exp(bw_prediction[0,0,0]))
        # mse and accuracy of this call
        model_predictions = np.asarray(model_predictions, dtype=np.float32)
        true_capacity = true_capacity / 1e6
        model_predictions = model_predictions / 1e6
        call_mse = []
        call_accuracy = []
        call_over_estimated_rate = []
        call_under_estimated_rate = []
        for true_bw, pre_bw in zip(true_capacity, model_predictions):
            if np.isnan(true_bw) or np.isnan(pre_bw):
                continue
            else:
                mse_ = (true_bw - pre_bw) ** 2
                call_mse.append(mse_)
                accuracy_ = max(0, 1 - abs(pre_bw - true_bw) / true_bw)
                call_accuracy.append(accuracy_)
                over_estimated = max(0, (pre_bw - true_bw) / true_bw)
                call_over_estimated_rate.append(over_estimated)
                under_estimated = max(0, (true_bw - pre_bw) / true_bw)
                call_under_estimated_rate.append(under_estimated)
        call_mse = np.asarray(call_mse, dtype=np.float32)
        every_call_mse.append(np.mean(call_mse))
        call_accuracy = np.asarray(call_accuracy,  dtype=np.float32)
        every_call_accuracy.append(np.mean(call_accuracy))
        call_over_estimated_rate = np.asarray(call_over_estimated_rate, dtype=np.float32)
        every_call_over_estimated_rate.append(np.mean(call_over_estimated_rate))
        call_under_estimated_rate = np.asarray(call_under_estimated_rate, dtype=np.float32)
        every_call_under_estimated_rate.append(np.mean(call_under_estimated_rate))
    every_call_mse = np.asarray(every_call_mse, dtype=np.float32)
    every_call_accuracy = np.asarray(every_call_accuracy, dtype=np.float32)
    every_call_over_estimated_rate = np.asarray(every_call_over_estimated_rate, dtype=np.float32)
    every_call_under_estimated_rate = np.asarray(every_call_under_estimated_rate, dtype=np.float32)
    return np.mean(every_call_mse), np.mean(every_call_accuracy), np.mean(every_call_over_estimated_rate), np.mean(every_call_under_estimated_rate)
    

@pyrallis.wrap()
def train(config: TrainConfig):
    state_dim = STATE_DIM
    action_dim = ACTION_DIM

    # testdataset_file = open(pickle_path, 'rb')
    # dataset = pickle.load(testdataset_file)
    # print('dataset loaded')

    # replay_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)
    # replay_buffer.load_dataset(dataset)

    replay_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)
    for i in range(6):
        print(f'policy_v{i} dataset loaded')
        dataset = pickle.load(open(f"/data2/kj/Schaferct/training_dataset_pickle/training_dataset_K_1000_v{i}_emulated.pickle", 'rb'))
        # if i == 2:
        #     replay_buffer.load_dataset(dataset, sample_num=int(dataset["observations"].shape[0] / 5), policy_id=i)
        # elif i == 5:
        #     replay_buffer.load_dataset(dataset, sample_num=int(dataset["observations"].shape[0] / 5), policy_id=i)
        # else:
        replay_buffer.load_dataset(dataset, sample_num=int(dataset["observations"].shape[0]), policy_id=i)

    max_action = MAX_ACTION

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed)

    # q_network = TwinQ(state_dim, action_dim).to(config.device)
    q_network = VectorizedQ(state_dim, action_dim, num_critics=TrainConfig.num_critics).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    # actor = GaussianPolicy_NewAct(state_dim, action_dim, max_action, dropout=config.actor_dropout).to(config.device)
    actor = GMMPolicy_NewAct(state_dim, action_dim, max_action, dropout=config.actor_dropout).to(config.device)
    # state_dict = torch.load("/data2/kj/SRPO/Encoder_model/large_2M/encoder_225.pth")
    # Rename keys as needed
    # new_state_dict = {}
    # for key, value in state_dict.items():
    #     if 'encoder' in key:
    #         new_key = key.replace('encoder.', '')
    #         new_state_dict[new_key] = value
    # q_network.encoder.load_state_dict(new_state_dict)
    # v_network.encoder.load_state_dict(new_state_dict)
    # actor.encoder.load_state_dict(new_state_dict)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
    }

    print("---------------------------------------")
    print(f"Training IQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ImplicitQLearning(**kwargs)

    # config.load_model = "/data2/kj/Schaferct/code/checkpoints_iql/riql-new_act-beta_3.0-quantile_0-sigma_0.5-K1800-few_state-act_80-delay-jitter_nolinear-gmm_4-argmax-v14-b312c7e0/all_checkpoint_1000000.pt"
    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    if USE_WANDB:
        wandb_init(asdict(config))

    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        if USE_WANDB:
            wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            
            pt_path = os.path.join(config.checkpoints_path, f"actor_checkpoint_{t + 1}.pt")
            onnx_path = os.path.join(config.checkpoints_path, f"actor_checkpoint_{t + 1}.onnx")
            all_pt_path = os.path.join(config.checkpoints_path, f"all_checkpoint_{t + 1}.pt")
            # save pt
            if config.checkpoints_path is not None:
                torch.save(trainer.state_dict()["actor"], pt_path)
                torch.save(trainer.state_dict(), all_pt_path)
            # save onnx
            export2onnx(pt_path, onnx_path)
            # evaluate
            mse_, accuracy_, over_estimated_rate, under_estimated_rate = evaluate(onnx_path)
            if USE_WANDB and trainer.total_it > 1000:
                wandb.log({"mse": mse_, "error_rate": 1 - accuracy_, "over-estimated_rate": over_estimated_rate, "under-estimated_rate": under_estimated_rate}, step=trainer.total_it)
                print({"mse": mse_, "error_rate": 1 - accuracy_, "over-estimated_rate": over_estimated_rate, "under-estimated_rate": under_estimated_rate})


if __name__ == "__main__":
    train()
