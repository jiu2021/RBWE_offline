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
from sklearn.linear_model import LinearRegression

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
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
DEVICE = "cuda:1"
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
    # Wandb logging
    project: str = "BWE"
    group: str = "IQL"
    name: str = f"new_act-beta-{beta}-v2&v5_1_5-wo_5-new_reward_nonlinear-few_state-clipped_q"

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
    def load_dataset(self, data: Dict[str, np.ndarray], sample_num: int = 2000000):
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

        last_loss = states[:, 100:101]
        last_delay = states[:, 30:31]
        last_delay = torch.clamp(last_delay, min=0, max=500)

        # last_delay_5 = states[:, 30:35]  # 获取五个值

        # # 拟合last_delay
        # x = torch.arange(5).repeat(512).reshape(512, 5).float()   # x轴为0到4
        # y = last_delay_5.cpu().numpy()  # 将last_delay转换为numpy数组
        # model = LinearRegression().fit(x, y.T)  # 拟合线性回归模型
        # trend = model.coef_.T  # 获取回归系数并转置

        # # 确保trend的形状为(512, 1)
        # trend = trend.reshape(512, 1)

        # 定义分段函数的系数
        delay_coeff = torch.zeros_like(last_delay)  # 初始化系数为0
        # delay_coeff = torch.where((last_delay > 20) & (last_delay <= 50) & (trend < 0), 0.3, delay_coeff)
        # delay_coeff = torch.where((last_delay > 20) & (last_delay <= 50), 0.3, delay_coeff)
        delay_coeff = torch.where((last_delay > 50) & (last_delay <= 100), 0.5, delay_coeff)
        delay_coeff = torch.where((last_delay > 100) & (last_delay <= 200), 0.8, delay_coeff)
        delay_coeff = torch.where(last_delay > 200, 1, delay_coeff)

        # 根据分段函数计算延迟惩罚
        delay_penalty = delay_coeff * (last_delay / 500.0)

        rewards = (rewards / 10.0 - 0.5 * last_loss - delay_penalty) * 10.0
        # rewards = (rewards / 10.0 - 0.5 * last_loss  - 0.3 * (last_delay / 500.0) ) * 10.0

        # 将states增加一个维度
        # states = torch.unsqueeze(states, 0)
        # 返回states, actions, rewards, next_states, dones
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError

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
    
class GaussianPolicy(nn.Module):
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

    def forward(self, obs: torch.Tensor, h, c):
        obs_ = torch.squeeze(obs, 0)
        obs_ = obs_ * self.encoder0
###
        mean = self.encoder1(obs_)
        mean, _ = self.gru(mean)
        mean = self.fc_mid(mean)
        mem1 = mean
        mean = self.rb1(mean) + mem1
        mem2 = mean
        mean = self.rb2(mean) + mem2
        mean = self.final(mean)
###
        mean = mean * self.max_action * 1e6  # Mbps -> bps
        mean = mean.clamp(min = 10)  # larger than 10bps
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        std = std.expand(mean.shape[0], 1)
        ret = torch.cat((mean, std), 1)
        ret = torch.unsqueeze(ret, 0)  # (1, bs, 2)
        return ret, h, c

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

    def forward(self, obs: torch.Tensor, h, c):
        obs_ = torch.squeeze(obs, 0)
        obs_ = obs_ * self.encoder0
###
        mean = self.encoder1(obs_)
        mean, _ = self.gru(mean)
        mean = self.fc_mid(mean)
        mem1 = mean
        mean = self.rb1(mean) + mem1
        mem2 = mean
        mean = self.rb2(mean) + mem2
        mean = self.final(mean)
###       
        mean = mean * 6.0  # Mbps -> bps
        # mean = mean.clamp(min = 10)  # larger than 10bps
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        std = std.expand(mean.shape[0], 1)
        ret = torch.cat((mean, std), 1)
        ret = torch.unsqueeze(ret, 0)  # (1, bs, 2)
        return ret, h, c

class MultiPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        n_policies: int = 6,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.max_action = max_action
        self.n_policies = n_policies
        self.encoder0 = nn.Parameter(torch.tensor([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 
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
        self.encoder0.requires_grad_(False)
        self.fc_head = nn.Linear(state_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.ac1 = nn.LeakyReLU()
        self.policy_list = []
        self.load_policy(state_dim, act_dim, max_action)
        
        # weight
        self.weight_fc_body = nn.Linear(hidden_dim, hidden_dim)
        self.weight_block = ResidualBlock(hidden_dim, hidden_dim)
        self.weight_output = nn.Linear(hidden_dim, n_policies)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, obs: torch.Tensor, h = torch.zeros((1, 1)), c = torch.zeros((1, 1))):
        obs_ = torch.squeeze(obs, 0)
        obs_ = obs_ * self.encoder0

        # 特征提取（前向传递到权重分支）
        x = self.fc_head(obs_)
        x = self.layer_norm(x)
        x = self.ac1(x)

        # 使用权重网络计算权重
        weight_out = self.weight_fc_body(x)
        weight_out = self.ac1(weight_out)
        weight_out = self.weight_block(weight_out)
        weight_out = self.weight_output(weight_out)
        weight_out = self.soft_max(weight_out)  # (batch_size, n_policies)

        # 遍历所有 policies，计算每个 policy 的动作
        policy_outputs = []
        for i, policy in enumerate(self.policy_list):
            policy_output, _, _ = policy(obs, h, c)  # 获取 policy 的动作输出
            mean = policy_output[:, :, 0]  # 提取均值部分 (1, batch_size)
            policy_outputs.append(mean)

        # 堆叠所有 policy 的输出，形状为 (batch_size, n_policies, act_dim)
        policy_outputs = torch.stack(policy_outputs, dim=1).squeeze(0)

        # 对每个动作维度进行加权和，形状为 (batch_size, act_dim) (512,1,1)->(512,1)
        weighted_output = torch.bmm(policy_outputs.view(-1, 1, self.n_policies), weight_out.view(-1, self.n_policies, 1)).squeeze(2) 
        # 动作裁剪到 [-max_action, max_action]
        weighted_output = torch.clamp(weighted_output, 10.0, 8 * 1e6)
    
        ret = torch.cat((weighted_output, weighted_output), 1)
        ret = torch.unsqueeze(ret, 0)  # (1, bs, 2)
        return ret, h, c

    def load_policy(self, state_dim, act_dim, max_action):
        for i in range(self.n_policies):
            policy = GaussianPolicy(state_dim, act_dim, max_action).to(DEVICE)
            policy.load_state_dict(torch.load(f'/data2/kj/Schaferct/code/policy_model/policy_v{i}.pt'))
            # 冻结策略的参数
            # for param in policy.parameters():
            #     param.requires_grad = False
            self.policy_list.append(policy)

class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )
    
class DeterministicPolicy1(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        multiple_output_num = 5,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.encoder = StateEncoder(state_dim, hidden_dim)
        self.decoder = MultiOutputDecoder(hidden_dim, hidden_dim, multiple_output_num)
        self.ac = nn.Tanh()
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, obs: torch.Tensor, h, c) -> torch.Tensor:
        x = self.encoder(obs)
        x = self.decoder(x)
        out = self.ac(x)
        mean = torch.exp(((out + 1.0) / 2 * ln_800 + ln_001)) * 1e6
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        std = std.expand(mean.shape[0], 1)
        ret = torch.cat((mean, std), 1)
        ret = torch.unsqueeze(ret, 0)
        return ret, h, c

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

    def both(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state_ = state * self.encoder
        sa = torch.cat([state_, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))

class TwinQ1(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, multiple_output_num=5, n_hidden: int = 2
    ):
        super().__init__()
        self.encoder = StateEncoder(state_dim, hidden_dim)
        self.q1 = MultiOutputDecoder(hidden_dim + action_dim, hidden_dim, multiple_output_num)
        self.q2 = MultiOutputDecoder(hidden_dim + action_dim, hidden_dim, multiple_output_num)

    def both(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state_ = self.encoder(state)
        sa = torch.cat([state_, action], 1)
        return self.q1(sa).squeeze(), self.q2(sa).squeeze()

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
    
class ValueFunction1(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, multiple_output_num = 5, n_hidden: int = 2):
        super().__init__()
        self.encoder = StateEncoder(state_dim, hidden_dim)
        self.decoder = MultiOutputDecoder(hidden_dim, hidden_dim, multiple_output_num)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.encoder(state)
        x = self.decoder(x)
        return x.squeeze()


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
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_v_clipped(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            original_target_q = self.q_target(observations, actions)  # 保留原始值
            target_q = original_target_q.clone()  # 创建副本用于修改
            
            #################################################
            # 新增非对称价值裁剪逻辑
            # 计算动态分位数阈值（70%分位数）
            quantile_val = torch.quantile(target_q, 0.7, keepdim=True)
            
            # 非对称裁剪公式：高估部分压缩，低估部分保留
            clipped_target_q = torch.where(
                target_q > quantile_val,
                quantile_val + 0.1 * (target_q - quantile_val),  # 高估区域压缩90%
                target_q  # 低估区域保持原样
            )
            target_q = clipped_target_q  # 替换原始target_q
            #################################################

        v = self.vf(observations)
        adv = target_q - v  # 这里使用的已经是裁剪后的target_q
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        
        # 新增监控指标
        log_dict["value_clip_rate"] = (target_q != original_target_q).float().mean().item()
        log_dict["quantile_0.7"] = quantile_val.mean().item()
        
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
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        # log_dict["q_rewards"] = torch.mean(rewards).item()
        # log_dict["q_targets"] = torch.mean(targets).item()
        log_dict["q_score"] = (torch.mean(qs[0]).item() + torch.mean(qs[1]).item()) / 2
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_q_pessimistic(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        # 原始目标值计算
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        
        # 获取两个Q网络的预测值
        qs = self.qf.both(observations, actions)
        q1, q2 = qs
        
        # --- 隐式悲观策略核心修改 ---
        # 计算Q值的逐样本方差（基于两个Q网络）
        q_variance = torch.var(torch.stack(qs), dim=0)  # shape: (batch_size, 1)
        # 确保方差为非负值
        q_variance = torch.clamp(q_variance, min=0.0)
        # 引入悲观惩罚项：目标值 = 原始目标值 - Q标准差
        targets = targets - torch.sqrt(q_variance)  # 悲观调整
        # --------------------------
        
        # 计算Q损失（保持原结构）
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        
        # 日志记录（新增方差指标）
        log_dict["q_variance"] = torch.mean(q_variance).item()
        log_dict["q_score"] = (torch.mean(q1).item() + torch.mean(q2).item()) / 2
        log_dict["q_loss"] = q_loss.item()
        
        # 反向传播与优化（保持原结构）
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # 更新目标Q网络（保持原结构）
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
        std = out_[0, 1:]
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
        # log_dict["actor_bc_loss"] = torch.mean(bc_losses).item()
        # log_dict["actor_expadv_loss"] = torch.mean(exp_adv).item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()


    def _update_policy_quantile(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: dict,
        quantile_tau: float = 0.25,  # 分位数
    ):
        # 计算指数加权的优势
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)

        # 获取策略网络的输出
        out_, _, _ = self.actor(observations, torch.zeros((1, 1)), torch.zeros((1, 1)))
        out_ = torch.squeeze(out_, 0)
        mean = out_[:, :1]  # 均值
        std = out_[0, 1:]   # 标准差
        policy_out = Normal(mean, std)

        # 行为克隆损失（基于分位数损失替代原来的 BC loss）
        if isinstance(policy_out, torch.distributions.Distribution):
            # 动作预测分布的均值
            predicted_actions = policy_out.mean

            # 自定义分位数损失
            def quantile_loss(predicted, target, tau):
                diff = target - predicted
                return torch.max(tau * diff, (tau - 1) * diff)

            # 计算分位数损失
            bc_losses = quantile_loss(predicted_actions, actions, quantile_tau)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape mismatch")
            # 如果是简单的张量，使用平方误差作为损失（兼容性）
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError

        # 结合优势加权
        policy_loss = torch.mean(exp_adv * bc_losses)

        # 日志记录
        log_dict["actor_all_loss"] = policy_loss.item()
        log_dict["actor_bc_loss"] = torch.mean(bc_losses).item()
        log_dict["actor_expadv_loss"] = torch.mean(exp_adv).item()

        # 反向传播并更新策略网络
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
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
        # adv = self._update_v(observations, actions, log_dict)
        adv = self._update_v_clipped(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # self._update_q_pessimistic(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict)
        # self._update_policy_quantile(adv, observations, actions, log_dict, quantile_tau=0.25)

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
    torchBwModel = GaussianPolicy_NewAct(STATE_DIM, ACTION_DIM, MAX_ACTION)
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
            onnx_estimate, onnx_hidden_state, onnx_cell_state = ort_session.run(None, feed_dict)
            assert np.allclose(torch_estimate.numpy(), onnx_estimate, atol=100), 'Failed to match model outputs!, {}, {}'.format(torch_estimate.numpy(), onnx_estimate)
            assert np.allclose(torch_hidden_state, onnx_hidden_state, atol=1e-7), 'Failed to match hidden state1'
            assert np.allclose(torch_cell_state, onnx_cell_state, atol=1e-7), 'Failed to match cell state!'
        
        assert np.allclose(torch_hidden_state, final_hidden_state, atol=1e-7), 'Failed to match final hidden state!'
        assert np.allclose(torch_cell_state, final_cell_state, atol=1e-7), 'Failed to match final cell state!'
        print("Torch and Onnx models outputs have been verified successfully!")

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
            bw_prediction, hidden_state, cell_state = ort_session.run(None, feed_dict)
            model_predictions.append(obss[0][0][5] * np.exp(bw_prediction[0,0,0]))
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
        dataset = pickle.load(open(f"/data2/kj/Schaferct/training_dataset_pickle/training_dataset_K_1800_v{i}_new_act.pickle", 'rb'))
        if i == 2 or i == 5:
            replay_buffer.load_dataset(dataset, sample_num=int(dataset["observations"].shape[0] / 5))
        else:
            replay_buffer.load_dataset(dataset, sample_num=dataset["observations"].shape[0])

    
    max_action = MAX_ACTION

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed)

    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    # actor = (
    #     DeterministicPolicy(
    #         state_dim, action_dim, max_action, dropout=config.actor_dropout
    #     )
    #     if config.iql_deterministic
    #     else GaussianPolicy(
    #         state_dim, action_dim, max_action, dropout=config.actor_dropout
    #     )
    # ).to(config.device)
    # actor = MultiPolicy(state_dim, action_dim, max_action, dropout=config.actor_dropout).to(config.device)
    actor = GaussianPolicy_NewAct(state_dim, action_dim, max_action, dropout=config.actor_dropout).to(config.device)
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

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    if USE_WANDB:
        wandb_init(asdict(config))

    for t in range(int(config.max_timesteps)):
        # 每个policy迭代1e4次
        # if t % (config.eval_freq * 2) == 0:
        #     dataset = pickle.load(open(f"/data2/kj/Schaferct/training_dataset_pickle/training_dataset_K_1800_v{policy_list[policy_index]}.pickle", 'rb'))
        #     replay_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)
        #     replay_buffer.load_dataset(dataset)
        #     print(f'policy_v{policy_index} dataset loaded')
        #     policy_index = (policy_index + 1) % 6

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
