# -*- coding: utf-8 -*-
# @Author  : n13eho
# @Time    : 2024.03.25

"""
Make training dataset out of K sessions from each policy type, you can modify K to get more or less data

output: new_training_dataset_pickle.pickle
"""


import json
import os
current_dir = os.path.split(os.path.abspath(__file__))[0]
project_root_path = current_dir.rsplit('/', 1)[0]
import numpy as np
import math
import pickle
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import PowerTransformer
from scipy.stats import skew, kurtosis

pickle_name = 'training_dataset_pickle_test.pickle'
K = 1000  # every 6 chunk, 6*K sessions
random.seed(0)
TESTBED_POLICY_TYPE_NUM = 6
testbed_dataset_dir_path = os.path.join(project_root_path, 'ALLdatasets', 'testbed_dataset_policy')

def load_bwec_dataset():
    obs_ = []
    action_ = []
    next_obs_ = []
    reward_ = []
    done_ = []

    for testbed_chunk_idx in range(TESTBED_POLICY_TYPE_NUM):
        # if testbed_chunk_idx == 5:
        #     print('dumping v0-v4...')
        #     dataset_file_path = os.path.join(project_root_path, 'training_dataset_pickle', 'training_dataset_pickle_K_1800_wo_v5.pickle')
        #     pkt_batch(obs_, action_, next_obs_, reward_, done_, dataset_file_path)

        sub_dir_path = os.path.join(testbed_dataset_dir_path, 'v' + str(testbed_chunk_idx))
        for session in tqdm(random.sample(os.listdir(sub_dir_path), K)[:5], desc='Processing policy_type ' + str(testbed_chunk_idx)):
            session_path = os.path.join(sub_dir_path, session)
            with open(session_path, 'r', encoding='utf-8') as jf:
                single_session_trajectory = json.load(jf)
                observations = single_session_trajectory['observations']
                actions = single_session_trajectory['bandwidth_predictions']
                quality_videos = single_session_trajectory['video_quality']
                quality_audios = single_session_trajectory['audio_quality']

                avg_q_v = np.nanmean(np.asarray(quality_videos, dtype=np.float32))
                avg_q_a = np.nanmean(np.asarray(quality_audios, dtype=np.float32))

                obs = []
                next_obs = []
                action = []
                reward = []
                for idx in range(len(observations)):

                    r_v = quality_videos[idx]
                    r_a = quality_audios[idx]
                    if math.isnan(quality_videos[idx]):
                        r_v = avg_q_v
                    if math.isnan(quality_audios[idx]):
                        r_a = avg_q_a
                    reward.append(r_v * 1.8 + r_a * 0.2)

                    obs.append(observations[idx])
                    if idx + 1 >= len(observations):
                        next_obs.append([-1] * len(observations[0]))  # s_terminal
                    else:
                        next_obs.append(observations[idx + 1])
                    action.append([actions[idx]])
                
                done_bool = [False] * (len(obs) - 1) + [True]

            # check dim
            assert len(obs) == len(next_obs) == len(action) == len(reward) == len(done_bool), 'DIM not match'

            # expaned into x_
            obs_.extend(obs)
            action_.extend(action)
            next_obs_.extend(next_obs)
            reward_.extend(reward)
            done_.extend(done_bool)
            # break
        
    
    print('dumping v0-v5...')
    dataset_file_path = os.path.join(project_root_path, 'training_dataset_pickle', pickle_name)
    pkt_batch(obs_, action_, next_obs_, reward_, done_, dataset_file_path)

    # return {
    #     'observations': np.array(obs_),
    #     'actions': np.array(action_),
    #     'next_observations': np.array(next_obs_),
    #     'rewards': np.array(reward_),
    #     'terminals': np.array(done_),
    # }


def pkt_batch(obs_, action_, next_obs_, reward_, done_, dataset_file_path):
    # 定义批次大小
    batch_size = 1000  # 根据内存大小调整批次

    # 打开文件以写入模式
    with open(dataset_file_path, 'wb') as dataset_file:
        # 分批写入 observations, actions, next_observations, rewards, terminals
        for i in range(0, len(obs_), batch_size):
            batch_obs = np.array(obs_[i:i + batch_size], dtype=np.float32)
            batch_actions = np.array(action_[i:i + batch_size], dtype=np.float32)
            batch_next_obs = np.array(next_obs_[i:i + batch_size], dtype=np.float32)
            batch_rewards = np.array(reward_[i:i + batch_size], dtype=np.float32)
            batch_terminals = np.array(done_[i:i + batch_size], dtype=np.bool_)

            # 创建批次数据字典
            batch_data = {
                'observations': batch_obs,
                'actions': batch_actions,
                'next_observations': batch_next_obs,
                'rewards': batch_rewards,
                'terminals': batch_terminals,
            }

            # 使用 pickle 追加写入批次数据
            pickle.dump(batch_data, dataset_file, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Dumped batch {i // batch_size + 1}/{len(obs_) // batch_size + 1}")

    print("Dumping completed.")



def view_actions():
    res = []
    res_transformed = []
    for testbed_chunk_idx in range(TESTBED_POLICY_TYPE_NUM):
        sub_dir_path = os.path.join(testbed_dataset_dir_path, 'v' + str(testbed_chunk_idx))
        actions_relative = []
        for session in tqdm(random.sample(os.listdir(sub_dir_path), K), desc='Processing policy_type ' + str(testbed_chunk_idx)):
            session_path = os.path.join(sub_dir_path, session)
            with open(session_path, 'r', encoding='utf-8') as jf:
                single_session_trajectory = json.load(jf)
                actions = single_session_trajectory['bandwidth_predictions']
                observations = single_session_trajectory['observations']
                # for i in range(len(observations) - 1):
                #     if actions[i] != 0:
                #         actions_relative.append(actions[i+1]/actions[i])
                #     else:
                #         print(f'{session_path}:idx:{i} zero!!')
                for i in range(len(observations)):
                    if observations[i][5] != 0:
                        actions_relative.append(np.log(actions[i]/observations[i][5]))
                    else:
                        print(f'{session_path}:idx:{i} zero!!')

                    # actions_relative.append(actions[i])

            
        actions_relative = np.sort(np.asarray(actions_relative))
        # 计算均值、方差、偏度、峰度
        mean = np.mean(actions_relative)
        variance = np.var(actions_relative)
        skewness = skew(actions_relative)
        kurt = kurtosis(actions_relative)

        print(f"Policy type {testbed_chunk_idx}: Mean: {mean}, Variance: {variance}, Skewness: {skewness}, Kurtosis: {kurt}")
        res.append(actions_relative)

        # min_val, max_val = 10, 8 * 1e6
        # actions_relative = np.clip(actions_relative, min_val, max_val)
        # transformed_data = 8 * ((actions_relative - min_val) / (max_val - min_val))
        # transformed_data = np.tanh(transformed_data)
        # # 创建 PowerTransformer 对象，使用 Yeo-Johnson 变换
        # scaler = PowerTransformer(method='yeo-johnson')
        # # 对数据进行 Yeo-Johnson 变换
        # transformed_data = scaler.fit_transform(actions_relative.reshape(-1, 1)).flatten()

        # transformed_data = yeo_johnson_transform(actions_relative)
        # transformed_data, best_lambda = stats.boxcox(actions_relative)
        # res_transformed.append(transformed_data)
        # print("最佳lambda:", best_lambda)

        # print("变换后的数据:", transformed_data)
        
        # cdf = np.cumsum(np.ones_like(actions_relative)) / len(actions_relative)
        # plt.figure()
        # plt.plot(actions_relative, cdf, label='CDF', color='blue', linewidth=2)
        # plt.savefig(f'./tmp/v{str(testbed_chunk_idx)}_actions_relative_rate.png')

        # 绘制直方图
        # plt.hist(actions_relative, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
        plt.figure()
        sns.kdeplot(actions_relative, label=f'v{str(testbed_chunk_idx)}', fill=True, alpha=0.5)
        # plt.title(f"Action Data Distribution", fontsize=16)
        plt.xlabel("Value of new action", fontsize=16)
        plt.ylabel("Density", fontsize=16)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.legend(fontsize=16)
        plt.savefig(f'./actions_plot/v{str(testbed_chunk_idx)}_actions_relative.png')

        # plt.figure()
        # sns.kdeplot(transformed_data, label=f'v{str(testbed_chunk_idx)}', fill=True, alpha=0.5)
        # plt.title(f"Action Data Distribution", fontsize=14)
        # plt.xlabel("Value", fontsize=12)
        # plt.ylabel("Density", fontsize=12)
        # plt.grid(axis='y', linestyle='--', alpha=0.6)
        # plt.legend()
        # plt.savefig(f'./actions_tanh/v{str(testbed_chunk_idx)}_actions_relative_transformed.png')
    
   
    # all_res = None
    # for i in range(len(res)):
    #     if all_res is None:
    #         all_res = res[i]
    #     else:
    #         all_res = np.concatenate((all_res, res[i]))
    #         res[i] = None
    # #     # sns.kdeplot(res[i], label=f'v{str(i)}', fill=True, alpha=0.5)
    #         # 计算均值、方差、偏度、峰度
    # mean = np.mean(all_res)
    # variance = np.var(all_res)
    # skewness = skew(all_res)
    # kurt = kurtosis(all_res)

    # print(f"All Policy: Mean: {mean}, Variance: {variance}, Skewness: {skewness}, Kurtosis: {kurt}")
    # # transformed_data, all_best_lambda = stats.boxcox(all_res)
    # # 创建 PowerTransformer 对象，使用 Yeo-Johnson 变换
    # scaler = PowerTransformer(method='yeo-johnson')
    # # 对数据进行 Yeo-Johnson 变换
    # transformed_data = scaler.fit_transform(all_res.reshape(-1, 1)).flatten()
    # print("ALL最佳lambda:", scaler.lambdas_)

    plt.figure(figsize=(10, 6),dpi=300) 
    for i in range(len(res)):
        sns.kdeplot(res[i], label=f'v{str(i)}', fill=True, alpha=0.4)
    # plt.title(f"Action Data Distribution", fontsize=14)
    plt.xlabel("Value of new action", fontsize=16)
    plt.ylabel("Density", fontsize=16)
    plt.tick_params(axis='y', labelcolor='black', labelsize=16)
    plt.tick_params(axis='x', labelcolor='black', labelsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(fontsize=16) 
    plt.savefig(f'./actions_plot/actions.png')

    # plt.figure(figsize=(15, 8)) 
    # for i in range(len(res_transformed)):
    #     sns.kdeplot(res_transformed[i], label=f'v{str(i)}', fill=True, alpha=0.5)
    # plt.title(f"Action Data Distribution", fontsize=14)
    # plt.xlabel("Value", fontsize=12)
    # plt.ylabel("Density", fontsize=12)
    # plt.grid(axis='y', linestyle='--', alpha=0.6)
    # plt.legend()
    # plt.savefig(f'./actions_tanh/actions_tanh.png')

# def plot_action_distribution():
#     import matplotlib.pyplot as plt
#     for testbed_chunk_idx in range(TESTBED_POLICY_TYPE_NUM):
#         sub_dir_path = os.path.join(testbed_dataset_dir_path, 'v' + str(testbed_chunk_idx))
#         for session in os.listdir(sub_dir_path):
#             session_path = os.path.join(sub_dir_path, session)
#             with open(session_path, 'r', encoding='utf-8') as jf:
#                 single_session_trajectory = json.load(jf)
#                 actions = single_session_trajectory['bandwidth_predictions']
#                 plt.plot(actions)
#     plt.show()

def yeo_johnson_transform(y):
    """
    使用指定的λ参数对数据y进行Yeo-Johnson正变换。
    
    参数:
      y: 原始数据（numpy数组）
      lam: 指定的λ参数
    返回:
      转换后的数据
    """
    # y = np.array(y)
    transformed = np.empty_like(y)
    
    # 对于y >= 0
    pos_mask = y >= 0
    transformed[pos_mask] = np.log(y[pos_mask] + 1)
    
    # 对于y < 0
    neg_mask = y < 0
    transformed[neg_mask] = - np.log(-y[neg_mask] + 1)
    
    return transformed

if __name__ == '__main__':
    # dataset = load_bwec_dataset()
    
    # dataset_file = open(dataset_file_path, 'wb')
    # pickle.dump(dataset, dataset_file)
    # dataset_file.close()

    view_actions()