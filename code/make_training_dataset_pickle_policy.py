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
from sklearn.preprocessing import PowerTransformer
from collections import deque


class SlidingWindowQueue:
    def __init__(self, max_len=50):
        self.queue = deque([0] * max_len, maxlen=50)

    def add_value(self, value):
        """添加一个新值到队列中"""
        self.queue.append(value)

    def get_recent_5(self):
        """返回最近 5 个值"""
        return list(self.queue)[-5:][::-1]
    
    def get_recent_1(self):
        """返回最近 5 个值"""
        return list(self.queue)[-1:][::-1]
    
    def get_recent_10(self):
        """返回最近 10 个值"""
        return list(self.queue)[-10:][::-1]

    def get_recent_5_averages(self):
        """
        返回包含 5 个值的数组：
        [最近 10 个值的均值,
         最近 10-20 个值的均值,
         最近 20-30 个值的均值,
         最近 30-40 个值的均值,
         最近 40-50 个值的均值]
        """
        if len(self.queue) < 50:
            raise ValueError("队列长度不足 50，无法计算所需均值")

        values = list(self.queue)
        result = []
        for i in range(5):
            chunk = values[-(i + 1) * 10 : -i * 10 if i > 0 else None]
            result.append(sum(chunk) / len(chunk))
        return result  # 反转顺序，使结果从最旧到最新




pickle_name = 'new_training_dataset_pickle'
K = 10  # every 6 chunk, 6*K sessions
random.seed(0)
# lambda_val = -0.255092930685282
# ALL最佳lambda: [0.93591732]
lambda_val = 0.93591732
TESTBED_POLICY_TYPE_NUM = 6
testbed_dataset_dir_path = os.path.join(project_root_path, 'ALLdatasets', 'testbed_dataset_policy')
# testbed_dataset_dir_path = os.path.join(project_root_path, 'ALLdatasets', 'emulated_dataset_policy')


def load_bwec_dataset():

    for testbed_chunk_idx in range(TESTBED_POLICY_TYPE_NUM):
        obs_ = []
        action_ = []
        next_obs_ = []
        reward_ = []
        done_ = []
        true_capacity_ = []
        sub_dir_path = os.path.join(testbed_dataset_dir_path, 'v' + str(testbed_chunk_idx))
        policy_data_len = len(os.listdir(sub_dir_path))
        print('policy_type:', 'v' + str(testbed_chunk_idx), 'data_len:', policy_data_len)
        # for session in tqdm(random.sample(os.listdir(sub_dir_path), min(K, policy_data_len)), desc='Processing policy_type ' + 'v' + str(testbed_chunk_idx)):
        for session in tqdm(random.sample(os.listdir(sub_dir_path), min(K, policy_data_len)), desc='Processing policy_type ' + 'v' + str(testbed_chunk_idx)):
            session_path = os.path.join(sub_dir_path, session)
            with open(session_path, 'r', encoding='utf-8') as jf:
                single_session_trajectory = json.load(jf)
                observations = single_session_trajectory['observations']
                actions = single_session_trajectory['bandwidth_predictions']
                quality_videos = single_session_trajectory['video_quality']
                quality_audios = single_session_trajectory['audio_quality']
                # true_capacitys = single_session_trajectory['true_capacity']
                avg_q_v = np.nanmean(np.asarray(quality_videos, dtype=np.float32))
                avg_q_a = np.nanmean(np.asarray(quality_audios, dtype=np.float32))

                obs = []
                next_obs = []
                action = []
                reward = []
                # action_queue = SlidingWindowQueue(max_len=10)
                # action_queue_next = SlidingWindowQueue(max_len=10)
                # action_queue_next.add_value(actions[0])
                true_capacity = []
                for idx in range(len(observations)):
                # for idx in range(len(observations) - 1):
                    
                    # 初始版本
                    # r_v = quality_videos[idx]
                    # r_a = quality_audios[idx]
                    # if math.isnan(quality_videos[idx]):
                    #     r_v = avg_q_v
                    # if math.isnan(quality_audios[idx]):
                    #     r_a = avg_q_a
                    # reward.append(r_v * 1.8 + r_a * 0.2)

                    # obs.append(observations[idx])
                    # if idx + 1 >= len(observations):
                    #     next_obs.append([-1] * len(observations[0]))  # s_terminal
                    # else:
                    #     next_obs.append(observations[idx + 1])
                    # action.append([actions[idx]])

                    # 修改动作分布版本
                    r_v = quality_videos[idx]
                    r_a = quality_audios[idx]
                    if math.isnan(quality_videos[idx]):
                        r_v = avg_q_v
                    if math.isnan(quality_audios[idx]):
                        r_a = avg_q_a
                    reward.append(r_v * 1.8 + r_a * 0.2)

                    # t_c = true_capacitys[idx]
                    # if math.isnan(true_capacitys[idx]):
                    #     new_idx = idx
                    #     while new_idx >= 0 and math.isnan(true_capacitys[new_idx]):
                    #         new_idx = new_idx - 1
                    #     print(new_idx)
                    #     t_c = true_capacitys[new_idx]
                    # true_capacity.append(t_c)

                    obs.append(observations[idx])
                    if idx + 1 >= len(observations):
                        next_obs.append([-1] * len(observations[0]))  # s_terminal
                    else:
                        next_obs.append(observations[idx + 1])

                    # box-cox变换版
                    # if observations[idx][5] != 0:
                    #     ratio = actions[idx] / observations[idx][5]
                    #     action.append([(np.power(ratio, lambda_val) - 1) / lambda_val])
                    # else:
                    #     # 推理时这个值乘以0
                    #     action.append([0])

                    if observations[idx][5] != 0:
                        # ratio = actions[idx] / observations[idx][5]
                        # action.append([(np.power(ratio, lambda_val) - 1) / lambda_val])
                        sub = actions[idx] - observations[idx][5]
                        # y_trans = yeo_johnson_transform([sub], lambda_val)
                        action.append([sub])
                    else:
                        # 推理时这个值乘以0
                        action.append([0])

                    # idx = idx + 1

                    # r_v = quality_videos[idx]
                    # r_a = quality_audios[idx]
                    # if math.isnan(quality_videos[idx]):
                    #     r_v = avg_q_v
                    # if math.isnan(quality_audios[idx]):
                    #     r_a = avg_q_a
                    # reward.append(r_v * 1.8 + r_a * 0.2)

                    # action_queue.add_value(actions[idx - 1])
                    # # last_action_short = action_queue.get_recent_5()
                    # # last_action_long = action_queue.get_recent_5_averages()
                    # # new_obs = last_action_short + last_action_long + observations[idx]
                    # last_action = action_queue.get_recent_10()
                    # new_obs = last_action + observations[idx]
                    # obs.append(new_obs)

                    # action_queue_next.add_value(actions[idx])
                    # # last_action_short_next = action_queue_next.get_recent_5()
                    # # last_action_long_next = action_queue_next.get_recent_5_averages()
                    # last_action_next = action_queue_next.get_recent_10()
                    

                    # if idx + 1 >= len(observations) - 1:
                    #     # next_obs.append([-1] * len(observations[0]))  # s_terminal
                    #     next_obs.append([-1] * 160)
                    # else:
                    #     # next_obs.append(observations[idx + 1])
                    #     # new_obs_next = last_action_short_next + last_action_long_next + observations[idx + 1]
                    #     new_obs_next = last_action_next + observations[idx + 1]
                    #     next_obs.append(new_obs_next)
                    
                    # if actions[idx - 1] != 0:
                    #     # action.append([np.log(actions[idx]/actions[idx - 1])])
                    #     action.append([actions[idx]])
                    # else:
                    #     action.append([0])
                
                done_bool = [False] * (len(obs) - 1) + [True]
            action = yeo_johnson_transform(action)
            # check dim
            assert len(obs) == len(next_obs) == len(action) == len(reward) == len(done_bool), 'DIM not match'

            # expaned into x_
            obs_.extend(obs)
            action_.extend(action)
            next_obs_.extend(next_obs)
            reward_.extend(reward)
            done_.extend(done_bool)
            # true_capacity_.extend(true_capacity)
            # break

        dataset_policy =  {
            'observations': np.array(obs_),
            'actions': np.array(action_),
            'next_observations': np.array(next_obs_),
            'rewards': np.array(reward_),
            # 'true_capacity': np.array(true_capacity_),
            'terminals': np.array(done_),
        }

        print('dumping...')
        dataset_file_path = os.path.join(project_root_path, 'training_dataset_pickle', f'training_dataset_K_{K}_' + 'v' + str(testbed_chunk_idx)+ "_y_lamda" + ".pickle")
        dataset_file = open(dataset_file_path, 'wb')
        pickle.dump(dataset_policy, dataset_file)
        dataset_file.close()

def yeo_johnson_transform(y):
    """
    使用指定的λ参数对数据y进行Yeo-Johnson正变换。
    
    参数:
      y: 原始数据（numpy数组）
      lam: 指定的λ参数
    返回:
      转换后的数据
    """
    y = np.array(y)
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
    load_bwec_dataset()

    # print('dumping...')
    # dataset_file_path = os.path.join(project_root_path, 'training_dataset_pickle', pickle_name)
    # dataset_file = open(dataset_file_path, 'wb')
    # pickle.dump(dataset, dataset_file)
    # dataset_file.close()