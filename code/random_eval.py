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
from tqdm import tqdm
import random
from matplotlib import pyplot as plt



pickle_name = 'new_training_dataset_pickle'
K = 1000  # every 6 chunk, 6*K sessions
random.seed(0)
# lambda_val = -0.255092930685282
# ALL最佳lambda: [0.93591732]
lambda_val = 0.93591732
TESTBED_POLICY_TYPE_NUM = 6
# testbed_dataset_dir_path = os.path.join(project_root_path, 'ALLdatasets', 'testbed_dataset_policy')
testbed_dataset_dir_path = os.path.join(project_root_path, 'ALLdatasets', 'emulated_dataset_policy')
bw_range = {0: "0-1", 1: "1-2", 2: "2-4", 3: "4-6", 4: "6-8"}
def cal_range(trace):
    mean = np.mean(trace)
    if mean <= 1:
        return 0
    elif mean <= 2:
        return 1
    elif mean <= 4:
        return 2
    elif mean <= 6:
        return 3
    else:
        return 4
def load_bwec_dataset():

    for testbed_chunk_idx in range(TESTBED_POLICY_TYPE_NUM):
        obs_ = []
        action_ = []
        next_obs_ = []
        reward_ = []
        done_ = []
        true_capacity_ = []
        bw = [[],[],[],[],[]]
        bw_jitter = [[],[],[],[],[]]
        stat_arr = []
        sub_dir_path = os.path.join(testbed_dataset_dir_path, 'v' + str(testbed_chunk_idx))
        policy_data_len = len(os.listdir(sub_dir_path))
         # 重新创建一个空目录
        os.makedirs(os.path.join("/data2/kj/Schaferct/code/eval_list3", 'v' + str(testbed_chunk_idx)), exist_ok=True)
        print('policy_type:', 'v' + str(testbed_chunk_idx), 'data_len:', policy_data_len)
        # for session in tqdm(random.sample(os.listdir(sub_dir_path), min(K, policy_data_len)), desc='Processing policy_type ' + 'v' + str(testbed_chunk_idx)):
        for session in tqdm(random.sample(os.listdir(sub_dir_path),  policy_data_len), desc='Processing policy_type ' + 'v' + str(testbed_chunk_idx)):
            session_path = os.path.join(sub_dir_path, session)
            with open(session_path, 'r', encoding='utf-8') as jf:
                single_session_trajectory = json.load(jf)
                observations = single_session_trajectory['observations']
                actions = single_session_trajectory['bandwidth_predictions']
                quality_videos = single_session_trajectory['video_quality']
                quality_audios = single_session_trajectory['audio_quality']
                true_capacitys = single_session_trajectory['true_capacity']
                avg_q_v = np.nanmean(np.asarray(quality_videos, dtype=np.float32))
                avg_q_a = np.nanmean(np.asarray(quality_audios, dtype=np.float32))
            
                true_capacitys = np.array(true_capacitys) / 1e6
                if np.max(true_capacitys) < 8 and np.mean(true_capacitys) >= 0.2:
                    bw_std = np.std(true_capacitys)
                    # stat_arr.append((np.mean(true_capacitys), np.std(true_capacitys)))
                    range_num = cal_range(true_capacitys)
                    # if len(bw[range_num]) < 20 or (range_num <= 1 and len(bw[range_num]) < 100) or (range_num == 2 and len(bw[range_num]) < 50):
                    if bw_std > 0.01:
                        bw_jitter[range_num].append(session)
                        bw[range_num].append(session)
                        plt.close()
                        plt.plot(range(len(true_capacitys)), true_capacitys)
                        save_path = os.path.join("/data2/kj/Schaferct/code/eval_list3", 'v' + str(testbed_chunk_idx), bw_range[range_num])
                        os.makedirs(save_path, exist_ok=True)
                        plt.savefig(os.path.join("/data2/kj/Schaferct/code/eval_list3", 'v' + str(testbed_chunk_idx), bw_range[range_num], session.split('.')[0]+'.png'))
                    elif len(bw_jitter[range_num]) >= 5:
                        bw[range_num].append(session)
                        plt.close()
                        plt.plot(range(len(true_capacitys)), true_capacitys)
                        save_path = os.path.join("/data2/kj/Schaferct/code/eval_list3", 'v' + str(testbed_chunk_idx), bw_range[range_num])
                        os.makedirs(save_path, exist_ok=True)
                        plt.savefig(os.path.join("/data2/kj/Schaferct/code/eval_list3", 'v' + str(testbed_chunk_idx), bw_range[range_num], session.split('.')[0]+'.png'))
                    elif range_num >= 3:
                        bw[range_num].append(session)
                        plt.close()
                        plt.plot(range(len(true_capacitys)), true_capacitys)
                        save_path = os.path.join("/data2/kj/Schaferct/code/eval_list3", 'v' + str(testbed_chunk_idx), bw_range[range_num])
                        os.makedirs(save_path, exist_ok=True)
                        plt.savefig(os.path.join("/data2/kj/Schaferct/code/eval_list3", 'v' + str(testbed_chunk_idx), bw_range[range_num], session.split('.')[0]+'.png'))

            # if len(bw[0]) + len(bw[1]) + len(bw[2]) + len(bw[3]) + len(bw[4])  >= 290:
            #     break
                    # os.system(f"cp {session_path} {os.path.join("/data2/kj/Schaferct/code/eval_list3", 'v' + str(testbed_chunk_idx), session)}")
        
        # # 提取均值和标准差
        # means, stds = zip(*stat_arr)

        # # 绘制散点图
        # plt.close()
        # plt.figure(figsize=(8, 5))
        # plt.scatter(means, stds, color="blue", alpha=0.6, label="Samples")

        # plt.xlabel("Mean Capacity")
        # plt.ylabel("Standard Deviation")
        # plt.title("Scatter Plot of Mean vs Standard Deviation")
        # plt.legend()
        # plt.grid(True)

        # plt.savefig(f"/data2/kj/Schaferct/ALLdatasets/emulated_dataset_policy/data_{str(testbed_chunk_idx)}.png")
        np.save(f"data_{str(testbed_chunk_idx)}.npy", np.array(bw, dtype=object))

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


def generate_eval():
    exist_trace = os.listdir("/data2/kj/Workspace/Pandia/docker_mnt/traffic_shell/trace_data")
    new_trace = [[],[],[],[],[],[]]
    bw_12_arr = []
    for testbed_chunk_idx in range(TESTBED_POLICY_TYPE_NUM):

        data = np.load(f"/data2/kj/Schaferct/code/data_{str(testbed_chunk_idx)}.npy", allow_pickle=True).tolist()
        bw1_arr = []
        for bw_1 in data[0]:
            if bw_1 not in exist_trace:
                bw1_arr.append(bw_1)

        # bw2_arr = []
        # for bw_2 in data[1]:
            # if bw_2 not in exist_trace:
            #     bw2_arr.append(bw_2)
        bw_12_arr.append(random.sample(data[1], min(len(data[1]), 10)))

    np.save(f"data_12.npy", np.array(bw_12_arr, dtype=object))
        # new_trace[testbed_chunk_idx].append(bw1_arr)
        # new_trace[testbed_chunk_idx].append(bw2_arr)

    # np.save(f"data_new.npy", np.array(new_trace, dtype=object))
    # trace_file = "/data2/kj/Schaferct/ALLdatasets/emulated_dataset_policy"
    # for policy in os.listdir("/data2/kj/Schaferct/code/eval_list3"):
    #     for bw_range in os.listdir(f"/data2/kj/Schaferct/code/eval_list3/{policy}"):
    #         if bw_range in ["4-6", "6-8"]:
    #             print(bw_range)
    #             continue
    #         for trace_name in os.listdir(f"/data2/kj/Schaferct/code/eval_list3/{policy}/{bw_range}"):
    #             trace = trace_file + f"/{policy}/{trace_name.split('.')[0]}.json"
    #             if os.path.exists(f"/data2/kj/Workspace/Pandia/docker_mnt/traffic_shell/trace_data_new/{trace_name.split('.')[0]}.json"):
    #                 continue
    #             os.system(f"cp {trace} /data2/kj/Workspace/Pandia/docker_mnt/traffic_shell/trace_data_new/")

if __name__ == '__main__':
    # dataset = load_bwec_dataset()
    # load_bwec_dataset()
    # generate_eval()
    # data = os.listdir("/data2/kj/Schaferct/code/eval_list2/v1/1-2")
    # print(len(data))
    # new_trace = np.load(f"/data2/kj/Schaferct/code/data_new.npy", allow_pickle=True).tolist()
    # last_eval = []
    # for policy in new_trace:
    #     last_eval.append(random.sample(policy[0], min(len(policy[0]), 50)))
    # np.save(f"data_new_eval.npy", np.array(last_eval, dtype=object))
    # print(new_trace)
    # print('dumping...')
    # dataset_file_path = os.path.join(project_root_path, 'training_dataset_pickle', pickle_name)
    # dataset_file = open(dataset_file_path, 'wb')
    # pickle.dump(dataset, dataset_file)
    # dataset_file.close()


    # trace_file = "/data2/kj/Schaferct/ALLdatasets/emulated_dataset_policy"
    # bw_1 =  np.load(f"/data2/kj/Schaferct/code/data_new_eval.npy", allow_pickle=True).tolist()
    # bw_2 =  np.load(f"/data2/kj/Schaferct/code/data_12.npy", allow_pickle=True).tolist()
    # for i in range(6):
    #     for j in range(len(bw_2[i])):
    #         trace = bw_2[i][j]
    #         if os.path.exists(f"/data2/kj/Workspace/Pandia/docker_mnt/traffic_shell/trace_data_last_eval/{trace}"):
    #                 continue
    #         os.system(f"cp /data2/kj/Schaferct/ALLdatasets/emulated_dataset_policy/v{i}/{trace} /data2/kj/Workspace/Pandia/docker_mnt/traffic_shell/trace_data_last_eval/")

    # data_new = os.listdir("/data2/kj/Workspace/Pandia/docker_mnt/traffic_shell/trace_data_last_eval")
    # data_old = os.listdir("/data2/kj/Workspace/Pandia/docker_mnt/traffic_shell/trace_data")
    # i = 0
    # for trace in data_new:
    #     if trace in data_old:
    #         print(trace)
    #         i+= 1
    # print("total:", i)
    # print(len(data))

    # for policy in os.listdir("/data2/kj/Schaferct/code/eval_list3"):
    #     for bw_range in os.listdir(f"/data2/kj/Schaferct/code/eval_list3/{policy}"):
    #         if bw_range in ["4-6", "6-8"]:
    #             print(bw_range)
    #             continue
    #         for trace_name in os.listdir(f"/data2/kj/Schaferct/code/eval_list3/{policy}/{bw_range}"):
    #             trace = trace_file + f"/{policy}/{trace_name.split('.')[0]}.json"
    #             if os.path.exists(f"/data2/kj/Workspace/Pandia/docker_mnt/traffic_shell/trace_data_new/{trace_name.split('.')[0]}.json"):
    #                 continue
    #             os.system(f"cp {trace} /data2/kj/Workspace/Pandia/docker_mnt/traffic_shell/trace_data_new/")



    # for file in ["08682.json", "05842.json", "01722.json", "01493.json", "02572.json", "00113.json", "00472.json", "05940.json", "08312.json", "07495.json", "05865.json", "01196.json", "01667.json", "01483.json", "00193.json", "02216.json", "02208.json", "01195.json", "04364.json", "02222.json", "06913.json", "06283.json", "08940.json", "07581.json"]:
    # for file in ["08682.json", "05842.json", "01722.json", "01493.json", "02572.json", "00113.json", "00472.json", "05940.json", "08312.json", "07495.json", "05865.json", "01196.json", "01667.json", "01483.json", "00193.json", "02216.json"]:
    #     if os.path.exists(f"/data2/kj/Workspace/Pandia/docker_mnt/traffic_shell/trace_data/{file.split('.')[0]}.json"):
    #         continue
    #     else:
    #         print("no file", file)
            