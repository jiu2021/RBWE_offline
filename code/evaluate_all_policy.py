# -*- coding: utf-8 -*-
# @Author  : n13eho
# @Time    : 2024.03.25

"""
Evaluate all models (baseline, new model, ...) on whole evaluation dataset
Metircs: error rate, mse, over-estimated rate
"""

import json
import os
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
import matplotlib.pyplot as plt
import pandas as pd
from utils import find_gmm_mode

humm = 1e6

current_dir = os.path.split(os.path.abspath(__file__))[0]
project_root_path = current_dir.rsplit('/', 1)[0]
emulate_dataset_dir_path = os.path.join(project_root_path, 'ALLdatasets', 'emulated_dataset_policy')
onnx_models_dir_path = os.path.join(project_root_path, 'onnx_model_for_evaluation')
# onnx_model_name = ['Schaferct_model', 'fast_and_furious_model', 'baseline', 'v0_truepoint_70000', 'v1_truepoint_255000', 'v2_truepoint_70000', 'v3_truepoint_100000', 'v4_truepoint_130000']  # < modify your onnx model names
# onnx_model_name = ['baseline', 'Schaferct_model', 'fast_and_furious_model', 'v2_checkpoint_105000', 'v4_checkpoint_40000']  # < modify your onnx model names
onnx_model_name = ['Schaferct_model', 'baseline', 'gauss', 'gmm4_535k']  # < modify your onnx model names
onnx_model_name = ['Schaferct_model', 'baseline', 'gmm4_535k']  # < modify your onnx model names

our_model = onnx_model_name[-1]
# gauss_model = onnx_model_name[-2]

def get_over_estimated_rate(x, y):
    l = [max((xx - yy) / (yy), 0) for xx, yy in zip(x, y)]
    l = np.asarray(l, dtype=np.float32)
    return [np.nanmean(l), np.nanstd(l)]

def get_under_estimated_rate(x, y):
    l = [max((yy - xx) / (yy), 0) for xx, yy in zip(x, y)]
    l = np.asarray(l, dtype=np.float32)
    return [np.nanmean(l), np.nanstd(l)]

def get_mse(x, y):
    l = [(xx - yy) ** 2 for xx, yy in zip(x, y)]
    l = np.asarray(l, dtype=np.float32)
    return [np.nanmean(l), np.nanstd(l)]

def get_error_rate(x, y):
    # error rate = min(1, |x-y| / y)
    l = [min(1, abs(xx - yy) / yy) for xx, yy in zip(x, y)]
    l = np.asarray(l, dtype=np.float32)
    return [np.nanmean(l), np.nanstd(l)]

def evaluate_every_f(e_f_path):
    # [behavior policy, baseline, m1, m2, m3, ...]
    er_perf = []
    mse_perf = []
    oer_perf = []
    uer_perf = []
    with open(e_f_path, "r") as file:
        call_data = json.load(file)

        observations_150 = np.asarray(call_data['observations'], dtype=np.float32)
        observations_120 = np.asarray(call_data['observations'], dtype=np.float32)[:, :120]

        behavior_policy = np.asarray(call_data['bandwidth_predictions'], dtype=np.float32) / humm
        true_capacity = np.asarray(call_data['true_capacity'], dtype=np.float32) / humm

        # first go with behavior policy
        er_perf.append(get_error_rate(behavior_policy, true_capacity))
        mse_perf.append(get_mse(behavior_policy, true_capacity))
        oer_perf.append(get_over_estimated_rate(behavior_policy, true_capacity))
        uer_perf.append(get_under_estimated_rate(behavior_policy, true_capacity))
        
        # then go with these models
        for onnx_name in onnx_model_name:
            if onnx_name == "checkpoint_580000":
                observations = observations_120
            else:
                observations = observations_150

            onnx_m_path = os.path.join(onnx_models_dir_path, onnx_name + '.onnx')
            ort_session = ort.InferenceSession(onnx_m_path)
            predictions = []
            if onnx_name == 'fast_and_furious_model':
                hidden_state, cell_state = np.zeros((1, 128), dtype=np.float32), np.zeros((1, 128), dtype=np.float32)
            else:
                hidden_state, cell_state = np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)
            # hc = np.zeros((1, 1), dtype=np.float32)
            for t in range(observations.shape[0]):
                feed_dict = {'obs': observations[t:t+1,:].reshape(1,1,-1),
                            'hidden_states': hidden_state,
                            'cell_states': cell_state
                            }
                # bw_prediction, hidden_state, cell_state = ort_session.run(None, feed_dict)
                if onnx_name == our_model:
                    mean, std, pi, _, _ = ort_session.run(None, feed_dict)
                    # 对于每个样本，找到权重最大的分支索引
                    max_component_indices = np.argmax(pi, axis=1)
                    # 利用高级索引选取对应分支的均值
                    batch_indices = np.arange(mean.shape[0])
                    selected_actions = mean[batch_indices, max_component_indices, :][0,0]
                    
                    mean = np.squeeze(mean)
                    std = np.squeeze(std)
                    pi = np.squeeze(pi)
                    act = find_gmm_mode(pi, mean, std)
                    if act:
                        predictions.append(observations[t, 5] * np.exp(act))
                    else:
                        predictions.append(observations[t, 5] * np.exp(selected_actions))

                    # bw_prediction, hidden_state, cell_state = ort_session.run(None, feed_dict)
                    # predictions.append(observations[t, 5] * np.exp(bw_prediction[0,0,0]))
                # elif onnx_name == gauss_model:
                #     bw_prediction, hidden_state, cell_state = ort_session.run(None, feed_dict)
                #     predictions.append(observations[t, 5] * np.exp(bw_prediction[0,0,0]))
                else:
                    bw_prediction, hidden_state, cell_state = ort_session.run(None, feed_dict)
                    predictions.append(bw_prediction[0,0,0])


            # if onnx_name == our_model or onnx_name == gauss_model:
            if onnx_name == our_model:
                predictions = np.asarray(predictions, dtype=np.float32) / humm
                # predictions = np.exp(((predictions + 1.0) / 2.0 * np.log(800.0) + np.log(0.01)))
            else:
                predictions = np.asarray(predictions, dtype=np.float32) / humm

            er_perf.append(get_error_rate(predictions, true_capacity))
            mse_perf.append(get_mse(predictions, true_capacity))
            oer_perf.append(get_over_estimated_rate(predictions, true_capacity))
            uer_perf.append(get_under_estimated_rate(predictions, true_capacity))
    return er_perf, mse_perf, oer_perf, uer_perf


def plot_data(mse, error_rate, over_estimated_rate, under_estimated_rate, data_policy_id):
    # 保存数据到 CSV 文件
    data = {
        'Model': ['behavior policy'] + onnx_model_name,
        'MSE': mse[:,0],
        'Error Rate': error_rate[:,0],
        'Over-estimated Rate': over_estimated_rate[:,0],
        'Under-estimated Rate': under_estimated_rate[:,0],
        'MSE std': mse[:,1],
        'Error Rate std': error_rate[:,1],
        'Over-estimated Rate std': over_estimated_rate[:,1],
        'Under-estimated Rate std': under_estimated_rate[:,1]
    }
    df = pd.DataFrame(data)
    res_dir = f"/data2/kj/Schaferct/result/{our_model}_emulated_new"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    csv_path = os.path.join(res_dir, data_policy_id + "_data_new_eval.csv")
    df.to_csv(csv_path, index=False)

    # 绘制图表
    # plot
    models_names = ['behavior policy']
    onnx_model_name_id = onnx_model_name
    models_names.extend(onnx_model_name_id)
    type_num = len(onnx_model_name_id) + 1
    x = np.arange(type_num)
    bar_width = 0.25
    
    fig, bar_rate = plt.subplots(figsize=(10, 5))
    bar_mse = bar_rate.twinx()
    rects1 = bar_rate.bar(x, error_rate[:,0], bar_width, color='c')
    rects2 = bar_rate.bar(x + bar_width, over_estimated_rate[:,0], bar_width, color='orange')
    rects4 = bar_rate.bar(x + bar_width * 2, under_estimated_rate[:,0], bar_width, color='b')
    rects3 = bar_mse.bar(x + bar_width * 3, mse[:,0], bar_width, color='green')
    
    bar_rate.bar_label(rects1, padding=1, fmt='%.2f')
    bar_rate.bar_label(rects2, padding=1, fmt='%.2f')
    bar_rate.bar_label(rects4, padding=1, fmt='%.2f')
    bar_mse.bar_label(rects3, padding=1, fmt='%.2f')
    
    bar_rate.set_ylabel('Error Rate / Over-estimated Rate (%)') 
    bar_mse.set_ylabel('MSE (Mbps^2)')

    bar_rate.set_xlabel("preditive models")
    plt.xticks(x + bar_width, models_names)
    plt.legend([rects1, rects2, rects4, rects3], ['error rate', 'over-estimated rate', 'under-estimated rate', 'mse'], 
            bbox_to_anchor=(0.5, 1.03), ncol=4, loc='center', frameon=False)
    
    plt.tight_layout()
    res_dir = f"/data2/kj/Schaferct/result/{our_model}_emulated_new"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    plt.savefig(os.path.join(res_dir, data_policy_id + "_data_new_eval.png"))
    plt.close()



if __name__ == "__main__":
    all_e_file_path_policy = []
    test_data_policy = ['v0', 'v1', 'v2', 'v3', 'v4']
    # test_data_policy = ['v1', 'v4']
    # put all evaluation json in a list
    for data_policy_id in test_data_policy:
        all_e_file_path = []
        sub_dir_path = os.path.join(emulate_dataset_dir_path, data_policy_id)
        for e_file in os.listdir(sub_dir_path):
            e_file_path = os.path.join(sub_dir_path, e_file)
            all_e_file_path.append(e_file_path)
        all_e_file_path_policy.append(all_e_file_path)
    # all_e_file_path = []
    # for sub_dir_name in tqdm(os.listdir(emulate_dataset_dir_path)):
    #     sub_dir_path = os.path.join(emulate_dataset_dir_path, sub_dir_name)
    #     for e_file in os.listdir(sub_dir_path):
    #         e_file_path = os.path.join(sub_dir_path, e_file)
    #         all_e_file_path.append(e_file_path)
                    
    # prepare data
    # [behavior policy, baseline, m1, m2, m3, ...]
    all_err_rate = []
    all_mse = []
    all_over_estimated_rate = []
    all_under_estimated_rate = []
    for all_e_file_path, data_policy_id in zip(all_e_file_path_policy, test_data_policy):
        print(f"Evaluating on {data_policy_id}")
        error_rate = []
        mse = []
        over_estimated_rate = []
        under_estimated_rate = []
        for e_file in tqdm(all_e_file_path, desc='Evaluating'):
            er_perf, mse_perf, oer_perf, uer_pref = evaluate_every_f(e_file)

            error_rate.append(er_perf)
            mse.append(mse_perf)
            over_estimated_rate.append(oer_perf)
            under_estimated_rate.append(uer_pref)
        # get avg
        error_rate = np.asarray(error_rate)
        mse = np.asarray(mse)
        over_estimated_rate = np.asarray(over_estimated_rate)
        under_estimated_rate = np.asarray(under_estimated_rate)
        error_rate = np.average(error_rate, axis=0)
        mse = np.average(mse, axis=0)
        over_estimated_rate = np.average(over_estimated_rate, axis=0)
        under_estimated_rate = np.average(under_estimated_rate, axis=0)

        all_err_rate.append(error_rate)
        all_mse.append(mse)
        all_over_estimated_rate.append(over_estimated_rate)
        all_under_estimated_rate.append(under_estimated_rate)

        plot_data(mse, error_rate, over_estimated_rate, under_estimated_rate, data_policy_id)

    all_err_rate = np.average(all_err_rate, axis=0)
    all_mse = np.average(all_mse, axis=0)
    all_over_estimated_rate = np.average(all_over_estimated_rate, axis=0)
    all_under_estimated_rate = np.average(all_under_estimated_rate, axis=0)

    plot_data(all_mse, all_err_rate, all_over_estimated_rate, all_under_estimated_rate, 'all')
        # plot
        # models_names = ['behavior policy']
        # onnx_model_name_id = ['Schaferct', 'fast_and_furious', 'baseline', 'our']
        # models_names.extend(onnx_model_name_id)
        # type_num = len(onnx_model_name_id) + 1
        # x = np.arange(type_num)
        # bar_width = 0.25
        
        # fig, bar_rate = plt.subplots(figsize=(10, 5))
        # bar_mse = bar_rate.twinx()
        # rects1 = bar_rate.bar(x, error_rate, bar_width, color='c')
        # rects2 = bar_rate.bar(x + bar_width, over_estimated_rate, bar_width, color='orange')
        # rects4 = bar_rate.bar(x + bar_width * 2, under_estimated_rate, bar_width, color='b')
        # rects3 = bar_mse.bar(x + bar_width * 3, mse, bar_width, color='green')
        
        # bar_rate.bar_label(rects1, padding=1, fmt='%.2f')
        # bar_rate.bar_label(rects2, padding=1, fmt='%.2f')
        # bar_rate.bar_label(rects4, padding=1, fmt='%.2f')
        # bar_mse.bar_label(rects3, padding=1, fmt='%.2f')
        
        # bar_rate.set_ylabel('Error Rate / Over-estimated Rate (%)') 
        # bar_mse.set_ylabel('MSE (Mbps^2)')

        # bar_rate.set_xlabel("preditive models")
        # plt.xticks(x + bar_width, models_names)
        # plt.legend([rects1, rects2, rects4, rects3], ['error rate', 'over-estimated rate', 'under-estimated rate', 'mse'], 
        #         bbox_to_anchor=(0.5, 1.03), ncol=4, loc='center', frameon=False)
        
        # plt.tight_layout()
        # plt.savefig(os.path.join(current_dir, data_policy_id + "_data.png"))
        # plt.close()


    
