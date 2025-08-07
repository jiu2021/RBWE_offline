# -*- coding: utf-8 -*-
# @Author  : n13eho
# @Time    : 2024.03.25

"""
Evaluate models on evaluation datasets in detail.
"""

import glob
import json
import os
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
import matplotlib.pyplot as plt
from utils import find_gmm_mode

current_dir = os.path.split(os.path.abspath(__file__))[0]
project_root_path = current_dir.rsplit('/', 1)[0]

plt.rcParams.clear()
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.labelsize'] = 17
plt.rcParams['ytick.labelsize'] = 17
plt.rcParams['axes.labelsize'] = 17
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


if __name__ == "__main__":

    data_dir = "../data"  # < modify the path to your data
    # onnx_models = ['baseline', 'Schaferct_model', 'checkpoint_580000']  # < modify your onnx model names
    # onnx_models = ['Schaferct_model', 'baseline', 'v0_checkpoint_95000', 'v1_checkpoint_175000', 'v2_checkpoint_105000', 'v3_checkpoint_165000', 'v4_checkpoint_40000', 'v5_checkpoint_95000']
    # onnx_models = ['Schaferct_model', 'fast_and_furious_model', 'baseline', 'v2_checkpoint_105000', 'v4_checkpoint_40000']
    onnx_models = ['Schaferct_model', 'baseline', 'gmm_4']
    onnx_models_label = ['Schaferct_model', 'baseline', 'gmm_4']
    onnx_models_dir = os.path.join(project_root_path, 'onnx_model')
    figs_dir = os.path.join(project_root_path, 'onnx_model_offline_evaluation', ('_'.join(onnx_models)))
    if not os.path.exists(figs_dir):
        os.mkdir(figs_dir)
    data_files = glob.glob(os.path.join(data_dir, f'*.json'), recursive=True)
    ort_sessions = []
    for m in onnx_models:
        m_path = os.path.join(onnx_models_dir, m + '.onnx')
        ort_sessions.append(ort.InferenceSession(m_path))

    for filename in tqdm(data_files, desc="Processing"):
        with open(filename, "r") as file:
            call_data = json.load(file)
        
        observations_120 = np.asarray(call_data['observations'], dtype=np.float32)[:, :120]
        observations_150 = np.asarray(call_data['observations'], dtype=np.float32)
        bandwidth_predictions = np.asarray(call_data['bandwidth_predictions'], dtype=np.float32)
        true_capacity = np.asarray(call_data['true_capacity'], dtype=np.float32)

        baseline_model_predictions = {}
        for m in onnx_models:
            baseline_model_predictions[m] = []
        
        hidden_state, cell_state = np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)
        hidden_state_farc, cell_state_farc = np.zeros((1, 128), dtype=np.float32), np.zeros((1, 128), dtype=np.float32)    
        for t in range(observations_120.shape[0]):
            obss_120 = observations_120[t:t+1,:].reshape(1,1,-1)
            feed_dict_120 = {'obs': obss_120,
                        'hidden_states': hidden_state,
                        'cell_states': cell_state
                        }
            
            obss_150 = observations_150[t:t+1,:].reshape(1,1,-1)
            feed_dict_150 = {'obs': obss_150,
                        'hidden_states': hidden_state,
                        'cell_states': cell_state
                        }
            for idx, orts in enumerate(ort_sessions):
                feed_dict = feed_dict_120 if onnx_models[idx] ==  'checkpoint_580000' else feed_dict_150
                if onnx_models[idx] == 'fast_and_furious_model':
                    feed_dict['hidden_states'] = hidden_state_farc
                    feed_dict['cell_states'] = cell_state_farc
                    bw_prediction, hidden_state_farc, cell_state_farc = orts.run(None, feed_dict)
                elif onnx_models[idx] != onnx_models[-1]:
                    feed_dict['hidden_states'] = hidden_state
                    feed_dict['cell_states'] = cell_state
                    bw_prediction, hidden_state, cell_state = orts.run(None, feed_dict)
                
                if onnx_models[idx] == onnx_models[-1]:
                    # model_predictions = obss_150[0][0][5] * np.exp(bw_prediction[0,0,0])
                    # # model_predictions = np.exp(((model_predictions + 1.0) / 2.0 * np.log(800.0) + np.log(0.01))) * 1e6
                    # baseline_model_predictions[onnx_models[idx]].append(model_predictions)

                    mean, std, pi, _, _ = orts.run(None, feed_dict)
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
                        baseline_model_predictions[onnx_models[idx]].append(obss_150[0][0][5] * np.exp(act))
                    else:
                        baseline_model_predictions[onnx_models[idx]].append(obss_150[0][0][5] * np.exp(selected_actions))
                else:
                    baseline_model_predictions[onnx_models[idx]].append(bw_prediction[0,0,0])
           
        
        for m in onnx_models:
            baseline_model_predictions[m] = np.asarray(baseline_model_predictions[m], dtype=np.float32)
            
        fig = plt.figure(figsize=(10, 5))
        time_s = np.arange(0, observations_150.shape[0]*60,60)/1000
        for idx, m in enumerate(onnx_models):
            plt.plot(time_s, baseline_model_predictions[m] / 1000, linestyle='-', label=onnx_models_label[idx], color='C' + str(idx))
            # plt.plot(time_s, baseline_model_predictions[m] / 1000, linestyle='-', label=['check_point'][idx], color='C' + str(idx))
        plt.plot(time_s, bandwidth_predictions/1000, linestyle='--', label='Estimator ' + call_data['policy_id'], color='C' + str(len(onnx_models)))
        plt.plot(time_s, true_capacity/1000, label='True Capacity', color='black')
        plt.xlim(0, 125)
        plt.ylim(0)
        plt.ylabel("Bandwidth (Kbps)")
        plt.xlabel("Duration (second)")
        plt.grid(True)
        
        plt.legend(bbox_to_anchor=(0.5, 1.05), ncol=4, handletextpad=0.1, columnspacing=0.5,
                    loc='center', frameon=False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, os.path.basename(filename).replace(".json",".pdf")), dpi=300)
        plt.close()