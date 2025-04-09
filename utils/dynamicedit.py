import os
import sys
import inspect

import json
import torch
import logging
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import cv2
import torch.nn.functional as F
import types
logging.getLogger().setLevel(logging.INFO)

class DynamicEdit():
    def __init__(self, model, top_k_ranks=None, top_k_ranks_truth=None, edit_layer_range=None):

        self.model = model
        self.model.model.eval()
        self.tokenizer = model.tokenizer

        model_config = getattr(model, 'model', None) and getattr(model.model, 'config', None)
        
        if model_config: # model.model.config.model_type
            model_type = getattr(model_config, 'model_type', None)
            self.D = model.model.config.hidden_size
            self.num_layers = model.model.config.num_hidden_layers
            self.E = model.model.lm_head
            self.lm_sep_idx = 2
            # print(f'self.model_name is {model_type}')

        else: # model.args.model_name
            self.D = model.num_lm_hidden_size
            self.num_layers = model.num_lm_layers
            self.E = model.lm_head
            if model.args.model_name == ('MiniGPT4' or 'LLaVA-7B-HF'):
                self.lm_sep_idx = 3
            else:
                self.lm_sep_idx = 2
            
        # print(f'args.model_name is {model.args.model_name}')

        self.top_k_ranks = top_k_ranks
        self.top_k_ranks_truth = top_k_ranks_truth
        self.hallu_vectors = None
        self.truth_vectors = None
        if edit_layer_range is None:
            self.edit_layer_range = np.arange(self.num_layers)
        else:
            self.edit_layer_range = edit_layer_range

        self.f = open(f'logit_lens_test_{model.args.model_name}.txt', 'w')

    def filter_key(self):
        key_dict = []
        for key in self.model.model.state_dict():
            if self.model.args.model_name == 'MiniGPT4':
               # 'llama_model.model.layers.2.mlp.down_proj.weight_format'
                if (
                    'weight' in key 
                    and 'mlp' in key 
                    and '_format' not in key 
                    and not 'visual_encoder' in key 
                    and not 'gate_proj' in key 
                    and not 'up_proj' in key
                ):
                    key_dict.append(key)
            elif self.model.args.model_name == 'Qwen_VL_Chat':
                if (
                    'mlp.c_proj.weight' in key 
                    and not 'visual' in key 
                ):
                    key_dict.append(key)
            elif self.model.args.model_name == 'mPLUG_Owl2':
                if (
                    'weight' in key 
                    and 'mlp' in key 
                    and not 'vision' in key 
                    and not 'gate_proj' in key 
                    and not 'up_proj' in key
                    and not 'visual' in key # owl2
                ):
                    key_dict.append(key)
            else:
                if (
                    'weight' in key 
                    and 'mlp' in key 
                    and not 'vision_tower' in key 
                    and not 'gate_proj' in key 
                    and not 'up_proj' in key
                ):
                    key_dict.append(key)
        return key_dict


    def edit(self, hallu_vectors, truth_vectors, edit_keys=True, edit_values=True):
            assert edit_keys or edit_values, 'At least one of edit_keys or edit_values should be True'
            # logging.info(f'Editing keys: {edit_keys}, Editing values: {edit_values}.')

            if self.edit_layer_range is None:
                self.edit_layer_range = np.arange(self.num_layers)
            # logging.info(f'Editing layers: {self.edit_layer_range}')

            edited_state_dict = self.model.model.state_dict()
            
            key_dict = self.filter_key()
   
            for key in key_dict:
                layer_num = int(key.split('.')[self.lm_sep_idx])
                if layer_num in self.edit_layer_range:

                    # logging.info(f'Editing: {key}')
                    
                    if self.top_k_ranks == 0:
                        hallu_filter = torch.eye(self.D)
                    else:
                        hallu_matrix = torch.zeros(self.D, self.D)
                        for rank in range(len(hallu_vectors[layer_num])):
                            hallu_vec = hallu_vectors[layer_num][rank]
                            hallu_matrix += hallu_vec @ hallu_vec.T
                        hallu_filter = torch.eye(self.D) - hallu_matrix

                    if self.top_k_ranks_truth == 0:
                        truth_filter = torch.eye(self.D)
                    else:
                        truth_matrix = torch.zeros(self.D, self.D)
                        for rank in range(len(truth_vectors[layer_num])):
                            truth_vec = truth_vectors[layer_num][rank]
                            truth_matrix += truth_vec @ truth_vec.T
                        truth_filter = truth_matrix
                    
                    P_filter_left = hallu_filter @ truth_filter
                    P_filter_right = truth_filter @ hallu_filter
                    if self.model.args.model_name == 'MiniGPT4':
                        P_filter_left = P_filter_left.to(edited_state_dict[key].device).to(self.model.model.llama_model.dtype)
                        P_filter_right = P_filter_right.to(edited_state_dict[key].device).to(self.model.model.llama_model.dtype)
                    else:
                        P_filter_left = P_filter_left.to(edited_state_dict[key].device).to(self.model.model.dtype)
                        P_filter_right = P_filter_right.to(edited_state_dict[key].device).to(self.model.model.dtype)

                    weight = edited_state_dict[key]
                    weight = weight.T

                    if edit_keys and 'up_proj' in key:
                        modified_weight = P_filter_left @ weight  # (D, D) @ (D, 4D) -> (D, 4D)
                    elif edit_values and 'down_proj' in key:
                        modified_weight = weight @ P_filter_right  # (4D, D) @ (D, D) -> (4D, D)
                    elif 'c_proj' in key: # Qwen_VL_Chat
                        print('c_proj')
                        modified_weight = weight @ P_filter_right
                    else:
                        print('no modified_weight')
                        continue

                    target_layer = self.model.model.model.layers[layer_num].mlp.down_proj
                    def new_forward(self, input):
                        # modified_weight = self.weight + modified_weight
                        return F.linear(input, modified_weight.T, self.bias)
                    target_layer.forward = types.MethodType(new_forward, target_layer)     
            #         if torch.allclose(weight, modified_weight) and ('gate_proj' not in key):
            #             # logging.warning(f'Module {key} not edited after projection.')
            #             print(f'Module {key} not edited after projection.')

            #         # if self.model_category in ['llama', 'mistral', 'opt', 'gptj']:
            #         modified_weight = modified_weight.T

            #         edited_state_dict[key] = modified_weight.to('cuda').contiguous()  # contiguous for saving to disk

            # self.model.model.load_state_dict(edited_state_dict, assign=True)
            # logging.info('Edited model created.')
            
            if not hasattr(self.model, 'chat'):
                raise AttributeError("The edited model does not have a 'chat' function.")
            return self.model