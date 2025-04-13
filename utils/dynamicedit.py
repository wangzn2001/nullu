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
    def __init__(self, model, top_k_ranks_hallu=None, top_k_ranks_truth=None, edit_layer_range=None, matrix=None):

        self.model = model
        self.model.model.eval()
        self.tokenizer = model.tokenizer
        self.device = next(self.model.model.parameters()).device
        self.matrix = matrix

        model_config = getattr(model, 'model', None) and getattr(model.model, 'config', None)
        
        self.key_dict = self.filter_key()
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

        self.top_k_ranks_hallu = top_k_ranks_hallu
        self.top_k_ranks_truth = top_k_ranks_truth
        self.hallu_vectors = None
        self.truth_vectors = None
        self.edit_layer_range = edit_layer_range


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
    
    def _get_difference_matrix(self, pos_data, neg_data):
        non_preferred_sent_embs = pos_data.permute(1, 0, 2)  # (L, N, D)
        preferred_sent_embs = neg_data.permute(1, 0, 2)  # (L, N, D)
        if self.matrix == 'difference':
            hallu_matrix = (preferred_sent_embs - non_preferred_sent_embs) / 2 # (L, N, D)
        elif self.matrix == 'hidden_states':
            hallu_matrix = non_preferred_sent_embs  # (L, N, D)
        truth_matrix = preferred_sent_embs

        logging.info('Matrix calculated.')
        return hallu_matrix, truth_matrix

    def get_ats(self, pos_data, neg_data):
        hallu_matrix, truth_matrix = self._get_difference_matrix(pos_data, neg_data)  # (L, N, D)
        ats_hallu = {}
        ats_truth = {}
        for key in self.key_dict:
            layer_num = int(key.split('.')[self.lm_sep_idx])  
            ats_hallu[key] = hallu_matrix[layer_num]
            ats_truth[key] = truth_matrix[layer_num]
        return ats_hallu, ats_truth
    
    def svd_on_ats(self, ats_hallu, ats_truth):
        '''
        Key(D, 4D) -> U(D, D) S(D) V^T(D, 4D)
        Value(4D, D) -> U(4D, D) S(4D) V^T(D, D)
        x_l (N, D) -> U(N, N); S(N,); V^T(N, D)

        Note: v @ v.T is not numerically I, but plotting it as a heatmap shows that it is close to I.
        '''
        svd_hallu = {}
        for key in ats_hallu:
            logging.debug(f'Calculating SVD for: {key}')
            M = ats_hallu[key].to(torch.float32)  # SVD function only works with float32
            u, s, vt = torch.linalg.svd(M.cuda(), full_matrices=False)  # Skinny SVD, vt is V^T
            svd_hallu[key] = {'u': u.cpu(), 's': s.cpu(), 'v': vt.T.cpu()}
        logging.info('SVD of ATS calculated.')

        svd_truth = {} 
        for key in ats_truth:
            logging.debug(f'Calculating SVD for truth: {key}')
            M = ats_truth[key].to(torch.float32)  # SVD function only works with float32
            u, s, vt = torch.linalg.svd(M.cuda(), full_matrices=False)  # Skinny SVD, vt is V^T
            svd_truth[key] = {'u': u.cpu(), 's': s.cpu(), 'v': vt.T.cpu()}
        logging.info('SVD of ATS truth calculated.')
        return svd_hallu, svd_truth
    
    def find_p_hallu(self, svd_hallu, svd_truth):
        hallu_vectors = {}
        for key in svd_hallu.keys():
            layer_num = int(key.split('.')[self.lm_sep_idx])  # Format: 'language_model.model.layers.0.mlp.up_proj.weight'
            
            singular_vectors = svd_hallu[key]['v']  # (D, N): N cols of (D,) vectors

            hallu_vec = []
            if self.top_k_ranks_hallu == 0:
                hallu_vec.append(torch.zeros(self.D))
                logging.info('top_k_ranks_hallu is zero !!!')
            else:
                hallu_rank_list = np.arange(self.top_k_ranks_hallu)  # [0, 1] by default
                for r in hallu_rank_list:
                    singular_vector = singular_vectors[:, r].unsqueeze(dim=1)  # (D, 1)
                    hallu_vec.append(singular_vector)
            
            hallu_vectors[layer_num] = torch.stack(hallu_vec, dim=0)
        logging.info('Hallu vectors caculated.')

        truth_vectors = {}
        for key in svd_truth.keys():
            layer_num = int(key.split('.')[self.lm_sep_idx])  # Format: 'language_model.model.layers.0.mlp.up_proj.weight'

            singular_vectors = svd_truth[key]['v']  # (D, N): N cols of (D,) vectors
            # singular_list.append(singular_vectors) 
            truth_vec = []
            if self.top_k_ranks_truth == 0:
                truth_vec.append(torch.zeros(self.D))
                logging.info('top_k_ranks_truth is zero !!!')
            else:    
                truth_rank_list = np.arange(self.top_k_ranks_truth)  # [0, 1] by default
                for r in truth_rank_list:
                    singular_vector = singular_vectors[:, r].unsqueeze(dim=1)  # (D, 1)
                    truth_vec.append(singular_vector)
            truth_vectors[layer_num] = torch.stack(truth_vec, dim=0)

        logging.info('Truth subspace calculated.')
        return hallu_vectors, truth_vectors

    def save_vectors(self, pos_data, neg_data):
        ats_hallu, ats_truth = self.get_ats(pos_data, neg_data)
        svd_hallu, svd_truth = self.svd_on_ats(ats_hallu, ats_truth)
        del ats_hallu
        del ats_truth
        hallu_vectors, truth_vectors = self.find_p_hallu(svd_hallu, svd_truth)
        del svd_hallu
        del svd_truth
        torch.cuda.empty_cache()
        return hallu_vectors, truth_vectors


    def edit(self, hallu_vectors, truth_vectors):

            edited_state_dict = self.model.model.state_dict()

            for key in self.key_dict:
                layer_num = int(key.split('.')[self.lm_sep_idx])
                if layer_num in self.edit_layer_range:
                    if self.top_k_ranks_hallu[layer_num][0] == -1:
                        hallu_filter = torch.eye(self.D).to(self.device)
                    else:
                        hallu_matrix = torch.zeros(self.D, self.D).to(self.device)
                        for rank in self.top_k_ranks_hallu[layer_num]:
                            hallu_vec = hallu_vectors[layer_num][rank].to(self.device)
                            hallu_matrix += hallu_vec @ hallu_vec.T
                        hallu_filter = torch.eye(self.D).to(self.device) - hallu_matrix

                    if self.top_k_ranks_truth[layer_num][0] == -1:
                        truth_filter = torch.eye(self.D).to(self.device)
                    else:
                        truth_matrix = torch.zeros(self.D, self.D).to(self.device)
                        for rank in self.top_k_ranks_truth[layer_num]:
                            truth_vec = truth_vectors[layer_num][rank].to(self.device)
                            truth_matrix += truth_vec @ truth_vec.T
                        truth_filter = truth_matrix

                    P_filter = truth_filter @ hallu_filter
                    if self.model.args.model_name == 'MiniGPT4':
                        P_filter = P_filter.to(edited_state_dict[key].device).to(self.model.model.llama_model.dtype)
                    else:
                        P_filter = P_filter.to(edited_state_dict[key].device).to(self.model.model.dtype)

                    weight = edited_state_dict[key]
                    weight = weight.T
 
                    if 'down_proj' in key:
                        modified_weight = weight @ P_filter
                        if self.model.args.model_name == 'MiniGPT4':
                            target_layer = self.model.model.llama_model.model.layers[layer_num].mlp.down_proj
                        else:
                            target_layer = self.model.model.model.layers[layer_num].mlp.down_proj # LLAVA

                        def new_forward(self, input, modified_weight=modified_weight):
                            return F.linear(input, modified_weight.T, self.bias)
                        
                        target_layer.forward = types.MethodType(new_forward, target_layer)     
                    elif 'c_proj' in key: # Qwen_VL_Chat
                        print('c_proj')
                        modified_weight = weight @ P_filter
                    else:
                        print('no modified_weight')
                        continue

            if not hasattr(self.model, 'chat'):
                raise AttributeError("The edited model does not have a 'chat' function.")
            return self.model