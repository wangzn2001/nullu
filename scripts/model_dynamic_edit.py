import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import time
import pickle
import cv2
import json
import numpy as np
from tqdm import tqdm

from model import build_model
from dataset import build_dataset

import random, torch
import torch.backends.cudnn as cudnn
from utils.dynamicedit import DynamicEdit
import torch.nn.functional as F
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'


def dynamic_edit(args, model, hallu_vectors, truth_vectors, rank_hallu, rank_truth):
    layer_range = np.arange(args.lowest_layer, args.highest_layer)

    editor = DynamicEdit(model=model, top_k_ranks_hallu=rank_hallu, top_k_ranks_truth=rank_truth, edit_layer_range=layer_range)
    
    edited_model = editor.edit(hallu_vectors, truth_vectors)

    return edited_model
    

def get_k_exceeding_threshold(row_tensor, threshold, nullu=False):
    if nullu is True:
        return 4
    if threshold < 0.0:
        return 0
    abs_row = row_tensor.abs()
    cumsum = abs_row.cumsum(dim=-1)
    
    # 找到第一个位置 k，使得和大于 threshold
    exceed_indices = (cumsum > threshold).nonzero(as_tuple=False)
    if exceed_indices.numel() == 0:
        return row_tensor.size(-1)  # 如果都没超过，返回最大长度
    else:
        return exceed_indices[0].item() + 1  # 加1表示前k个数（包含这个）
    

def calculate_rank(args, hidden_states, hallu_hidden_states, truth_hidden_states):
    rank_hallu = {}
    rank_truth = {}

    for layer in range(args.lowest_layer, args.highest_layer):
        hallu_state = hallu_hidden_states[layer]
        truth_state = truth_hidden_states[layer]
        hidden_state = hidden_states[layer]
        similarity_hallu = F.cosine_similarity(hidden_state, torch.squeeze(hallu_state), dim=1)
        similarity_truth = F.cosine_similarity(hidden_state, torch.squeeze(truth_state), dim=1)
        if args.original is True:
            k_hallu = get_k_exceeding_threshold(similarity_hallu, threshold=-1.0)
            k_truth = get_k_exceeding_threshold(similarity_truth, threshold=-1.0)
        elif args.nullu is True:
            k_hallu = get_k_exceeding_threshold(similarity_hallu, threshold=args.threshold_hallu, nullu=True)
            k_truth = get_k_exceeding_threshold(similarity_truth, threshold=-2.0)
        else:
            k_hallu = get_k_exceeding_threshold(similarity_hallu, threshold=args.threshold_hallu)
            k_truth = get_k_exceeding_threshold(similarity_truth, threshold=args.threshold_truth)
        
        rank_hallu[layer] = k_hallu
        rank_truth[layer] = k_truth

    return rank_hallu, rank_truth


def get_hidden_states(args, model, image_path, prompt, device):
    outputs = model._basic_forward(image_path, prompt, None, return_dict=True)
    hidden_states = torch.stack(outputs.hidden_states)[1:, 0]   # [32, seq_len, 4096]
    if args.nullu is True:
        return hidden_states.mean(1).to(device)
    else:
        return hidden_states[:, -1].to(device)


def get_model_answer_chair(args, data, model, answer_file, hallu_vectors, truth_vectors, device):

    with open(answer_file, 'w') as ans_file:
        for ins in tqdm(data):
            image_id = ins['image_id']
            image_path = ins['image_path']
            prompt = ins['question']
            hidden_states = get_hidden_states(args, model, image_path, prompt, device)
            rank_hallu, rank_truth = calculate_rank(args, hidden_states, hallu_vectors, truth_vectors)

            edited_model = dynamic_edit(args, model, hallu_vectors, truth_vectors, rank_hallu, rank_truth)

            response = edited_model.chat(image_path, prompt)

            out = {
                "image_id": image_id,
                "model_name": args.model_name,
                "question": prompt,
                "caption": response,
                "rank_hallu": rank_hallu,
                "rank_truth": rank_truth
            }
            ans_file.write(json.dumps(out) + "\n")
    print(f'----CHAIR----\nSaved responses to {answer_file}')


def get_model_answer_pope(args, data, model, answer_file, hallu_vectors, truth_vectors, device):

    for strategy, sub_data in data.items():
        chat_save_file = answer_file.replace('_chat.jsonl', f'_{strategy}_{args.threshold_hallu}_{args.threshold_truth}_chat.jsonl')

        with open(chat_save_file, 'w') as ans_file:
            for ins in tqdm(sub_data):
                image_path = ins['image_path']
                prompt = ins['question']

                hidden_states = get_hidden_states(args, model, image_path, prompt, device)
                rank_hallu, rank_truth = calculate_rank(args, hidden_states, hallu_vectors, truth_vectors)

                edited_model = dynamic_edit(args, model, hallu_vectors, truth_vectors, rank_hallu, rank_truth)

                response = edited_model.chat(image_path, prompt).strip()

                ins['image_path'] = os.path.basename(ins['image_path'])
                ins['response'] = response
                ins['answer'] = 'no' if any(kw in response.lower() for kw in ["no", "not", "false", f"n't"]) else 'yes'
                ins['rank_hallu'] = rank_hallu
                ins['rank_truth'] = rank_truth

                ans_file.write(json.dumps(ins) + '\n')

    print(f'----POPE----\nSaved responses to {answer_file}')
        

def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def main(args):

    setup_seeds(args.seed)

    model = build_model(args)
    
    device = next(model.model.parameters()).device

    data = build_dataset(args.dataset, args.split, args.sampling, args.num_samples)
    hallu_path = os.path.join(args.tensors_path, 'hallu_vectors.pth')
    truth_path = os.path.join(args.tensors_path, 'truth_vectors.pth')
    hallu_vectors = torch.load(hallu_path)
    truth_vectors = torch.load(truth_path)
    hallu_vectors = {k: v.to(device) for k, v in hallu_vectors.items()}
    truth_vectors = {k: v.to(device) for k, v in truth_vectors.items()}
    
    if args.original is True:
        save_dir = f"./eval/{args.dataset}/{args.model_name}/"
    else:
        save_dir = f"./eval/{args.dataset}/{args.tensors_path.split('/')[-1]}--layer{args.lowest_layer}_{args.highest_layer}/"
    
    os.makedirs(save_dir, exist_ok=True)

    model_tag = f"_beam{args.num_beams}_token{args.max_length}"
    sampling_tag = f"_{args.sampling}{args.num_samples}" if args.num_samples else ""

    save_file = os.path.join(
        save_dir,
        f"{args.split}{sampling_tag}{model_tag}_seed{args.seed}_chat.jsonl"
    )
    
    if args.dataset == "chair":
        get_model_answer_chair(args, data, model, save_file, hallu_vectors, truth_vectors, device)

        from calculate_chair import chair_calculation
        chair_calculation(save_file)

    elif args.dataset == "pope":
        get_model_answer_pope(args, data, model, save_file, hallu_vectors, truth_vectors, device)

        from calculate_pope import pope_calculation
        pope_calculation(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Run a model')
    parser.add_argument("--model_name", choices=['LLaVA-7B', 'MiniGPT4', 'mPLUG_Owl2', 'Qwen_VL_Chat'], default="MiniGPT4") 
    parser.add_argument("--model_path", default="/workspace/Nullu/output/edited_model/MiniGPT4-top4-16-32-test") 
    parser.add_argument("--tensors_path", type=str, default=None)

    parser.add_argument("--dataset", choices=['chair', 'pope'], default="pope")
    parser.add_argument("--split", default="val")

    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--sampling", choices=['first', 'random'], default='random')

    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=64)

    parser.add_argument("--seed", type=int, default=114514)

    parser.add_argument("--threshold_hallu", type=float, default=4) #
    parser.add_argument("--threshold_truth", type=float, default=4) #

    parser.add_argument("--lowest_layer", type=int, default=16) # 31-32,16-32,16-24,24-32
    parser.add_argument("--highest_layer", type=int, default=32) #
    
    parser.add_argument("--original", type=bool, default=False) #
    parser.add_argument("--nullu", type=bool, default=False) #
    # MME
    # parser.add_argument("--reference_dir", default="/data/MME_Benchmark_release_version/eval_tool/Your_Results")
    # parser.add_argument("--base_dir", default="/workspace/MME")
    
    main(parser.parse_args())