import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import time

import cv2
import json
import numpy as np
from tqdm import tqdm

from model import build_model
from dataset import build_dataset

import random, torch
import torch.backends.cudnn as cudnn
from utils.dynamicedit import DynamicEdit

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'


def dynamic_edit(args, model):
    if args.lowest_layer == -1 or args.highest_layer == -1:
        layer_range = None
    else:
        layer_range = np.arange(args.lowest_layer, args.highest_layer)

    editor = DynamicEdit(model=model, top_k_ranks=args.top_k_ranks, top_k_ranks_truth=args.top_k_ranks_truth, edit_layer_range=layer_range)
    hallu_path = os.path.join(args.tensors_path, 'hallu_vectors.pth')
    truth_path = os.path.join(args.tensors_path, 'truth_vectors.pth')
    hallu_vectors = torch.load(hallu_path)
    truth_vectors = torch.load(truth_path)

    edited_model = editor.edit(hallu_vectors, truth_vectors, edit_keys=args.edit_keys, edit_values=args.edit_values)

    return edited_model
    

def get_model_answer_chair(args, data, model, answer_file):
    
    with open(answer_file, 'w') as ans_file:
        for ins in tqdm(data):
            image_id = ins['image_id']
            image_path = ins['image_path']
            prompt = ins['question']
            
            edited_model = dynamic_edit(args, model)

            response = edited_model.chat(image_path, prompt)

            out = {
                "image_id": image_id,
                "model_name": args.model_name,
                "question": prompt,
                "caption": response,
            }

            ans_file.write(json.dumps(out) + "\n")

    print(f'----CHAIR----\nSaved responses to {answer_file}')


def get_model_answer_pope(args, data, model, answer_file):

    for strategy, sub_data in data.items():

        chat_save_file = answer_file.replace('_chat.jsonl', f'_{strategy}_chat.jsonl')
        result_save_file = answer_file.replace('_chat.jsonl', f'_{strategy}_result.json')
        
        label_list, pred_list = [], []
        with open(chat_save_file, 'w') as ans_file:
            for ins in tqdm(sub_data):

                edited_model = dynamic_edit(args, model)

                response = edited_model.chat(ins['image_path'], ins['question']).strip()

                ins['image_path'] = os.path.basename(ins['image_path'])
                ins['response'] = response
                ins['answer'] = 'no' if any(kw in response.lower() for kw in ["no", "not", "false", f"n't"]) else 'yes'

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
    
    data = build_dataset(args.dataset, args.split, args.sampling, args.num_samples)    

    save_dir = f"./eval/{args.dataset}/{args.tensors_path.split('/')[-1]}_test/"
    os.makedirs(save_dir, exist_ok=True)

    model_tag = (
        f"_t={args.temperature}_" if args.temperature != 0.0 else ""
    ) + f"_beam{args.num_beams}_num{args.max_length}"
    sampling_tag = f"_{args.sampling}{args.num_samples}" if args.num_samples else ""

    save_file = os.path.join(
        save_dir,
        f"{args.split}{sampling_tag}{model_tag}_{args.seed}_chat.jsonl"
    )

    if args.dataset == "chair":
        get_model_answer_chair(args, data, model, save_file)

        from calculate_chair import chair_calculation
        chair_calculation(save_file)

    elif args.dataset == "pope":
        get_model_answer_pope(args, data, model, save_file)

        from calculate_pope import pope_calculation
        pope_calculation(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Run a model')
    parser.add_argument("--model_name", choices=['LLaVA-7B', 'MiniGPT4', 'mPLUG_Owl2', 'Qwen_VL_Chat'], default="MiniGPT4") 
    parser.add_argument("--model_path", default="/workspace/Nullu/output/edited_model/MiniGPT4-top4-16-32-test") 
    parser.add_argument("--dataset", choices=['chair', 'pope'], default="pope")
    parser.add_argument("--split", default="val")

    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--sampling", choices=['first', 'random'], default='random')

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=float, default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")

    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=64)

    parser.add_argument("--seed", type=int, default=114514)

    parser.add_argument("--top_k_ranks", type=int, default=4) #
    parser.add_argument("--top_k_ranks_truth", type=int, default=4) #
    parser.add_argument("--lowest_layer", type=int, default=16) # 31-32,16-32,16-24,24-32
    parser.add_argument("--highest_layer", type=int, default=32) #
    
    # MME
    parser.add_argument("--reference_dir", default="/data/MME_Benchmark_release_version/eval_tool/Your_Results")
    parser.add_argument("--base_dir", default="/workspace/MME")
    
    parser.add_argument("--tensors_path", type=str, default=None)
    parser.add_argument("--edit_keys", action="store_true", default=False)
    parser.add_argument("--edit_values", action="store_true", default=True)
    
    main(parser.parse_args())