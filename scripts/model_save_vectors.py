import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shutil
import argparse
import cv2
import json
import numpy as np
import random
import pickle
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from model import build_model
from utils.dynamicedit import DynamicEdit

def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


# pos: halluciation, neg: non_halluciation
def load_embedding_data(pkl_path, loc):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"File not found: {pkl_path}")
    
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)

    pos_data, neg_data = [], []
    for entry in data:
        if entry['label'] == 0:
            pos_data.append(entry[loc])
        else:
            neg_data.append(entry[loc])

    if not pos_data:
        raise ValueError("No positive data found.")
    if not neg_data:
        raise ValueError("No negative data found.")

    pos_data = torch.stack(pos_data).float()
    neg_data = torch.stack(neg_data).float()

    if pos_data.size(0) != neg_data.size(0):
        raise ValueError("Positive and negative data sizes do not match.")

    return pos_data, neg_data


def save_model_and_config(hallu_vectors, truth_vectors, save_path):

    os.makedirs(save_path, exist_ok=True)
    
    torch.save(hallu_vectors, os.path.join(save_path, 'hallu_vectors.pth'))
    torch.save(truth_vectors, os.path.join(save_path, 'truth_vectors.pth'))
    print(f'Saved vectors to {save_path}')
    # print(f'Saved edited model to {save_path}')


def main(args):

    setup_seeds()

    model = build_model(args)
    
    if args.emb_path is not None:
        loc = {
            'mean': 'hidden_states_mean',
            'last': 'hidden_states',
            'mlp_residual': 'mlp_residual'
        }.get(args.ebd)

        pos_data, neg_data = load_embedding_data(args.emb_path, loc=loc)
        print(f'Loading offline embeddings from {args.emb_path}')

    output_dir = os.path.join("./output", args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    
    editor = DynamicEdit(model=model, top_k_ranks_hallu=args.top_k_ranks_hallu, top_k_ranks_truth=args.top_k_ranks_truth, matrix=args.matrix)

    hallu_vectors, truth_vectors = editor.save_vectors(pos_data, neg_data)
    
    # Save edited model
    save_dir = args.tensors_path
    os.makedirs(save_dir, exist_ok=True)

    save_name = f"{args.model_name}-top{args.top_k_ranks_hallu}hallu-top{args.top_k_ranks_truth}truth-{args.ebd}-{args.matrix}"
    save_path = os.path.join(args.tensors_path, save_name)

    save_model_and_config(hallu_vectors, truth_vectors, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Run a model')

    parser.add_argument("--model_name", choices=['LLaVA-7B', 'MiniGPT4', 'Qwen_VL_Chat'], default="LLaVA-7B") 
    parser.add_argument("--model_path", default="/workspace/data1/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590")
    parser.add_argument(
        "--emb_path", type=str, 
        default="./output/MiniGPT4/lure_train_first20_1_42_activations.pkl"
    ) 

    parser.add_argument("--top_k_ranks_hallu", type=int, default=4) #
    parser.add_argument("--top_k_ranks_truth", type=int, default=4) #

    parser.add_argument("--ebd", choices=['mean', 'last', 'mlp_residual'], default='mean')
    parser.add_argument("--matrix", choices=['hidden_states', 'difference'], default="hidden_states") 

    parser.add_argument("--tensors_path", type=str, default="./output/saved_tensors")

    main(parser.parse_args())
