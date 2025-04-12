import torch
import json

def get_k_exceeding_threshold(row_tensor, threshold):
    abs_row = row_tensor.abs()
    cumsum = abs_row.cumsum(dim=-1)
    
    # 找到第一个位置 k，使得和大于 threshold
    exceed_indices = (cumsum > threshold).nonzero(as_tuple=False)
    if exceed_indices.numel() == 0:
        return 100  # 如果都没超过，返回最大长度
    else:
        return exceed_indices[0].item() + 1  # 加1表示前k个数（包含这个）

file_path = 'test.jsonl'  # 替换为你的文件路径

with open(file_path, 'r') as f:
    for line_num, line in enumerate(f, start=1):
        data = json.loads(line)
        
        # 转为tensor
        rank_hallu_tensor = torch.tensor(data['rank_hallu'])[1]
        rank_truth_tensor = torch.tensor(data['rank_truth'])[1]
        print(get_k_exceeding_threshold(rank_hallu_tensor, 0.8))
        print(get_k_exceeding_threshold(rank_truth_tensor, 3))