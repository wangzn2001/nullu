import json
import matplotlib.pyplot as plt
import os
from collections import Counter
import numpy as np

# === 配置 ===
chat_file = 'eval/pope/llava-v1.5-7b/val_random500_beam1_num10_42_popular_chat.jsonl'   # 替换为第一个文件路径
file1 = 'eval/pope/LLaVA-7B-top100-top100truth-30-32--last/val_random500_beam3_num10_42_random_chat_rank.jsonl'   # 替换为第一个文件路径
file2 = 'eval/pope/LLaVA-7B-top100-top100truth-30-32--last/val_random500_beam3_num10_42_popular_chat_rank.jsonl'   # 替换为第二个文件路径
save_dir = 'eval/pope/LLaVA-7B-top100-top100truth-30-32--last'                # 保存图像的文件夹
os.makedirs(save_dir, exist_ok=True)

def load_chat(chat_path, rank_path):
    data = []
    idx_true = []
    idx_false = []
    n = 0
    with open(chat_path, 'r') as f:
        for line in f:
            tmp = json.loads(line)
            if(tmp['answer'] == 'yes') :
                idx_true.append(n)
            else:
                idx_false.append(n)
            n += 1
    with open(rank_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    rank_hallu = [[item['rank_hallu'][i] for item in data] for i in range(2)]
    rank_truth = [[item['rank_truth'][i] for item in data] for i in range(2)]

    
    rank_hallu_true = [
    [rank_hallu[0][i] for i in idx_true],  # 第0列在idx_true上的子集
    [rank_hallu[1][i] for i in idx_true]
    ]
    rank_truth_true = [
        [rank_truth[0][i] for i in idx_true],
        [rank_truth[1][i] for i in idx_true]
    ]

    rank_hallu_false = [
        [rank_hallu[0][i] for i in idx_false],
        [rank_hallu[1][i] for i in idx_false]
    ]
    rank_truth_false = [
        [rank_truth[0][i] for i in idx_false],
        [rank_truth[1][i] for i in idx_false]
    ]

    
    return rank_hallu_true, rank_truth_true, rank_hallu_false, rank_truth_false

def load_ranks(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    rank_hallu = [[item['rank_hallu'][i] for item in data] for i in range(2)]
    rank_truth = [[item['rank_truth'][i] for item in data] for i in range(2)]
    return rank_hallu, rank_truth

# === 加载两个文件
rank_hallu1, rank_truth1 = load_ranks(file1)
rank_hallu2, rank_truth2 = load_ranks(file2)

rank_hallu_true, rank_truth_true, rank_hallu_false, rank_truth_false = load_chat(chat_file, file1)

# === 对比绘图函数 ===
def plot_rank_comparison_multiple(datasets, labels, title, save_name):
    # datasets: List of rank lists，e.g. [[file1_hallu], [file1_truth], [file2_hallu], [file2_truth]]
    counters = [Counter(r) for r in datasets]

    # 所有出现的 rank 值
    all_ranks = sorted(set().union(*[c.keys() for c in counters]))
    x = np.arange(len(all_ranks))
    bar_width = 0.2

    plt.figure(figsize=(10, 5))

    # 画每一组数据
    for i, (counter, label) in enumerate(zip(counters, labels)):
        values = [counter.get(rank, 0) for rank in all_ranks]
        plt.bar(x + (i - 1.5) * bar_width, values, width=bar_width, label=label, edgecolor='black')

    plt.xticks(x, all_ranks, rotation=45)
    plt.xlabel('Rank Value')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()

    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved: {save_path}")

# === 绘图：rank[0]
plot_rank_comparison_multiple(
    datasets=[
        rank_hallu1[0], rank_truth1[0],
        rank_hallu2[0], rank_truth2[0]
    ],
    labels=[
        'random_hallu[0]', 'random_truth[0]',
        'popular_hallu[0]', 'popular_truth[0]'
    ],
    title='Rank[0] Distribution Comparison (random vs popular)',
    save_name='rank_0_comparison_multi.png'
)

# === 绘图：rank[1]
plot_rank_comparison_multiple(
    datasets=[
        rank_hallu1[1], rank_truth1[1],
        rank_hallu2[1], rank_truth2[1]
    ],
    labels=[
        'file1_hallu[1]', 'file1_truth[1]',
        'file2_hallu[1]', 'file2_truth[1]'
    ],
    title='Rank[1] Distribution Comparison (File1 vs File2)',
    save_name='rank_1_comparison_multi.png'
)

plot_rank_comparison_multiple(
    datasets=[
        rank_hallu_true[0], rank_truth_true[0],
        rank_hallu_false[0], rank_truth_false[0]
    ],
    labels=[
        'rank_hallu_true[0]', 'rank_truth_true[0]',
        'rank_hallu_false[0]', 'rank_truth_false[0]'
    ],
    title='Rank[0] TF Distribution Comparison (random vs popular)',
    save_name='rank_0_comparison_multi_tf_popular.png'
)

plot_rank_comparison_multiple(
    datasets=[
        rank_hallu_true[1], rank_truth_true[1],
        rank_hallu_false[1], rank_truth_false[1]
    ],
    labels=[
        'rank_hallu_true[1]', 'rank_truth_true[1]',
        'rank_hallu_false[1]', 'rank_truth_false[1]'
    ],
    title='Rank[1] TF Distribution Comparison (random vs popular)',
    save_name='rank_1_comparison_multi_tf_popular.png'
)