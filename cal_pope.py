import json
from sklearn.metrics import f1_score, confusion_matrix

def compute_f1_score(jsonl_path):
    labels = []
    answers = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            labels.append(1 if data["label"].lower() == "yes" else 0)
            answers.append(1 if data["answer"].lower() == "yes" else 0)
    f1 = f1_score(labels, answers)
    tn, fp, fn, tp = confusion_matrix(labels, answers).ravel()
    
    print(f"F1 Score: {f1:.4f}")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives (TN): {tn}")

# 替换为你的文件路径
jsonl_file_path = "eval/pope/llava-v1.5-7b/val_random500_beam1_num128_42_random_chat.jsonl"

compute_f1_score(jsonl_file_path)