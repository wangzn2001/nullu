import json

def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

def compute_marco(tp, fp, tn, fn):
    f1_pos = compute_f1(tp, fp, fn)
    f1_neg = compute_f1(tn, fn, fp)
    return (f1_pos + f1_neg) /2

strategy = "random"
model = "LLaVA-7B"
original_model = {
    "LLaVA-7B": 'llava-v1.5-7b',
    "MiniGPT4": 'Llama-2-7b-chat-hf'
}
n = 0
with open(f"eval/pope/{model}-top100-top100truth-30-32--last__dif/val_random500_beam3_num10_42_{strategy}_chat_vec_0.3_2.0.jsonl", "r") as f:

    TN = 0
    TP = 0
    FN = 0
    FP = 0
    for line in f:
        data = json.loads(line)
        if data['label']=='yes' and data['answer']=='yes':
            TP += 1
        elif data['label']=='no' and data['answer']=='yes':
            FP += 1
        elif data['label']=='no' and data['answer']=='no':
            TN += 1
        elif data['label']=='yes' and data['answer']=='no':
            FN += 1
        
        # rank_hallu = torch.tensor(data["rank_hallu"])  # [2, 100]
        # rank_truth = torch.tensor(data["rank_truth"])  # [2, 100]

        # k_hallu = [get_k_exceeding_threshold(row, threshold=1.0) for row in rank_hallu]
        # k_truth = [get_k_exceeding_threshold(row, threshold=2.0) for row in rank_truth]

        # results.append({
        #     "k_hallu": k_hallu,  # [k_row0, k_row1]
        #     "k_truth": k_truth
        # })
        n+=1
    print("----------Dynamic------------")
    f1 = compute_f1(TP, FP, FN)
    ba = compute_marco(TP, FP, TN, FN)
    print(TP)
    print(FP)
    print(TN)
    print(FN)
    print(f1)
    print(ba)

with open(f"eval/pope/{model}-top100-top100truth-30-32--last/val_random500_beam3_num10_42_{strategy}_chat_vec_0.3_1.0.jsonl", "r") as f:
    cnt = 0
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    for line in f:
        cnt += 1
        if cnt > n: break
        data = json.loads(line)
        if data['label']=='yes' and data['answer']=='yes':
            TP += 1
        elif data['label']=='no' and data['answer']=='yes':
            FP += 1
        elif data['label']=='no' and data['answer']=='no':
            TN += 1
        elif data['label']=='yes' and data['answer']=='no':
            FN += 1
        
        # rank_hallu = torch.tensor(data["rank_hallu"])  # [2, 100]
        # rank_truth = torch.tensor(data["rank_truth"])  # [2, 100]

        # k_hallu = [get_k_exceeding_threshold(row, threshold=1.0) for row in rank_hallu]
        # k_truth = [get_k_exceeding_threshold(row, threshold=2.0) for row in rank_truth]

        # results.append({
        #     "k_hallu": k_hallu,  # [k_row0, k_row1]
        #     "k_truth": k_truth
        # })
    print("----------stable 100------------")
    f1 = compute_f1(TP, FP, FN)
    ba = compute_marco(TP, FP, TN, FN)
    print(TP)
    print(FP)
    print(TN)
    print(FN)
    print(f1)
    print(ba)


with open(f"eval/pope/{original_model[model]}/val_random500_beam1_num10_42_{strategy}_chat.jsonl", "r") as f:
    cnt = 0
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    for line in f:
        cnt += 1
        if cnt > n: break
        data = json.loads(line)
        if data['label']=='yes' and data['answer']=='yes':
            TP += 1
        elif data['label']=='no' and data['answer']=='no':
            TN += 1
        elif data['label']=='no' and data['answer']=='yes':
            FP += 1
        elif data['label']=='yes' and data['answer']=='no':
            FN += 1
        
        # rank_hallu = torch.tensor(data["rank_hallu"])  # [2, 100]
        # rank_truth = torch.tensor(data["rank_truth"])  # [2, 100]

        # k_hallu = [get_k_exceeding_threshold(row, threshold=1.0) for row in rank_hallu]
        # k_truth = [get_k_exceeding_threshold(row, threshold=2.0) for row in rank_truth]

        # results.append({
        #     "k_hallu": k_hallu,  # [k_row0, k_row1]
        #     "k_truth": k_truth
        # })
    print("----------Vanilla------------")
    f1 = compute_f1(TP, FP, FN)
    ba = compute_marco(TP, FP, TN, FN)
    print(TP)
    print(FP)
    print(TN)
    print(FN)
    print(f1)
    print(ba)


# 示例打印结果
# for r in results:
#     print(r)