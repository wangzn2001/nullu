import torch

path = './output/edited_model/LLaVA-7B-top4-top0truth-16-32--31truth-32truth---mean/hallu_vectors.pth'
vectors = torch.load(path)

for key, tensor in vectors.items():
    print(f"Key: {key}, Shape: {tensor.shape}")
print(vectors[30][0].shape)