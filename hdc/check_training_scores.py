import os
import sys
import numpy as np
import torch
from sklearn.preprocessing import Normalizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# Match the exact config used during training
HDC_DIM = 5000
HDC_LR = 0.01
HDC_EPOCHS = 400
SEED = 42

CACHE_DIR = os.path.join(ROOT, 'embeddings')
CKPT_DIR = os.path.join(HERE, 'checkpoints')
TRAIN_EMB_CACHE = os.path.join(CACHE_DIR, 'esm2_t12_35M_UR50D_poreforming_amp_cpp.npz')
HDC_CKPT = os.path.join(
    CKPT_DIR, f'esm2_hdc_amp_vs_cpp_d{HDC_DIM}_lr{HDC_LR}_ep{HDC_EPOCHS}_seed{SEED}.pt')

# ---- Load the same training embeddings used to fit the model ----
d = np.load(TRAIN_EMB_CACHE)
X_train, y_train = d['X'], d['y']
print(f"Loaded training embeddings: {X_train.shape[0]} peptides")

# ---- Same L2 normalization as training ----
Xtr = Normalizer(norm='l2').fit(X_train).transform(X_train)

# ---- Load the fitted model ----
model = torch.load(HDC_CKPT, map_location=torch.device('cpu'), weights_only=False)
model.eval() if hasattr(model, 'eval') else None

Xtr_tensor = torch.from_numpy(Xtr).float()

# ---- Get per-sample scores ----
# onlinehd's .probabilities() returns softmax-style class probabilities per sample
probs = model.probabilities(Xtr_tensor).detach().cpu().numpy()
preds = probs.argmax(axis=1)

correct = preds == y_train
print(f"\nTraining accuracy (sanity check): {correct.mean():.4f}")

# Confidence = how far the winning class's probability is from 0.5 (max uncertainty)
winning_prob = probs.max(axis=1)
print(f"\nWinning-class probability stats (training set):")
print(f"  Mean: {winning_prob.mean():.4f}")
print(f"  Min:  {winning_prob.min():.4f}")
print(f"  Max:  {winning_prob.max():.4f}")
print(f"  Std:  {winning_prob.std():.4f}")

print(f"\nSplit by correct vs incorrect predictions:")
print(f"  Correct predictions   -- mean winning prob: {winning_prob[correct].mean():.4f}")
print(f"  Incorrect predictions -- mean winning prob: {winning_prob[~correct].mean():.4f}")
