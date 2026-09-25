import os
import onlinehd
import torch
import numpy as np
from helpers import (load_sequences, extract_embeddings, run_kfold_hdc,
                     save_report, hdc_grid_search, save_grid_csv,
                     load_esm2_model, get_esm2_embedding)

MODEL_NAME = "esm2_t12_35M_UR50D"
REPO_ID = "facebook/" + MODEL_NAME
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS = 5
HDC_DIMS = [5000, 10000, 15000]
HDC_LRS = [0.01, 0.035, 0.1]
HDC_EPOCHS_LIST = [20, 50, 100, 200, 400]
HDC_NORMALIZER = 'l2'  # 'l2' (per-sample L2), 'standard' (z-score), or 'none'
REPORTS_DIR = "reports_epoch_sweep"

def load_model(repo_id):
    return load_esm2_model(repo_id, device=DEVICE)

def get_embedding(sequence):
    return get_esm2_embedding(sequence, tokenizer, model)

def main():
    global model, tokenizer
    print(f"Device: {DEVICE}")
    amp_seqs = load_sequences(os.path.join(DATA_DIR, 'Poreforming_AMPS.txt'))
    cpp_seqs = load_sequences(os.path.join(DATA_DIR, 'Poreforming_CPPs.txt'))
    print(f"  Poreforming_AMPS: {len(amp_seqs)}  Poreforming_CPPs: {len(cpp_seqs)}")

    tokenizer, model = load_model(REPO_ID)
    amp_emb = extract_embeddings(get_embedding, amp_seqs, "AMP")
    cpp_emb = extract_embeddings(get_embedding, cpp_seqs, "CPP")
    X = np.vstack([amp_emb, cpp_emb])
    y = np.array([0] * len(amp_seqs) + [1] * len(cpp_seqs))

    results, best = hdc_grid_search(X, y, MODEL_NAME + "_HDC",
                                    dims=HDC_DIMS, lrs=HDC_LRS,
                                    epochs_list=HDC_EPOCHS_LIST,
                                    normalizer=HDC_NORMALIZER)
    save_grid_csv(results, MODEL_NAME + "_HDC", out_dir=REPORTS_DIR)

    # Save full report for the best config
    print(f"\nRe-running best config for full report: {best}")
    model_name = f"{MODEL_NAME}_HDC_best"
    fold_accuracies, all_y_test, all_y_pred = run_kfold_hdc(
        X, y, model_name, N_FOLDS,
        dim=best['dim'], lr=best['lr'], epochs=best['epochs'],
        normalizer=HDC_NORMALIZER)
    save_report(fold_accuracies, all_y_test, all_y_pred, model_name,
                extra_info={"ESM2 variant": REPO_ID, "Device": DEVICE,
                            "HDC dim": best['dim'], "HDC lr": best['lr'],
                            "HDC epochs": best['epochs'],
                            "HDC normalizer": HDC_NORMALIZER},
                out_dir=REPORTS_DIR)

if __name__ == '__main__':
    main()
