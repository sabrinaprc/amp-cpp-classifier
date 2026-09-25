import os
import onlinehd
import importlib.util
import torch
import numpy as np
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
from helpers import (load_sequences, extract_embeddings, run_kfold_hdc,
                     save_report, hdc_grid_search, save_grid_csv)

MODEL_NAME = "PepBERT-large-UniParc"
REPO_ID = "dzjxzyd/" + MODEL_NAME
WEIGHTS_FILE = "tmodel_17.pt"
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS = 5
HDC_DIMS = [5000, 10000, 15000]
HDC_LRS = [0.01, 0.035, 0.1]
HDC_EPOCHS_LIST = [20, 50, 100, 200, 400]
HDC_NORMALIZER = 'l2'  # 'l2' (per-sample L2), 'standard' (z-score), or 'none'
REPORTS_DIR = "reports_epoch_sweep"

def load_module_from_hub(repo_id, filename):
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_model(repo_id, weights_file):
    print(f"Loading PepBERT: {repo_id}")
    model_module = load_module_from_hub(repo_id, "model.py")
    config_module = load_module_from_hub(repo_id, "config.py")
    build_transformer = model_module.build_transformer
    get_config = config_module.get_config
    tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    weights_path = hf_hub_download(repo_id=repo_id, filename=weights_file)
    config = get_config()
    print(f"Embedding dimension: {config['d_model']}  Max seq_len: {config['seq_len']}")
    model = build_transformer(
        src_vocab_size=tokenizer.get_vocab_size(),
        src_seq_len=config["seq_len"],
        d_model=config["d_model"]
    )
    state = torch.load(weights_path, map_location=torch.device(DEVICE))
    model.load_state_dict(state["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, tokenizer, config["seq_len"]

def get_embedding(sequence):
    raw_ids = tokenizer.encode(sequence).ids[: max_seq_len - 2]
    encoded_ids = (
        [tokenizer.token_to_id("[SOS]")]
        + raw_ids
        + [tokenizer.token_to_id("[EOS]")]
    )
    input_ids = torch.tensor([encoded_ids], dtype=torch.int64).to(DEVICE)
    encoder_mask = torch.ones((1, 1, 1, input_ids.size(1)), dtype=torch.int64).to(DEVICE)
    with torch.no_grad():
        emb = model.encode(input_ids, encoder_mask)
        emb_avg = emb[:, 1:-1, :].mean(dim=1)
    return emb_avg.squeeze(0).cpu().numpy()

def main():
    global model, tokenizer, max_seq_len
    print(f"Device: {DEVICE}")
    amp_seqs = load_sequences(os.path.join(DATA_DIR, 'Poreforming_AMPS.txt'))
    cpp_seqs = load_sequences(os.path.join(DATA_DIR, 'Poreforming_CPPs.txt'))
    print(f"  Poreforming_AMPS: {len(amp_seqs)}  Poreforming_CPPs: {len(cpp_seqs)}")

    model, tokenizer, max_seq_len = load_model(REPO_ID, WEIGHTS_FILE)
    n_truncated = sum(1 for s in amp_seqs + cpp_seqs
                      if len(tokenizer.encode(s).ids) > max_seq_len - 2)
    if n_truncated:
        print(f"  Warning: {n_truncated} sequence(s) longer than {max_seq_len-2} tokens will be truncated")
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
                extra_info={"PepBERT variant": REPO_ID, "Weights": WEIGHTS_FILE,
                            "Device": DEVICE, "HDC dim": best['dim'],
                            "HDC lr": best['lr'], "HDC epochs": best['epochs'],
                            "HDC normalizer": HDC_NORMALIZER},
                out_dir=REPORTS_DIR)

if __name__ == '__main__':
    main()
