import argparse
import numpy as np
import torch
from transformers import AutoTokenizer
from data_utils import load_df, get_sequences

MODEL_ID = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
BATCH_SIZE = 64
OUTPUT_PATH = "embeddings.npy"


def mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def extract(
    sequences: list[str],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        encoded = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        pooled = mean_pool(hidden, attention_mask)
        all_embeddings.append(pooled.cpu().float().numpy())
        if (i // batch_size) % 10 == 0:
            print(f"  {i + len(batch_seqs)}/{len(sequences)} sequences embedded")
    return np.concatenate(all_embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_23k_v1.csv")
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    import sys, os, importlib.util

    model_dir = os.path.abspath(args.model)

    def _load_module(name, path):
        spec = importlib.util.spec_from_file_location(
            name, path, submodule_search_locations=[model_dir]
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    esm_config_mod = _load_module(
        "esm_config", os.path.join(model_dir, "esm_config.py")
    )
    modeling_mod = _load_module(
        "modeling_esm", os.path.join(model_dir, "modeling_esm.py")
    )
    NTEsmConfig = esm_config_mod.EsmConfig
    NTEsmModel = modeling_mod.EsmModel
    config = NTEsmConfig.from_pretrained(args.model)
    model = NTEsmModel.from_pretrained(args.model, config=config)
    model = model.half().to(device).eval()

    print("Loading data...")
    df = load_df(args.data)
    sequences = get_sequences(df)
    print(f"Sequences to embed: {len(sequences)}")

    print("Extracting embeddings...")
    embeddings = extract(sequences, model, tokenizer, device, args.batch_size)
    print(f"Embeddings shape: {embeddings.shape}")

    np.save(args.output, embeddings)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
