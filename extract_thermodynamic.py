import argparse
import numpy as np
import RNA
from data_utils import load_df, get_sequences

OUTPUT_PATH = "thermo_features.npy"


def dna_to_rna(seq: str) -> str:
    return seq.replace("T", "U")


def compute_thermo(seq: str) -> list[float]:
    rna = dna_to_rna(seq)
    _, mfe = RNA.fold(rna)
    fc = RNA.fold_compound(rna)
    ensemble_energy = fc.pf()[1]
    return [mfe, ensemble_energy]


def extract_all(sequences: list[str]) -> np.ndarray:
    results = []
    for i, seq in enumerate(sequences):
        feats = compute_thermo(seq)
        results.append(feats)
        if i % 1000 == 0:
            print(f"  {i}/{len(sequences)} sequences processed")
    return np.array(results, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_23k_v1.csv")
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    print("Loading data...")
    df = load_df(args.data)
    sequences = get_sequences(df)
    print(f"Sequences to process: {len(sequences)}")

    print("Computing thermodynamic features (MFE, ensemble free energy)...")
    feats = extract_all(sequences)
    print(f"Thermo features shape: {feats.shape}")

    np.save(args.output, feats)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
