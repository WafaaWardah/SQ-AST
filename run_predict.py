# Wafaa Wardah, TU-Berlin, 2025

import multiprocessing as mp, threading, sys, gc, logging, os
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
import argparse
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from transformers import ASTModel, ASTFeatureExtractor, logging as hf_logging



#hf_logging.set_verbosity_error()  # Silence unnecessary warnings from huggingface
torch.multiprocessing.set_sharing_strategy('file_system')



class ASTXL(torch.nn.Module):
    """Model-XL with individual AST for each dimension."""

    PRETRAINED_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

    def __init__(self, pretrained_model: str = PRETRAINED_MODEL) -> None:
        super(ASTXL, self).__init__()
        try:
            self.ast = ASTModel.from_pretrained(pretrained_model)
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained model from {pretrained_model}") from e

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(768, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden_state = self.ast(features).pooler_output
        pred = self.fc(hidden_state).squeeze()
        return pred



class ASTVal(Dataset):
    def __init__(self, df, data_dir, db_mean, db_std):
        self.df = df
        self.data_dir = data_dir
        self.feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.feature_extractor.sampling_rate = 48000  # Set to 48 kHz for our 48k audio
        self.feature_extractor.max_length = 1024      # Truncate inputs after 1024 patches
        self.feature_extractor.num_mel_bins = 128     # Customize if needed; keep original as default
        self.feature_extractor.return_attention_mask = True
        self.feature_extractor.mean = db_mean
        self.feature_extractor.std = db_std

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):

        TARGET_SR = 48_000

        file_name = os.path.join(self.data_dir, self.df['file_path'].iloc[index])
        waveform, sample_rate = torchaudio.load(file_name)

        waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze()

        if sample_rate != TARGET_SR:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=TARGET_SR
            )
            waveform = resampler(waveform)
            sample_rate = TARGET_SR

        features = self.feature_extractor(
            waveform, 
            sampling_rate=sample_rate, 
            return_attention_mask=True, 
            return_tensors="pt"
        )['input_values']

        features = features.squeeze()
        return index, features



def main():
    parser = argparse.ArgumentParser(description="Run the prediction script.")

    parser.add_argument("--path", type=str, help="Path to the wav file or to the data directory")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory if saving the results", default=None)

    parser.add_argument(
        "--weights_dir",
        type=str,
        default="./weights",
        help="Directory containing model weight files (mos.pth, noi.pth, ...). Default: ./weights"
    )
    
    parser.add_argument(
        "--dims",
        nargs="+",                     # allows multiple values (e.g. --dims mos noi)
        choices=["mos", "noi", "dis", "col", "loud"],
        default=["mos", "noi", "dis", "col", "loud"],  # default: all
        help=(
            "Dimensions to predict. "
            "Choose from: mos, noi, dis, col, loud. "
            "Default: all"
        ),
    )

    parser.add_argument(
        "--overwrite",
        type=str,
        default="true",
        help="Whether to recompute scores for files already present in the output CSV (default: true)"
    )

    parser.add_argument("--device", type=str, help="Device: cpu or gpu, defaults is set to cpu", default="cpu")
    parser.add_argument("--bs", type=str, help="Batch size, default is set to 1", default=1)
    parser.add_argument("--nw", type=str, help="Number of workers, default is set to 0", default=0)
    parser.add_argument("--print", type=str, help="Option to print the predicted scores out, default is set to False", default=False)

    args = parser.parse_args()

    overwrite = str(args.overwrite).lower() in ["true", "1", "yes"]
    print_enabled = args.print in [True, "true", "True", "1", 1]

    weights_path = args.weights_dir
    if not os.path.isdir(weights_path):
        raise FileNotFoundError(f"Weights directory not found: {weights_path}")

    db_mean = -10.25446422 # calculated from the validation datasets from July 2025
    db_std = 4.205750774 # calculated from the validation datasets from July 2025
    
    print("=" * 80)
    print("    P.SAMD (v1 2025) - Speech Quality Inference")
    print("=" * 80)
    
    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    if args.path.endswith(".wav"):
        wav_path = args.path
        data_dir = None
    else:
        data_dir = args.path
        wav_path = None

    if data_dir:
        db_name = os.path.basename(data_dir)
        file_paths = sorted([f for f in os.listdir(data_dir) if f.endswith(".wav")])
        df = pd.DataFrame({
            "db": db_name,
            "file_path": file_paths,
            "file_num": range(1, len(file_paths) + 1),
            "db_mean": db_mean,
            "db_std": db_std
        })
        print(f"Predicting quality for directory: {data_dir}")

    else:
        db_name = os.path.basename(wav_path).replace(".wav", "")
        df = pd.DataFrame({
            "db": [db_name],
            "file_path": [os.path.basename(wav_path)],
            "file_num": [1],
            "db_mean": [db_mean],
            "db_std": [db_std]
        })
        print(f"Predicting quality for single file")

    # check for existing results or create output file
    outfile = None
    processed_files_set = set()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        outfile = os.path.join(
            args.output_dir,
            f"{db_name}_prediction_per_file.csv"
        )

        if os.path.exists(outfile):
            if overwrite:
                os.remove(outfile)
            else:
                existing = pd.read_csv(outfile)
                processed_files_set = set(
                    existing["file_path"].astype(str).apply(os.path.basename)
                )

        if not os.path.exists(outfile):
            header = [
                "db",
                "file_path",
                "file_num",
                "mos_pred",
                "noi_pred",
                "dis_pred",
                "col_pred",
                "loud_pred"
            ]
            pd.DataFrame(columns=header).to_csv(outfile, index=False)

    skipped = 0

    if processed_files_set and not overwrite:
        df["file_path"] = df["file_path"].astype(str).apply(os.path.basename)
        before = len(df)
        df = df[~df["file_path"].isin(processed_files_set)].reset_index(drop=True)
        skipped = before - len(df)
        if skipped > 0:
            print(f"Skipping {skipped} already processed files.")

    ds = ASTVal(df, data_dir if data_dir else os.path.dirname(wav_path), db_mean, db_std)

    dl = DataLoader(
        dataset=ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.nw
    )

    ALL_DIMS = ["mos", "noi", "dis", "col", "loud"]
    COL_IDX = {d: i for i, d in enumerate(ALL_DIMS)}

    models_by_dim = {}
    for dim in args.dims:
        model = ASTXL()
        state = torch.load(
            os.path.join(weights_path, f"{dim}.pth"),
            map_location=torch.device(device),
            weights_only=True
        )
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        models_by_dim[dim] = model

    print(f"Model loaded") 

    total_files = skipped + len(ds)
    processed_counter = skipped
     
    print("\nCalculating quality scores...")

    with torch.no_grad():
        for index, batch_features in dl:
            batch_features = batch_features.float().to(device)
            y_hat = torch.full((len(index), 5), -0.25)

            for dim, model in models_by_dim.items():
                pred = model(batch_features).squeeze(-1).cpu()
                y_hat[:, COL_IDX[dim]] = pred

            for i, idx in enumerate(index.tolist()):
                processed_counter += 1

                # compute descaled scores for this file
                scores = (y_hat[i] * 4 + 1).numpy()
                file_path = ds.df.loc[idx, "file_path"]

                if print_enabled:
                    parts = [f"{d.upper()}: {scores[COL_IDX[d]]:.2f}" for d in args.dims]
                    print(
                        f"({processed_counter}/{total_files}) "
                        f"{os.path.basename(file_path)} | "
                        + ", ".join(parts)
                    )

                # write CSV after each processed file
                if outfile:
                    row = {
                        "db": ds.df.loc[idx, "db"],
                        "file_path": file_path,
                        "file_num": ds.df.loc[idx, "file_num"],
                        "mos_pred": scores[0],
                        "noi_pred": scores[1],
                        "dis_pred": scores[2],
                        "col_pred": scores[3],
                        "loud_pred": scores[4],
                    }

                    pd.DataFrame([row]).to_csv(
                        outfile,
                        mode="a",
                        header=False,
                        index=False
                    )

    # provide user confirmation about saved output file
    if outfile:
        print("\nSaved predicted scores:", outfile)



if __name__ == "__main__":
    main()

    # --- Graceful shutdown ---
    # 1) Close logging and flush stdio
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()

    # 2) Ensure DataLoader workers are gone
    kids = mp.active_children()
    for p in kids:
        p.terminate()
    for p in kids:
        p.join(timeout=1)

    # 3) Join any non-main Python threads briefly
    for t in [t for t in threading.enumerate() if t.name != "MainThread"]:
        t.join(timeout=1)

    # 4) GC to release file descriptors
    gc.collect()

    # 5) Last-resort hard exit if something still blocks (Mac-safe)
    os._exit(0)
