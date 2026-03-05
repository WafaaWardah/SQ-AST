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
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

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

    parser.add_argument("--device", type=str, help="Device: cpu or gpu, defaults is set to cpu", default="cpu")
    parser.add_argument("--bs", type=str, help="Batch size, default is set to 1", default=1)
    parser.add_argument("--nw", type=str, help="Number of workers, default is set to 0", default=0)

    parser.add_argument("--print", type=str, help="Option to print the predicted scores out, default is set to False", default=False)

    parser.add_argument(
        "--overwrite",
        type=str,
        default="true",
        help="Whether to recompute scores for files already present in the output CSV (default: true)"
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    dims = args.dims
    bs = int(args.bs)
    num_workers = int(args.nw)

    if str(args.path).endswith('.wav'): 
        wav_path = args.path
        data_dir = None
    else: 
        data_dir = args.path
        wav_path = None

    weights_path = args.weights_dir
    if not os.path.isdir(weights_path):
        raise FileNotFoundError(f"Weights directory not found: {weights_path}")

    overwrite = str(args.overwrite).lower() in ["true", "1", "yes"]

    db_mean = -10.25446422 # calculated from the validation datasets from July 2025
    db_std = 4.205750774 # calculated from the validation datasets from July 2025
    
    if args.device == "gpu" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print('=' * 80)
    print("    P.SAMD (v1 2025) - Speech Quality Inference")
    print('=' * 80)
    
    print(f"\nUsing device: {device}")

    dtype_dict = {
            'db': str,
            'file_path': str,
            'file_num': float,
            'db_mean': float,
            'db_std': float
        }

    if data_dir:
        db_name = os.path.basename(data_dir)

        print(f"Predicting quality {dims} for audio files in directory...")

        df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in dtype_dict.items()})
        file_paths = [os.path.basename(f) for f in os.listdir(data_dir) if f.endswith('.wav')]

        df['file_path'] = file_paths
        df['db'] = db_name # data_dir basename
        df['file_num'] = range(1, len(file_paths) + 1)
        df['db_mean'] = db_mean
        df['db_std'] = db_std
   
   
        ds = ASTVal(df, data_dir, db_mean, db_std)

    else: # single file
        db_name = os.path.basename(str(wav_path).replace('.wav',''))

        print(f"Predicting quality {dims} for single audio file...")

        df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in dtype_dict.items()})

        df = pd.DataFrame({
            'db': [db_name],
            'file_path': [os.path.basename(wav_path)],
            'file_num': [1],
            'db_mean': [db_mean],
            'db_std': [db_std]
        })

        ds = ASTVal(df, os.path.dirname(wav_path), db_mean, db_std)

    # check for existing results or create output file
    outfile = None
    existing_results = None

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        outfile = os.path.join(
            output_dir,
            db_name + '_prediction_per_file_' + current_time + '.csv'
        )

        # if overwrite disabled, try loading an existing file
        if not overwrite:
            existing_files = [
                f for f in os.listdir(output_dir)
                if f.startswith(db_name + "_prediction_per_file_") and f.endswith(".csv")
            ]
            if existing_files:
                existing_files.sort()
                outfile = os.path.join(output_dir, existing_files[-1])
                existing_results = pd.read_csv(outfile)
    
    # initialize the columns once before inference so the CSV always has the same structure
    for c in ['mos_pred','noi_pred','dis_pred','col_pred','loud_pred']:
        ds.df[c] = None

    # handle existing results
    if existing_results is not None:
        processed = set(existing_results["file_path"].values)
        before = len(ds.df)
        ds.df = ds.df[~ds.df["file_path"].isin(processed)].reset_index(drop=True)
        skipped = before - len(ds.df)

        if skipped > 0:
            print(f"Skipping {skipped} already processed files.")

    dl = DataLoader(
        dataset=ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers
    )

    ALL_DIMS = ["mos", "noi", "dis", "col", "loud"]
    COL_IDX = {d: i for i, d in enumerate(ALL_DIMS)}

    models_by_dim = {}
    for dim in dims:  # dims comes from argparse
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

    y_hat_val = torch.full((len(ds), 5), -0.25, device='cpu') # Stores the validation outputs, later filled into ds_val df

    total_files = len(ds.df) 
    processed_files = 0       

    with torch.no_grad():  # Disable gradient tracking for inference
        print("\nCalculating quality scores...")
        for b, (index, batch_features) in enumerate(dl):

            batch_features = batch_features.float().to(device)

            for dim, model in models_by_dim.items():
                pred = model(batch_features).squeeze(-1).detach().cpu()
                y_hat_val[index, COL_IDX[dim]] = pred 

            for idx in index.tolist():
                processed_files += 1

                # compute descaled scores for this file
                scores = (y_hat_val[idx] * 4 + 1).numpy()

                ds.df.loc[int(idx), 'mos_pred'] = scores[0]
                ds.df.loc[int(idx), 'noi_pred'] = scores[1]
                ds.df.loc[int(idx), 'dis_pred'] = scores[2]
                ds.df.loc[int(idx), 'col_pred'] = scores[3]
                ds.df.loc[int(idx), 'loud_pred'] = scores[4]

                if args.print in [True, 'true', 'True', '1', 1]:
                    file_path = ds.df.loc[int(idx), "file_path"]
                    parts = [f"{d.upper()}: {scores[COL_IDX[d]]:.2f}" for d in dims]
                    print(f"({processed_files}/{total_files}) {os.path.basename(file_path)} | " + ", ".join(parts))

                # write CSV after each processed file
                if outfile is not None:
                    tmp_df = ds.df.drop(columns=['db_mean', 'db_std'], errors='ignore')

                    if existing_results is not None:
                        combined = pd.concat([existing_results, tmp_df], ignore_index=True)
                        combined.drop_duplicates(subset=["file_path"], keep="last", inplace=True)
                        combined.to_csv(outfile, index=False)
                    else:
                        tmp_df.to_csv(outfile, index=False)

    # provide user confirmation about saved output file
    print("Saved predicted scores:", os.path.join(output_dir, outfile))



if __name__ == "__main__":
    main()

    # --- Graceful shutdown ---
    # 1) Close logging and flush stdio
    logging.shutdown()
    sys.stdout.flush(); sys.stderr.flush()

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
