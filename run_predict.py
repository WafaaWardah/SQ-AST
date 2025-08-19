# prediction script for Model-XL (no DB parameter calliberated normalization)
# 10.01.2025
import os
import pandas as pd
import torch
from datasets import ASTVal
from models import ASTXL
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
from datetime import datetime

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the prediction script.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory")
    parser.add_argument("data_dir", type=str, help="Path to the data directory")
    args = parser.parse_args()

    output_dir = args.output_dir
    data_dir = args.data_dir

    # Debug prints (optional)
    print(f"Output Directory: {output_dir}")
    print(f"Data Directory: {data_dir}")

    # Set this according to your hardware
    bs = 32
    num_workers = 8

    db_name = os.path.basename(data_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nPrediction begins...")
    print(f"Using device: {device}")

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    dtype_dict = {
        'db': str,
        'file_path': str,
        'file_num': float,
        'db_mean': float,
        'db_std': float
    }

    df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in dtype_dict.items()})
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]

    df['file_path'] = file_paths
    df['db'] = db_name
    df['file_num'] = range(1, len(file_paths) + 1)

    output_file = os.path.join(output_dir, f"{db_name}_data_file.csv")
  
    # Add fixed db_mean and db_std columns to df
    df['db_mean'] = -10.25446422 # AST paper: -4.2677393
    df['db_std'] = 4.205750774 # AST paper: 4.5689974
   
    ds = ASTVal(df, data_dir)

    dl = DataLoader(
        dataset=ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers
    )

    # Load MOS model
    model_mos = ASTXL()
    mos_state_dict = torch.load("weights/mos.pth", map_location=torch.device(device), weights_only=True)
    model_mos.load_state_dict(mos_state_dict)
    model_mos.eval()

    # Load NOI model
    model_noi = ASTXL()
    noi_state_dict = torch.load("weights/noi.pth", map_location=torch.device(device), weights_only=True)
    model_noi.load_state_dict(noi_state_dict)
    model_noi.eval()

    # Load DIS model
    model_dis = ASTXL()
    dis_state_dict = torch.load("weights/dis.pth", map_location=torch.device(device), weights_only=True)
    model_dis.load_state_dict(dis_state_dict)
    model_dis.eval()

    # Load COL model
    model_col = ASTXL()
    col_state_dict = torch.load("weights/col.pth", map_location=torch.device(device), weights_only=True)
    model_col.load_state_dict(col_state_dict)
    model_col.eval()

    # Load LOUD model
    model_loud = ASTXL()
    loud_state_dict = torch.load("weights/loud.pth", map_location=torch.device(device), weights_only=True)
    model_loud.load_state_dict(loud_state_dict)
    model_loud.eval()

    print(f"\nModel loaded") 

    y_hat_val = torch.full((len(ds), 5), -0.25, device='cpu') # Stores the validation outputs, later filled into ds_val df

    total_files = len(ds.df) 
    processed_files = 0       

    with torch.no_grad():  # Disable gradient tracking for inference
        print("\nCalculating quality scores...")
        for b, (index, batch_features) in enumerate(dl):

            batch_features = batch_features.float().to(device)

            # Forward pass ---------------------------------------
            mos_pred = model_mos(batch_features)
            noi_pred = model_noi(batch_features)
            dis_pred = model_dis(batch_features)
            col_pred = model_col(batch_features)
            loud_pred = model_loud(batch_features)
            
            # Stack predictions for each dimension
            y_hat_batch = torch.stack([mos_pred, noi_pred, dis_pred, col_pred, loud_pred], dim=1).squeeze().to('cpu')
            y_hat_val[index, :] = y_hat_batch

            # Iterate through current batch to print scores with file paths
            for idx, scores in zip(index, y_hat_batch):
                idx = int(idx)  # Convert PyTorch tensor to native Python integer
                file_path = ds.df.loc[idx, 'file_path']  # Retrieve file path using index

                processed_files += 1
                # Descale predictions for display
                descaled_scores = scores * 4 + 1
                print(f"({processed_files}/{total_files}) {os.path.basename(file_path)} | MOS: {descaled_scores[0]:.2f}, "
                    f"NOI: {descaled_scores[1]:.2f}, DIS: {descaled_scores[2]:.2f}, "
                    f"COL: {descaled_scores[3]:.2f}, LOUD: {descaled_scores[4]:.2f}")

    # Scale predictions once all batches are processed
    y_hat_val_descaled = y_hat_val * 4 + 1 # On CPU
    y_hat_val_descaled = y_hat_val_descaled.detach().numpy() # On CPU

    # Convert predictions into DataFrame columns on CPU
    ds.df['mos_pred'] = y_hat_val_descaled[:, 0]
    ds.df['noi_pred'] = y_hat_val_descaled[:, 1]
    ds.df['dis_pred'] = y_hat_val_descaled[:, 2]
    ds.df['col_pred'] = y_hat_val_descaled[:, 3]
    ds.df['loud_pred'] = y_hat_val_descaled[:, 4]

    filtered_val_df = ds.df.loc[
        (ds.df['mos_pred'] != 0.0) &
        (ds.df['noi_pred'] != 0.0) &
        (ds.df['dis_pred'] != 0.0) &
        (ds.df['col_pred'] != 0.0) &
        (ds.df['loud_pred'] != 0.0)
    ]

    filtered_val_df.to_csv(os.path.join(output_dir, db_name + '_prediction_per_file_' + current_time + '.csv'), index=False)  
    print("Saved predicted scores:", os.path.join(output_dir, db_name + '_prediction_per_file_' + current_time + '.csv'))
