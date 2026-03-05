# SQ-AST  
Transformer-Based Speech Quality Prediction

SQ-AST is a transformer-based model for predicting perceptual speech quality from audio signals.  
The model estimates multiple quality dimensions including overall quality (MOS), noisiness, distortion, coloration, and loudness.

If you use this model in your research, please cite the publication listed below.

---

# Quick Start

1. Install dependencies

```
pip install -r requirements.txt
```

2. Download the model weights

Download the `.pth` files and place them in a directory named `weights` next to `run_predict.py`.

Recommended weights (improved version):  
https://tubcloud.tu-berlin.de/s/K9owXP3Fnj4pnJg

3. Run prediction on a sample file

```
python run_predict.py --path samples --print True
```

Expected output:

```
(1/2) c00007_P501_C_english_m2_FB_48k.wav | MOS: 4.32, NOI: 4.58, DIS: 4.37, COL: 4.62, LOUD: 4.71
(2/2) c00001_P501_C_english_f1_FB_48k.wav | MOS: 4.43, NOI: 4.62, DIS: 4.27, COL: 4.71, LOUD: 4.69
```

---

# Installation

Clone the repository and install dependencies.

```
git clone <repo_url>
cd SQ-AST
pip install -r requirements.txt
```

The only script required for inference is:

```
run_predict.py
```

---

# Model Weights

Each perceptual dimension uses a separate weight file:

```
mos.pth
noi.pth
dis.pth
col.pth
loud.pth
```

Two versions are available.

Original Interspeech 2025 release:

https://tubcloud.tu-berlin.de/s/rik9dQaR66R8w5A

Improved version (recommended):

https://tubcloud.tu-berlin.de/s/K9owXP3Fnj4pnJg

The improved weights were evaluated by **ITU‑T SG12 Q9** and approved for standardization.

Place all weight files in:

```
./weights
```

or specify another directory using `--weights_dir`.

---

# Running Predictions

The script can process:

• a single `.wav` file  
• all `.wav` files inside a directory

## Single File

Predict all quality dimensions:

```
python run_predict.py --path /path/to/file.wav --print True
```

Predict selected dimensions:

```
python run_predict.py --path /path/to/file.wav --dims mos noi --print True
```

Save results to CSV:

```
python run_predict.py --path /path/to/file.wav --output_dir results
```

---

## Directory Processing

Predict quality for all WAV files in a directory:

```
python run_predict.py --path /path/to/directory --print True
```

Example using GPU and batching:

```
python run_predict.py \
  --path /path/to/directory \
  --output_dir results \
  --device gpu \
  --bs 64 \
  --nw 4
```

---

# Resuming Interrupted Runs

When an output directory is specified, predictions are written **incrementally after each processed file**.

This enables safe resuming of long runs.

Example:

```
python run_predict.py \
  --path dataset \
  --output_dir results \
  --overwrite false
```

Behavior:

• previously processed files are detected automatically  
• they are skipped on subsequent runs  
• processing continues with the next unprocessed file

---

# Command Line Options

## Input

`--path`

Path to a `.wav` file or directory containing `.wav` files.

---

## Output

`--output_dir`

Directory where prediction CSV files are written.  
If omitted, results are only printed to the console.

`--print`

Print predictions to the console while processing.

---

## Model Options

`--dims`

Select quality dimensions to predict.

Possible values:

```
mos noi dis col loud
```

Default: all dimensions.

`--weights_dir`

Directory containing the model weight files.

Default:

```
./weights
```

---

## Performance Options

`--device`

Processing device.

```
cpu
gpu
```

Default: `cpu`.

`--bs`

Batch size used for inference.

Default: `1`.

`--nw`

Number of dataloader workers.

Default: `0`.

---

## Run Control

`--overwrite`

Controls behavior when an output CSV already exists.

```
true   overwrite existing results
false  resume from previous run
```

Default: `true`.

---

# Output Format

If `--output_dir` is specified, predictions are written to:

```
<dataset_name>_prediction_per_file.csv
```

Example:

```
db,file_path,file_num,mos_pred,noi_pred,dis_pred,col_pred,loud_pred
dataset,a.wav,1,4.32,4.58,4.37,4.62,4.71
```

Each row corresponds to one processed audio file.

---

# Interpretation of Scores

The predicted scores follow the perceptual speech quality scales defined in ITU‑T recommendations, including:

• ITU‑T P.800 (subjective listening tests)  
• ITU‑T P.835 (speech quality dimensions)  
• ITU‑T P.863 / P.1204 related modeling frameworks

Scores typically lie in the range:

```
1 (bad)  →  5 (excellent)
```

for the predicted perceptual quality dimensions.

---

# Citation

If you use SQ‑AST in your research, please cite:

Wardah, W., Spang, R.P., Barriac, V., Reimes, J., Llagostera, A., Berger, J., Möller, S. (2025)  
SQ-AST: A Transformer-Based Model for Speech Quality Prediction.  
Proc. Interspeech 2025, 2335–2339.  
doi: 10.21437/Interspeech.2025-2683

https://www.isca-archive.org/interspeech_2025/wardah25_interspeech.html
