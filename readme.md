# SQ-AST
A transformer-based speech quality prediction model. Please cite [this paper](https://www.isca-archive.org/interspeech_2025/wardah25_interspeech.html) when using this model.

Cite as: Wardah, W., Spang, R.P., Barriac, V., Reimes, J., Llagostera, A., Berger, J., Möller, S. (2025) SQ-AST: A Transformer-Based Model for Speech Quality Prediction. Proc. Interspeech 2025, 2335-2339, doi: 10.21437/Interspeech.2025-2683

## To set up

Install the dependencies (requirements.txt) and download the run_predict.py script. This is the only script needed for inference. 

### Download Weights
The weight .pth files can be downloaded from [here](https://tubcloud.tu-berlin.de/s/rik9dQaR66R8w5A). There are 5 of them for each dimension. Download and save them in the same directory as the run_predict.py script.

## Running Predictions

You can run prediction for either one .wav file or for all .wav files in a directory. You can also select if you want to only predict some dimensions to save compute time. 

### Command for inference of one .wav file:

Predict all dimensions for a single .wav file and show the results on screen only (no output file will be created):

```bash
python run_predict.py --path /path/to/file.wav --print True
```

Predict selected dimensions (e.g., overall quality mos and noisiness) for a single .wav file and show the results on screen only (no output file will be created):

```bash
python run_predict.py --path /path/to/file.wav --dims mos noi --print True
```

Predict all dimensions for a single .wav file without printing the results, and save the predictions to an output file:

```bash
python run_predict.py --path /path/to/file.wav --output_dir /outputs/directory
```

### Command for inference of all wav files in a directory:

Predict all dimensions for all .wav files in a directory and display the predictions on screen only (no output files will be created):

```bash
python run_predict.py --path /path/to/directory --print True
```

Predict all dimensions for all .wav files in a directory using a specified batch size and number of workers, enable GPU processing, and save the predictions to an output file without printing them on screen:

```bash
python run_predict.py \
--path /path/to/directory \
--output_dir /outputs/directory \
--bs 64 \
--nw 4 \
--device gpu
```