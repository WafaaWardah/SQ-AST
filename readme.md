# SQ-AST
A transformer-based speech quality prediction model. Please cite [this paper](https://www.isca-archive.org/interspeech_2025/wardah25_interspeech.html) when using this model.

Cite as: Wardah, W., Spang, R.P., Barriac, V., Reimes, J., Llagostera, A., Berger, J., Möller, S. (2025) SQ-AST: A Transformer-Based Model for Speech Quality Prediction. Proc. Interspeech 2025, 2335-2339, doi: 10.21437/Interspeech.2025-2683

## To set up

Intall the dependencies (requirements.txt) and download the run_predict.py script. This is the only script needed for inference. 

### Download Weights
The weight .pth files can be downloaded from [here](https://tubcloud.tu-berlin.de/s/rik9dQaR66R8w5A). There are 5 of them for each dimension. Download and save them in the same directory as the run:predict.py script.

## Running Predictions

You can run prediction for either one .wav file or for all wav files in a directory. You can also select if you want to only predict some dimensions to save compute time. 

### Command for inference of one wav file:

To predict all dimensions for one wav file, print out the prediction, don’t save the prediction:

```bash
python python run_predict.py --path /path/to/file.wav --print True
```

To predict some dimensions (example overall quality mos and noisiness) for one wav file, print out the prediction, don’t save the prediction:

```bash
python python run_predict.py --path /path/to/file.wav --dims mos noi --print True
```

To predict all dimensions for one wav file, don’t print out the prediction, but save the prediction:

```bash
python python run_predict.py --path /path/to/file.wav --output_dir /outputs/directory
```

### Command for inference of all wav files in a directory:

To predict all dimensions for all wav files in a directory, print out the predictions, but don’t save the predictions:

```bash
python run_predict.py --path /path/to/directory --print True
```

To predict all dimensions for all wav files in a directory, set batch size and number of workers, use gpu,  don’t print out the predictions, and save the prediction:

```bash
python run_predict.py \
--path /path/to/directory \
--output_dir /outputs/directory \
--bs 64 \
--nw 4 \
--device gpu
```