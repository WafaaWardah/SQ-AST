# SQ-AST
A transformer-based speech quality prediction model. Please cite [this paper](https://www.isca-archive.org/interspeech_2025/wardah25_interspeech.html) when using this model.

Cite as: Wardah, W., Spang, R.P., Barriac, V., Reimes, J., Llagostera, A., Berger, J., Möller, S. (2025) SQ-AST: A Transformer-Based Model for Speech Quality Prediction. Proc. Interspeech 2025, 2335-2339, doi: 10.21437/Interspeech.2025-2683

## Running Predictions

All wav files to be labeled should be placed in one directory. Prediction for individual wav file is not available yet, but if the directory contains only one wav file then that will be processed.

The predicted labels will be displayed and also saved as a prediction_per_file csv file at the end of the prediction. 

These variables might need to be modified in the run_predict.py file according to your hardware: bs, num_workers. If gpu is available, it will be used, otherwise cpu will be used.

To run predictions, use the following command:

```bash
python run_predict.py /path/to/output/dir /path/to/data/dir
```
## Download Weights
The weight .pth files can be downloaded from [here](https://tubcloud.tu-berlin.de/s/rik9dQaR66R8w5A). There are 5 of them for each dimension. Download and save them in the .weights/ directory.
