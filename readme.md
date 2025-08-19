## Running Predictions

All wav files to be labeled should be placed in one directory. Prediction for individual wav file is not available yet, but if the directory contains only one wav file then that will be processed.

The predicted labels will be displayed and also saved as a prediction_per_file csv file at the end of the prediction. 

These variables might need to be modified in the run_predict.py file according to your hardware: bs, num_workers. If gpu is available, it will be used, otherwise cpu will be used.

To run predictions, use the following command:

```bash
python run_predict.py /path/to/output/dir /path/to/data/dir
```
## Download Weights
The weight .pth files can be downloaded from [here](https://tubcloud.tu-berlin.de/s/s4bKTRxWkSTTzFr). There are 5 of them for each dimension. Download and save them in the .weights/ directory.
