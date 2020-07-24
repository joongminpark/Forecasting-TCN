# Load forecasting with Temporal Convolutional Networks

## Requirements
- python 3.6.9    
- pytorch 1.2   
- cuda 10.0


### Data
```
load-forecasting
├── README.md
│
├── experiments.txt
│
├── main.py
├── directstep_test.py
│
└── src
    ├── data_loader.py
    ├── TCN1B.py
    ├── TCN2B.py
    └── train.py
```

## Usage
The file `experiments/experiments.txt` contains the hyperparameters I
used to obtain the results given below.


## Result
- Achieve good performance (MAPE, RMSE) compare to other models
- Searching taget for journal (Completed draft)

**Boxplot for each TCN module**  
<img src="./assets/boxplot.PNG" width="60%">

## modal architecture

**Temporal Convolution Network**  
<img src="./assets/tcn.PNG" width="60%">

**Output module**  
<img src="./assets/output_module.PNG" width="60%">

**Inside layer module**  
<img src="./assets/within_module.PNG" width="60%">

**Outside layer module**  
<img src="./assets/across_module.PNG" width="60%">
