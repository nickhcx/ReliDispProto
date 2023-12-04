## Requirements 

* `pytorch = 1.0.1`
* `json`
* `argparse`
* `sklearn`
* `transformers==3.0.2`

## Usage 

### BaseStage Training
Open the directory "ReliDispProto/BaseStage", and run the python file "train_demo.py". 
The saved model are in the directory "ReliDispProto/BaseStage/checkpoint/". 
The specific commands and several important hyper-parameters are as follows:

```bash
 python3 train_demo.py --embedding_type bert --lr 1e-2 --max_length 60
```

### StreamingSession Training 
Open the directory "ReliDispProto/StreamingSession", and run the python file "train_demo.py". 
The saved model are in the directory "ReliDispProto/StreamingSessioncheckpoint/". 
The specific commands and several important hyper-parameters are as follows:

```bash
python3 train_demo.py --embedding_type bert \
--pl_weight 0.1 --base_lr 0.01 \
--learn_rate 2e-3 --K 5 --val_step 100 --val_iter 100 --max_length 60
```
