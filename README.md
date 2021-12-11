# UMB-SR
Incorporating User Micro-behaviors into Multi-task Learning for Session-based Recommendation


# Module Data and Code
This code is based on the SIGIR2020 Paper: Incorporating User Micro-behaviors and Item Knowledge into Multi-task Learning for Session-based Recommendation[1]. Our module is based on the MKM-SR module implemented in this paper[2].

The dataset we used for this module is from: https://www.kaggle.com/retailrocket/ecommerce-dataset


# Blog
For a detailed explanation and report of this module, please refer to our blog: https://medium.com/@elisha.pz.0108/user-micro-behaviors-session-based-recommendation-systems-2358e1fcfb45


# Usage
Our code has already included a processed dataset that is ready to be used by the module. However if the user would like to make changes to the events.csv file, then they will need to
run the command

```
usage: data_processing.py
command: python ./data_processing.py

Then you can run the file ```main.py``` to train the model.
```
```
usage: main.py 
optional arguments:
  --dataset             dataset name
  --batchSize           input batch size
  --hiddenSize          hidden state size
  --epoch EPOCH         the number of epochs to train for
  --lr LR               learning rate
  --l2 L2               l2 penalty
  --step STEP           gnn propogation steps
  --patience PATIENCE   the number of epoch to wait before early stop
  

```

# Reference
[1]Wenjing Meng, Deqing Yang and Yanghua Xiao. 2020. Incorporating User Micro-behaviors and Item Knowledge into Multi-task Learning for Session-based Recommendation. In Proceedings of the 43rd International ACM.

[2]https://github.com/ciecus/MKM-SR
