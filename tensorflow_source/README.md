# Tensorflow implementation of Energy-efficient parking analytics system using deep reinforcement learning

By Yoones Rezaei, Stephen Lee, Daniel Mosse from University of Pittsburgh

# Citation

If you find our paper helpful in your work, please consider citing:

```
@inproceedings{rezaei2021energy,
  title={Energy-efficient parking analytics system using deep reinforcement learning},
  author={Rezaei, Yoones and Lee, Stephen and Mosse, Daniel},
  booktitle={Proceedings of the 8th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation},
  pages={81--90},
  year={2021}
}
```

# Introduction

In this repository we release the code and data for our paper "Energy-efficient parking analytics system using deep reinforcement learning" in BuildSys 2021. You can find the paper [here](https://arxiv.org/pdf/2202.08973). 

This version of the code is not supported anymore. We've added a new Pytorch version of the code which can be accessed from [here](https://github.com/pittcps/rl-parking). 


# Installation

This code uses Python 3, tensorflow 1.1, keras, and the keras-RL library. The requirements are as follows:

- tensorflow 1.14
- keras 2.3.1
- keras-rl 0.4.2
- gym
- [gin-config](https://github.com/google/gin-config) 0.3.0


# Data
We use the parking dataset from [data.melbourne.vic.gov.au](https://data.melbourne.vic.gov.au/Transport/Parking-bay-arrivals-and-departures-2014/mq3i-cbxd) which has been collected from central business district in Melbourne, Australia. You can download the processed version of the data from [Here](https://drive.google.com/file/d/1oajRsAdkDz6xw5kzT0p4oqrN3O60N4Yw/view?usp=sharing) which can be used directly with out model.


# Usage
First set up desired settings in the config file similar to [default.gin](https://github.com/pittcps/rl-parking/blob/main/configs/default.gin). Please note this version of the code only supports the "cloud" mode.

Next download the data and extract it into your project directory.

To train the model, from the pytorch_source directory, run the following command: 

```
python run.py --mode="train"
```

To test the model, from the pytorch_source directory, run the following command:  
```
python test.py --mode="test" --ckpt="dir/to/model"
```


