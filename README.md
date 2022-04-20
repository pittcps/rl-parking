![Alt text](/img/solar_desgin.JPG?raw=true)

# Solar-powered Parking Analytics System using Deep Reinforcement Learning

By Yoones Rezaei, Talha Khan, Stephen Lee, Daniel Mosse from University of Pittsburgh

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

In this repository we release the code and data for our paper "Energy-efficient parking analytics system using deep reinforcement learning" in BuildSys 2021 and its extension. You can find the original paper [here](https://arxiv.org/pdf/2202.08973).

The old version of the code for the original paper can be found in tensorflow_source directory.


# Installation

The updated version of our code uses Python 3,  Pytorch, and the [PFRL](https://github.com/pfnet/pfrl) library and is tested with an Nvidia RTX 3090 GPU. 
The list of the requirements to run this code are:

- PFRL 0.3.0
- torch 1.9.0+cu111
- gym 0.22.0
- [gin-config](https://github.com/google/gin-config) 0.3.0

# Data
We use the parking dataset from [data.melbourne.vic.gov.au](https://data.melbourne.vic.gov.au/Transport/Parking-bay-arrivals-and-departures-2014/mq3i-cbxd) which has been collected from central business district in Melbourne, Australia. In addition we use the [AusGrid](https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data) solar data, collected by [Ratnam et al.](https://www.tandfonline.com/doi/abs/10.1080/14786451.2015.1100196?journalCode=gsol20) and collect weather data using [DarkSky](https://darksky.net/forecast/40.7127,-74.0059/us12/en) API. You can download the processed version of the data from [Here](https://drive.google.com/file/d/1oajRsAdkDz6xw5kzT0p4oqrN3O60N4Yw/view?usp=sharing) which can be used directly with out model.

# Usage

First set up desired settings in the config file similar to [default.gin](https://github.com/pittcps/rl-parking/blob/main/configs/default.gin).
Next download the data and extract it into your project directory.

To train the model from scratch, from the pytorch_source directory, run the following command: 

```
python train.py --config="dir/to/configFile"
```

To continue the training from an existing model, from the pytorch_source directory, run the following command: 

```
python train.py --config="dir/to/configFile" --ckpt="dir/to/model"
```

To test the model, from the pytorch_source directory, run the following command:  
```
python test.py --config="dir/to/configFile" --ckpt="dir/to/model"
```


