![alt text](https://github.com/pittcps/rl-parking/img/solar_design.jpg?raw=true)

# Solar-powered Parking Analytics System using Deep Reinforcement Learning

By Yoones Rezaei, Talha Khan, Stephen Lee, Daniel Mosse from University of Pittsburgh

# Citation

If you find our paper helpful in your work, please consider citing:

'''
@inproceedings{rezaei2021energy,
  title={Energy-efficient parking analytics system using deep reinforcement learning},
  author={Rezaei, Yoones and Lee, Stephen and Mosse, Daniel},
  booktitle={Proceedings of the 8th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation},
  pages={81--90},
  year={2021}
}
'''

# Introduction

In this repository we release the code and data for our paper "Energy-efficient parking analytics system using deep reinforcement learning" in BuildSys 2021 and its extension. You can find the original paper [here](https://arxiv.org/pdf/2202.08973).

The old version of the code for the original paper can be found in tensorflow_source directory.


# Installation

The updated version of our code uses Python 3,  Pytorch, and the [PFRL](https://github.com/pfnet/pfrl) library and is tested with an Nvidia RTX 3090 GPU. 
The list of the requirements to run this code are:

PFRL 0.3.0
torch 1.9.0+cu111
gym 0.22.0
[gin-config](https://github.com/google/gin-config) 0.3.0

# Data



# Usage

First set up desired settings in the config file similar to [default.gin](https://github.com/pittcps/rl-parking/configs/default.gin).
Next download the data and extract it into your project directory.

To train the model from scratch, from the pytorch_source directory run the following command: 

'''
python train.py --config="dir/to/configFile"
'''

To continue the training from an existing model, from the pytorch_source directory run the following command: 

'''
python train.py --config="dir/to/configFile --ckpt="dir/to/model"
'''

To test the model, from the pytorch_source directory run the following command:  
'''
python test.py --config="dir/to/configFile --ckpt="dir/to/model"
'''


