# TM-SNN: Threshold Modulated Spiking Neural Network for Multi-task Learning

This repository contains code implementation for the paper "TM-SNN: Threshold Modulated Spiking Neural Network for Multi-task Learning". The code uses threshold modulation to implement spiking neural networks that change their behavior. TM-SNN is tested for solving multitask classification on the NMNIST neuromorphic dataset. The results show that TM-SNN can actually learn different tasks through modifying its dynamics via modulation of the neurons’ firing threshold.

## Installation

### Prerequisites

The code runs in Python3.9 using Intel's Lava neuromorphic framework and the Lava-DL package. They can be found at https://github.com/lava-nc/lava and https://github.com/lava-nc/lava-dl.

### Clone

Clone this repo to your local machine using `https://github.com/PaoloGCD/MultiTask-SNN.git`

### Dataset

The experiments are performed on the NMNIST dataset. Download and extract the dataset to '.data/NMNIST' folder from https://www.garrickorchard.com/datasets/n-mnist.

## Running the tests

To run the SNN for single task classification (base-case), execute:

```shell
$ sh ./experiments/NMNIST-base-case.sh
```

To run the SNN for two-task classification using threshold control, execute:

```shell
$ sh ./experiments/NMNIST-threshold-two-blocks.sh
```

To run the SNN for two-task classification using threshold control whit auxiliary block, execute:

```shell
$ sh ./experiments/NMNIST-threshold-three-blocks.sh
```

## Authors

* **Paolo G. Cachi** - *Virginia Commonwealth University* - USA
