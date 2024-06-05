# ECQx: Explainability-Driven Quantization for Low-Bit and Sparse DNNs

- ECQ: **E**ntropy-**C**onstrained (trained) **Q**uantization as described in [(Becking et al., 2020)](https://www.researchgate.net/profile/Daniel-Becking/publication/354987516_Finding_Storage-_and_Compute-Efficient_Convolutional_Neural_Networks/links/6156f041a6fae644fbb6a2a8/Finding-Storage-and-Compute-Efficient-Convolutional-Neural-Networks.pdf), and applied to Ternary Networks [(Marban et al., 2020)](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Marban_Learning_Sparse__Ternary_Neural_Networks_With_Entropy-Constrained_Trained_Ternarization_CVPRW_2020_paper.html) and a 4bit Hardware-Software Co-Design [(Wiedemann et al., 2021)](https://ieeexplore.ieee.org/abstract/document/9440253).

- ECQx is an e**X**plainability-driven version of ECQ which corrects cluster assignments based on their relevance [(Becking et al., 2022)](https://link.springer.com/chapter/10.1007/978-3-031-04083-2_14).

<div align="center">
<img src="https://github.com/d-becking/ECQx/assets/56083075/67e3aff6-25d6-4771-97f0-6651b3ef2737"  width="600">
</div>

## Table of Contents
- [Information](#information)
- [Installation](#installation)
- [ECQx Usage](#ecqx-usage):
  * [Reproducibility of paper results](#reproducibility-of-paper-results)
  * [Demo](#demo)
- [Citation and Publications](#citation-and-publications)
- [License](#license)

## Information

This repository demonstrates ECQx using ResNet architectures to solve CIFAR and PASCAL VOC. 
How to run the code, reproduce paper results and run the demo is described in the [ECQx Usage](#ecqx-usage) section.

[TBD: ADD MORE INFO]

## Installation

The software provides python packages which can be installed using pip. However, core technologies are implemented using C++, which requires a C++ compiler for the installation process.

The software has been tested on different target platforms (mainly Linux and macOS).

### Requirements

- python >= 3.6 (recommended versions 3.6, 3.7, 3.8, and 3.9) with working pip
- **Windows:** Microsoft Visual Studio 2015 Update 3 or later

### Package installation
**Note**: If your NVIDIA GPU does not support CUDA 12, add the following line to the `requirements.txt` file:

`--extra-index-url https://download.pytorch.org/whl/cu118` 
- On **_Linux/Mac_**, running the script `create_env.sh` sets up a virtual python environment "env" and installs all required packages and the software itself, automatically.

- For **_Windows_**, it is recommended to issue from the root of the cloned repository:
    ```
    pip install wheel
    pip install -r requirements.txt
    pip install .
    ```
 For activating this environment, issue:
```
source env/bin/activate
```

**Note**: For further information on how to set up a virtual python environment (also on **Windows**) refer to https://docs.python.org/3/library/venv.html .


## ECQx Usage

Before running the code, first create the environment as described in [Installation](#installation), and activate it.

Execute 
```shell
python run.py --help
```
for parser argument descriptions.

### Reproducibility of paper results

[TBD]

#### CIFAR-10 experiments (w/ ResNet20):

`--dataset_path` must be specified in accordance with your local data directories. For the CIFAR experiment, the data will be downloaded (< 200MB) to --dataset_path if the data is not already available there (defaults to "../data").

Basic setting with default hyperparameters and parser arguments for running 4bit ECQ (without x) on CIFAR10 with an already pre-trained ResNet20:

```shell
python run.py --model_path=./models/pretrained/resnet20.pt --dataset_path=<YOUR_PATH> --verbose
```

Basic setting for running 4bit ECQx on CIFAR10 with an already pre-trained ResNet20:

```shell
python run.py --lrp --model_path=./models/pretrained/resnet20.pt --dataset_path=<YOUR_PATH> --verbose
```
The above command generates LRP relevances using the default "_resnet_" `--canonizer`, and the "_epsilon_plus_flat_bn_pass_" `--lrp_composite`.

For simple network architectures, e.g., without BatchNorm modules and without residual connections, it is recommended to use "_vgg_" or "_resnetcifar_" `--canonizer`s, otherwise "_resnet_" or "_mobilenet_" `--canonizer`s.

Investigating different `--lrp_composite`s can also improve the ECQx performance. We recommend "_epsilon_plus_flat_bn_pass_", "_epsilon_plus_flat_", "_alpha2_beta1_flat_bn_pass_", and "_alpha2_beta1_flat_".

Increasing the `--Lambda` hyperparameter will intensify the entropy constraint and thus lead to a higher sparsity (and thus performance degradation, which can be compensated to a certain extent by ECQx).

#### Pascal VOC experiments:

[TBD]

### Logging results using Weights & Biases

We used Weights & Biases (wandb) for experiment logging. Enabling `--wandb`. If you want to use it, add your `--wandb_key` and optionally an experiment identifier for the run (`--wandb_run_name`).


### Demo

[TBD]



## Citation and Publications
If you use ECQx in your work, please cite:
```
@inproceedings{becking2022ecqx,
  author={Becking, Daniel and Dreyer, Maximilian and Samek, Wojciech and M{\"u}ller, Karsten and Lapuschkin, Sebastian},
  title={{ECQ}$^{\text{X}}$: Explainability-{D}riven {Q}uantization for {L}ow-{B}it and {S}parse {DNN}s},
  booktitle={xxAI - Beyond Explainable AI, Lecture Notes in Computer Science (LNAI Vol. 13200), Springer International Publishing},
  pages={271--296},
  year={2022},
  doi={10.1007/978-3-031-04083-2_14}
}
```

### Publications
- [(Becking et al., 2022)](https://link.springer.com/chapter/10.1007/978-3-031-04083-2_14) - **"ECQx: Explainability-Driven Quantization for Low-Bit and Sparse DNNs"**, in xxAI - Beyond Explainable AI, Lecture Notes in Computer Science (LNAI Vol. 13200), Springer
International Publishing, pp. 271–296, 2022
- [(Becking et al., 2023)](https://openreview.net/forum?id=5VgMDKUgX0) - **"NNCodec: An Open Source Software Implementation of the Neural Network Coding ISO/IEC Standard"**, 40th International Conference on Machine Learning (ICML), Neural Compression Workshop (Spotlight), 2023
- [(Anders et al., 2021](https://arxiv.org/abs/2106.13200) - **"Software for dataset-wide XAI: from local explanations to global insights with Zennit, CoRelAy, and ViRelAy"**, arXiv preprint arXiv:2106.13200, 2021
- [(Wiedemann et al., 2021)](https://ieeexplore.ieee.org/abstract/document/9440253) - **"FantastIC4: A Hardware-Software Co-Design Approach for Efficiently Running 4Bit-Compact Multilayer Perceptrons"**, in IEEE Open Journal of Circuits and Systems, Vol. 2, pp. 407-419, 2021
- [(Marban et al., 2020)](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Marban_Learning_Sparse__Ternary_Neural_Networks_With_Entropy-Constrained_Trained_Ternarization_CVPRW_2020_paper.html) - **"Learning Sparse & Ternary Neural Networks With Entropy-Constrained Trained Ternarization (EC2T)"**, Proceedings of the IEEE/CVF CVPR Workshops, pp. 722-723, 2020
- [(Becking et al., 2020)](https://www.researchgate.net/profile/Daniel-Becking/publication/354987516_Finding_Storage-_and_Compute-Efficient_Convolutional_Neural_Networks/links/6156f041a6fae644fbb6a2a8/Finding-Storage-and-Compute-Efficient-Convolutional-Neural-Networks.pdf) - **"Finding Storage-and Compute-Efficient Convolutional Neural Networks"**, Master Thesis, Technical University of Berlin, 2020

## License

Please see [LICENSE.txt](./LICENSE.txt) file for the terms of the use of the contents of this repository.
For Zennit and NNCodec licences please also check the license files in the according subdirectories and the current github repositories:

[![Conference](https://img.shields.io/badge/github_fraunhoferhhi-NNCodec-green)](https://github.com/fraunhoferhhi/nncodec)
[![Conference](https://img.shields.io/badge/github_chr5tphr-Zennit-red)](https://github.com/chr5tphr/zennit)

For more information and bug reports, please contact: daniel.becking@hhi.fraunhofer.de

**Copyright (c) 2019-2024, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V.**

**All rights reserved.**
