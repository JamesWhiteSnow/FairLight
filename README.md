# FairLight
Although reinforcement learning (RL) approaches are promising in autonomous traffic signal control (TSC), they often suffer from the unfairness problem that causes extremely long waiting time at intersections for partial vehicles. This is mainly because the traditional RL methods focus on optimizing the overall traffic performance, while the fairness of individual vehicles is neglected. To address this problem, we propose a novel RL-based method named FairLight for the fair and efficient control of traffic with variable phase duration. Inspired by the concept of user satisfaction index (USI) proposed in the transportation field, we introduce a fairness index in the design of key RL elements, which specially considers the travel quality (e.g., fairness). Based on our proposed hierarchical action space method, FairLight can accurately allocate the duration of traffic lights for selected phases. Experimental results obtained from various well-known traffic benchmarks show that, compared with the state-of-the-art RL-based TSC methods, FairLight can not only achieve better fairness performance but also improve the control quality from the perspectives of the average travel time of vehicles and RL convergence speed.

In this project, we open-source the source code of our FairLight approach. 

On Git Hub, we will introduce how to reproduce the results of our experiments in the paper.

For details of our method, please see our [original paper](https://ieeexplore.ieee.org/abstract/document/9969874) at IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (IEEE TCAD).

Welcome to cite our paper!

```
@article{ye2023fairlight,
  title={FairLight: Fairness-Aware Autonomous Traffic Signal Control With Hierarchical Action Space},
  author={Ye, Yutong and Ding, Jiepin and Wang, Ting and Zhou, Junlong and Wei, Xian and Chen, Mingsong},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (IEEE TCAD)},
  year={2023},
  volume={42},
  number={8},
  pages={2434-2446},
  doi={10.1109/TCAD.2022.3226673}
}
```

## Requirements
Under the root directory, execute the following conda commands to configure the Python environment.
``conda create --name <new_environment_name> --file requirements.txt``

``conda activate <new_environment_name>``

### Simulator installation
Our experiments are implemented on top of the traffic simulator Cityflow. Detailed installation guide files can be found in https://cityflow-project.github.io/

#### 1. Install cpp dependencies
``sudo apt update && sudo apt install -y build-essential cmake``

#### 2. Clone CityFlow project from github
``git clone https://github.com/cityflow-project/CityFlow.git``

#### 3. Go to CityFlow project’s root directory and run
``pip install .``

#### 4. Wait for installation to complete and CityFlow should be successfully installed
``import cityflow``

``eng = cityflow.Engine``

## Run the code
#### Execute the following command to run the experiment over the specified dataset.
``python train.py --d <dataset_name>``

## Datasets
For the experiments, we used both synthetic and real-world traffic datasets provided by https://traffic-signal-control.github.io/dataset.html.
| Dataset Name | Dataset Type | # of intersections |
| :-----------: | :-----------: | :-----------: |
| Syn1 | Synthetic | 1×3 |
| Syn2 | Synthetic | 2×2 |
| Syn3 | Synthetic | 3×3 |
| Syn4 | Synthetic | 4×4 |
| NY9th | Real-world | 1×16 |
| NY11th | Real-world | 1×16 |
| Hangzhou | Real-world | 4×4 |
| Jinan | Real-world | 4×3 |
