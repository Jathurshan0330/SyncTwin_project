# SyncTwin_project
Assignment: This repository contains the code for paper implementation of the "SyncTwin: Treatment Effect Estimation with Longitudinal Outcomes" paper published at NeurIPS 2021. The official implementation of the paper can be found at https://github.com/ZhaozhiQIAN/SyncTwin-NeurIPS-2021


# Getting Started

### Installation Guide
Run the algorithm using Pytorch and CUDA https://pytorch.org/

```
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

```
pip install -r requirements.txt
```



# Data generation for the simulation study
To maintain similarity in the dataset and reproduce the results of the simulation study conducted in the paper, data generation scripts from the official implementation of the paper (https://github.com/ZhaozhiQIAN/SyncTwin-NeurIPS-2021) was utilized. 

Run the following script for data generation:

For conditions m=0, S=25,p=[0.1,0.25,0.5]  ==> For different confounding bias
```
python -u -m pkpd_sim3_bias_generation --sim_id=sync6d-p10 --control_sample=1000 --control_c1=100 --train_step=25 --step=30 --seed=100

python -u -m pkpd_sim3_bias_generation --sim_id=sync6d-p25 --control_sample=1000 --control_c1=250 --train_step=25 --step=30 --seed=100


python -u -m pkpd_sim3_bias_generation --sim_id=sync6d-p50 --control_sample=1000 --control_c1=500 --train_step=25 --step=30 --seed=100
```



For conditions p=0.5, S=25, m = [0.3,0.5,0.7] ==> Irregularly observed covariates
```
python -u -m pkpd_sim3_irregular_generation --sim_id=sync6d --seed=100 --missing_pct=0.3

python -u -m pkpd_sim3_irregular_generation --sim_id=sync6d --seed=100 --missing_pct=0.5

python -u -m pkpd_sim3_irregular_generation --sim_id=sync6d --seed=100 --missing_pct=0.7
```

# Demonstration
A demonstration notebook (Demo.ipynb) is included to provide a hands-on experience to showcase the implementation of the method and produce results for various experiment settings.

# Reference
```
@article{qian2021synctwin,
  title={Synctwin: Treatment effect estimation with longitudinal outcomes},
  author={Qian, Zhaozhi and Zhang, Yao and Bica, Ioana and Wood, Angela and van der Schaar, Mihaela},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={3178--3190},
  year={2021}
}
```
