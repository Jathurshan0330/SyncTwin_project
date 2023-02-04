# SyncTwin_project
Assignment: This repository contains the code for paper implementation of the "SyncTwin: Treatment Effect Estimation with Longitudinal Outcomes" paper published at NeurIPS 2021. 





# Data generation for the stimulation study
To maintain similarity in the dataset and reproduce the results of the stimulation study conducted in the paper, data generation scripts from the official implementation of the paper (https://github.com/ZhaozhiQIAN/SyncTwin-NeurIPS-2021) was utilized. 

Run the following script for data generation:
```
python -u -m pkpd_sim3_bias_generation --sim_id=sync6d-p10 --control_sample=1000 --control_c1=100 --train_step=25 --step=30 --seed=100

```

