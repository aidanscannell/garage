
# Create training curves from garage results

Results were obtained by running
```sh
python cluster_train.py -m ++main_config.env_name=HalfCheetahVelEnv,HopperV2,HalfCheetahV2,Walker2DV2 ++main_config.seed=42,100,148,432,427,94,1039,6432,39,12032
```

The inverted pendulm results were obtained with 
```sh
python cluster_train_inv_pend.py -m ++main_config.env_name=InvertedPendulumV2 ++main_config.num_test_tasks=8 ++main_config.num_train_tasks=8 ++main_config._target_="cluster_train_inv_pend.main" ++main_config.latent_size=2 ++main_config.reward_scale=1 ++main_config.meta_batch_size=8 ++main_config.seed=42,100,148,432,427,94,1039,6432,39,12032
```
and with 
```sh
python cluster_train_inv_pend.py -m ++main_config.env_name=InvertedPendulumV2 ++main_config.num_test_tasks=8 ++main_config.num_train_tasks=8 ++main_config._target_="cluster_train_inv_pend.main" ++main_config.latent_size=2 ++main_config.reward_scale=1.0  ++main_config.meta_batch_size=8 ++main_config.seed=321,435,645,543,0987,364,1,83,5,111
```
Results were copied with
```sh
rsync triton:/scratch/work/scannea1/python-projects/garage-pearl/src/garage/examples/torch/output/cluster_train_inv_pend/2023-12-11_16-08-58/9/data/local/experiment/main/progress.csv ./inv-pend-pearl-seed-10-second-try.csv
```
