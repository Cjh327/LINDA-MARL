# LINDA-MARL
Code for "LINDA: Multi-Agent Local Information Decomposition for Awareness of Teammates"

## Install
```
sh install_sc2.sh

conda create -n pymarl python=3.8 -y
conda activate pymarl

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch -y
pip install sacred numpy scipy matplotlib seaborn pyyaml pygame pytest probscale imageio snakeviz tensorboard-logger
pip install git+https://github.com/oxwhirl/smac.git
```

## Run
Run single experiment
```
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z
```

Run script
```
python run_experiments.py --map 2s3z --alg qmix --repeat 2 --cuda 0
```
