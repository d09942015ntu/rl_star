# [Implementation] RL-STaR: Theoretical Analysis of Reinforcement Learning Frameworks for Self-Taught Reasoner

[https://openreview.net/pdf?id=Oo2XthxKB9](https://openreview.net/pdf?id=Oo2XthxKB9)

## Install

```sh
python3 -m venv venv_rlstar
source venv_rlstar/bin/activate 
pip install -r requirements.txt 
```

## Quick Run
check whether everything is correctly installed

```sh
bash quick_run.sh
```

## Show Theoretical Values
```sh
python3 RLstar_theory.py
```

## Generate pre-trained data
```sh
bash run_gen_pretrain.sh 
```

## Run RL-STaR
```sh
bash run_RLstar.sh 
```

## Plot RL-STaR result

```sh
python3 vis_rlstar_accuracy.py 
````

