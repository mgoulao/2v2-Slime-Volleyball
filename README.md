# 2v2 Slime Volley

## Available Agents

* PPO
* PPO with Attacker/Defender roles
* PPO with Top/Bottom roles
* PPO with Leader

## Structure

```
--
 |- slimevolleygymrepo (Fork of the original environment with our changes)
 |- study (all scripts to train and evaluate the agents)
```

## Requirements

### Languages
* Python 3 (we used 3.8, but most versions should work as well)
### Packages (conda or pip):
* pytorch (with CUDA for better performance)
* numpy
* gym
* tensorboard

## How to train

We have a python script to train each agent they all have the following format: `train_*.py`. They all have two flags `--render` to render the environment and see what is happening, and `--noselfplay` to disable selfplay. A training session is 10 million timesteps, during our tests this is equivalent to 6 hours of training. 

```shell
$ python train_ppo.py
```

## Evaluate

To evaluate different agents against each other we use the script `eval_agents.py`, this script will make the agents play 1000 games and compute the average and std of points that the right agent scored.   

Options:
* baseline
* ppo
* ppo_*

```shell
$ python eval_agents.py --left=baseline --right=ppo
```
