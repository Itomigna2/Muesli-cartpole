# Muesli-cartpole

This repository is deprecated. I am working now on https://github.com/Itomigna2/Muesli-lunarlander 

## Links
Colab demo link : https://colab.research.google.com/drive/1lckCJbZAUgUMhGMtb-v9IZb4QrQYhGSM?usp=sharing

Muesli paper link : https://arxiv.org/abs/2104.06159

CartPole-v1 env document : https://www.gymlibrary.dev/environments/classic_control/cart_pole/

## Implemented
- [x] MuZero network
- [x] 5 step unroll
- [x] L_pg+cmpo
- [x] L_v
- [x] L_r
- [x] L_m (5 step)
- [x] Stacking 8 observations
- [x] Mini-batch update 
- [x] Hidden state scaled within [-1,1]
- [x] Gradient clipping by value [-1,1]
- [x] Dynamics network gradient scale 1/2
- [x] Target network(prior parameters) moving average update
- [x] Categorical representation (value, reward model)
- [x] Normalized advantage
- [x] Tensorboard monitoring

## Differences from paper
- [x] self play follow main network inferenced policy (originally follow target network)

## Memo
This code(.ipynb) is executable in Google Colab. Requirements.txt is from Colab CPU compute backend.


