# Muesli-cartpole

## Links
Muesli paper link : https://arxiv.org/abs/2104.06159

CartPole-v1 env document : https://www.gymlibrary.dev/environments/classic_control/cart_pole/

## Implemented
- [x] MuZero network
- [x] L_pg+cmpo
- [x] L_v
- [x] L_r
- [x] L_m
- [x] replay buffer & replay proportion
- [x] minibatch update (CPU)
- [x] gradient clipping by value [-1,1]
- [x] dynamics network gradient scale 1/2
- [x] target network(prior parameters) moving average update

## Todo
- [ ] Tensor to GPU
- [ ] minibatch update (GPU)
- [ ] parallel loss compute
- [ ] Retrace estimator
- [ ] normalized advantage
- [ ] hyperparameter tuning
- [ ] categorical representation (value, reward)

## Differences from paper
- [x] self play follow main network inferenced policy (originally follow target network)
- [x] code [if i%30==0:] in main loop is not in official pseudocode. 

## Memo
This code(.ipynb) is executable in Google Colab. Requirements.txt is from Colab CPU compute backend.


