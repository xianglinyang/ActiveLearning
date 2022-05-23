# ActiveLearning
active learning SOTA query strategies
## Strategies
- baseline
  > - "random"
- diversity based (density based)
  > - "coreset"
  > - "coresetMIP"
- uncertainty based
  > - "LeastConfidence", uncertainty sampling with minimal top confidence
  > - "entropy": uncertainty sampling with maximal entropy
  > - "LL4AL": Learning a loss_pred_net to predict uncertainty.
  > - "Margin": minimal margin, top1-top2
  > - "adversarial": adversarial active learning using DeepFool
  > - "Discriminative": discriminative active learning with raw pixels as the representation
  > - "DiscriminativeAE": discriminative active learning with an autoencoder embedding as the representation
  > - "DiscriminativeLearned": discriminative active learning with the learned representation from the model as the representation
  > - "DiscriminativeStochastic": discriminative active learning with the learned representation as the representation and sampling proportionally to the confidence as being "unlabeled".
  > - vaal (ICCV, 2019) [[code](https://github.com/sinhasam/vaal)]
  > - sraal (CVPR oral, 2020) [[code](https://github.com/Beichen1996/SRAAL)]
  > - ta-vaal (CVPR, 2021) [[code](https://github.com/cubeyoung/TA-VAAL)]
- influence based
  > - "ISAL": (ICCV, 2021) Influence Selection for Active Learning [[code](https://github.com/dragonlzm/ISAL)]
  > - "EGL": estimated gradient length
- Bayesian based
  > - "Bayesian": Bayesian uncertainty sampling with minimal top confidence (least confidence with dropout)
  > - "BayesianEntropy": Bayesian uncertainty sampling with maximal entropy (highest entropy with dropout)
  > - "Bayesian Active Learning Disagreement": Deep Bayesian Active Learning with Image Data
- hybrid
  > two methods hybrid
  > - waal


## Dependencies
```
Keras==2.4.3
scipy==1.5.2
numpy==1.19.5
torch==1.6.0+cu101
torchvision==0.4.1
tqdm==4.50.0
Pillow==8.2.0
```
Please run the following commands to install all dependencies:
```console
~$ pip install -r requirements.txt
```
## Run different strategies
see ```Run``` Dir to choose different strategies. 
Example:
```console
~$ python run_coreset.py --dataset "CIFAR10"
```
