# ActiveLearning
active learning SOTA query strategies
## Strategies
- **baseline**
  > - "random"
- **diversity based** (density based)
  > - "coreset"
  > - "coresetMIP"
- **uncertainty based**
  > - "LeastConfidence", uncertainty sampling with minimal top confidence
  > - "entropy": uncertainty sampling with maximal entropy
  > - "Margin": minimal margin, top1-top2
  > - "DFAL": adversarial active learning using DeepFool
  > - "ALFA-Mix": (CVPR 2022), close to DFAL, but work on latent space
- **model based**
  > - "LL4AL": (CVPR, 2019) Learning a loss_pred_net to predict uncertainty.
  > - vaal (ICCV, 2019) [[code](https://github.com/sinhasam/vaal)]
  > - sraal (CVPR oral, 2020) [[code](https://github.com/Beichen1996/SRAAL)]
  > - ta-vaal (CVPR, 2021) [[code](https://github.com/cubeyoung/TA-VAAL)]
  > – GCNAL: (CVPR 2021) A model-based approach that learns a graph convolutional network to measures the relation between labelled and unlabelled instances
  >> shortcoming: these AL methods do not consider the diversity of the selected samples and are prone to selecting samples with repetitive patterns
- **influence based**
  > - "ISAL": (ICCV, 2021) Influence Selection for Active Learning [[code](https://github.com/dragonlzm/ISAL)]
  > forgetting sample and data diet
  > - (data diet): (nips, 2021) [[code](https://github.com/mansheej/data_diet)]
  > - (forgetting samples): (ICLR, 2019) [[code](https://github.com/mtoneva/example_forgetting)]
  > - "EGL": estimated gradient length, similar to data diet *GraN* method
- **Bayesian based**
  > - "Bayesian": Bayesian uncertainty sampling with minimal top confidence (least confidence with dropout)
  > - "BayesianEntropy": Bayesian uncertainty sampling with maximal entropy (highest entropy with dropout)
  > - "BALD": (ICML 2017)

- **hybrid**: Hybrid AL methods exploit both diversity and uncertainty in their sample selection methodologies
  > - two methods hybrid
  > - waal: (AISATAS, 2020) [[code](https://github.com/cjshui/WAAL)]
  > - BADGE: (ICLR 2020) A hybrid approach that queries the centroids obtained from the clustering of the gradient embeddings [[code](https://github.com/JordanAsh/badge)]
  


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
