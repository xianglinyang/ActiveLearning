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
  > - "LL4AL"
  > - "Margin Sampling": Active Hidden Markov Models for Information Extraction, IDA, 2001
  > - "entropy": uncertainty sampling with maximal entropy
  > - "adversarial": adversarial active learning using DeepFool
  > - "EGL": estimated gradient length
  > - "Discriminative": discriminative active learning with raw pixels as the representation
  > - "DiscriminativeAE": discriminative active learning with an autoencoder embedding as the representation
  > - "DiscriminativeLearned": discriminative active learning with the learned representation from the model as the representation
  > - "DiscriminativeStochastic": discriminative active learning with the learned representation as the representation and sampling proportionally to the confidence as being "unlabeled".
  > - "Uncertainty Sampling with Dropout Estimation": Deep Bayesian Active Learning with Image Data
- Bayesian based
  > - "Bayesian": Bayesian uncertainty sampling with minimal top confidence
  > - "BayesianEntropy": Bayesian uncertainty sampling with maximal entropy
  > - Bayesian Active Learning Disagreement: Deep Bayesian Active Learning with Image Data
- hybrid


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
