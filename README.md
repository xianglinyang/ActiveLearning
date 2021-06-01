# ActiveLearning
active learning SOTA query strategies
## Strategies
- "LeastConfidence"
- "coreset"
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
