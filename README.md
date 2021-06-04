# Similarity and Matching of Neural Network Representations

## Install

### Conda
```console
conda env create -f environment.yml
```

### Pip
```console
pip install -r requirements.txt
```
*Note: make sure you have cuda driver & toolkit installed if applicable*

## Setup

There's a config file which tells the script where it can find or download the datasets to. Please edit `config/default.env`:
```bash
[dataset_root]
pytorch = '/data/pytorch' # path to pytroch datasets such as cifar10
celeba = '/data/celeba' # path to celeba dataset
```

## Docker
If you prefer docker, you can also build docker image:
```console
cd container
./build.sh
cd ..
```

After you can run any command inside container with:
```console
container/docker_run.sh -g 0 -c "python python/file/to/run.py"
```
* **-g** Which GPU to use for training, list GPU IDs here
* **-c** the command to run inside the container

The config file needs to be set properly. 
As you can see in [docer_run.sh](./container/docker_run.sh) the ```/home/${USER}/cache``` folder is mapped to 
the ```/cache``` folder inside, so it is recommended to store your data in 
```/home/${USER}/cache/data/pytorch``` and ```/home/${USER}/cache/data/celeba``` folders and leave the config file 
with the default settings:
```bash
[dataset_root]
pytorch = '/cache/data/pytorch' # path to pytroch datasets such as cifar10
celeba = '/cache/data/celeba' # path to celeba dataset
```

## Train

### Train a neural net

If you want to skip this step, just use the pretrained neural networks uploaded under *archive/* folder. Otherwise train a new network by

```console
python src/bin/train.py
```
#### Settings

* **-h, --help** Get help about parameters
* **-m, --model** Model to train. Please choose from: *lenet, tiny10, resnet_w1, resnet_w2, resnet_w3, resnet_w4, resnet_w5*. Default: lenet
* **-d, --dataset** Dataset to learn on. Please choose from: *cifar10, cifar100, mnist*. Default: mnist
* **-e, --epochs**  Number of epochs to train. Default: 300
* **-lr, --lr**     Learning rate. Default: 1e-1
* **-o, --out-dir** Folder to save networks to. Default: snapshots/
* **-b, --batch-size**     Batch size. Default: 128
* **-s, --save-frequency** How often to save model in iterations. Default: 10000
* **-i, --init-noise**     Initialisation seed. Default: 0
* **-g, --gradient-noise** Image batch order seed. Default: 0
* **-wd, --weight-decay**  Weight decay to use. Deault: 1e-4
* **-opt, --optimizer**    Name of the optimizer. Please choose from: *adam, sgd*. Default: sgd

There's a default schedule in the learning, the learning rate is divided by 10 at 1/3 of the training, and with another 10 at the 2/3 of the training.

Your models are going to be saved under *snapshots/* by default. 

### Train a stitching layer
```console
python src/bin/find_transform.py path/to/model1.pt /path/to/model2.pt layer1 layer2 dataset
```
where layer1 corresponds to a layer of model1, and layer2 to model2. Example:
```console
python src/bin/find_transform.py archive/Tiny10/CIFAR10/in0-gn0/110000.pt archive/Tiny10/CIFAR10/in1-gn1/110000.pt bn3 bn3 cifar10
```
#### Settings

* **-h, --help** Get help about parameters
* **--run-name** The name of the subfolder to save to. If not given, it defaults to the current date-time.
* **-e, --epochs** Number of epochs to train. Default: 30
* **-lr, --lr** Learning rate. Default: 1e-3
* **-o, --out-dir** Folder to save networks to. Default: snapshots/
* **-b, --batch-size** Batch size. Default: 128
* **-s, --save-frequency** How often to save the transformation matrix in iterations.
                       This number is multiplied by the number of epochs. Default: 10
* **--seed** Seed of the run. Default: 0
* **-wd, --weight-decay** Weight decay to use. Deault: 1e-4
* **--optimizer** Name of the optimizer. Please choose from: adam, sgd. Default: adam
* **--debug** Either to run in debug mode or not. Default: False
* **--flatten** Either to flatten layers around transformation. NOTE: not used in the paper, hardly ever used,
             it might be buggy. Default: False
* **--l1** l1 regularization used on transformation matrix. Default: 0
* **--cka-reg** CKA regularisation used on transformation matrix. Default: 0
* **-i, --init** Initialisation of transformation matrix. Options:
   * random: random initialisation. Default.
   * perm: random permutation
   * eye: identity matrix
   * ps_inv: pseudo inverse initialisation
   * ones-zeros: weight matrix is all 1, bias is all 0.
* **-m, --mask** Any mask applied on transformation. Options:
   * identity: All values are 1 in mask. Default.
   * semi-match: Based on correlation choose the best pairs.
   * abs-semi-match: Semi-match between absolute correlations.
   * random-permuation: A random permutation matrix.
* **--target-type** The loss to apply at logits. Options:
   * hard: Use true labels. Default.
   * soft_1: Use soft crossentropy loss to model1.
   * soft_2: Use soft crossentropy loss to model2.
   * soft_12: Use soft crossentropy loss to the mean of model1 and model2.
   * soft_1_plus_2: Use soft crossentropy loss to the sum of model1 and model2.
* **--temperature** The temperature to use if target type is a soft label. Default: 1.
           

There's a default schedule in the learning, the learning rate is divided by 10 at 1/3 of the training, and with another 10 at the 2/3 of the training.

You will find the results of your runs under *results/* folder by default.

#### Results: Matrix folder

In the matrix folder, you'll find pickle files that contain a lot of information
about your run. E.g. the bias & weights of the stitching layer, accuracy, crossentropy, etc.

#### Results: Pdf folder

There is a human readable format of the results, in a pdf version, under
the pdf/ folder.


## Layer information

If you're not aware of the available layer names to a given model, you can check our cheat sheet:

```console
python src/bin/layer_info.py model_name
```
Example: 
```bash
python src/bin/layer_info.py resnet_w3
```



