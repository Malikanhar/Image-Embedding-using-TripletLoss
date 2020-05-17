# Image Embedding using Triplet Loss
Facenet--Triplet loss for image embedding using Cifar100 dataset

## Problem Analysis
Image embedding is a computer vision tasks using convolutional neural networks to convert an image into an array of size (1, n) where n is the size of the embedding. This can be done by selecting 3 images, where the first is the anchor that will be the reference image, the second is a similar image with the anchor (same class), and the negative which has a different class from the anchor.

![Triplet Loss](https://github.com/Malikanhar/Image-Similarity-using-Facenet/blob/master/assets/triplet_loss.PNG)

The loss function is defined as :

![Triplet Formula](https://github.com/Malikanhar/Image-Similarity-using-Facenet/blob/master/assets/triplet_formula.png)

Where `d(a, p)` and `d(a, n)` represent the Euclidean distances between the Anchor and the Positive and Negative pairs. margin is a parameter helping the network learning a specific distance between positive and negative samples (using the anchor).

By using this formula, the network will learn to produce the smallest distance between Positive-Anchor and the largest distance between Anchor-Negative as illustrated in the following figure.

![Triplet Loss](https://github.com/Malikanhar/Image-Similarity-using-Facenet/blob/master/assets/triplet_learning.PNG)

## Getting Started
First, you have to download Cifar100 dataset for training and Cifar10 dataset for the validation from [here](https://www.cs.toronto.edu/~kriz/cifar.html). You can also using imagenet dataset for the better model performance.

### Re-order Dataset Directory
Facenet requires a dataset directory as the following:
```bash
dataset
    |-- airplane
    |   |-- airplane_0001.png
    |   |-- airplane_0002.png
    |   '-- airplane_0003.png
    |-- cat
    |   |-- cat_0001.png
    |   |-- cat_0002.png
    |   '-- cat_0003.png
    |-- frog
    |   |-- frog_0001.png
    |   |-- frog_0002.png
    |   '-- frog_0003.png
    '-- truck
        |-- truck_0001.png
        |-- truck_0002.png
        '-- truck_0003.png
```

### Generate Validation Pairs
Generate a pairs.txt file for the validation
<pre>
python src/generate_pairs.py
  --data_dir cifar10
</pre>

### Start Training
Before strat the training, you need to add the src for the PYTHONPATH by running the following command:
<pre>export PYTHONPATH=src</pre>

Start the training by running the following command:
<pre>
python src/train_tripletloss.py 
  --models_base_dir models
  --data_dir dataset/cifar100
  --image_size 32 
  --model_def models.squeezenet 
  --optimizer ADAGRAD
  --max_nrof_epochs 100
  --lfw_pairs dataset/cifar10_pairs.txt 
  --lfw_dir /content/facenet/dataset/cifar10
</pre>

### Freeze Graph
<pre>
python src/freeze_graph.py
  checkpoint/
  frozen_graph.pb
</pre>

where the `checkpoint/` is a directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters, and `frozen_graph.pb` is the filename for the exported graphdef protobuf (.pb).
