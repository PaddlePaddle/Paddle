# Recoginize Digits on IPU 

<p align="center">
<img src="https://github.com/PaddlePaddle/book/blob/develop/02.recognize_digits/image/mnist_example_image.png?raw=true" width="400"><br/>
Figure 1. Example of a MNIST picture
</p>

## Accuracy Alignment with IPU

Here we use LeNet-5, which consists 5 building blocks which will be tansformed and fused in modern convolution building blocks.

Trainning `mnist` on CPU achievs 98.3x% accuracy in 4 epochs, while training in IPU (after compilation) convergents faster, achieves 99.9y% accuracy in 4 epoches with the same partition of the dataset.

```bash
Epoch 4, batch 100, Cost 0.054509, Validation Cost 0.000850
Epoch 4, batch 200, Cost 0.006553, Validation Cost 0.000850
Epoch 4, batch 300, Cost 0.012887, Validation Cost 0.000850
Epoch 4, batch 400, Cost 0.005729, Validation Cost 0.000850
Epoch 4, batch 500, Cost 0.011038, Validation Cost 0.000850
Epoch 4, batch 600, Cost 0.031277, Validation Cost 0.000850
Epoch 4, batch 700, Cost 0.032082, Validation Cost 0.000850
Epoch 4, batch 800, Cost 0.054537, Validation Cost 0.000850
Epoch 4, batch 900, Cost 0.026884, Validation Cost 0.000850
Best pass is 4, validation average loss is 0.000850238038765383
The classification accuracy is 99.98%
```
<p align="center">
 Training MNIST on IPU
</p>

## Training on IPU

> bash train_with_ipu.sh

#### Options
The program has a few comand line options.

For MNIST dataset, it is enough to train on a single IPU, it is enough for precision alignmnet. We will support Model replica and data parallel for feature test purpose soon.

`-h`                   Show usage information.  

`--use_ipu`            Train on IPU.

`--num_ipus`           Number of IPUs for training, defaults to 1.

`--no_pipelining`      Tasks on IPUs will not be pipelined if graph is sharded on different IPUs, defaults to True.

`--replication-factor` Number of times to replicate the graph to perform data parallel training. This must be a factor of the number of IPUs, defaults to 1.

Examples:

    python train.py \
    --use_ipu True \
    --num_ipus 1 \
    --no_pipelining 


## Inference on IPU with python frontend

> bash infer_with_ipu.sh

Example:

```
(py37_paddle-ipu) [docker-λ>] leiw@gbnwx-pod006-3-in_docker_dev:~/Paddle/python/paddle/fluid/tests/unittests/ipu/test_dataset/mnist$ bash infer_with_ipu.sh
[09/18 07:14:40] mnist:infer INFO: Reading data ...
[09/18 07:14:40] mnist:infer INFO: Complete reading image infer_3.png
[09/18 07:14:40] mnist:infer INFO: Constructing the computation graph ...
[09/18 07:15:12] mnist:infer INFO: Computation graph built.
[09/18 07:15:12] mnist:infer INFO: Change batch size of var %s from %d to %d
[09/18 07:15:12] mnist:infer INFO: Drawing IR graph ...
[09/18 07:15:12] mnist:infer INFO: Complete drawing.
digit hand write number picture is recognized as : 3
```

## Inference on IPU with Analysis API (c++) backend

We will add this example in the future