## Image
We testing AlexNet, GooleNet and a small network which refer the config of cifar10 in Caffe.

Benchmark config:

- AlexNet
  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet

- GoogleNet v1
goolenet.prototxt: https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet .
But remove loss1 and loss2 when testing benchmark.

- SmallNet
smallnet_mnist_cifar: https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_quick_train_test.prototxt


## RNN
We use lstm network for text classfication to test benchmark.

### Data Set
- IMDB
- Sequence legth=100, in fact, PaddlePaddle support training with variable-length sequence. But tensorflow need to pad, in order to compare, we also pad sequence length to 100 in PaddlePaddle.
- Dictionary size=30000 
- peephole connection is used both in PaddlePaddle and tensorflow.

### Single GPU
  We testing network for different numbers of lstm layer, hidden size, batch size.

### Multi GPU
  We fixed hidden size as 512, testing network for different numbers of lstm layer and batch size.
