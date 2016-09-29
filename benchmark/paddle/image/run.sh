set -e

function train() {
  cfg=$1
  thread=$2
  bz=$3
  args="batch_size=$3"
  prefix=$4
  paddle train --job=time \
    --config=$cfg \
    --use_gpu=True \
    --trainer_count=$thread \
    --log_period=10 \
    --test_period=100 \
    --config_args=$args \
    > logs/$prefix-${thread}gpu-$bz.log 2>&1 
}

if [ ! -d "logs" ]; then
  mkdir logs
fi

#========single-gpu=========#
# alexnet
train alexnet.py 1 64 alexnet
train alexnet.py 1 128 alexnet
train alexnet.py 1 256 alexnet
train alexnet.py 1 512 alexnet

# googlenet
train googlenet.py 1 64 goolenet
train googlenet.py 1 128 goolenet

# smallnet
train smallnet_mnist_cifar.py 1 64 smallnet
train smallnet_mnist_cifar.py 1 128 smallnet
train smallnet_mnist_cifar.py 1 256 smallnet
train smallnet_mnist_cifar.py 1 512 smallnet
