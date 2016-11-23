set -e

# If use `paddle train` to run, it must use DataProvider to
# pass the data type to PaddlePaddle system.
# And PaddlePaddle requires training set list (train.list),
function gen_file() {
  if [ ! -d "train.txt" ]; then
    for ((i=1;i<=1024;i++))
    do
      echo "train/n09246464/n09246464_38735.jpeg 972" >> train.txt
    done
  fi

  if [ ! -d "train.list" ]; then
    echo "train.txt" > train.list
  fi
}

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

gen_file
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
train googlenet.py 1 64 googlenet
train googlenet.py 1 128 googlenet
train googlenet.py 1 256 googlenet

# smallnet
train smallnet_mnist_cifar.py 1 64 smallnet
train smallnet_mnist_cifar.py 1 128 smallnet
train smallnet_mnist_cifar.py 1 256 smallnet
train smallnet_mnist_cifar.py 1 512 smallnet


############################
#========multi-gpus=========#
train alexnet.py 4 512 alexnet
train alexnet.py 4 1024 alexnet

train googlenet.py 4 512 googlenet 
train googlenet.py 4 1024 googlenet
