set -e

function test() {
  cfg=$1
  batch_size=$2
  prefix=$3
  python $cfg --batch_size=$batch_size > logs/${prefix}-1gpu-${batch_size}.log 2>&1
}

if [ ! -d "logs" ]; then
  mkdir logs
fi

# alexnet
test alexnet.py 64 alexnet
test alexnet.py 128 alexnet
test alexnet.py 256 alexnet
test alexnet.py 512 alexnet

# googlenet
test googlenet.py 64 googlenet
test googlenet.py 128 googlenet

# smallnet 
test smallnet_mnist_cifar.py 64 smallnet
test smallnet_mnist_cifar.py 128 smallnet
test smallnet_mnist_cifar.py 256 smallnet
test smallnet_mnist_cifar.py 512 smallnet
