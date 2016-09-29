set -e

function test() {
  cfg=$1
  batch=$2
  prefix=$3
  sed -i "/input: \"data\"/{n;s/^input_dim.*/input_dim: $batch/g}" $cfg 
  sed -i "/input: \"label\"/{n;s/^input_dim.*/input_dim: $batch/g}" $cfg
  caffe time --model=$cfg --iterations=50 --gpu 0 --logtostderr=1 > logs/$prefix-1gpu-batch${batch}.log 2>&1
}

if [ ! -d "logs" ]; then
  mkdir logs
fi

# alexnet
test config/alexnet.prototxt 64 alexnet 
test config/alexnet.prototxt 128 alexnet 
test config/alexnet.prototxt 256 alexnet 
test config/alexnet.prototxt 512 alexnet 

# googlenet
test config/googlenet.prototxt 64 googlenet 
test config/googlenet.prototxt 128 googlenet 

# small net 
test config/smallnet_mnist_cifar.prototxt 64 smallnet 
test config/smallnet_mnist_cifar.prototxt 128 smallnet 
test config/smallnet_mnist_cifar.prototxt 256 smallnet 
test config/smallnet_mnist_cifar.prototxt 512 smallnet 
