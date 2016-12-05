set -e

function test() {
  cfg=$1
  num_gpu=$2
  batch_size=$3
  batch_per_gpu=`expr ${batch_size} / ${num_gpu}`
  prefix=$4
  python $cfg --num_gpus=$num_gpu --batch_size=${batch_per_gpu} > logs/${prefix}-4gpu-${batch_size}.log 2>&1
}

if [ ! -d "logs" ]; then
  mkdir logs
fi

# alexnet
test alexnet_multi_gpu.py 4 512 alexnet
test alexnet_multi_gpu.py 4 1024 alexnet

# googlenet 
test googlenet_multi_gpu.py 4 512 alexnet
test googlenet_multi_gpu.py 4 1024 alexnet
