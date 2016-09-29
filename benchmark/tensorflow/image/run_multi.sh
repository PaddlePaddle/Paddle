set -e

function test() {
  cfg=$1
  num_gpu=$2
  batch_size=$2
  batch_per_gpu=`expr ${batch_size} / ${num_gpu}`
  prefix=$3
  python $cfg --num_gpus=$num_gpu --batch_size=${batch_per_gpu} > logs/${prefix}-1gpu-${batch_size}.log 2>&1
}

if [ ! -d "logs" ]; then
  mkdir logs
fi

# alexnet
test alexnet_multi_gpu.py 4 128 alexnet
test alexnet_multi_gpu.py 4 256 alexnet
test alexnet_multi_gpu.py 4 512 alexnet
