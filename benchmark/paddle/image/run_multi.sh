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

#========multi-gpus=========#
# multi-gpu 4gpus
train alexnet.py 4 128 alexnet
train alexnet.py 4 256 alexnet
train alexnet.py 4 512 alexnet

# multi-gpu 4gpus
train alexnet.py 2 128 alexnet
train alexnet.py 2 256 alexnet
train alexnet.py 2 512 alexnet
