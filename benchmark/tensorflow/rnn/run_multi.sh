set -e

function test() {
  num_gpu=$1
  lstm_num=$2
  hid_size=$3
  batch_per_gpu=`expr ${batch_size} / ${num_gpu}`
  batch_size=$4
  python rnn_multi_gpu.py --num_layers=${lstm_num} --batch_size=$batch_per_gpu \
      --num_gpus=${num_gpu} \
      --hidden_size=${hid_size} \
      --forward_backward_only=1 \
      > logs/${num_gpu}gpu-${lstm_num}lstm-hid${hid_size}-batch${batch_size}.log 2>&1
}

if [ ! -d "logs" ]; then
  mkdir logs
fi

#--num_gpus--lstm_num--hiddne_size--batch_size--#
test 4 2 256 128 
test 4 2 256 256 
test 4 2 256 512 

test 4 2 512 128 
test 4 2 512 256 
test 4 2 512 512 
