set -e

function test() {
  num_gpu=$1
  lstm_num=$2
  hid_size=$3
  batch_size=$4
  python rnn_multi_gpu.py --num_layers=${lstm_num} --batch_size=$batch_size \
      --num_gpus=${num_gpu} \
      --hidden_size=${hid_size} \
      --forward_backward_only=1 \
      > logs/${num_gpu}gpu-${lstm_num}lstm-batch${batch_size}-hid${hid_size}.log 2>&1
}

if [ ! -d "logs" ]; then
  mkdir logs
fi

#--num_gpus--lstm_num--hiddne_size--batch_size--#
test 4 2 512 128 
test 4 2 512 256 
test 4 2 512 512 

test 4 4 512 128 
test 4 4 512 256 
test 4 4 512 512 
