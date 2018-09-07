set -e

function test() {
  lstm_num=$1
  batch_size=$2
  hid_size=$3
  prefix=$4
  python rnn.py --num_layers=${lstm_num} --batch_size=$batch_size \
      --hidden_size=${hid_size} \
      --forward_backward_only=1 \
       > logs/1gpu-${lstm_num}lstm-batch${batch_size}-hid${hid_size}.log 2>&1
}

if [ ! -d "logs" ]; then
  mkdir logs
fi

#--lstm_num--batch_size--hidden_size--#
test 2 64 256 
test 2 64 512 
test 2 64 1280 

test 2 128 256 
test 2 128 512 
test 2 128 1280 

test 2 256 256 
test 2 256 512 
test 2 256 1280 
