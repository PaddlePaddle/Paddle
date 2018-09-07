set -e

function train() {
  cfg=$1
  thread=$2
  args="lstm_num=${3},seq_pad=${4},hidden_size=${5},batch_size=${6}"
  paddle train --job=time \
    --config=$cfg \
    --use_gpu=1 \
    --trainer_count=$thread \
    --log_period=10 \
    --test_period=100 \
    --num_passes=1 \
    --feed_data=1 \
    --config_args=$args \
    >logs/rnn-pad${4}-${thread}gpu-lstm${3}-batch${6}-hid${5}.log 2>&1
}

if [ ! -d "logs" ]; then
  mkdir logs
fi

## padding, single gpu
#-----config--gpu--lstm_num--padding--hidden_size--batch_size
## lstm_num=2, batch_size=64
train rnn.py 1 2 1 256 64 
train rnn.py 1 2 1 512 64 
train rnn.py 1 2 1 1280 64 

## lstm_num=2, batch_size=128
train rnn.py 1 2 1 256 128 
train rnn.py 1 2 1 512 128 
train rnn.py 1 2 1 1280 128 

## lstm_num=4, batch_size=256
train rnn.py 1 2 1 256 256 
train rnn.py 1 2 1 512 256 
train rnn.py 1 2 1 1280 256 


#==================multi gpus=====================#
# hidden_size=256, lstm_num=2, different batch size
train rnn.py 4 2 1 256 128 
train rnn.py 4 2 1 256 256 
train rnn.py 4 2 1 256 512 

# hidden_size=512, lstm_num=4, different batch size
train rnn.py 4 2 1 512 128 
train rnn.py 4 2 1 512 256 
train rnn.py 4 2 1 512 512 
