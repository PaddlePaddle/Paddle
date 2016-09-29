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
    >rnn-pad${4}-${thread}gpu-batch${6}-hid${5}-lstm${3}.log 2>&1
}

#-----config--gpu--lstm_num--padding--hidden_size--batch_size
#==================multi gpus=====================#
## hidden_size=512, lstm_num=2, different batch size
train rnn.py 4 2 1 512 128 
train rnn.py 4 2 1 512 256 
train rnn.py 4 2 1 512 512 

## hidden_size=512, lstm_num=4, different batch size
train rnn.py 4 4 1 512 128 
train rnn.py 4 4 1 512 256 
train rnn.py 4 4 1 512 512 


#==================single gpu======================#
#=========for compare, testing single gpu again====#
## hidden_size=512, lstm_num=2, different batch size
train rnn.py 1 2 1 512 128 
train rnn.py 1 2 1 512 256 
train rnn.py 1 2 1 512 512 

## hidden_size=512, lstm_num=4, different batch size
train rnn.py 1 4 1 512 128 
train rnn.py 1 4 1 512 256 
train rnn.py 1 4 1 512 512 
