set -e

unset OMP_NUM_THREADS MKL_NUM_THREADS
export OMP_DYNAMIC="FALSE"
export KMP_AFFINITY="granularity=fine,compact,0,0"

function train() {
  topology=$1
  bs=$2
  thread=1
  if [ $3 ]; then
    thread=$3
  fi
  if [ $thread -eq 1 ]; then
    use_mkldnn=1
    log="logs/${topology}-mkldnn-${bs}.log"
  else
    use_mkldnn=0
    log="logs/${topology}-${thread}mklml-${bs}.log"
  fi
  args="batch_size=${bs}"
  config="${topology}.py"
  paddle train --job=time \
    --config=$config \
    --use_mkldnn=$use_mkldnn \
    --use_gpu=False \
    --trainer_count=$thread \
    --log_period=10 \
    --test_period=100 \
    --config_args=$args \
    2>&1 | tee ${log} 
}

if [ ! -d "train.list" ]; then
  echo " " > train.list
fi
if [ ! -d "logs" ]; then
  mkdir logs
fi

#========= mkldnn =========#
# vgg
train vgg 64 
train vgg 128
train vgg 256

#========== mklml ===========#
train vgg 64 16
train vgg 128 16
train vgg 256 16
