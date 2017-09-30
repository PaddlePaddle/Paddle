set -e

function train() {
  unset OMP_NUM_THREADS MKL_NUM_THREADS
  export OMP_DYNAMIC="FALSE"
  export KMP_AFFINITY="granularity=fine,compact,0,0"
  topology=$1
  bs=$2
  use_mkldnn=$3
  if [ $3 == "True" ]; then
    thread=1
    log="logs/${topology}-mkldnn-${bs}.log"
  elif [ $3 == "False" ]; then
    thread=`nproc`
    # each trainer_count use only 1 core to avoid conflict
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    log="logs/${topology}-${thread}mklml-${bs}.log"
  else
    echo "Wrong input $3, use True or False."
    exit 0
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

#========== mkldnn ==========#
train vgg 64 True
train vgg 128 True
train vgg 256 True

#========== mklml ===========#
train vgg 64 False
train vgg 128 False
train vgg 256 False
