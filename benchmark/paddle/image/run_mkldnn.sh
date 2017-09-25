set -e

unset OMP_NUM_THREADS MKL_NUM_THREADS
export OMP_DYNAMIC="FALSE"
export KMP_AFFINITY="granularity=fine,compact,0,0"

function train() {
  topology=$1
  bs=$2
  use_mkldnn=$3
  if [ $3 == "True" ]; then
    use_mkldnn=$3
    thread=1
    log="logs/${topology}-mkldnn-${bs}.log"
  elif [ $3 == "False" ]; then
    use_mkldnn=$3
    thread=`nproc`
    log="logs/${topology}-${thread}mklml-${bs}.log"
  else
    echo "Wrong input $3, use True or False."
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
train vgg 64 True
train vgg 128 True
train vgg 256 True

#========== mklml ===========#
train vgg 64 False
train vgg 128 False
train vgg 256 False
