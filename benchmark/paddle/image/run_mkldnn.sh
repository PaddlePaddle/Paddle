set -e

function train() {
  unset OMP_NUM_THREADS MKL_NUM_THREADS
  export OMP_DYNAMIC="FALSE"
  export KMP_AFFINITY="granularity=fine,compact,0,0"
  topology=$1
  layer_num=$2
  bs=$3
  use_mkldnn=$4
  if [ $4 == "True" ]; then
    thread=1
    log="logs/${topology}-${layer_num}-mkldnn-${bs}.log"
  elif [ $4 == "False" ]; then
    thread=`nproc`
    # each trainer_count use only 1 core to avoid conflict
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    log="logs/${topology}-${layer_num}-${thread}mklml-${bs}.log"
  else
    echo "Wrong input $3, use True or False."
    exit 0
  fi
  args="batch_size=${bs},layer_num=${layer_num}"
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

for use_mkldnn in True False; do
  for batchsize in 64 128 256; do
    train vgg 19 $batchsize $use_mkldnn
    train resnet 50  $batchsize $use_mkldnn
  done
done
