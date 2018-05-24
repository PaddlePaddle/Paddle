set -e

function clock_to_seconds() {
  hours=`echo $1 | awk -F ':' '{print $1}'`
  mins=`echo $1 | awk -F ':' '{print $2}'`
  secs=`echo $1 | awk -F ':' '{print $3}'`
  echo `awk 'BEGIN{printf "%.2f",('$secs' + '$mins' * 60 + '$hours' * 3600)}'`
}

function infer() {
  unset OMP_NUM_THREADS MKL_NUM_THREADS OMP_DYNAMIC KMP_AFFINITY
  topology=$1
  layer_num=$2
  bs=$3
  use_mkldnn=$4
  if [ $4 == "True" ]; then
    thread=1
    log="logs/infer-${topology}-${layer_num}-mkldnn-${bs}.log"
  elif [ $4 == "False" ]; then
    thread=`nproc`
    if [ $thread -gt $bs ]; then
      thread=$bs
    fi
    log="logs/infer-${topology}-${layer_num}-${thread}mklml-${bs}.log"
  else
    echo "Wrong input $4, use True or False."
    exit 0
  fi

  models_in="models/${topology}-${layer_num}/pass-00000/"
  if [ ! -d $models_in ]; then
    echo "Training model ${topology}_${layer_num}"
    paddle train --job=train \
      --config="${topology}.py" \
      --use_mkldnn=True \
      --use_gpu=False \
      --trainer_count=1 \
      --num_passes=1 \
      --save_dir="models/${topology}-${layer_num}" \
      --config_args="batch_size=128,layer_num=${layer_num},num_samples=256" \
      > /dev/null 2>&1
    echo "Done"
  fi
  log_period=$((256 / bs))
  paddle train --job=test \
    --config="${topology}.py" \
    --use_mkldnn=$use_mkldnn \
    --use_gpu=False \
    --trainer_count=$thread \
    --log_period=$log_period \
    --config_args="batch_size=${bs},layer_num=${layer_num},is_infer=True" \
    --init_model_path=$models_in \
    2>&1 | tee ${log}

  # calculate the last 5 logs period time of 1280 samples,
  # the time before are burning time.
  start=`tail ${log} -n 7 | head -n 1 | awk -F ' ' '{print $2}' | xargs`
  end=`tail ${log} -n 2 | head -n 1 | awk -F ' ' '{print $2}' | xargs`
  start_sec=`clock_to_seconds $start`
  end_sec=`clock_to_seconds $end`
  fps=`awk 'BEGIN{printf "%.2f",(1280 / ('$end_sec' - '$start_sec'))}'`
  echo "Last 1280 samples start: ${start}(${start_sec} sec), end: ${end}(${end_sec} sec;" >> ${log}
  echo "FPS: $fps images/sec" 2>&1 | tee -a ${log}
}

if [ ! -f "train.list" ]; then
  echo " " > train.list
fi
if [ ! -f "test.list" ]; then
  echo " " > test.list
fi
if [ ! -d "logs" ]; then
  mkdir logs
fi
if [ ! -d "models" ]; then
  mkdir -p models
fi

# inference benchmark
for use_mkldnn in True False; do
  for batchsize in 1 2 4 8 16; do
    infer vgg 19 $batchsize $use_mkldnn
    infer resnet 50 $batchsize $use_mkldnn
    infer googlenet v1 $batchsize $use_mkldnn
    infer alexnet 2 $batchsize $use_mkldnn
  done
done
