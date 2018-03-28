set -e

function clock_to_seconds() {
  hours=`echo $1 | awk -F ':' '{print $1}'`
  mins=`echo $1 | awk -F ':' '{print $2}'`
  secs=`echo $1 | awk -F ':' '{print $3}'`
  echo `awk 'BEGIN{printf "%.2f",('$secs' + '$mins' * 60 + '$hours' * 3600)}'`
}

function infer() {
  export OPENBLAS_MAIN_FREE=1
  topology=$1
  layer_num=$2
  bs=$3
  trainers=`nproc`
  if [ $trainers -gt $bs ]; then
    trainers=$bs
  fi
  log="logs/infer-${topology}-${layer_num}-${trainers}openblas-${bs}.log"
  threads=$((`nproc` / trainers))
  if [ $threads -eq 0 ]; then
    threads=1
  fi
  export OPENBLAS_NUM_THREADS=$threads

  models_in="models/${topology}-${layer_num}/pass-00000/"
  if [ ! -d $models_in ]; then
    echo "./run_mkl_infer.sh to save the model first"
    exit 0
  fi
  log_period=$((32 / bs))
  paddle train --job=test \
    --config="${topology}.py" \
    --use_mkldnn=False \
    --use_gpu=False \
    --trainer_count=$trainers \
    --log_period=$log_period \
    --config_args="batch_size=${bs},layer_num=${layer_num},is_infer=True,num_samples=256" \
    --init_model_path=$models_in \
    2>&1 | tee ${log}

  # calculate the last 5 logs period time of 160(=32*5) samples,
  # the time before are burning time.
  start=`tail ${log} -n 7 | head -n 1 | awk -F ' ' '{print $2}' | xargs`
  end=`tail ${log} -n 2 | head -n 1 | awk -F ' ' '{print $2}' | xargs`
  start_sec=`clock_to_seconds $start`
  end_sec=`clock_to_seconds $end`
  fps=`awk 'BEGIN{printf "%.2f",(160 / ('$end_sec' - '$start_sec'))}'`
  echo "Last 160 samples start: ${start}(${start_sec} sec), end: ${end}(${end_sec} sec;" >> ${log}
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

# inference benchmark
for batchsize in 1 2 4 8 16; do
  infer vgg 19 $batchsize
  infer resnet 50 $batchsize 
  infer googlenet v1 $batchsize
  infer alexnet 2 $batchsize
done
