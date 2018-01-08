set -e

function train() {
  export OPENBLAS_NUM_THREADS=1
  topology=$1
  layer_num=$2
  bs=$3
  thread=`nproc`
  # each trainer_count use only 1 core to avoid conflict
  log="logs/train-${topology}-${layer_num}-${thread}openblas-${bs}.log"
  args="batch_size=${bs},layer_num=${layer_num}"
  config="${topology}.py"
  paddle train --job=time \
    --config=$config \
    --use_mkldnn=False \
    --use_gpu=False \
    --trainer_count=$thread \
    --log_period=3 \
    --test_period=30 \
    --config_args=$args \
    2>&1 | tee ${log} 

  avg_time=`tail ${log} -n 1 | awk -F ' ' '{print $8}' | sed 's/avg=//'`
  fps=`awk 'BEGIN{printf "%.2f",('$bs' / '$avg_time' * 1000)}'`
  echo "FPS: $fps images/sec" 2>&1 | tee -a ${log}
}

if [ ! -f "train.list" ]; then
  echo " " > train.list
fi
if [ ! -d "logs" ]; then
  mkdir logs
fi

# training benchmark
for batchsize in 64 128 256; do
  train vgg 19 $batchsize
  train resnet 50 $batchsize
  train googlenet v1 $batchsize
  train alexnet 2 $batchsize
done
