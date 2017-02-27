set -e

if [ ! -d "train.list" ]; then
  echo " " > train.list
fi

paddle train \
     --config=config.py \
     --use_gpu=true \
     --trainer_count=1 \
     --log_period=1 \
