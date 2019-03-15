#!/bin/bash

numactl -l python script/train_resnet.py \
    --skip_batch_num=10 \
    --device=CPU \
    --iterations=400 \
    --pass_num=1 \
    --batch_size=128 \
    --model=resnet_imagenet \
    --data_set=imagenet \
    --use_fake_data \
    --save_model \
    --save_model_path=<path_to_save_model>

#numactl --membind=0 --physcpubind=0-28 python script/infer_image_classification.py \
#    --device=CPU \
#    --skip_batch_num=10 \
#    --iterations=400 \
#    --batch_size=1 \
#    --data_set=imagenet \
#    --use_fake_data \
#    --infer_model_path=<path_to_model_file>
