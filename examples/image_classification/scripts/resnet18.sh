export CUDA_VISIBLE_DEVICES=0,1,2,3 

# 默认imagenet数据存储在data/ILSVRC2012/下，去除-d便使用静态图模式运行
python -m paddle.distributed.launch main.py \
        --arch resnet18 \
        --epoch 120 \
        --batch-size 64 \
        --learning-rate 0.1 \
        --lr-scheduler cosine \
        -d \
        data/ILSVRC2012/