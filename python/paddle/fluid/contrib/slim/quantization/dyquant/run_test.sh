export CUDA_VISIBLE_DEVICES=5,6
export FLAGS_fraction_of_gpu_memory_to_use=0.1

model="mobilenet_v2"
epoch=10
num_workers=5
batch_size=128
train_batchs=50     # -1 means use all samples
test_batchs=-1
lr=0.0001

python test.py \
    --arch=${model} \
    --epoch=${epoch} \
    --num_workers=${num_workers} \
    --train_batchs=${train_batchs} \
    --test_batchs=${test_batchs} \
    --batch_size=${batch_size} \
    --lr=${lr} \
    --enable_quant
