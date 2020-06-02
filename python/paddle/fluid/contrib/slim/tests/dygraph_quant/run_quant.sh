export CUDA_VISIBLE_DEVICES=4
export FLAGS_fraction_of_gpu_memory_to_use=0.3

epoch=1
batch_size=64
lr=0.0001
is_fast_test=false

#for model_name in mobilenet_v1 resnet50
for model_name in mobilenet_v1
do
    if [ "$is_fast_test" = true ];
    then
        python quant_dygraph.py \
            --model_name=${model_name} \
            --epoch=${epoch} \
            --batch_size=${batch_size} \
            --lr=${lr} \
            --action_fast_test
    else
        python quant_dygraph.py \
            --model_name=${model_name} \
            --epoch=${epoch} \
            --batch_size=${batch_size} \
            --lr=${lr}
    fi
done
