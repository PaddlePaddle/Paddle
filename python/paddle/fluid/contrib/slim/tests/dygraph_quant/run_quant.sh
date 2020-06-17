export CUDA_VISIBLE_DEVICES=2
export FLAGS_fraction_of_gpu_memory_to_use=0.8
export FLAGS_cudnn_deterministic=1

data_dir='/work/datasets/ILSVRC2012/'
out_dir='output_0610'
epoch=10
batch_size=128
lr=0.0001
is_fast_test=false

#for model_name in mobilenet_v1 resnet50
for model_name in mobilenet_v1
do
    if [ $is_fast_test = true ];
    then
        echo "is_fast_test=true"
        python -u quant_dygraph.py \
            --model_name=${model_name} \
            --data_path=${data_dir} \
            --output_dir=${out_dir} \
            --epoch=${epoch} \
            --batch_size=${batch_size} \
            --lr=${lr} \
            --action_fast_test
    else
        echo "is_fast_test=false"
        python -u quant_dygraph.py \
            --model_name=${model_name} \
            --data_path=${data_dir} \
            --output_dir=${out_dir} \
            --epoch=${epoch} \
            --batch_size=${batch_size} \
            --lr=${lr}
    fi
done
