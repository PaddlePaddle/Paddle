export CUDA_VISIBLE_DEVICES=6

load_model_path=$1
data_dir='/work/datasets/ILSVRC2012/'
eval_test_samples=-1  # if set as -1, eval all test samples

echo "--------eval model: ${model_name}-------------"
python -u eval.py \
   --use_gpu=True \
   --class_dim=1000 \
   --image_shape=3,224,224 \
   --data_dir=${data_dir} \
   --test_samples=${eval_test_samples} \
   --inference_model=$load_model_path 

echo "\n\n"
