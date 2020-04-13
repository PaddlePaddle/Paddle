#!/bin/bash
BERT_BASE_PATH="./data/pretrained_models/uncased_L-12_H-768_A-12/"
TASK_NAME='MNLI'
DATA_PATH="./data/glue_data/MNLI/"
CKPT_PATH="./data/saved_model/mnli_models"

export CUDA_VISIBLE_DEVICES=7

# start fine-tuning
python3.7 bert_classifier.py\
    --use_cuda true \
    --do_train true \
    --do_test true \
    --batch_size 64 \
    --init_pretraining_params ${BERT_BASE_PATH}/dygraph_params/ \
    --data_dir ${DATA_PATH} \
    --vocab_path ${BERT_BASE_PATH}/vocab.txt \
    --checkpoints ${CKPT_PATH} \
    --save_steps 1000 \
    --weight_decay  0.01 \
    --warmup_proportion 0.1 \
    --validation_steps 100 \
    --epoch 3 \
    --max_seq_len 128 \
    --bert_config_path ${BERT_BASE_PATH}/bert_config.json \
    --learning_rate 5e-5 \
    --skip_steps 10 \
    --shuffle true

