export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m paddle.distributed.launch --selected_gpus=0,1,2,3  train.py \
          --train_file ./data/train.tsv \
		  --test_file  ./data/test.tsv \
          --word_dict_path ./data/word.dic \
          --label_dict_path ./data/tag.dic \ 
          --word_rep_dict_path ./data/q2b.dic \
          --device gpu \
          --grnn_hidden_dim 128 \
          --word_emb_dim 128 \
          --bigru_num 2 \
          --base_learning_rate 1e-3 \
          --batch_size 300 \
          --epoch 10 \
          --save_dir   ./model \
          -d
