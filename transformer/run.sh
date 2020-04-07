python -u train.py \
 --epoch 30 \
 --src_vocab_fpath wmt16_ende_data_bpe/vocab_all.bpe.32000 \
 --trg_vocab_fpath wmt16_ende_data_bpe/vocab_all.bpe.32000 \
 --special_token '<s>' '<e>' '<unk>' \
 --training_file wmt16_ende_data_bpe/train.tok.clean.bpe.32000.en-de.tiny \
 --validation_file wmt16_ende_data_bpe/newstest2014.tok.bpe.32000.en-de \
 --batch_size 4096 \
 --print_step 1 \
 --use_cuda True \
 --random_seed 1000 \
 --save_step 10 \
 --eager_run True
 #--init_from_pretrain_model base_model_dygraph/step_100000/ \
 #--init_from_checkpoint trained_models/step_200/transformer
 #--n_head 16 \
 #--d_model 1024 \
 #--d_inner_hid 4096 \
 #--prepostprocess_dropout 0.3
exit

echo `date`

python -u predict.py \
  --src_vocab_fpath wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --trg_vocab_fpath wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --predict_file wmt16_ende_data_bpe/newstest2014.tok.bpe.32000.en-de \
  --batch_size 64 \
  --init_from_params base_model_dygraph/step_100000/ \
  --beam_size 5 \
  --max_out_len 255 \
  --output_file predict.txt \
  --eager_run True
  #--max_length 500 \
  #--n_head 16 \
  #--d_model 1024 \
  #--d_inner_hid 4096 \
  #--prepostprocess_dropout 0.3

echo `date`