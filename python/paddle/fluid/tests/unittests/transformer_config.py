# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os.path import expanduser


class TrainTaskConfig(object):
    # only support GPU currently
    use_gpu = True
    # the epoch number to train.
    pass_num = 1
    # the number of sequences contained in a mini-batch.
    # deprecated, set batch_size in args.
    batch_size = 20
    # the hyper parameters for Adam optimizer.
    # This static learning_rate will be multiplied to the LearningRateScheduler
    # derived learning rate the to get the final learning rate.
    learning_rate = 1
    beta1 = 0.9
    beta2 = 0.98
    eps = 1e-9
    # the parameters for learning rate scheduling.
    warmup_steps = 4000
    # the weight used to mix up the ground-truth distribution and the fixed
    # uniform distribution in label smoothing when training.
    # Set this as zero if label smoothing is not wanted.
    label_smooth_eps = 0.1
    # the directory for saving trained models.
    model_dir = "trained_models"
    # the directory for saving checkpoints.
    ckpt_dir = "trained_ckpts"
    # the directory for loading checkpoint.
    # If provided, continue training from the checkpoint.
    ckpt_path = None
    # the parameter to initialize the learning rate scheduler.
    # It should be provided if use checkpoints, since the checkpoint doesn't
    # include the training step counter currently.
    start_step = 0

    check_acc = True

    data_path = expanduser("~") + (
        "/.cache/paddle/dataset/test_dist_transformer/")
    src_vocab_fpath = data_path + "vocab.bpe.32000"
    trg_vocab_fpath = data_path + "vocab.bpe.32000"
    train_file_pattern = data_path + "train.tok.clean.bpe.32000.en-de"
    val_file_pattern = data_path + "newstest2013.tok.bpe.32000.en-de"
    pool_size = 2000
    sort_type = None
    local = True
    shuffle = False
    shuffle_batch = False
    special_token = ['<s>', '<e>', '<unk>']
    token_delimiter = ' '
    use_token_batch = False


class InferTaskConfig(object):
    use_gpu = True
    # the number of examples in one run for sequence generation.
    batch_size = 10
    # the parameters for beam search.
    beam_size = 5
    max_out_len = 256
    # the number of decoded sentences to output.
    n_best = 1
    # the flags indicating whether to output the special tokens.
    output_bos = False
    output_eos = False
    output_unk = True
    # the directory for loading the trained model.
    model_path = "trained_models/pass_1.infer.model"


class ModelHyperParams(object):
    # These following five vocabularies related configurations will be set
    # automatically according to the passed vocabulary path and special tokens.
    # size of source word dictionary.
    src_vocab_size = 10000
    # size of target word dictionay
    trg_vocab_size = 10000
    # index for <bos> token
    bos_idx = 0
    # index for <eos> token
    eos_idx = 1
    # index for <unk> token
    unk_idx = 2
    # max length of sequences deciding the size of position encoding table.
    # Start from 1 and count start and end tokens in.
    max_length = 256
    # the dimension for word embeddings, which is also the last dimension of
    # the input and output of multi-head attention, position-wise feed-forward
    # networks, encoder and decoder.
    d_model = 512
    # size of the hidden layer in position-wise feed-forward networks.
    d_inner_hid = 2048
    # the dimension that keys are projected to for dot-product attention.
    d_key = 64
    # the dimension that values are projected to for dot-product attention.
    d_value = 64
    # number of head used in multi-head attention.
    n_head = 8
    # number of sub-layers to be stacked in the encoder and decoder.
    n_layer = 6
    # dropout rate used by all dropout layers.
    dropout = 0.0  # no random
    # random seed used in dropout for CE.
    dropout_seed = None
    # the flag indicating whether to share embedding and softmax weights.
    # vocabularies in source and target should be same for weight sharing.
    weight_sharing = True


def merge_cfg_from_list(cfg_list, g_cfgs):
    """
    Set the above global configurations using the cfg_list. 
    """
    assert len(cfg_list) % 2 == 0
    for key, value in zip(cfg_list[0::2], cfg_list[1::2]):
        for g_cfg in g_cfgs:
            if hasattr(g_cfg, key):
                try:
                    value = eval(value)
                except Exception:  # for file path
                    pass
                setattr(g_cfg, key, value)
                break


# The placeholder for batch_size in compile time. Must be -1 currently to be
# consistent with some ops' infer-shape output in compile time, such as the
# sequence_expand op used in beamsearch decoder.
batch_size = -1
# The placeholder for squence length in compile time.
seq_len = ModelHyperParams.max_length
# Here list the data shapes and data types of all inputs.
# The shapes here act as placeholder and are set to pass the infer-shape in
# compile time.
input_descs = {
    # The actual data shape of src_word is:
    # [batch_size * max_src_len_in_batch, 1]
    "src_word": [(batch_size, seq_len, 1L), "int64", 2],
    # The actual data shape of src_pos is:
    # [batch_size * max_src_len_in_batch, 1]
    "src_pos": [(batch_size, seq_len, 1L), "int64"],
    # This input is used to remove attention weights on paddings in the
    # encoder.
    # The actual data shape of src_slf_attn_bias is:
    # [batch_size, n_head, max_src_len_in_batch, max_src_len_in_batch]
    "src_slf_attn_bias": [(batch_size, ModelHyperParams.n_head, seq_len,
                           seq_len), "float32"],
    # The actual data shape of trg_word is:
    # [batch_size * max_trg_len_in_batch, 1]
    "trg_word": [(batch_size, seq_len, 1L), "int64",
                 2],  # lod_level is only used in fast decoder.
    # The actual data shape of trg_pos is:
    # [batch_size * max_trg_len_in_batch, 1]
    "trg_pos": [(batch_size, seq_len, 1L), "int64"],
    # This input is used to remove attention weights on paddings and
    # subsequent words in the decoder.
    # The actual data shape of trg_slf_attn_bias is:
    # [batch_size, n_head, max_trg_len_in_batch, max_trg_len_in_batch]
    "trg_slf_attn_bias": [(batch_size, ModelHyperParams.n_head, seq_len,
                           seq_len), "float32"],
    # This input is used to remove attention weights on paddings of the source
    # input in the encoder-decoder attention.
    # The actual data shape of trg_src_attn_bias is:
    # [batch_size, n_head, max_trg_len_in_batch, max_src_len_in_batch]
    "trg_src_attn_bias": [(batch_size, ModelHyperParams.n_head, seq_len,
                           seq_len), "float32"],
    # This input is used in independent decoder program for inference.
    # The actual data shape of enc_output is:
    # [batch_size, max_src_len_in_batch, d_model]
    "enc_output": [(batch_size, seq_len, ModelHyperParams.d_model), "float32"],
    # The actual data shape of label_word is:
    # [batch_size * max_trg_len_in_batch, 1]
    "lbl_word": [(batch_size * seq_len, 1L), "int64"],
    # This input is used to mask out the loss of paddding tokens.
    # The actual data shape of label_weight is:
    # [batch_size * max_trg_len_in_batch, 1]
    "lbl_weight": [(batch_size * seq_len, 1L), "float32"],
    # These inputs are used to change the shape tensor in beam-search decoder.
    "trg_slf_attn_pre_softmax_shape_delta": [(2L, ), "int32"],
    "trg_slf_attn_post_softmax_shape_delta": [(4L, ), "int32"],
    "init_score": [(batch_size, 1L), "float32"],
}

# Names of word embedding table which might be reused for weight sharing.
word_emb_param_names = (
    "src_word_emb_table",
    "trg_word_emb_table", )
# Names of position encoding table which will be initialized externally.
pos_enc_param_names = (
    "src_pos_enc_table",
    "trg_pos_enc_table", )
# separated inputs for different usages.
encoder_data_input_fields = (
    "src_word",
    "src_pos",
    "src_slf_attn_bias", )
decoder_data_input_fields = (
    "trg_word",
    "trg_pos",
    "trg_slf_attn_bias",
    "trg_src_attn_bias",
    "enc_output", )
label_data_input_fields = (
    "lbl_word",
    "lbl_weight", )
# In fast decoder, trg_pos (only containing the current time step) is generated
# by ops and trg_slf_attn_bias is not needed.
fast_decoder_data_input_fields = (
    "trg_word",
    "init_score",
    "trg_src_attn_bias", )
# fast_decoder_util_input_fields = (
#     "trg_slf_attn_pre_softmax_shape_delta",
#     "trg_slf_attn_post_softmax_shape_delta", )
