#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import functools
import glob
import os
import random
import tarfile
import time
from functools import partial
from os.path import expanduser

import numpy as np
from test_dist_base import RUN_STEP, TestDistRunnerBase, runtime_main

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.nn.functional as F

const_para_attr = fluid.ParamAttr(initializer=fluid.initializer.Constant(0.001))
const_bias_attr = const_para_attr

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1


# from transformer_config import ModelHyperParams, TrainTaskConfig, merge_cfg_from_list
class TrainTaskConfig:
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
        "/.cache/paddle/dataset/test_dist_transformer/"
    )
    src_vocab_fpath = data_path + "vocab.bpe.32000"
    trg_vocab_fpath = data_path + "vocab.bpe.32000"
    train_file_pattern = data_path + "train.tok.clean.bpe.32000.en-de"
    val_file_pattern = data_path + "newstest2013.tok.bpe.32000.en-de.cut"
    pool_size = 2000
    sort_type = None
    local = True
    shuffle = False
    shuffle_batch = False
    special_token = ['<s>', '<e>', '<unk>']
    token_delimiter = ' '
    use_token_batch = False


class InferTaskConfig:
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


class ModelHyperParams:
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
    "src_word": [(batch_size, seq_len, 1), "int64", 2],
    # The actual data shape of src_pos is:
    # [batch_size * max_src_len_in_batch, 1]
    "src_pos": [(batch_size, seq_len, 1), "int64"],
    # This input is used to remove attention weights on paddings in the
    # encoder.
    # The actual data shape of src_slf_attn_bias is:
    # [batch_size, n_head, max_src_len_in_batch, max_src_len_in_batch]
    "src_slf_attn_bias": [
        (batch_size, ModelHyperParams.n_head, seq_len, seq_len),
        "float32",
    ],
    # The actual data shape of trg_word is:
    # [batch_size * max_trg_len_in_batch, 1]
    "trg_word": [
        (batch_size, seq_len, 1),
        "int64",
        2,
    ],  # lod_level is only used in fast decoder.
    # The actual data shape of trg_pos is:
    # [batch_size * max_trg_len_in_batch, 1]
    "trg_pos": [(batch_size, seq_len, 1), "int64"],
    # This input is used to remove attention weights on paddings and
    # subsequent words in the decoder.
    # The actual data shape of trg_slf_attn_bias is:
    # [batch_size, n_head, max_trg_len_in_batch, max_trg_len_in_batch]
    "trg_slf_attn_bias": [
        (batch_size, ModelHyperParams.n_head, seq_len, seq_len),
        "float32",
    ],
    # This input is used to remove attention weights on paddings of the source
    # input in the encoder-decoder attention.
    # The actual data shape of trg_src_attn_bias is:
    # [batch_size, n_head, max_trg_len_in_batch, max_src_len_in_batch]
    "trg_src_attn_bias": [
        (batch_size, ModelHyperParams.n_head, seq_len, seq_len),
        "float32",
    ],
    # This input is used in independent decoder program for inference.
    # The actual data shape of enc_output is:
    # [batch_size, max_src_len_in_batch, d_model]
    "enc_output": [(batch_size, seq_len, ModelHyperParams.d_model), "float32"],
    # The actual data shape of label_word is:
    # [batch_size * max_trg_len_in_batch, 1]
    "lbl_word": [(batch_size * seq_len, 1), "int64"],
    # This input is used to mask out the loss of padding tokens.
    # The actual data shape of label_weight is:
    # [batch_size * max_trg_len_in_batch, 1]
    "lbl_weight": [(batch_size * seq_len, 1), "float32"],
    # These inputs are used to change the shape tensor in beam-search decoder.
    "trg_slf_attn_pre_softmax_shape_delta": [(2,), "int32"],
    "trg_slf_attn_post_softmax_shape_delta": [(4,), "int32"],
    "init_score": [(batch_size, 1), "float32"],
}

# Names of word embedding table which might be reused for weight sharing.
word_emb_param_names = (
    "src_word_emb_table",
    "trg_word_emb_table",
)
# Names of position encoding table which will be initialized externally.
pos_enc_param_names = (
    "src_pos_enc_table",
    "trg_pos_enc_table",
)
# separated inputs for different usages.
encoder_data_input_fields = (
    "src_word",
    "src_pos",
    "src_slf_attn_bias",
)
decoder_data_input_fields = (
    "trg_word",
    "trg_pos",
    "trg_slf_attn_bias",
    "trg_src_attn_bias",
    "enc_output",
)
label_data_input_fields = (
    "lbl_word",
    "lbl_weight",
)
# In fast decoder, trg_pos (only containing the current time step) is generated
# by ops and trg_slf_attn_bias is not needed.
fast_decoder_data_input_fields = (
    "trg_word",
    "init_score",
    "trg_src_attn_bias",
)

# fast_decoder_util_input_fields = (
#     "trg_slf_attn_pre_softmax_shape_delta",
#     "trg_slf_attn_post_softmax_shape_delta", )


# from optim import LearningRateScheduler
class LearningRateScheduler:
    """
    Wrapper for learning rate scheduling as described in the Transformer paper.
    LearningRateScheduler adapts the learning rate externally and the adapted
    learning rate will be fed into the main_program as input data.
    """

    def __init__(
        self,
        d_model,
        warmup_steps,
        learning_rate=0.001,
        current_steps=0,
        name="learning_rate",
    ):
        self.current_steps = current_steps
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.static_lr = learning_rate
        self.learning_rate = paddle.static.create_global_var(
            name=name,
            shape=[1],
            value=float(learning_rate),
            dtype="float32",
            persistable=True,
        )

    def update_learning_rate(self):
        self.current_steps += 1
        lr_value = (
            np.power(self.d_model, -0.5)
            * np.min(
                [
                    np.power(self.current_steps, -0.5),
                    np.power(self.warmup_steps, -1.5) * self.current_steps,
                ]
            )
            * self.static_lr
        )
        return np.array([lr_value], dtype="float32")


# from transformer_train import train_loop
def pad_batch_data(
    insts,
    pad_idx,
    n_head,
    is_target=False,
    is_label=False,
    return_attn_bias=True,
    return_max_len=True,
    return_num_token=False,
):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    num_token = (
        functools.reduce(lambda x, y: x + y, [len(inst) for inst in insts])
        if return_num_token
        else 0
    )
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.
    inst_data = np.array(
        [inst + [pad_idx] * (max_len - len(inst)) for inst in insts]
    )
    return_list += [inst_data.astype("int64").reshape([-1, 1])]
    if is_label:  # label weight
        inst_weight = np.array(
            [
                [1.0] * len(inst) + [0.0] * (max_len - len(inst))
                for inst in insts
            ]
        )
        return_list += [inst_weight.astype("float32").reshape([-1, 1])]
    else:  # position data
        inst_pos = np.array(
            [
                list(range(1, len(inst) + 1)) + [0] * (max_len - len(inst))
                for inst in insts
            ]
        )
        return_list += [inst_pos.astype("int64").reshape([-1, 1])]
    if return_attn_bias:
        if is_target:
            # This is used to avoid attention on paddings and subsequent
            # words.
            slf_attn_bias_data = np.ones((inst_data.shape[0], max_len, max_len))
            slf_attn_bias_data = np.triu(slf_attn_bias_data, 1).reshape(
                [-1, 1, max_len, max_len]
            )
            slf_attn_bias_data = np.tile(
                slf_attn_bias_data, [1, n_head, 1, 1]
            ) * [-1e9]
        else:
            # This is used to avoid attention on paddings.
            slf_attn_bias_data = np.array(
                [
                    [0] * len(inst) + [-1e9] * (max_len - len(inst))
                    for inst in insts
                ]
            )
            slf_attn_bias_data = np.tile(
                slf_attn_bias_data.reshape([-1, 1, 1, max_len]),
                [1, n_head, max_len, 1],
            )
        return_list += [slf_attn_bias_data.astype("float32")]
    if return_max_len:
        return_list += [max_len]
    if return_num_token:
        return_list += [num_token]
    return return_list if len(return_list) > 1 else return_list[0]


def prepare_batch_input(
    insts, data_input_names, src_pad_idx, trg_pad_idx, n_head, d_model
):
    """
    Put all padded data needed by training into a dict.
    """
    src_word, src_pos, src_slf_attn_bias, src_max_len = pad_batch_data(
        [inst[0] for inst in insts], src_pad_idx, n_head, is_target=False
    )
    src_word = src_word.reshape(-1, src_max_len, 1)
    src_pos = src_pos.reshape(-1, src_max_len, 1)
    trg_word, trg_pos, trg_slf_attn_bias, trg_max_len = pad_batch_data(
        [inst[1] for inst in insts], trg_pad_idx, n_head, is_target=True
    )
    trg_word = trg_word.reshape(-1, trg_max_len, 1)
    trg_pos = trg_pos.reshape(-1, trg_max_len, 1)

    trg_src_attn_bias = np.tile(
        src_slf_attn_bias[:, :, ::src_max_len, :], [1, 1, trg_max_len, 1]
    ).astype("float32")

    lbl_word, lbl_weight, num_token = pad_batch_data(
        [inst[2] for inst in insts],
        trg_pad_idx,
        n_head,
        is_target=False,
        is_label=True,
        return_attn_bias=False,
        return_max_len=False,
        return_num_token=True,
    )

    data_input_dict = dict(
        list(
            zip(
                data_input_names,
                [
                    src_word,
                    src_pos,
                    src_slf_attn_bias,
                    trg_word,
                    trg_pos,
                    trg_slf_attn_bias,
                    trg_src_attn_bias,
                    lbl_word,
                    lbl_weight,
                ],
            )
        )
    )
    return data_input_dict, np.asarray([num_token], dtype="float32")


def read_multiple(reader, count, clip_last=True):
    """
    Stack data from reader for multi-devices.
    """

    def __impl__():
        res = []
        for item in reader():
            res.append(item)
            if len(res) == count:
                yield res
                res = []
        if len(res) == count:
            yield res
        elif not clip_last:
            data = []
            for item in res:
                data += item
            if len(data) > count:
                inst_num_per_part = len(data) // count
                yield [
                    data[inst_num_per_part * i : inst_num_per_part * (i + 1)]
                    for i in range(count)
                ]

    return __impl__


def split_data(data, num_part):
    """
    Split data for each device.
    """
    if len(data) == num_part:
        return data
    data = data[0]
    inst_num_per_part = len(data) // num_part
    return [
        data[inst_num_per_part * i : inst_num_per_part * (i + 1)]
        for i in range(num_part)
    ]


def test_context(
    test_program,
    avg_cost,
    train_exe,
    dev_count,
    data_input_names,
    sum_cost,
    token_num,
):
    val_data = DataReader(
        src_vocab_fpath=TrainTaskConfig.src_vocab_fpath,
        trg_vocab_fpath=TrainTaskConfig.trg_vocab_fpath,
        fpattern=TrainTaskConfig.val_file_pattern,
        token_delimiter=TrainTaskConfig.token_delimiter,
        use_token_batch=TrainTaskConfig.use_token_batch,
        batch_size=TrainTaskConfig.batch_size
        * (1 if TrainTaskConfig.use_token_batch else dev_count),
        pool_size=TrainTaskConfig.pool_size,
        sort_type=TrainTaskConfig.sort_type,
        start_mark=TrainTaskConfig.special_token[0],
        end_mark=TrainTaskConfig.special_token[1],
        unk_mark=TrainTaskConfig.special_token[2],
        # count start and end tokens out
        max_length=ModelHyperParams.max_length - 2,
        clip_last_batch=False,
        shuffle=False,
        shuffle_batch=False,
    )

    build_strategy = fluid.BuildStrategy()

    strategy = fluid.ExecutionStrategy()
    strategy.num_threads = 1

    test_exe = fluid.ParallelExecutor(
        use_cuda=TrainTaskConfig.use_gpu,
        main_program=test_program,
        share_vars_from=train_exe,
        build_strategy=build_strategy,
        exec_strategy=strategy,
    )

    def test(exe=test_exe):
        test_total_cost = 0
        test_total_token = 0
        test_data = read_multiple(
            reader=val_data.batch_generator,
            count=dev_count if TrainTaskConfig.use_token_batch else 1,
        )
        for batch_id, data in enumerate(test_data()):
            feed_list = []
            for place_id, data_buffer in enumerate(
                split_data(data, num_part=dev_count)
            ):
                data_input_dict, _ = prepare_batch_input(
                    data_buffer,
                    data_input_names,
                    ModelHyperParams.eos_idx,
                    ModelHyperParams.eos_idx,
                    ModelHyperParams.n_head,
                    ModelHyperParams.d_model,
                )
                feed_list.append(data_input_dict)

            outs = exe.run(
                feed=feed_list, fetch_list=[sum_cost.name, token_num.name]
            )
            sum_cost_val, token_num_val = np.array(outs[0]), np.array(outs[1])
            test_total_cost += sum_cost_val.sum()
            test_total_token += token_num_val.sum()
        test_avg_cost = test_total_cost / test_total_token
        test_ppl = np.exp([min(test_avg_cost, 100)])
        return test_avg_cost, test_ppl

    return test


def train_loop(
    exe,
    train_progm,
    dev_count,
    sum_cost,
    avg_cost,
    lr_scheduler,
    token_num,
    predict,
    test_program,
):
    # Initialize the parameters.
    if TrainTaskConfig.ckpt_path:
        lr_scheduler.current_steps = TrainTaskConfig.start_step
    else:
        exe.run(fluid.framework.default_startup_program())

    train_data = DataReader(
        src_vocab_fpath=TrainTaskConfig.src_vocab_fpath,
        trg_vocab_fpath=TrainTaskConfig.trg_vocab_fpath,
        fpattern=TrainTaskConfig.train_file_pattern,
        token_delimiter=TrainTaskConfig.token_delimiter,
        use_token_batch=TrainTaskConfig.use_token_batch,
        batch_size=TrainTaskConfig.batch_size
        * (1 if TrainTaskConfig.use_token_batch else dev_count),
        pool_size=TrainTaskConfig.pool_size,
        sort_type=TrainTaskConfig.sort_type,
        shuffle=TrainTaskConfig.shuffle,
        shuffle_batch=TrainTaskConfig.shuffle_batch,
        start_mark=TrainTaskConfig.special_token[0],
        end_mark=TrainTaskConfig.special_token[1],
        unk_mark=TrainTaskConfig.special_token[2],
        # count start and end tokens out
        max_length=ModelHyperParams.max_length - 2,
        clip_last_batch=False,
    )
    train_data = read_multiple(
        reader=train_data.batch_generator,
        count=dev_count if TrainTaskConfig.use_token_batch else 1,
    )

    build_strategy = fluid.BuildStrategy()
    # Since the token number differs among devices, customize gradient scale to
    # use token average cost among multi-devices. and the gradient scale is
    # `1 / token_number` for average cost.
    build_strategy.gradient_scale_strategy = (
        fluid.BuildStrategy.GradientScaleStrategy.Customized
    )

    strategy = fluid.ExecutionStrategy()
    strategy.num_threads = 1

    train_exe = fluid.ParallelExecutor(
        use_cuda=TrainTaskConfig.use_gpu,
        loss_name=sum_cost.name,
        main_program=train_progm,
        build_strategy=build_strategy,
        exec_strategy=strategy,
    )

    data_input_names = (
        encoder_data_input_fields
        + decoder_data_input_fields[:-1]
        + label_data_input_fields
    )

    if TrainTaskConfig.val_file_pattern is not None:
        test = test_context(
            test_program,
            avg_cost,
            train_exe,
            dev_count,
            data_input_names,
            sum_cost,
            token_num,
        )

    # the best cross-entropy value with label smoothing
    loss_normalizer = -(
        (1.0 - TrainTaskConfig.label_smooth_eps)
        * np.log((1.0 - TrainTaskConfig.label_smooth_eps))
        + TrainTaskConfig.label_smooth_eps
        * np.log(
            TrainTaskConfig.label_smooth_eps
            / (ModelHyperParams.trg_vocab_size - 1)
            + 1e-20
        )
    )
    init = False
    for pass_id in range(TrainTaskConfig.pass_num):
        pass_start_time = time.time()
        for batch_id, data in enumerate(train_data()):
            if batch_id >= RUN_STEP:
                break

            feed_list = []
            total_num_token = 0

            if TrainTaskConfig.local:
                lr_rate = lr_scheduler.update_learning_rate()

            for place_id, data_buffer in enumerate(
                split_data(data, num_part=dev_count)
            ):
                data_input_dict, num_token = prepare_batch_input(
                    data_buffer,
                    data_input_names,
                    ModelHyperParams.eos_idx,
                    ModelHyperParams.eos_idx,
                    ModelHyperParams.n_head,
                    ModelHyperParams.d_model,
                )
                total_num_token += num_token
                feed_kv_pairs = list(data_input_dict.items())
                if TrainTaskConfig.local:
                    feed_kv_pairs += list(
                        {lr_scheduler.learning_rate.name: lr_rate}.items()
                    )
                feed_list.append(dict(feed_kv_pairs))

                if not init:
                    for pos_enc_param_name in pos_enc_param_names:
                        pos_enc = position_encoding_init(
                            ModelHyperParams.max_length + 1,
                            ModelHyperParams.d_model,
                        )
                        feed_list[place_id][pos_enc_param_name] = pos_enc

            if not TrainTaskConfig.check_acc:
                for feed_dict in feed_list:
                    feed_dict[sum_cost.name + "@GRAD"] = 1.0 / total_num_token
            else:
                b = 100 * TrainTaskConfig.batch_size
                a = np.asarray([b], dtype="float32")
                for feed_dict in feed_list:
                    feed_dict[sum_cost.name + "@GRAD"] = 1.0 / a

            outs = train_exe.run(
                fetch_list=[sum_cost.name, token_num.name], feed=feed_list
            )

            sum_cost_val, token_num_val = np.array(outs[0]), np.array(outs[1])
            total_sum_cost = sum_cost_val.sum()
            total_token_num = token_num_val.sum()
            total_avg_cost = total_sum_cost / total_token_num

            init = True

            # Validate and save the model for inference.
            if TrainTaskConfig.val_file_pattern is not None:
                val_avg_cost, val_ppl = test()
                print("[%f]" % val_avg_cost)
            else:
                assert False


# import transformer_reader as reader
class SortType:
    GLOBAL = 'global'
    POOL = 'pool'
    NONE = "none"


class Converter:
    def __init__(self, vocab, beg, end, unk, delimiter):
        self._vocab = vocab
        self._beg = beg
        self._end = end
        self._unk = unk
        self._delimiter = delimiter

    def __call__(self, sentence):
        return (
            [self._beg]
            + [
                self._vocab.get(w, self._unk)
                for w in sentence.split(self._delimiter)
            ]
            + [self._end]
        )


class ComposedConverter:
    def __init__(self, converters):
        self._converters = converters

    def __call__(self, parallel_sentence):
        return [
            self._converters[i](parallel_sentence[i])
            for i in range(len(self._converters))
        ]


class SentenceBatchCreator:
    def __init__(self, batch_size):
        self.batch = []
        self._batch_size = batch_size

    def append(self, info):
        self.batch.append(info)
        if len(self.batch) == self._batch_size:
            tmp = self.batch
            self.batch = []
            return tmp


class TokenBatchCreator:
    def __init__(self, batch_size):
        self.batch = []
        self.max_len = -1
        self._batch_size = batch_size

    def append(self, info):
        cur_len = info.max_len
        max_len = max(self.max_len, cur_len)
        if max_len * (len(self.batch) + 1) > self._batch_size:
            result = self.batch
            self.batch = [info]
            self.max_len = cur_len
            return result
        else:
            self.max_len = max_len
            self.batch.append(info)


class SampleInfo:
    def __init__(self, i, max_len, min_len):
        self.i = i
        self.min_len = min_len
        self.max_len = max_len


class MinMaxFilter:
    def __init__(self, max_len, min_len, underlying_creator):
        self._min_len = min_len
        self._max_len = max_len
        self._creator = underlying_creator

    def append(self, info):
        if info.max_len > self._max_len or info.min_len < self._min_len:
            return
        else:
            return self._creator.append(info)

    @property
    def batch(self):
        return self._creator.batch


class DataReader:
    """
    The data reader loads all data from files and produces batches of data
    in the way corresponding to settings.

    An example of returning a generator producing data batches whose data
    is shuffled in each pass and sorted in each pool:

    ```
    train_data = DataReader(
        src_vocab_fpath='data/src_vocab_file',
        trg_vocab_fpath='data/trg_vocab_file',
        fpattern='data/part-*',
        use_token_batch=True,
        batch_size=2000,
        pool_size=10000,
        sort_type=SortType.POOL,
        shuffle=True,
        shuffle_batch=True,
        start_mark='<s>',
        end_mark='<e>',
        unk_mark='<unk>',
        clip_last_batch=False).batch_generator
    ```

    :param src_vocab_fpath: The path of vocabulary file of source language.
    :type src_vocab_fpath: basestring
    :param trg_vocab_fpath: The path of vocabulary file of target language.
    :type trg_vocab_fpath: basestring
    :param fpattern: The pattern to match data files.
    :type fpattern: basestring
    :param batch_size: The number of sequences contained in a mini-batch.
        or the maximum number of tokens (include paddings) contained in a
        mini-batch.
    :type batch_size: int
    :param pool_size: The size of pool buffer.
    :type pool_size: int
    :param sort_type: The grain to sort by length: 'global' for all
        instances; 'pool' for instances in pool; 'none' for no sort.
    :type sort_type: basestring
    :param clip_last_batch: Whether to clip the last uncompleted batch.
    :type clip_last_batch: bool
    :param tar_fname: The data file in tar if fpattern matches a tar file.
    :type tar_fname: basestring
    :param min_length: The minimum length used to filt sequences.
    :type min_length: int
    :param max_length: The maximum length used to filt sequences.
    :type max_length: int
    :param shuffle: Whether to shuffle all instances.
    :type shuffle: bool
    :param shuffle_batch: Whether to shuffle the generated batches.
    :type shuffle_batch: bool
    :param use_token_batch: Whether to produce batch data according to
        token number.
    :type use_token_batch: bool
    :param field_delimiter: The delimiter used to split source and target in
        each line of data file.
    :type field_delimiter: basestring
    :param token_delimiter: The delimiter used to split tokens in source or
        target sentences.
    :type token_delimiter: basestring
    :param start_mark: The token representing for the beginning of
        sentences in dictionary.
    :type start_mark: basestring
    :param end_mark: The token representing for the end of sentences
        in dictionary.
    :type end_mark: basestring
    :param unk_mark: The token representing for unknown word in dictionary.
    :type unk_mark: basestring
    :param seed: The seed for random.
    :type seed: int
    """

    def __init__(
        self,
        src_vocab_fpath,
        trg_vocab_fpath,
        fpattern,
        batch_size,
        pool_size,
        sort_type=SortType.GLOBAL,
        clip_last_batch=True,
        tar_fname=None,
        min_length=0,
        max_length=100,
        shuffle=True,
        shuffle_batch=False,
        use_token_batch=False,
        field_delimiter="\t",
        token_delimiter=" ",
        start_mark="<s>",
        end_mark="<e>",
        unk_mark="<unk>",
        seed=0,
    ):
        self._src_vocab = self.load_dict(src_vocab_fpath)
        self._only_src = True
        if trg_vocab_fpath is not None:
            self._trg_vocab = self.load_dict(trg_vocab_fpath)
            self._only_src = False
        self._pool_size = pool_size
        self._batch_size = batch_size
        self._use_token_batch = use_token_batch
        self._sort_type = sort_type
        self._clip_last_batch = clip_last_batch
        self._shuffle = shuffle
        self._shuffle_batch = shuffle_batch
        self._min_length = min_length
        self._max_length = max_length
        self._field_delimiter = field_delimiter
        self._token_delimiter = token_delimiter
        self.load_src_trg_ids(
            end_mark, fpattern, start_mark, tar_fname, unk_mark
        )
        self._random = random.Random(x=seed)

    def load_src_trg_ids(
        self, end_mark, fpattern, start_mark, tar_fname, unk_mark
    ):
        converters = [
            Converter(
                vocab=self._src_vocab,
                beg=self._src_vocab[start_mark],
                end=self._src_vocab[end_mark],
                unk=self._src_vocab[unk_mark],
                delimiter=self._token_delimiter,
            )
        ]
        if not self._only_src:
            converters.append(
                Converter(
                    vocab=self._trg_vocab,
                    beg=self._trg_vocab[start_mark],
                    end=self._trg_vocab[end_mark],
                    unk=self._trg_vocab[unk_mark],
                    delimiter=self._token_delimiter,
                )
            )

        converters = ComposedConverter(converters)

        self._src_seq_ids = []
        self._trg_seq_ids = None if self._only_src else []
        self._sample_infos = []

        for i, line in enumerate(self._load_lines(fpattern, tar_fname)):
            src_trg_ids = converters(line)
            self._src_seq_ids.append(src_trg_ids[0])
            lens = [len(src_trg_ids[0])]
            if not self._only_src:
                self._trg_seq_ids.append(src_trg_ids[1])
                lens.append(len(src_trg_ids[1]))
            self._sample_infos.append(SampleInfo(i, max(lens), min(lens)))

    def _load_lines(self, fpattern, tar_fname):
        fpaths = glob.glob(fpattern)

        if len(fpaths) == 1 and tarfile.is_tarfile(fpaths[0]):
            if tar_fname is None:
                raise Exception("If tar file provided, please set tar_fname.")

            f = tarfile.open(fpaths[0], "r")
            for line in f.extractfile(tar_fname):
                line = line.decode()
                fields = line.strip("\n").split(self._field_delimiter)
                if (not self._only_src and len(fields) == 2) or (
                    self._only_src and len(fields) == 1
                ):
                    yield fields
        else:
            for fpath in fpaths:
                if not os.path.isfile(fpath):
                    raise IOError("Invalid file: %s" % fpath)

                with open(fpath, "rb") as f:
                    for line in f:
                        line = line.decode()
                        fields = line.strip("\n").split(self._field_delimiter)
                        if (not self._only_src and len(fields) == 2) or (
                            self._only_src and len(fields) == 1
                        ):
                            yield fields

    @staticmethod
    def load_dict(dict_path, reverse=False):
        word_dict = {}
        with open(dict_path, "rb") as fdict:
            for idx, line in enumerate(fdict):
                line = line.decode()
                if reverse:
                    word_dict[idx] = line.strip("\n")
                else:
                    word_dict[line.strip("\n")] = idx
        return word_dict

    def batch_generator(self):
        # global sort or global shuffle
        if self._sort_type == SortType.GLOBAL:
            infos = sorted(
                self._sample_infos, key=lambda x: x.max_len, reverse=True
            )
        else:
            if self._shuffle:
                infos = self._sample_infos
                self._random.shuffle(infos)
            else:
                infos = self._sample_infos

            if self._sort_type == SortType.POOL:
                for i in range(0, len(infos), self._pool_size):
                    infos[i : i + self._pool_size] = sorted(
                        infos[i : i + self._pool_size], key=lambda x: x.max_len
                    )

        # concat batch
        batches = []
        batch_creator = (
            TokenBatchCreator(self._batch_size)
            if self._use_token_batch
            else SentenceBatchCreator(self._batch_size)
        )
        batch_creator = MinMaxFilter(
            self._max_length, self._min_length, batch_creator
        )

        for info in infos:
            batch = batch_creator.append(info)
            if batch is not None:
                batches.append(batch)

        if not self._clip_last_batch and len(batch_creator.batch) != 0:
            batches.append(batch_creator.batch)

        if self._shuffle_batch:
            self._random.shuffle(batches)

        for batch in batches:
            batch_ids = [info.i for info in batch]

            if self._only_src:
                yield [[self._src_seq_ids[idx]] for idx in batch_ids]
            else:
                yield [
                    (
                        self._src_seq_ids[idx],
                        self._trg_seq_ids[idx][:-1],
                        self._trg_seq_ids[idx][1:],
                    )
                    for idx in batch_ids
                ]


# from transformer_model import transformer
def position_encoding_init(n_position, d_pos_vec):
    """
    Generate the initial values for the sinusoid position encoding table.
    """
    position_enc = np.array(
        [
            [
                pos / np.power(10000, 2 * (j // 2) / d_pos_vec)
                for j in range(d_pos_vec)
            ]
            if pos != 0
            else np.zeros(d_pos_vec)
            for pos in range(n_position)
        ]
    )
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return position_enc.astype("float32")


def multi_head_attention(
    queries,
    keys,
    values,
    attn_bias,
    d_key,
    d_value,
    d_model,
    n_head=1,
    dropout_rate=0.0,
    cache=None,
):
    """
    Multi-Head Attention. Note that attn_bias is added to the logit before
    computing softmax activiation to mask certain selected positions so that
    they will not considered in attention weights.
    """
    if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
        raise ValueError(
            "Inputs: queries, keys and values should all be 3-D tensors."
        )

    def __compute_qkv(queries, keys, values, n_head, d_key, d_value):
        """
        Add linear projection to queries, keys, and values.
        """
        q = layers.fc(
            input=queries,
            size=d_key * n_head,
            num_flatten_dims=2,
            param_attr=const_para_attr,
            bias_attr=const_bias_attr,
        )
        k = layers.fc(
            input=keys,
            size=d_key * n_head,
            num_flatten_dims=2,
            param_attr=const_para_attr,
            bias_attr=const_bias_attr,
        )
        v = layers.fc(
            input=values,
            size=d_value * n_head,
            num_flatten_dims=2,
            param_attr=const_para_attr,
            bias_attr=const_bias_attr,
        )
        return q, k, v

    def __split_heads(x, n_head):
        """
        Reshape the last dimension of input tensor x so that it becomes two
        dimensions and then transpose. Specifically, input a tensor with shape
        [bs, max_sequence_length, n_head * hidden_dim] then output a tensor
        with shape [bs, n_head, max_sequence_length, hidden_dim].
        """
        if n_head == 1:
            return x

        hidden_size = x.shape[-1]
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        reshaped = paddle.reshape(
            x=x, shape=[0, 0, n_head, hidden_size // n_head]
        )

        # permute the dimensions into:
        # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        return paddle.transpose(x=reshaped, perm=[0, 2, 1, 3])

    def __combine_heads(x):
        """
        Transpose and then reshape the last two dimensions of input tensor x
        so that it becomes one dimension, which is reverse to __split_heads.
        """
        if len(x.shape) == 3:
            return x
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = paddle.transpose(x, perm=[0, 2, 1, 3])
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        return paddle.reshape(
            x=trans_x,
            shape=list(map(int, [0, 0, trans_x.shape[2] * trans_x.shape[3]])),
        )

    def scaled_dot_product_attention(q, k, v, attn_bias, d_model, dropout_rate):
        """
        Scaled Dot-Product Attention
        """
        scaled_q = paddle.scale(x=q, scale=d_model**-0.5)
        product = paddle.matmul(x=scaled_q, y=k, transpose_y=True)
        if attn_bias:
            product += attn_bias
        weights = paddle.nn.functional.softmax(product)
        if dropout_rate:
            weights = paddle.nn.functional.dropout(
                weights,
                p=dropout_rate,
            )
        out = paddle.matmul(weights, v)
        return out

    q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)

    if cache is not None:  # use cache and concat time steps
        k = cache["k"] = layers.concat([cache["k"], k], axis=1)
        v = cache["v"] = layers.concat([cache["v"], v], axis=1)

    q = __split_heads(q, n_head)
    k = __split_heads(k, n_head)
    v = __split_heads(v, n_head)

    ctx_multiheads = scaled_dot_product_attention(
        q, k, v, attn_bias, d_model, dropout_rate
    )

    out = __combine_heads(ctx_multiheads)

    # Project back to the model size.
    proj_out = layers.fc(
        input=out,
        size=d_model,
        num_flatten_dims=2,
        param_attr=const_para_attr,
        bias_attr=const_bias_attr,
    )
    return proj_out


def positionwise_feed_forward(x, d_inner_hid, d_hid):
    """
    Position-wise Feed-Forward Networks.
    This module consists of two linear transformations with a ReLU activation
    in between, which is applied to each position separately and identically.
    """
    hidden = layers.fc(
        input=x,
        size=d_inner_hid,
        num_flatten_dims=2,
        act="relu",
        param_attr=const_para_attr,
        bias_attr=const_bias_attr,
    )
    out = layers.fc(
        input=hidden,
        size=d_hid,
        num_flatten_dims=2,
        param_attr=const_para_attr,
        bias_attr=const_bias_attr,
    )
    return out


def pre_post_process_layer(prev_out, out, process_cmd, dropout_rate=0.0):
    """
    Add residual connection, layer normalization and droput to the out tensor
    optionally according to the value of process_cmd.
    This will be used before or after multi-head attention and position-wise
    feed-forward networks.
    """
    for cmd in process_cmd:
        if cmd == "a":  # add residual connection
            out = out + prev_out if prev_out else out
        elif cmd == "n":  # add layer normalization
            out = layers.layer_norm(
                out,
                begin_norm_axis=len(out.shape) - 1,
                param_attr=fluid.initializer.Constant(1.0),
                bias_attr=fluid.initializer.Constant(0.0),
            )
        elif cmd == "d":  # add dropout
            if dropout_rate:
                out = paddle.nn.functional.dropout(
                    out,
                    p=dropout_rate,
                )
    return out


pre_process_layer = partial(pre_post_process_layer, None)
post_process_layer = pre_post_process_layer


def prepare_encoder(
    src_word,
    src_pos,
    src_vocab_size,
    src_emb_dim,
    src_max_len,
    dropout_rate=0.0,
    word_emb_param_name=None,
    pos_enc_param_name=None,
):
    """Add word embeddings and position encodings.
    The output tensor has a shape of:
    [batch_size, max_src_length_in_batch, d_model].
    This module is used at the bottom of the encoder stacks.
    """
    if TrainTaskConfig.check_acc:
        src_word_emb = layers.embedding(
            src_word,
            size=[src_vocab_size, src_emb_dim],
            param_attr=fluid.ParamAttr(
                name=word_emb_param_name,
                initializer=fluid.initializer.ConstantInitializer(0.001),
            ),
        )
    else:
        src_word_emb = layers.embedding(
            src_word,
            size=[src_vocab_size, src_emb_dim],
            param_attr=fluid.ParamAttr(
                name=word_emb_param_name,
                initializer=fluid.initializer.Normal(0.0, src_emb_dim**-0.5),
            ),
        )

    src_word_emb = paddle.scale(x=src_word_emb, scale=src_emb_dim**0.5)
    src_pos_enc = layers.embedding(
        src_pos,
        size=[src_max_len, src_emb_dim],
        param_attr=fluid.ParamAttr(
            name=pos_enc_param_name,
            trainable=False,
            initializer=fluid.initializer.ConstantInitializer(0.001),
        ),
    )
    src_pos_enc.stop_gradient = True
    enc_input = src_word_emb + src_pos_enc
    return (
        paddle.nn.functional.dropout(
            enc_input,
            p=dropout_rate,
        )
        if dropout_rate
        else enc_input
    )


prepare_encoder = partial(
    prepare_encoder, pos_enc_param_name=pos_enc_param_names[0]
)
prepare_decoder = partial(
    prepare_encoder, pos_enc_param_name=pos_enc_param_names[1]
)


def encoder_layer(
    enc_input,
    attn_bias,
    n_head,
    d_key,
    d_value,
    d_model,
    d_inner_hid,
    dropout_rate=0.0,
):
    """The encoder layers that can be stacked to form a deep encoder.
    This module consits of a multi-head (self) attention followed by
    position-wise feed-forward networks and both the two components companied
    with the post_process_layer to add residual connection, layer normalization
    and droput.
    """
    attn_output = multi_head_attention(
        enc_input,
        enc_input,
        enc_input,
        attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        dropout_rate,
    )
    attn_output = post_process_layer(
        enc_input, attn_output, "dan", dropout_rate
    )
    ffd_output = positionwise_feed_forward(attn_output, d_inner_hid, d_model)
    return post_process_layer(attn_output, ffd_output, "dan", dropout_rate)


def encoder(
    enc_input,
    attn_bias,
    n_layer,
    n_head,
    d_key,
    d_value,
    d_model,
    d_inner_hid,
    dropout_rate=0.0,
):
    """
    The encoder is composed of a stack of identical layers returned by calling
    encoder_layer.
    """
    for i in range(n_layer):
        enc_output = encoder_layer(
            enc_input,
            attn_bias,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            dropout_rate,
        )
        enc_input = enc_output
    return enc_output


def decoder_layer(
    dec_input,
    enc_output,
    slf_attn_bias,
    dec_enc_attn_bias,
    n_head,
    d_key,
    d_value,
    d_model,
    d_inner_hid,
    dropout_rate=0.0,
    cache=None,
):
    """The layer to be stacked in decoder part.
    The structure of this module is similar to that in the encoder part except
    a multi-head attention is added to implement encoder-decoder attention.
    """
    slf_attn_output = multi_head_attention(
        dec_input,
        dec_input,
        dec_input,
        slf_attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        dropout_rate,
        cache,
    )
    slf_attn_output = post_process_layer(
        dec_input,
        slf_attn_output,
        "dan",  # residual connection + dropout + layer normalization
        dropout_rate,
    )
    enc_attn_output = multi_head_attention(
        slf_attn_output,
        enc_output,
        enc_output,
        dec_enc_attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        dropout_rate,
    )
    enc_attn_output = post_process_layer(
        slf_attn_output,
        enc_attn_output,
        "dan",  # residual connection + dropout + layer normalization
        dropout_rate,
    )
    ffd_output = positionwise_feed_forward(
        enc_attn_output,
        d_inner_hid,
        d_model,
    )
    dec_output = post_process_layer(
        enc_attn_output,
        ffd_output,
        "dan",  # residual connection + dropout + layer normalization
        dropout_rate,
    )
    return dec_output


def decoder(
    dec_input,
    enc_output,
    dec_slf_attn_bias,
    dec_enc_attn_bias,
    n_layer,
    n_head,
    d_key,
    d_value,
    d_model,
    d_inner_hid,
    dropout_rate=0.0,
    caches=None,
):
    """
    The decoder is composed of a stack of identical decoder_layer layers.
    """
    for i in range(n_layer):
        cache = None
        if caches is not None:
            cache = caches[i]

        dec_output = decoder_layer(
            dec_input,
            enc_output,
            dec_slf_attn_bias,
            dec_enc_attn_bias,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            dropout_rate,
            cache=cache,
        )
        dec_input = dec_output
    return dec_output


def make_all_inputs(input_fields):
    """
    Define the input data layers for the transformer model.
    """
    inputs = []
    for input_field in input_fields:
        input_var = layers.data(
            name=input_field,
            shape=input_descs[input_field][0],
            dtype=input_descs[input_field][1],
            lod_level=input_descs[input_field][2]
            if len(input_descs[input_field]) == 3
            else 0,
            append_batch_size=False,
        )
        inputs.append(input_var)
    return inputs


def transformer(
    src_vocab_size,
    trg_vocab_size,
    max_length,
    n_layer,
    n_head,
    d_key,
    d_value,
    d_model,
    d_inner_hid,
    dropout_rate,
    weight_sharing,
    label_smooth_eps,
):
    if weight_sharing:
        assert (
            src_vocab_size == src_vocab_size
        ), "Vocabularies in source and target should be same for weight sharing."
    enc_inputs = make_all_inputs(encoder_data_input_fields)

    enc_output = wrap_encoder(
        src_vocab_size,
        max_length,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        dropout_rate,
        weight_sharing,
        enc_inputs,
    )

    dec_inputs = make_all_inputs(decoder_data_input_fields[:-1])

    predict = wrap_decoder(
        trg_vocab_size,
        max_length,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        dropout_rate,
        weight_sharing,
        dec_inputs,
        enc_output,
    )

    # Padding index do not contribute to the total loss. The weights is used to
    # cancel padding index in calculating the loss.
    label, weights = make_all_inputs(label_data_input_fields)
    if label_smooth_eps:
        label = F.label_smooth(
            label=layers.one_hot(input=label, depth=trg_vocab_size),
            epsilon=label_smooth_eps,
        )

    cost = paddle.nn.functional.softmax_with_cross_entropy(
        logits=paddle.reshape(predict, shape=[-1, trg_vocab_size]),
        label=label,
        soft_label=True if label_smooth_eps else False,
    )
    weighted_cost = cost * weights
    sum_cost = paddle.sum(weighted_cost)
    token_num = paddle.sum(weights)
    avg_cost = sum_cost / token_num
    avg_cost.stop_gradient = True
    return sum_cost, avg_cost, predict, token_num


def wrap_encoder(
    src_vocab_size,
    max_length,
    n_layer,
    n_head,
    d_key,
    d_value,
    d_model,
    d_inner_hid,
    dropout_rate,
    weight_sharing,
    enc_inputs=None,
):
    """
    The wrapper assembles together all needed layers for the encoder.
    """
    if enc_inputs is None:
        # This is used to implement independent encoder program in inference.
        src_word, src_pos, src_slf_attn_bias = make_all_inputs(
            encoder_data_input_fields
        )
    else:
        src_word, src_pos, src_slf_attn_bias = enc_inputs
    enc_input = prepare_encoder(
        src_word,
        src_pos,
        src_vocab_size,
        d_model,
        max_length,
        dropout_rate,
        word_emb_param_name=word_emb_param_names[0],
    )
    enc_output = encoder(
        enc_input,
        src_slf_attn_bias,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        dropout_rate,
    )
    return enc_output


def wrap_decoder(
    trg_vocab_size,
    max_length,
    n_layer,
    n_head,
    d_key,
    d_value,
    d_model,
    d_inner_hid,
    dropout_rate,
    weight_sharing,
    dec_inputs=None,
    enc_output=None,
    caches=None,
):
    """
    The wrapper assembles together all needed layers for the decoder.
    """
    if dec_inputs is None:
        # This is used to implement independent decoder program in inference.
        (
            trg_word,
            trg_pos,
            trg_slf_attn_bias,
            trg_src_attn_bias,
            enc_output,
        ) = make_all_inputs(decoder_data_input_fields)
    else:
        trg_word, trg_pos, trg_slf_attn_bias, trg_src_attn_bias = dec_inputs

    dec_input = prepare_decoder(
        trg_word,
        trg_pos,
        trg_vocab_size,
        d_model,
        max_length,
        dropout_rate,
        word_emb_param_name=word_emb_param_names[0]
        if weight_sharing
        else word_emb_param_names[1],
    )
    dec_output = decoder(
        dec_input,
        enc_output,
        trg_slf_attn_bias,
        trg_src_attn_bias,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        dropout_rate,
        caches=caches,
    )
    # Return logits for training and probs for inference.
    if weight_sharing:
        predict = paddle.matmul(
            x=dec_output,
            y=fluid.framework._get_var(word_emb_param_names[0]),
            transpose_y=True,
        )
    else:
        predict = layers.fc(
            input=dec_output,
            size=trg_vocab_size,
            num_flatten_dims=2,
            param_attr=const_para_attr,
            bias_attr=const_bias_attr,
        )
    if dec_inputs is None:
        predict = paddle.nn.functional.softmax(predict)
    return predict


def get_model(is_dist, is_async):
    sum_cost, avg_cost, predict, token_num = transformer(
        ModelHyperParams.src_vocab_size,
        ModelHyperParams.trg_vocab_size,
        ModelHyperParams.max_length + 1,
        ModelHyperParams.n_layer,
        ModelHyperParams.n_head,
        ModelHyperParams.d_key,
        ModelHyperParams.d_value,
        ModelHyperParams.d_model,
        ModelHyperParams.d_inner_hid,
        ModelHyperParams.dropout,
        ModelHyperParams.weight_sharing,
        TrainTaskConfig.label_smooth_eps,
    )

    local_lr_scheduler = LearningRateScheduler(
        ModelHyperParams.d_model,
        TrainTaskConfig.warmup_steps,
        TrainTaskConfig.learning_rate,
    )
    # Context to do validation.
    test_program = fluid.default_main_program().clone(for_test=True)

    if not is_dist:
        optimizer = fluid.optimizer.Adam(
            learning_rate=local_lr_scheduler.learning_rate,
            beta1=TrainTaskConfig.beta1,
            beta2=TrainTaskConfig.beta2,
            epsilon=TrainTaskConfig.eps,
        )
        optimizer.minimize(sum_cost)
    elif is_async:
        optimizer = fluid.optimizer.SGD(0.003)
        optimizer.minimize(sum_cost)
    else:
        lr_decay = fluid.layers.learning_rate_scheduler.noam_decay(
            ModelHyperParams.d_model, TrainTaskConfig.warmup_steps
        )

        optimizer = fluid.optimizer.Adam(
            learning_rate=lr_decay,
            beta1=TrainTaskConfig.beta1,
            beta2=TrainTaskConfig.beta2,
            epsilon=TrainTaskConfig.eps,
        )
        optimizer.minimize(sum_cost)

    return (
        sum_cost,
        avg_cost,
        predict,
        token_num,
        local_lr_scheduler,
        test_program,
    )


def update_args():
    src_dict = DataReader.load_dict(TrainTaskConfig.src_vocab_fpath)
    trg_dict = DataReader.load_dict(TrainTaskConfig.trg_vocab_fpath)
    dict_args = [
        "src_vocab_size",
        str(len(src_dict)),
        "trg_vocab_size",
        str(len(trg_dict)),
        "bos_idx",
        str(src_dict[TrainTaskConfig.special_token[0]]),
        "eos_idx",
        str(src_dict[TrainTaskConfig.special_token[1]]),
        "unk_idx",
        str(src_dict[TrainTaskConfig.special_token[2]]),
    ]
    merge_cfg_from_list(dict_args, [TrainTaskConfig, ModelHyperParams])


class DistTransformer2x2(TestDistRunnerBase):
    def run_pserver(self, args):
        get_model(True, not args.sync_mode)
        t = self.get_transpiler(
            args.trainer_id,
            fluid.default_main_program(),
            args.endpoints,
            args.trainers,
            args.sync_mode,
        )
        pserver_prog = t.get_pserver_program(args.current_endpoint)
        startup_prog = t.get_startup_program(
            args.current_endpoint, pserver_prog
        )

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_prog)
        exe.run(pserver_prog)

    def run_trainer(self, args):
        TrainTaskConfig.use_gpu = args.use_cuda
        (
            sum_cost,
            avg_cost,
            predict,
            token_num,
            local_lr_scheduler,
            test_program,
        ) = get_model(args.is_dist, not args.sync_mode)

        if args.is_dist:
            t = self.get_transpiler(
                args.trainer_id,
                fluid.default_main_program(),
                args.endpoints,
                args.trainers,
                args.sync_mode,
            )
            trainer_prog = t.get_trainer_program()
            TrainTaskConfig.batch_size = 10
            TrainTaskConfig.train_file_pattern = (
                TrainTaskConfig.data_path
                + "train.tok.clean.bpe.32000.en-de.train_{}".format(
                    args.trainer_id
                )
            )
        else:
            TrainTaskConfig.batch_size = 20
            trainer_prog = fluid.default_main_program()

        if args.use_cuda:
            place = fluid.CUDAPlace(0)
        else:
            place = fluid.CPUPlace()

        startup_exe = fluid.Executor(place)

        TrainTaskConfig.local = not args.is_dist

        train_loop(
            startup_exe,
            trainer_prog,
            1,
            sum_cost,
            avg_cost,
            local_lr_scheduler,
            token_num,
            predict,
            test_program,
        )


if __name__ == "__main__":
    update_args()
    runtime_main(DistTransformer2x2)
