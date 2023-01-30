# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import pickle
import warnings
<<<<<<< HEAD
from functools import partial

import numpy as np

import paddle
import paddle.dataset.wmt16 as wmt16
import paddle.fluid as fluid
=======
import six
from functools import partial
import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.dataset.wmt16 as wmt16
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def get_input_descs(args, mode="train"):

    batch_size = args.batch_size  # TODO None(before)
    seq_len = None
    n_head = getattr(args, "n_head", 8)
    d_model = getattr(args, "d_model", 512)
    input_descs_train = {
        "src_word": [(batch_size, seq_len), "int64", 2],
        "src_pos": [(batch_size, seq_len), "int64"],
<<<<<<< HEAD
        "src_slf_attn_bias": [
            (batch_size, n_head, seq_len, seq_len),
            "float32",
        ],
        "trg_word": [(batch_size, seq_len), "int64", 2],
        "trg_pos": [(batch_size, seq_len), "int64"],
        "trg_slf_attn_bias": [
            (batch_size, n_head, seq_len, seq_len),
            "float32",
        ],
        "trg_src_attn_bias": [
            (batch_size, n_head, seq_len, seq_len),
            "float32",
        ],  # TODO: 1 for predict, seq_len for train
=======
        "src_slf_attn_bias": [(batch_size, n_head, seq_len, seq_len),
                              "float32"],
        "trg_word": [(batch_size, seq_len), "int64", 2],
        "trg_pos": [(batch_size, seq_len), "int64"],
        "trg_slf_attn_bias": [(batch_size, n_head, seq_len, seq_len),
                              "float32"],
        "trg_src_attn_bias":
        [(batch_size, n_head, seq_len, seq_len),
         "float32"],  # TODO: 1 for predict, seq_len for train
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        "enc_output": [(batch_size, seq_len, d_model), "float32"],
        "lbl_word": [(None, 1), "int64"],
        "lbl_weight": [(None, 1), "float32"],
        "init_score": [(batch_size, 1), "float32", 2],
<<<<<<< HEAD
        "init_idx": [(batch_size,), "int32"],
=======
        "init_idx": [(batch_size, ), "int32"],
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    }
    input_descs_predict = {
        "src_word": [(batch_size, seq_len), "int64", 2],
        "src_pos": [(batch_size, seq_len), "int64"],
<<<<<<< HEAD
        "src_slf_attn_bias": [
            (batch_size, n_head, seq_len, seq_len),
            "float32",
        ],
        "trg_word": [(batch_size, seq_len), "int64", 2],
        "trg_pos": [(batch_size, seq_len), "int64"],
        "trg_slf_attn_bias": [
            (batch_size, n_head, seq_len, seq_len),
            "float32",
        ],
=======
        "src_slf_attn_bias": [(batch_size, n_head, seq_len, seq_len),
                              "float32"],
        "trg_word": [(batch_size, seq_len), "int64", 2],
        "trg_pos": [(batch_size, seq_len), "int64"],
        "trg_slf_attn_bias": [(batch_size, n_head, seq_len, seq_len),
                              "float32"],
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        "trg_src_attn_bias": [(batch_size, n_head, 1, seq_len), "float32"],
        "enc_output": [(batch_size, seq_len, d_model), "float32"],
        "lbl_word": [(None, 1), "int64"],
        "lbl_weight": [(None, 1), "float32"],
        "init_score": [(batch_size, 1), "float32", 2],
<<<<<<< HEAD
        "init_idx": [(batch_size,), "int32"],
=======
        "init_idx": [(batch_size, ), "int32"],
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    }

    return input_descs_train if mode == "train" else input_descs_predict


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
fast_decoder_data_input_fields = (
    "trg_word",
    "trg_src_attn_bias",
)


<<<<<<< HEAD
class ModelHyperParams:
=======
class ModelHyperParams(object):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    print_step = 2
    save_dygraph_model_path = "dygraph_trained_models"
    save_static_model_path = "static_trained_models"
    inference_model_dir = "infer_model"
    output_file = "predict.txt"
    batch_size = 5
    epoch = 1
    learning_rate = 2.0
    beta1 = 0.9
    beta2 = 0.997
    eps = 1e-9
    warmup_steps = 8000
    label_smooth_eps = 0.1
    beam_size = 5
    max_out_len = 5  # small number to avoid the unittest timeout
    n_best = 1
    src_vocab_size = 36556
    trg_vocab_size = 36556
    bos_idx = 0  # index for <bos> token
    eos_idx = 1  # index for <eos> token
    unk_idx = 2  # index for <unk> token
    max_length = 256
    d_model = 512
    d_inner_hid = 2048
    d_key = 64
    d_value = 64
    n_head = 8
    n_layer = 6
    prepostprocess_dropout = 0.1
    attention_dropout = 0.1
    relu_dropout = 0.1
    preprocess_cmd = "n"  # layer normalization
    postprocess_cmd = "da"  # dropout + residual connection
    weight_sharing = True


<<<<<<< HEAD
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
    return_list = []
    max_len = max(len(inst) for inst in insts)
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
                list(range(0, len(inst))) + [0] * (max_len - len(inst))
                for inst in insts
            ]
        )
=======
def pad_batch_data(insts,
                   pad_idx,
                   n_head,
                   is_target=False,
                   is_label=False,
                   return_attn_bias=True,
                   return_max_len=True,
                   return_num_token=False):
    return_list = []
    max_len = max(len(inst) for inst in insts)
    inst_data = np.array(
        [inst + [pad_idx] * (max_len - len(inst)) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, 1])]
    if is_label:  # label weight
        inst_weight = np.array([[1.] * len(inst) + [0.] * (max_len - len(inst))
                                for inst in insts])
        return_list += [inst_weight.astype("float32").reshape([-1, 1])]
    else:  # position data
        inst_pos = np.array([
            list(range(0, len(inst))) + [0] * (max_len - len(inst))
            for inst in insts
        ])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return_list += [inst_pos.astype("int64").reshape([-1, 1])]
    if return_attn_bias:
        if is_target:
            slf_attn_bias_data = np.ones((inst_data.shape[0], max_len, max_len))
<<<<<<< HEAD
            slf_attn_bias_data = np.triu(slf_attn_bias_data, 1).reshape(
                [-1, 1, max_len, max_len]
            )
            slf_attn_bias_data = np.tile(
                slf_attn_bias_data, [1, n_head, 1, 1]
            ) * [-1e9]
        else:
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
=======
            slf_attn_bias_data = np.triu(slf_attn_bias_data,
                                         1).reshape([-1, 1, max_len, max_len])
            slf_attn_bias_data = np.tile(slf_attn_bias_data,
                                         [1, n_head, 1, 1]) * [-1e9]
        else:
            slf_attn_bias_data = np.array([[0] * len(inst) + [-1e9] *
                                           (max_len - len(inst))
                                           for inst in insts])
            slf_attn_bias_data = np.tile(
                slf_attn_bias_data.reshape([-1, 1, 1, max_len]),
                [1, n_head, max_len, 1])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return_list += [slf_attn_bias_data.astype("float32")]
    if return_max_len:
        return_list += [max_len]
    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]
    return return_list if len(return_list) > 1 else return_list[0]


def prepare_train_input(insts, src_pad_idx, trg_pad_idx, n_head):
    src_word, src_pos, src_slf_attn_bias, src_max_len = pad_batch_data(
<<<<<<< HEAD
        [inst[0] for inst in insts], src_pad_idx, n_head, is_target=False
    )
    src_word = src_word.reshape(-1, src_max_len)
    src_pos = src_pos.reshape(-1, src_max_len)
    trg_word, trg_pos, trg_slf_attn_bias, trg_max_len = pad_batch_data(
        [inst[1] for inst in insts], trg_pad_idx, n_head, is_target=True
    )
    trg_word = trg_word.reshape(-1, trg_max_len)
    trg_pos = trg_pos.reshape(-1, trg_max_len)

    trg_src_attn_bias = np.tile(
        src_slf_attn_bias[:, :, ::src_max_len, :], [1, 1, trg_max_len, 1]
    ).astype("float32")
=======
        [inst[0] for inst in insts], src_pad_idx, n_head, is_target=False)
    src_word = src_word.reshape(-1, src_max_len)
    src_pos = src_pos.reshape(-1, src_max_len)
    trg_word, trg_pos, trg_slf_attn_bias, trg_max_len = pad_batch_data(
        [inst[1] for inst in insts], trg_pad_idx, n_head, is_target=True)
    trg_word = trg_word.reshape(-1, trg_max_len)
    trg_pos = trg_pos.reshape(-1, trg_max_len)

    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :],
                                [1, 1, trg_max_len, 1]).astype("float32")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    lbl_word, lbl_weight, num_token = pad_batch_data(
        [inst[2] for inst in insts],
        trg_pad_idx,
        n_head,
        is_target=False,
        is_label=True,
        return_attn_bias=False,
        return_max_len=False,
<<<<<<< HEAD
        return_num_token=True,
    )
=======
        return_num_token=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    lbl_word = lbl_word.reshape(-1, 1)
    lbl_weight = lbl_weight.reshape(-1, 1)

    data_inputs = [
<<<<<<< HEAD
        src_word,
        src_pos,
        src_slf_attn_bias,
        trg_word,
        trg_pos,
        trg_slf_attn_bias,
        trg_src_attn_bias,
        lbl_word,
        lbl_weight,
=======
        src_word, src_pos, src_slf_attn_bias, trg_word, trg_pos,
        trg_slf_attn_bias, trg_src_attn_bias, lbl_word, lbl_weight
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    ]

    return data_inputs


def prepare_infer_input(insts, src_pad_idx, bos_idx, n_head):
    src_word, src_pos, src_slf_attn_bias, src_max_len = pad_batch_data(
<<<<<<< HEAD
        [inst[0] for inst in insts], src_pad_idx, n_head, is_target=False
    )
    # start tokens
    trg_word = np.asarray([[bos_idx]] * len(insts), dtype="int64")
    trg_src_attn_bias = np.tile(
        src_slf_attn_bias[:, :, ::src_max_len, :], [1, 1, 1, 1]
    ).astype("float32")
=======
        [inst[0] for inst in insts], src_pad_idx, n_head, is_target=False)
    # start tokens
    trg_word = np.asarray([[bos_idx]] * len(insts), dtype="int64")
    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :],
                                [1, 1, 1, 1]).astype("float32")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    trg_word = trg_word.reshape(-1, 1)
    src_word = src_word.reshape(-1, src_max_len)
    src_pos = src_pos.reshape(-1, src_max_len)

    data_inputs = [
<<<<<<< HEAD
        src_word,
        src_pos,
        src_slf_attn_bias,
        trg_word,
        trg_src_attn_bias,
=======
        src_word, src_pos, src_slf_attn_bias, trg_word, trg_src_attn_bias
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    ]
    return data_inputs


def get_feed_data_reader(args, mode='train'):
<<<<<<< HEAD
    def __for_train__():
        train_reader = paddle.batch(
            wmt16.train(args.src_vocab_size, args.trg_vocab_size),
            batch_size=args.batch_size,
        )
        for batch in train_reader():
            tensors = prepare_train_input(
                batch, args.eos_idx, args.eos_idx, args.n_head
            )
            yield tensors

    def __for_test__():
        test_reader = paddle.batch(
            wmt16.test(args.src_vocab_size, args.trg_vocab_size),
            batch_size=args.batch_size,
        )
        for batch in test_reader():
            tensors = prepare_infer_input(
                batch, args.eos_idx, args.eos_idx, args.n_head
            )
=======

    def __for_train__():
        train_reader = paddle.batch(wmt16.train(args.src_vocab_size,
                                                args.trg_vocab_size),
                                    batch_size=args.batch_size)
        for batch in train_reader():
            tensors = prepare_train_input(batch, args.eos_idx, args.eos_idx,
                                          args.n_head)
            yield tensors

    def __for_test__():
        test_reader = paddle.batch(wmt16.test(args.src_vocab_size,
                                              args.trg_vocab_size),
                                   batch_size=args.batch_size)
        for batch in test_reader():
            tensors = prepare_infer_input(batch, args.eos_idx, args.eos_idx,
                                          args.n_head)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            yield tensors

    return __for_train__ if mode == 'train' else __for_test__


<<<<<<< HEAD
class InputField:
=======
class InputField(object):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self, input_slots):
        self.feed_list = []
        for slot in input_slots:
            self.feed_list.append(
<<<<<<< HEAD
                paddle.static.data(
                    name=slot['name'],
                    shape=slot['shape'],
                    dtype=slot['dtype'],
                    lod_level=slot.get('lod_level', 0),
                )
            )
=======
                fluid.layers.data(name=slot['name'],
                                  shape=slot['shape'],
                                  dtype=slot['dtype'],
                                  lod_level=slot.get('lod_level', 0),
                                  append_batch_size=False))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def load(program, model_path, executor=None, var_list=None):
    """
    To load python2 saved models in python3.
    """
    try:
        fluid.load(program, model_path, executor, var_list)
    except UnicodeDecodeError:
        warnings.warn(
            "An UnicodeDecodeError is catched, which might be caused by loading "
            "a python2 saved model. Encoding of pickle.load would be set and "
<<<<<<< HEAD
            "load again automatically."
        )
        load_bak = pickle.load
        pickle.load = partial(load_bak, encoding="latin1")
        fluid.load(program, model_path, executor, var_list)
        pickle.load = load_bak
=======
            "load again automatically.")
        if six.PY3:
            load_bak = pickle.load
            pickle.load = partial(load_bak, encoding="latin1")
            fluid.load(program, model_path, executor, var_list)
            pickle.load = load_bak
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def load_dygraph(model_path, keep_name_table=False):
    """
    To load python2 saved models in python3.
    """
    try:
<<<<<<< HEAD
        para_dict = paddle.load(
            model_path + '.pdparams', keep_name_table=keep_name_table
        )
        opti_dict = paddle.load(
            model_path + '.pdopt', keep_name_table=keep_name_table
        )
=======
        para_dict, opti_dict = fluid.load_dygraph(
            model_path, keep_name_table=keep_name_table)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return para_dict, opti_dict
    except UnicodeDecodeError:
        warnings.warn(
            "An UnicodeDecodeError is catched, which might be caused by loading "
            "a python2 saved model. Encoding of pickle.load would be set and "
<<<<<<< HEAD
            "load again automatically."
        )
        load_bak = pickle.load
        pickle.load = partial(load_bak, encoding="latin1")
        para_dict = paddle.load(
            model_path + '.pdparams', keep_name_table=keep_name_table
        )
        opti_dict = paddle.load(
            model_path + '.pdopt', keep_name_table=keep_name_table
        )
        pickle.load = load_bak
        return para_dict, opti_dict
=======
            "load again automatically.")
        if six.PY3:
            load_bak = pickle.load
            pickle.load = partial(load_bak, encoding="latin1")
            para_dict, opti_dict = fluid.load_dygraph(
                model_path, keep_name_table=keep_name_table)
            pickle.load = load_bak
            return para_dict, opti_dict
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
