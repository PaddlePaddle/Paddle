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
from functools import partial

import glob
import six
import os
import tarfile

import numpy as np
import paddle.fluid as fluid


def get_input_descs(args):

    batch_size = args.batch_size  # TODO None(before)
    # The placeholder for squence length in compile time.
    seq_len = None  # TODO: None (before)
    # The head number.
    n_head = getattr(args, "n_head", 8)
    # The model dim.
    d_model = getattr(args, "d_model", 512)

    # Here list the data shapes and data types of all inputs.
    # The shapes here act as placeholder and are set to pass the infer-shape in
    # compile time.
    input_descs = {
        # The actual data shape of src_word is:
        # [batch_size, max_src_len_in_batch]
        "src_word": [(batch_size, seq_len), "int64", 2],
        # The actual data shape of src_pos is:
        # [batch_size, max_src_len_in_batch, 1]
        "src_pos": [(batch_size, seq_len), "int64"],
        # This input is used to remove attention weights on paddings in the
        # encoder.
        # The actual data shape of src_slf_attn_bias is:
        # [batch_size, n_head, max_src_len_in_batch, max_src_len_in_batch]
        "src_slf_attn_bias":
        [(batch_size, n_head, seq_len, seq_len), "float32"],
        # The actual data shape of trg_word is:
        # [batch_size, max_trg_len_in_batch, 1]
        "trg_word": [(batch_size, seq_len), "int64",
                     2],  # lod_level is only used in fast decoder.
        # The actual data shape of trg_pos is:
        # [batch_size, max_trg_len_in_batch, 1]
        "trg_pos": [(batch_size, seq_len), "int64"],
        # This input is used to remove attention weights on paddings and
        # subsequent words in the decoder.
        # The actual data shape of trg_slf_attn_bias is:
        # [batch_size, n_head, max_trg_len_in_batch, max_trg_len_in_batch]
        "trg_slf_attn_bias":
        [(batch_size, n_head, seq_len, seq_len), "float32"],
        # This input is used to remove attention weights on paddings of the source
        # input in the encoder-decoder attention.
        # The actual data shape of trg_src_attn_bias is:
        # [batch_size, n_head, max_trg_len_in_batch, max_src_len_in_batch]
        "trg_src_attn_bias": [(batch_size, n_head, 1, seq_len), "float32"
                              ],  # TODO: 1 for predict, seq_len for train
        # This input is used in independent decoder program for inference.
        # The actual data shape of enc_output is:
        # [batch_size, max_src_len_in_batch, d_model]
        "enc_output": [(batch_size, seq_len, d_model), "float32"],
        # The actual data shape of label_word is:
        # [batch_size * max_trg_len_in_batch, 1]
        "lbl_word": [(None, 1), "int64"],
        # This input is used to mask out the loss of paddding tokens.
        # The actual data shape of label_weight is:
        # [batch_size * max_trg_len_in_batch, 1]
        "lbl_weight": [(None, 1), "float32"],
        # This input is used in beam-search decoder.
        "init_score": [(batch_size, 1), "float32", 2],
        # This input is used in beam-search decoder for the first gather
        # (cell states updation)
        "init_idx": [(batch_size, ), "int32"],
    }

    return input_descs


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
    "trg_src_attn_bias", )


class InputField(object):
    """
    A high-level API for handling inputs in PaddlePaddle.
    """

    def __init__(self, input_slots, build_pyreader=False, capacity=100):
        self.feed_list = []
        self.loader = None
        for slot in input_slots:
            self.feed_list.append(
                fluid.layers.data(
                    name=slot['name'],
                    shape=slot['shape'],
                    dtype=slot['dtype'],
                    lod_level=slot.get('lod_level', 0),
                    append_batch_size=False))

        self.loader = fluid.io.DataLoader.from_generator(
            feed_list=self.feed_list,
            capacity=capacity,
            iterable=(not build_pyreader),
            use_double_buffer=True)


class ModelHyperParams(object):
    # used for continuous evaluation
    enable_ce = False

    # The frequency to save trained models when training.
    save_step = 10000
    # The frequency to fetch and print output when training.
    print_step = 2
    # path of trained parameter, to make prediction
    init_from_params = "trained_models/step_10/"
    # the directory for saving model
    save_model = "trained_models"
    # the directory for saving inference model.
    inference_model_dir = "infer_model"
    # Set seed for CE or debug
    random_seed = None
    # The pattern to match training data files.
    training_file = "train.tok.clean.bpe.1000.en-de"
    # The pattern to match test data files.
    predict_file = "newstest2016.tok.bpe.32000.en-de"
    # The file to output the translation results of predict_file to.
    output_file = "predict.txt"
    # The path of vocabulary file of source language.
    src_vocab_fpath = "vocab_all.bpe.32000"
    # The path of vocabulary file of target language.
    trg_vocab_fpath = "vocab_all.bpe.32000"
    # The <bos>, <eos> and <unk> tokens in the dictionary.
    special_token = ["<s>", "<e>", "<unk>"]
    # whether to use cuda
    use_cuda = True

    # args for reader, see reader.py for details
    token_delimiter = " "
    use_token_batch = True
    pool_size = 200000
    sort_type = "pool"
    shuffle = False
    shuffle_batch = False
    batch_size = 5

    # Hyparams for training:
    # the number of epoches for training
    epoch = 1
    # the hyper parameters for Adam optimizer.
    # This static learning_rate will be multiplied to the LearningRateScheduler
    # derived learning rate the to get the final learning rate.
    learning_rate = 2.0
    beta1 = 0.9
    beta2 = 0.997
    eps = 1e-9
    # the parameters for learning rate scheduling.
    warmup_steps = 8000
    # the weight used to mix up the ground-truth distribution and the fixed
    # uniform distribution in label smoothing when training.
    # Set this as zero if label smoothing is not wanted.
    label_smooth_eps = 0.1

    # Hyparams for generation:
    # the parameters for beam search.
    beam_size = 5
    max_out_len = 256
    # the number of decoded sentences to output.
    n_best = 1

    # Hyparams for model:
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
    # dropout rates of different modules.
    prepostprocess_dropout = 0.1
    attention_dropout = 0.1
    relu_dropout = 0.1
    # to process before each sub-layer
    preprocess_cmd = "n"  # layer normalization
    # to process after each sub-layer
    postprocess_cmd = "da"  # dropout + residual connection
    # the flag indicating whether to share embedding and softmax weights.
    # vocabularies in source and target should be same for weight sharing.
    weight_sharing = True


def pad_batch_data(insts,
                   pad_idx,
                   n_head,
                   is_target=False,
                   is_label=False,
                   return_attn_bias=True,
                   return_max_len=True,
                   return_num_token=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.
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
        return_list += [inst_pos.astype("int64").reshape([-1, 1])]
    if return_attn_bias:
        if is_target:
            # This is used to avoid attention on paddings and subsequent
            # words.
            slf_attn_bias_data = np.ones((inst_data.shape[0], max_len, max_len))
            slf_attn_bias_data = np.triu(slf_attn_bias_data,
                                         1).reshape([-1, 1, max_len, max_len])
            slf_attn_bias_data = np.tile(slf_attn_bias_data,
                                         [1, n_head, 1, 1]) * [-1e9]
        else:
            # This is used to avoid attention on paddings.
            slf_attn_bias_data = np.array([[0] * len(inst) + [-1e9] *
                                           (max_len - len(inst))
                                           for inst in insts])
            slf_attn_bias_data = np.tile(
                slf_attn_bias_data.reshape([-1, 1, 1, max_len]),
                [1, n_head, max_len, 1])
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
    """
    Put all padded data needed by training into a list.
    """
    src_word, src_pos, src_slf_attn_bias, src_max_len = pad_batch_data(
        [inst[0] for inst in insts], src_pad_idx, n_head, is_target=False)
    src_word = src_word.reshape(-1, src_max_len)
    src_pos = src_pos.reshape(-1, src_max_len)
    trg_word, trg_pos, trg_slf_attn_bias, trg_max_len = pad_batch_data(
        [inst[1] for inst in insts], trg_pad_idx, n_head, is_target=True)
    trg_word = trg_word.reshape(-1, trg_max_len)
    trg_pos = trg_pos.reshape(-1, trg_max_len)

    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :],
                                [1, 1, trg_max_len, 1]).astype("float32")

    lbl_word, lbl_weight, num_token = pad_batch_data(
        [inst[2] for inst in insts],
        trg_pad_idx,
        n_head,
        is_target=False,
        is_label=True,
        return_attn_bias=False,
        return_max_len=False,
        return_num_token=True)
    lbl_word = lbl_word.reshape(-1, 1)
    lbl_weight = lbl_weight.reshape(-1, 1)

    data_inputs = [
        src_word, src_pos, src_slf_attn_bias, trg_word, trg_pos,
        trg_slf_attn_bias, trg_src_attn_bias, lbl_word, lbl_weight
    ]

    return data_inputs


def prepare_infer_input(insts, src_pad_idx, bos_idx, n_head, place):
    """
    Put all padded data needed by beam search decoder into a list.
    """
    src_word, src_pos, src_slf_attn_bias, src_max_len = pad_batch_data(
        [inst[0] for inst in insts], src_pad_idx, n_head, is_target=False)
    # start tokens
    trg_word = np.asarray([[bos_idx]] * len(insts), dtype="int64")
    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :],
                                [1, 1, 1, 1]).astype("float32")
    trg_word = trg_word.reshape(-1, 1)
    src_word = src_word.reshape(-1, src_max_len)
    src_pos = src_pos.reshape(-1, src_max_len)

    data_inputs = [
        src_word, src_pos, src_slf_attn_bias, trg_word, trg_src_attn_bias
    ]
    return data_inputs


class SortType(object):
    GLOBAL = 'global'
    POOL = 'pool'
    NONE = "none"


class Converter(object):
    def __init__(self, vocab, beg, end, unk, delimiter, add_beg):
        self._vocab = vocab
        self._beg = beg
        self._end = end
        self._unk = unk
        self._delimiter = delimiter
        self._add_beg = add_beg

    def __call__(self, sentence):
        return ([self._beg] if self._add_beg else []) + [
            self._vocab.get(w, self._unk)
            for w in sentence.split(self._delimiter)
        ] + [self._end]


class ComposedConverter(object):
    def __init__(self, converters):
        self._converters = converters

    def __call__(self, parallel_sentence):
        return [
            self._converters[i](parallel_sentence[i])
            for i in range(len(self._converters))
        ]


class SentenceBatchCreator(object):
    def __init__(self, batch_size):
        self.batch = []
        self._batch_size = batch_size

    def append(self, info):
        self.batch.append(info)
        if len(self.batch) == self._batch_size:
            tmp = self.batch
            self.batch = []
            return tmp


class TokenBatchCreator(object):
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


class SampleInfo(object):
    def __init__(self, i, max_len, min_len):
        self.i = i
        self.min_len = min_len
        self.max_len = max_len


class MinMaxFilter(object):
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


class DataProcessor(object):
    def __init__(self,
                 src_vocab_fpath,
                 trg_vocab_fpath,
                 fpattern,
                 batch_size,
                 device_count,
                 n_head,
                 pool_size,
                 sort_type=SortType.GLOBAL,
                 clip_last_batch=False,
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
                 only_src=False,
                 seed=0):
        # convert str to bytes, and use byte data
        field_delimiter = field_delimiter.encode("utf8")
        token_delimiter = token_delimiter.encode("utf8")
        start_mark = start_mark.encode("utf8")
        end_mark = end_mark.encode("utf8")
        unk_mark = unk_mark.encode("utf8")
        self._src_vocab = self.load_dict(src_vocab_fpath)
        self._trg_vocab = self.load_dict(trg_vocab_fpath)
        self._bos_idx = self._src_vocab[start_mark]
        self._eos_idx = self._src_vocab[end_mark]
        self._unk_idx = self._src_vocab[unk_mark]
        self._only_src = only_src
        self._pool_size = pool_size
        self._batch_size = batch_size
        self._device_count = device_count
        self._n_head = n_head
        self._use_token_batch = use_token_batch
        self._sort_type = sort_type
        self._clip_last_batch = clip_last_batch
        self._shuffle = shuffle
        self._shuffle_batch = shuffle_batch
        self._min_length = min_length
        self._max_length = max_length
        self._field_delimiter = field_delimiter
        self._token_delimiter = token_delimiter
        self.load_src_trg_ids(fpattern, tar_fname)
        self._random = np.random
        self._random.seed(seed)

    def load_src_trg_ids(self, fpattern, tar_fname):
        converters = [
            Converter(
                vocab=self._src_vocab,
                beg=self._bos_idx,
                end=self._eos_idx,
                unk=self._unk_idx,
                delimiter=self._token_delimiter,
                add_beg=False)
        ]
        if not self._only_src:
            converters.append(
                Converter(
                    vocab=self._trg_vocab,
                    beg=self._bos_idx,
                    end=self._eos_idx,
                    unk=self._unk_idx,
                    delimiter=self._token_delimiter,
                    add_beg=True))

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
        assert len(fpaths) > 0, "no matching file to the provided data path"

        if len(fpaths) == 1 and tarfile.is_tarfile(fpaths[0]):
            if tar_fname is None:
                raise Exception("If tar file provided, please set tar_fname.")

            f = tarfile.open(fpaths[0], "rb")
            for line in f.extractfile(tar_fname):
                fields = line.strip(b"\n").split(self._field_delimiter)
                if (not self._only_src and len(fields) == 2) or (
                        self._only_src and len(fields) == 1):
                    yield fields
        else:
            for fpath in fpaths:
                if not os.path.isfile(fpath):
                    raise IOError("Invalid file: %s" % fpath)

                with open(fpath, "rb") as f:
                    for line in f:
                        fields = line.strip(b"\n").split(self._field_delimiter)
                        if (not self._only_src and len(fields) == 2) or (
                                self._only_src and len(fields) == 1):
                            yield fields

    @staticmethod
    def load_dict(dict_path, reverse=False):
        word_dict = {}
        with open(dict_path, "rb") as fdict:
            for idx, line in enumerate(fdict):
                if reverse:
                    word_dict[idx] = line.strip(b"\n")
                else:
                    word_dict[line.strip(b"\n")] = idx
        return word_dict

    def batch_generator(self, batch_size, use_token_batch):
        def __impl__():
            # global sort or global shuffle
            if self._sort_type == SortType.GLOBAL:
                infos = sorted(self._sample_infos, key=lambda x: x.max_len)
            else:
                if self._shuffle:
                    infos = self._sample_infos
                    self._random.shuffle(infos)
                else:
                    infos = self._sample_infos

                if self._sort_type == SortType.POOL:
                    reverse = True
                    for i in range(0, len(infos), self._pool_size):
                        # to avoid placing short next to long sentences
                        reverse = not reverse
                        infos[i:i + self._pool_size] = sorted(
                            infos[i:i + self._pool_size],
                            key=lambda x: x.max_len,
                            reverse=reverse)

            # concat batch
            batches = []
            batch_creator = TokenBatchCreator(
                batch_size) if use_token_batch else SentenceBatchCreator(
                    batch_size)
            batch_creator = MinMaxFilter(self._max_length, self._min_length,
                                         batch_creator)

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
                    yield [(self._src_seq_ids[idx], self._trg_seq_ids[idx][:-1],
                            self._trg_seq_ids[idx][1:]) for idx in batch_ids]

        return __impl__

    @staticmethod
    def split(data_reader, count):
        def __impl__():
            for item in data_reader():
                inst_num_per_part = len(item) // count
                for i in range(count):
                    yield item[inst_num_per_part * i:inst_num_per_part * (i + 1
                                                                          )]

        return __impl__

    def data_generator(self, phase, place=None):
        # Any token included in dict can be used to pad, since the paddings' loss
        # will be masked out by weights and make no effect on parameter gradients.
        src_pad_idx = trg_pad_idx = self._eos_idx
        bos_idx = self._bos_idx
        n_head = self._n_head
        data_reader = self.batch_generator(
            self._batch_size *
            (1 if self._use_token_batch else self._device_count),
            self._use_token_batch)
        if not self._use_token_batch:
            # to make data on each device have similar token number
            data_reader = self.split(data_reader, self._device_count)

        def __for_train__():
            for data in data_reader():
                data_inputs = prepare_train_input(data, src_pad_idx,
                                                  trg_pad_idx, n_head)
                yield data_inputs

        def __for_predict__():
            for data in data_reader():
                data_inputs = prepare_infer_input(data, src_pad_idx, bos_idx,
                                                  n_head, place)
                yield data_inputs

        return __for_train__ if phase == "train" else __for_predict__

    def get_vocab_summary(self):
        return len(self._src_vocab), len(
            self._trg_vocab), self._bos_idx, self._eos_idx, self._unk_idx


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
            "load again automatically.")
        if six.PY3:
            load_bak = pickle.load
            pickle.load = partial(load_bak, encoding="latin1")
            fluid.load(program, model_path, executor, var_list)
            pickle.load = load_bak


def load_dygraph(model_path, keep_name_table=False):
    """
    To load python2 saved models in python3.
    """
    try:
        para_dict, opti_dict = fluid.load_dygraph(model_path, keep_name_table)
        return para_dict, opti_dict
    except UnicodeDecodeError:
        warnings.warn(
            "An UnicodeDecodeError is catched, which might be caused by loading "
            "a python2 saved model. Encoding of pickle.load would be set and "
            "load again automatically.")
        if six.PY3:
            load_bak = pickle.load
            pickle.load = partial(load_bak, encoding="latin1")
            para_dict, opti_dict = fluid.load_dygraph(model_path,
                                                      keep_name_table)
            pickle.load = load_bak
            return para_dict, opti_dict
