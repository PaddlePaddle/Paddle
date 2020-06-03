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

from __future__ import division
from __future__ import print_function

import unittest
import random

import numpy as np

import paddle.fluid as fluid
from paddle.fluid.dygraph import Embedding, Linear, Layer
from paddle.fluid.layers import BeamSearchDecoder
from paddle.incubate.hapi.model import Model, Input, set_device
from paddle.incubate.hapi.text import *


class ModuleApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._np_rand_state = np.random.get_state()
        cls._py_rand_state = random.getstate()
        cls._random_seed = 123
        np.random.seed(cls._random_seed)
        random.seed(cls._random_seed)

        cls.model_cls = type(cls.__name__ + "Model", (Model, ), {
            "__init__": cls.model_init_wrapper(cls.model_init),
            "forward": cls.model_forward
        })

    @classmethod
    def tearDownClass(cls):
        np.random.set_state(cls._np_rand_state)
        random.setstate(cls._py_rand_state)

    @staticmethod
    def model_init_wrapper(func):
        def __impl__(self, *args, **kwargs):
            Model.__init__(self)
            func(self, *args, **kwargs)

        return __impl__

    @staticmethod
    def model_init(model, *args, **kwargs):
        raise NotImplementedError(
            "model_init acts as `Model.__init__`, thus must implement it")

    @staticmethod
    def model_forward(model, *args, **kwargs):
        return model.module(*args, **kwargs)

    def make_inputs(self):
        # TODO(guosheng): add default from `self.inputs`
        raise NotImplementedError(
            "model_inputs makes inputs for model, thus must implement it")

    def setUp(self):
        """
        For the model which wraps the module to be tested:
            Set input data by `self.inputs` list
            Set init argument values by `self.attrs` list/dict
            Set model parameter values by `self.param_states` dict
            Set expected output data by `self.outputs` list
        We can create a model instance and run once with these.
        """
        self.inputs = []
        self.attrs = {}
        self.param_states = {}
        self.outputs = []

    def _calc_output(self, place, mode="test", dygraph=True):
        if dygraph:
            fluid.enable_dygraph(place)
        else:
            fluid.disable_dygraph()
        fluid.default_main_program().random_seed = self._random_seed
        fluid.default_startup_program().random_seed = self._random_seed
        model = self.model_cls(**self.attrs) if isinstance(
            self.attrs, dict) else self.model_cls(*self.attrs)
        model.prepare(inputs=self.make_inputs(), device=place)
        if self.param_states:
            model.load(self.param_states, optim_state=None)
        return model.test_batch(self.inputs)

    def check_output_with_place(self, place, mode="test"):
        dygraph_output = self._calc_output(place, mode, dygraph=True)
        stgraph_output = self._calc_output(place, mode, dygraph=False)
        expect_output = getattr(self, "outputs", None)
        for actual_t, expect_t in zip(dygraph_output, stgraph_output):
            self.assertTrue(np.allclose(actual_t, expect_t, rtol=1e-5, atol=0))
        if expect_output:
            for actual_t, expect_t in zip(dygraph_output, expect_output):
                self.assertTrue(
                    np.allclose(
                        actual_t, expect_t, rtol=1e-5, atol=0))

    def check_output(self):
        devices = ["CPU", "GPU"] if fluid.is_compiled_with_cuda() else ["CPU"]
        for device in devices:
            place = set_device(device)
            self.check_output_with_place(place)


class TestBasicLSTM(ModuleApiTest):
    def setUp(self):
        # TODO(guosheng): Change to big size. Currently bigger hidden size for
        # LSTM would fail, the second static graph run might get diff output
        # with others.
        shape = (2, 4, 16)
        self.inputs = [np.random.random(shape).astype("float32")]
        self.outputs = None
        self.attrs = {"input_size": 16, "hidden_size": 16}
        self.param_states = {}

    @staticmethod
    def model_init(model, input_size, hidden_size):
        model.lstm = RNN(
            BasicLSTMCell(
                input_size,
                hidden_size,
                param_attr=fluid.ParamAttr(name="lstm_weight"),
                bias_attr=fluid.ParamAttr(name="lstm_bias")))

    @staticmethod
    def model_forward(model, inputs):
        return model.lstm(inputs)[0]

    def make_inputs(self):
        inputs = [
            Input(
                [None, None, self.inputs[-1].shape[-1]],
                "float32",
                name="input"),
        ]
        return inputs

    def test_check_output(self):
        self.check_output()


class TestBasicGRU(ModuleApiTest):
    def setUp(self):
        shape = (2, 4, 128)
        self.inputs = [np.random.random(shape).astype("float32")]
        self.outputs = None
        self.attrs = {"input_size": 128, "hidden_size": 128}
        self.param_states = {}

    @staticmethod
    def model_init(model, input_size, hidden_size):
        model.gru = RNN(BasicGRUCell(input_size, hidden_size))

    @staticmethod
    def model_forward(model, inputs):
        return model.gru(inputs)[0]

    def make_inputs(self):
        inputs = [
            Input(
                [None, None, self.inputs[-1].shape[-1]],
                "float32",
                name="input"),
        ]
        return inputs

    def test_check_output(self):
        self.check_output()


class TestBeamSearch(ModuleApiTest):
    def setUp(self):
        shape = (8, 32)
        self.inputs = [
            np.random.random(shape).astype("float32"),
            np.random.random(shape).astype("float32")
        ]
        self.outputs = None
        self.attrs = {
            "vocab_size": 100,
            "embed_dim": 32,
            "hidden_size": 32,
        }
        self.param_states = {}

    @staticmethod
    def model_init(self,
                   vocab_size,
                   embed_dim,
                   hidden_size,
                   bos_id=0,
                   eos_id=1,
                   beam_size=4,
                   max_step_num=20):
        embedder = Embedding(size=[vocab_size, embed_dim])
        output_layer = Linear(hidden_size, vocab_size)
        cell = BasicLSTMCell(embed_dim, hidden_size)
        decoder = BeamSearchDecoder(
            cell,
            start_token=bos_id,
            end_token=eos_id,
            beam_size=beam_size,
            embedding_fn=embedder,
            output_fn=output_layer)
        self.beam_search_decoder = DynamicDecode(
            decoder, max_step_num=max_step_num, is_test=True)

    @staticmethod
    def model_forward(model, init_hidden, init_cell):
        return model.beam_search_decoder([init_hidden, init_cell])[0]

    def make_inputs(self):
        inputs = [
            Input(
                [None, self.inputs[0].shape[-1]], "float32",
                name="init_hidden"),
            Input(
                [None, self.inputs[1].shape[-1]], "float32", name="init_cell"),
        ]
        return inputs

    def test_check_output(self):
        self.check_output()


class TestTransformerEncoder(ModuleApiTest):
    def setUp(self):
        self.inputs = [
            # encoder input: [batch_size, seq_len, hidden_size]
            np.random.random([2, 4, 512]).astype("float32"),
            # self attention bias: [batch_size, n_head, seq_len, seq_len]
            np.random.randint(0, 1, [2, 8, 4, 4]).astype("float32") * -1e9
        ]
        self.outputs = None
        self.attrs = {
            "n_layer": 2,
            "n_head": 8,
            "d_key": 64,
            "d_value": 64,
            "d_model": 512,
            "d_inner_hid": 1024
        }
        self.param_states = {}

    @staticmethod
    def model_init(model,
                   n_layer,
                   n_head,
                   d_key,
                   d_value,
                   d_model,
                   d_inner_hid,
                   prepostprocess_dropout=0.1,
                   attention_dropout=0.1,
                   relu_dropout=0.1,
                   preprocess_cmd="n",
                   postprocess_cmd="da",
                   ffn_fc1_act="relu"):
        model.encoder = TransformerEncoder(
            n_layer, n_head, d_key, d_value, d_model, d_inner_hid,
            prepostprocess_dropout, attention_dropout, relu_dropout,
            preprocess_cmd, postprocess_cmd, ffn_fc1_act)

    @staticmethod
    def model_forward(model, enc_input, attn_bias):
        return model.encoder(enc_input, attn_bias)

    def make_inputs(self):
        inputs = [
            Input(
                [None, None, self.inputs[0].shape[-1]],
                "float32",
                name="enc_input"),
            Input(
                [None, self.inputs[1].shape[1], None, None],
                "float32",
                name="attn_bias"),
        ]
        return inputs

    def test_check_output(self):
        self.check_output()


class TestTransformerDecoder(TestTransformerEncoder):
    def setUp(self):
        self.inputs = [
            # decoder input: [batch_size, seq_len, hidden_size]
            np.random.random([2, 4, 512]).astype("float32"),
            # encoder output: [batch_size, seq_len, hidden_size]
            np.random.random([2, 5, 512]).astype("float32"),
            # self attention bias: [batch_size, n_head, seq_len, seq_len]
            np.random.randint(0, 1, [2, 8, 4, 4]).astype("float32") * -1e9,
            # cross attention bias: [batch_size, n_head, seq_len, seq_len]
            np.random.randint(0, 1, [2, 8, 4, 5]).astype("float32") * -1e9
        ]
        self.outputs = None
        self.attrs = {
            "n_layer": 2,
            "n_head": 8,
            "d_key": 64,
            "d_value": 64,
            "d_model": 512,
            "d_inner_hid": 1024
        }
        self.param_states = {}

    @staticmethod
    def model_init(model,
                   n_layer,
                   n_head,
                   d_key,
                   d_value,
                   d_model,
                   d_inner_hid,
                   prepostprocess_dropout=0.1,
                   attention_dropout=0.1,
                   relu_dropout=0.1,
                   preprocess_cmd="n",
                   postprocess_cmd="da"):
        model.decoder = TransformerDecoder(
            n_layer, n_head, d_key, d_value, d_model, d_inner_hid,
            prepostprocess_dropout, attention_dropout, relu_dropout,
            preprocess_cmd, postprocess_cmd)

    @staticmethod
    def model_forward(model,
                      dec_input,
                      enc_output,
                      self_attn_bias,
                      cross_attn_bias,
                      caches=None):
        return model.decoder(dec_input, enc_output, self_attn_bias,
                             cross_attn_bias, caches)

    def make_inputs(self):
        inputs = [
            Input(
                [None, None, self.inputs[0].shape[-1]],
                "float32",
                name="dec_input"),
            Input(
                [None, None, self.inputs[0].shape[-1]],
                "float32",
                name="enc_output"),
            Input(
                [None, self.inputs[-1].shape[1], None, None],
                "float32",
                name="self_attn_bias"),
            Input(
                [None, self.inputs[-1].shape[1], None, None],
                "float32",
                name="cross_attn_bias"),
        ]
        return inputs

    def test_check_output(self):
        self.check_output()


class TestTransformerBeamSearchDecoder(ModuleApiTest):
    def setUp(self):
        self.inputs = [
            # encoder output: [batch_size, seq_len, hidden_size]
            np.random.random([2, 5, 128]).astype("float32"),
            # cross attention bias: [batch_size, n_head, seq_len, seq_len]
            np.random.randint(0, 1, [2, 2, 1, 5]).astype("float32") * -1e9
        ]
        self.outputs = None
        self.attrs = {
            "vocab_size": 100,
            "n_layer": 2,
            "n_head": 2,
            "d_key": 64,
            "d_value": 64,
            "d_model": 128,
            "d_inner_hid": 128
        }
        self.param_states = {}

    @staticmethod
    def model_init(model,
                   vocab_size,
                   n_layer,
                   n_head,
                   d_key,
                   d_value,
                   d_model,
                   d_inner_hid,
                   prepostprocess_dropout=0.1,
                   attention_dropout=0.1,
                   relu_dropout=0.1,
                   preprocess_cmd="n",
                   postprocess_cmd="da",
                   bos_id=0,
                   eos_id=1,
                   beam_size=4,
                   max_step_num=20):
        model.beam_size = beam_size

        def embeder_init(self, size):
            Layer.__init__(self)
            self.embedder = Embedding(size)

        Embedder = type("Embedder", (Layer, ), {
            "__init__": embeder_init,
            "forward": lambda self, word, pos: self.embedder(word)
        })
        embedder = Embedder(size=[vocab_size, d_model])
        output_layer = Linear(d_model, vocab_size)
        model.decoder = TransformerDecoder(
            n_layer, n_head, d_key, d_value, d_model, d_inner_hid,
            prepostprocess_dropout, attention_dropout, relu_dropout,
            preprocess_cmd, postprocess_cmd)
        transformer_cell = TransformerCell(model.decoder, embedder,
                                           output_layer)
        model.beam_search_decoder = DynamicDecode(
            TransformerBeamSearchDecoder(
                transformer_cell, bos_id, eos_id, beam_size,
                var_dim_in_state=2),
            max_step_num,
            is_test=True)

    @staticmethod
    def model_forward(model, enc_output, trg_src_attn_bias):
        caches = model.decoder.prepare_incremental_cache(enc_output)
        enc_output = TransformerBeamSearchDecoder.tile_beam_merge_with_batch(
            enc_output, model.beam_size)
        trg_src_attn_bias = TransformerBeamSearchDecoder.tile_beam_merge_with_batch(
            trg_src_attn_bias, model.beam_size)
        static_caches = model.decoder.prepare_static_cache(enc_output)
        rs, _ = model.beam_search_decoder(
            inits=caches,
            enc_output=enc_output,
            trg_src_attn_bias=trg_src_attn_bias,
            static_caches=static_caches)
        return rs

    def make_inputs(self):
        inputs = [
            Input(
                [None, None, self.inputs[0].shape[-1]],
                "float32",
                name="enc_output"),
            Input(
                [None, self.inputs[1].shape[1], None, None],
                "float32",
                name="trg_src_attn_bias"),
        ]
        return inputs

    def test_check_output(self):
        self.check_output()


class TestSequenceTagging(ModuleApiTest):
    def setUp(self):
        self.inputs = [
            np.random.randint(0, 100, (2, 8)).astype("int64"),
            np.random.randint(1, 8, (2)).astype("int64"),
            np.random.randint(0, 5, (2, 8)).astype("int64")
        ]
        self.outputs = None
        self.attrs = {"vocab_size": 100, "num_labels": 5}
        self.param_states = {}

    @staticmethod
    def model_init(model,
                   vocab_size,
                   num_labels,
                   word_emb_dim=128,
                   grnn_hidden_dim=128,
                   emb_learning_rate=0.1,
                   crf_learning_rate=0.1,
                   bigru_num=2,
                   init_bound=0.1):
        model.tagger = SequenceTagging(vocab_size, num_labels, word_emb_dim,
                                       grnn_hidden_dim, emb_learning_rate,
                                       crf_learning_rate, bigru_num, init_bound)

    @staticmethod
    def model_forward(model, word, lengths, target=None):
        return model.tagger(word, lengths, target)

    def make_inputs(self):
        inputs = [
            Input(
                [None, None], "int64", name="word"),
            Input(
                [None], "int64", name="lengths"),
            Input(
                [None, None], "int64", name="target"),
        ]
        return inputs

    def test_check_output(self):
        self.check_output()


class TestSequenceTaggingInfer(TestSequenceTagging):
    def setUp(self):
        super(TestSequenceTaggingInfer, self).setUp()
        self.inputs = self.inputs[:2]  # remove target

    def make_inputs(self):
        inputs = super(TestSequenceTaggingInfer,
                       self).make_inputs()[:2]  # remove target
        return inputs


class TestStackedRNN(ModuleApiTest):
    def setUp(self):
        shape = (2, 4, 16)
        self.inputs = [np.random.random(shape).astype("float32")]
        self.outputs = None
        self.attrs = {"input_size": 16, "hidden_size": 16, "num_layers": 2}
        self.param_states = {}

    @staticmethod
    def model_init(model, input_size, hidden_size, num_layers):
        cells = [
            BasicLSTMCell(input_size, hidden_size),
            BasicLSTMCell(hidden_size, hidden_size)
        ]
        stacked_cell = StackedRNNCell(cells)
        model.lstm = RNN(stacked_cell)

    @staticmethod
    def model_forward(self, inputs):
        return self.lstm(inputs)[0]

    def make_inputs(self):
        inputs = [
            Input(
                [None, None, self.inputs[-1].shape[-1]],
                "float32",
                name="input"),
        ]
        return inputs

    def test_check_output(self):
        self.check_output()


class TestLSTM(ModuleApiTest):
    def setUp(self):
        shape = (2, 4, 16)
        self.inputs = [np.random.random(shape).astype("float32")]
        self.outputs = None
        self.attrs = {"input_size": 16, "hidden_size": 16, "num_layers": 2}
        self.param_states = {}

    @staticmethod
    def model_init(model, input_size, hidden_size, num_layers):
        model.lstm = LSTM(input_size, hidden_size, num_layers=num_layers)

    @staticmethod
    def model_forward(model, inputs):
        return model.lstm(inputs)[0]

    def make_inputs(self):
        inputs = [
            Input(
                [None, None, self.inputs[-1].shape[-1]],
                "float32",
                name="input"),
        ]
        return inputs

    def test_check_output(self):
        self.check_output()


class TestBiLSTM(ModuleApiTest):
    def setUp(self):
        shape = (2, 4, 16)
        self.inputs = [np.random.random(shape).astype("float32")]
        self.outputs = None
        self.attrs = {"input_size": 16, "hidden_size": 16, "num_layers": 2}
        self.param_states = {}

    @staticmethod
    def model_init(model,
                   input_size,
                   hidden_size,
                   num_layers,
                   merge_mode="concat",
                   merge_each_layer=False):
        model.bilstm = BidirectionalLSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            merge_mode=merge_mode,
            merge_each_layer=merge_each_layer)

    @staticmethod
    def model_forward(model, inputs):
        return model.bilstm(inputs)[0]

    def make_inputs(self):
        inputs = [
            Input(
                [None, None, self.inputs[-1].shape[-1]],
                "float32",
                name="input"),
        ]
        return inputs

    def test_check_output_merge0(self):
        self.check_output()

    def test_check_output_merge1(self):
        self.attrs["merge_each_layer"] = True
        self.check_output()


class TestGRU(ModuleApiTest):
    def setUp(self):
        shape = (2, 4, 64)
        self.inputs = [np.random.random(shape).astype("float32")]
        self.outputs = None
        self.attrs = {"input_size": 64, "hidden_size": 128, "num_layers": 2}
        self.param_states = {}

    @staticmethod
    def model_init(model, input_size, hidden_size, num_layers):
        model.gru = GRU(input_size, hidden_size, num_layers=num_layers)

    @staticmethod
    def model_forward(model, inputs):
        return model.gru(inputs)[0]

    def make_inputs(self):
        inputs = [
            Input(
                [None, None, self.inputs[-1].shape[-1]],
                "float32",
                name="input"),
        ]
        return inputs

    def test_check_output(self):
        self.check_output()


class TestBiGRU(ModuleApiTest):
    def setUp(self):
        shape = (2, 4, 64)
        self.inputs = [np.random.random(shape).astype("float32")]
        self.outputs = None
        self.attrs = {"input_size": 64, "hidden_size": 128, "num_layers": 2}
        self.param_states = {}

    @staticmethod
    def model_init(model,
                   input_size,
                   hidden_size,
                   num_layers,
                   merge_mode="concat",
                   merge_each_layer=False):
        model.bigru = BidirectionalGRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            merge_mode=merge_mode,
            merge_each_layer=merge_each_layer)

    @staticmethod
    def model_forward(model, inputs):
        return model.bigru(inputs)[0]

    def make_inputs(self):
        inputs = [
            Input(
                [None, None, self.inputs[-1].shape[-1]],
                "float32",
                name="input"),
        ]
        return inputs

    def test_check_output_merge0(self):
        self.check_output()

    def test_check_output_merge1(self):
        self.attrs["merge_each_layer"] = True
        self.check_output()


class TestCNNEncoder(ModuleApiTest):
    def setUp(self):
        shape = (2, 32, 8)  # [N, C, H]
        self.inputs = [np.random.random(shape).astype("float32")]
        self.outputs = None
        self.attrs = {"num_channels": 32, "num_filters": 64, "num_layers": 2}
        self.param_states = {}

    @staticmethod
    def model_init(model, num_channels, num_filters, num_layers):
        model.cnn_encoder = CNNEncoder(
            num_layers=2,
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=[2, 3],
            pool_size=[7, 6])

    @staticmethod
    def model_forward(model, inputs):
        return model.cnn_encoder(inputs)

    def make_inputs(self):
        inputs = [
            Input(
                [None, self.inputs[-1].shape[1], None], "float32",
                name="input"),
        ]
        return inputs

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
