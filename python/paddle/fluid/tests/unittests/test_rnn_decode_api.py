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

import random
import unittest
import numpy as np

import paddle
import paddle.nn as nn
from paddle import Model, set_device
from paddle.static import InputSpec as Input
from paddle.fluid.dygraph import Layer
from paddle.nn import BeamSearchDecoder, dynamic_decode

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core

from paddle.fluid.executor import Executor
from paddle.fluid.framework import _test_eager_guard

paddle.enable_static()


class EncoderCell(layers.RNNCell):
    def __init__(self, num_layers, hidden_size, dropout_prob=0.0):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.lstm_cells = [
            layers.LSTMCell(hidden_size) for i in range(num_layers)
        ]

    def call(self, step_input, states):
        new_states = []
        for i in range(self.num_layers):
            out, new_state = self.lstm_cells[i](step_input, states[i])
            step_input = (
                layers.dropout(out, self.dropout_prob)
                if self.dropout_prob > 0
                else out
            )
            new_states.append(new_state)
        return step_input, new_states

    @property
    def state_shape(self):
        return [cell.state_shape for cell in self.lstm_cells]


class DecoderCell(layers.RNNCell):
    def __init__(self, num_layers, hidden_size, dropout_prob=0.0):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.lstm_cells = [
            layers.LSTMCell(hidden_size) for i in range(num_layers)
        ]

    def attention(self, hidden, encoder_output, encoder_padding_mask):
        query = layers.fc(
            hidden, size=encoder_output.shape[-1], bias_attr=False
        )
        attn_scores = layers.matmul(
            layers.unsqueeze(query, [1]), encoder_output, transpose_y=True
        )
        if encoder_padding_mask is not None:
            attn_scores = layers.elementwise_add(
                attn_scores, encoder_padding_mask
            )
        attn_scores = layers.softmax(attn_scores)
        attn_out = layers.squeeze(
            layers.matmul(attn_scores, encoder_output), [1]
        )
        attn_out = layers.concat([attn_out, hidden], 1)
        attn_out = layers.fc(attn_out, size=self.hidden_size, bias_attr=False)
        return attn_out

    def call(
        self, step_input, states, encoder_output, encoder_padding_mask=None
    ):
        lstm_states, input_feed = states
        new_lstm_states = []
        step_input = layers.concat([step_input, input_feed], 1)
        for i in range(self.num_layers):
            out, new_lstm_state = self.lstm_cells[i](step_input, lstm_states[i])
            step_input = (
                layers.dropout(out, self.dropout_prob)
                if self.dropout_prob > 0
                else out
            )
            new_lstm_states.append(new_lstm_state)
        out = self.attention(step_input, encoder_output, encoder_padding_mask)
        return out, [new_lstm_states, out]


class Encoder:
    def __init__(self, num_layers, hidden_size, dropout_prob=0.0):
        self.encoder_cell = EncoderCell(num_layers, hidden_size, dropout_prob)

    def __call__(self, src_emb, src_sequence_length):
        encoder_output, encoder_final_state = layers.rnn(
            cell=self.encoder_cell,
            inputs=src_emb,
            sequence_length=src_sequence_length,
            is_reverse=False,
        )
        return encoder_output, encoder_final_state


class Decoder:
    def __init__(
        self,
        num_layers,
        hidden_size,
        dropout_prob,
        decoding_strategy="infer_sample",
        max_decoding_length=20,
    ):
        self.decoder_cell = DecoderCell(num_layers, hidden_size, dropout_prob)
        self.decoding_strategy = decoding_strategy
        self.max_decoding_length = (
            None
            if (self.decoding_strategy == "train_greedy")
            else max_decoding_length
        )

    def __call__(
        self,
        decoder_initial_states,
        encoder_output,
        encoder_padding_mask,
        **kwargs
    ):
        output_layer = kwargs.pop("output_layer", None)
        if self.decoding_strategy == "train_greedy":
            # for teach-forcing MLE pre-training
            helper = layers.TrainingHelper(**kwargs)
        elif self.decoding_strategy == "infer_sample":
            helper = layers.SampleEmbeddingHelper(**kwargs)
        elif self.decoding_strategy == "infer_greedy":
            helper = layers.GreedyEmbeddingHelper(**kwargs)

        if self.decoding_strategy == "beam_search":
            beam_size = kwargs.get("beam_size", 4)
            encoder_output = (
                layers.BeamSearchDecoder.tile_beam_merge_with_batch(
                    encoder_output, beam_size
                )
            )
            encoder_padding_mask = (
                layers.BeamSearchDecoder.tile_beam_merge_with_batch(
                    encoder_padding_mask, beam_size
                )
            )
            decoder = layers.BeamSearchDecoder(
                cell=self.decoder_cell, output_fn=output_layer, **kwargs
            )
        else:
            decoder = layers.BasicDecoder(
                self.decoder_cell, helper, output_fn=output_layer
            )

        (
            decoder_output,
            decoder_final_state,
            dec_seq_lengths,
        ) = layers.dynamic_decode(
            decoder,
            inits=decoder_initial_states,
            max_step_num=self.max_decoding_length,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask,
            impute_finished=False  # for test coverage
            if self.decoding_strategy == "beam_search"
            else True,
            is_test=True if self.decoding_strategy == "beam_search" else False,
            return_length=True,
        )
        return decoder_output, decoder_final_state, dec_seq_lengths


class Seq2SeqModel:
    """Seq2Seq model: RNN encoder-decoder with attention"""

    def __init__(
        self,
        num_layers,
        hidden_size,
        dropout_prob,
        src_vocab_size,
        trg_vocab_size,
        start_token,
        end_token,
        decoding_strategy="infer_sample",
        max_decoding_length=20,
        beam_size=4,
    ):
        self.start_token, self.end_token = start_token, end_token
        self.max_decoding_length, self.beam_size = (
            max_decoding_length,
            beam_size,
        )
        self.src_embeder = paddle.nn.Embedding(
            src_vocab_size,
            hidden_size,
            weight_attr=fluid.ParamAttr(name="source_embedding"),
        )
        self.trg_embeder = paddle.nn.Embedding(
            trg_vocab_size,
            hidden_size,
            weight_attr=fluid.ParamAttr(name="target_embedding"),
        )
        self.encoder = Encoder(num_layers, hidden_size, dropout_prob)
        self.decoder = Decoder(
            num_layers,
            hidden_size,
            dropout_prob,
            decoding_strategy,
            max_decoding_length,
        )
        self.output_layer = lambda x: layers.fc(
            x,
            size=trg_vocab_size,
            num_flatten_dims=len(x.shape) - 1,
            param_attr=fluid.ParamAttr(),
            bias_attr=False,
        )

    def __call__(self, src, src_length, trg=None, trg_length=None):
        # encoder
        encoder_output, encoder_final_state = self.encoder(
            self.src_embeder(src), src_length
        )

        decoder_initial_states = [
            encoder_final_state,
            self.decoder.decoder_cell.get_initial_states(
                batch_ref=encoder_output, shape=[encoder_output.shape[-1]]
            ),
        ]
        src_mask = layers.sequence_mask(
            src_length, maxlen=layers.shape(src)[1], dtype="float32"
        )
        encoder_padding_mask = (src_mask - 1.0) * 1e9
        encoder_padding_mask = layers.unsqueeze(encoder_padding_mask, [1])

        # decoder
        decoder_kwargs = (
            {
                "inputs": self.trg_embeder(trg),
                "sequence_length": trg_length,
            }
            if self.decoder.decoding_strategy == "train_greedy"
            else (
                {
                    "embedding_fn": self.trg_embeder,
                    "beam_size": self.beam_size,
                    "start_token": self.start_token,
                    "end_token": self.end_token,
                }
                if self.decoder.decoding_strategy == "beam_search"
                else {
                    "embedding_fn": self.trg_embeder,
                    "start_tokens": layers.fill_constant_batch_size_like(
                        input=encoder_output,
                        shape=[-1],
                        dtype=src.dtype,
                        value=self.start_token,
                    ),
                    "end_token": self.end_token,
                }
            )
        )
        decoder_kwargs["output_layer"] = self.output_layer

        (decoder_output, decoder_final_state, dec_seq_lengths) = self.decoder(
            decoder_initial_states,
            encoder_output,
            encoder_padding_mask,
            **decoder_kwargs
        )
        if self.decoder.decoding_strategy == "beam_search":  # for inference
            return decoder_output
        logits, samples, sample_length = (
            decoder_output.cell_outputs,
            decoder_output.sample_ids,
            dec_seq_lengths,
        )
        probs = layers.softmax(logits)
        return probs, samples, sample_length


class PolicyGradient:
    """policy gradient"""

    def __init__(self, lr=None):
        self.lr = lr

    def learn(self, act_prob, action, reward, length=None):
        """
        update policy model self.model with policy gradient algorithm
        """
        self.reward = fluid.layers.py_func(
            func=reward_func, x=[action, length], out=reward
        )
        neg_log_prob = layers.cross_entropy(act_prob, action)
        cost = neg_log_prob * reward
        cost = (
            (layers.reduce_sum(cost) / layers.reduce_sum(length))
            if length is not None
            else layers.reduce_mean(cost)
        )
        optimizer = fluid.optimizer.Adam(self.lr)
        optimizer.minimize(cost)
        return cost


def reward_func(samples, sample_length):
    """toy reward"""

    def discount_reward(reward, sequence_length, discount=1.0):
        return discount_reward_1d(reward, sequence_length, discount)

    def discount_reward_1d(reward, sequence_length, discount=1.0, dtype=None):
        if sequence_length is None:
            raise ValueError(
                'sequence_length must not be `None` for 1D reward.'
            )
        reward = np.array(reward)
        sequence_length = np.array(sequence_length)
        batch_size = reward.shape[0]
        max_seq_length = np.max(sequence_length)
        dtype = dtype or reward.dtype
        if discount == 1.0:
            dmat = np.ones([batch_size, max_seq_length], dtype=dtype)
        else:
            steps = np.tile(np.arange(max_seq_length), [batch_size, 1])
            mask = np.asarray(
                steps < (sequence_length - 1)[:, None], dtype=dtype
            )
            # Make each row = [discount, ..., discount, 1, ..., 1]
            dmat = mask * discount + (1 - mask)
            dmat = np.cumprod(dmat[:, ::-1], axis=1)[:, ::-1]
        disc_reward = dmat * reward[:, None]
        disc_reward = mask_sequences(disc_reward, sequence_length, dtype=dtype)
        return disc_reward

    def mask_sequences(sequence, sequence_length, dtype=None, time_major=False):
        sequence = np.array(sequence)
        sequence_length = np.array(sequence_length)
        rank = sequence.ndim
        if rank < 2:
            raise ValueError("`sequence` must be 2D or higher order.")
        batch_size = sequence.shape[0]
        max_time = sequence.shape[1]
        dtype = dtype or sequence.dtype
        if time_major:
            sequence = np.transpose(sequence, axes=[1, 0, 2])
        steps = np.tile(np.arange(max_time), [batch_size, 1])
        mask = np.asarray(steps < sequence_length[:, None], dtype=dtype)
        for _ in range(2, rank):
            mask = np.expand_dims(mask, -1)
        sequence = sequence * mask
        if time_major:
            sequence = np.transpose(sequence, axes=[1, 0, 2])
        return sequence

    samples = np.array(samples)
    sample_length = np.array(sample_length)
    # length reward
    reward = (5 - np.abs(sample_length - 5)).astype("float32")
    # repeat punishment to trapped into local minima getting all same words
    # beam search to get more than one sample may also can avoid this
    for i in range(reward.shape[0]):
        reward[i] += (
            -10
            if sample_length[i] > 1
            and np.all(samples[i][: sample_length[i] - 1] == samples[i][0])
            else 0
        )
    return discount_reward(reward, sample_length, discount=1.0).astype(
        "float32"
    )


class MLE:
    """teacher-forcing MLE training"""

    def __init__(self, lr=None):
        self.lr = lr

    def learn(self, probs, label, weight=None, length=None):
        loss = layers.cross_entropy(input=probs, label=label, soft_label=False)
        max_seq_len = layers.shape(probs)[1]
        mask = layers.sequence_mask(length, maxlen=max_seq_len, dtype="float32")
        loss = loss * mask
        loss = layers.reduce_mean(loss, dim=[0])
        loss = layers.reduce_sum(loss)
        optimizer = fluid.optimizer.Adam(self.lr)
        optimizer.minimize(loss)
        return loss


class SeqPGAgent:
    def __init__(
        self,
        model_cls,
        alg_cls=PolicyGradient,
        model_hparams={},
        alg_hparams={},
        executor=None,
        main_program=None,
        startup_program=None,
        seed=None,
    ):
        self.main_program = (
            fluid.Program() if main_program is None else main_program
        )
        self.startup_program = (
            fluid.Program() if startup_program is None else startup_program
        )
        if seed is not None:
            self.main_program.random_seed = seed
            self.startup_program.random_seed = seed
        self.build_program(model_cls, alg_cls, model_hparams, alg_hparams)
        self.executor = executor

    def build_program(self, model_cls, alg_cls, model_hparams, alg_hparams):
        with fluid.program_guard(self.main_program, self.startup_program):
            source = fluid.data(name="src", shape=[None, None], dtype="int64")
            source_length = fluid.data(
                name="src_sequence_length", shape=[None], dtype="int64"
            )
            # only for teacher-forcing MLE training
            target = fluid.data(name="trg", shape=[None, None], dtype="int64")
            target_length = fluid.data(
                name="trg_sequence_length", shape=[None], dtype="int64"
            )
            label = fluid.data(
                name="label", shape=[None, None, 1], dtype="int64"
            )
            self.model = model_cls(**model_hparams)
            self.alg = alg_cls(**alg_hparams)
            self.probs, self.samples, self.sample_length = self.model(
                source, source_length, target, target_length
            )
            self.samples.stop_gradient = True
            self.reward = fluid.data(
                name="reward",
                shape=[None, None],  # batch_size, seq_len
                dtype=self.probs.dtype,
            )
            self.samples.stop_gradient = False
            self.cost = self.alg.learn(
                self.probs, self.samples, self.reward, self.sample_length
            )

        # to define the same parameters between different programs
        self.pred_program = self.main_program._prune_with_input(
            [source.name, source_length.name],
            [self.probs, self.samples, self.sample_length],
        )

    def predict(self, feed_dict):
        samples, sample_length = self.executor.run(
            self.pred_program,
            feed=feed_dict,
            fetch_list=[self.samples, self.sample_length],
        )
        return samples, sample_length

    def learn(self, feed_dict, fetch_list):
        results = self.executor.run(
            self.main_program, feed=feed_dict, fetch_list=fetch_list
        )
        return results


class TestDynamicDecode(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.model_hparams = {
            "num_layers": 2,
            "hidden_size": 32,
            "dropout_prob": 0.1,
            "src_vocab_size": 100,
            "trg_vocab_size": 100,
            "start_token": 0,
            "end_token": 1,
            "decoding_strategy": "infer_greedy",
            "max_decoding_length": 10,
        }

        self.iter_num = iter_num = 2
        self.batch_size = batch_size = 4
        src_seq_len = 10
        trg_seq_len = 12
        self.data = {
            "src": np.random.randint(
                2,
                self.model_hparams["src_vocab_size"],
                (iter_num * batch_size, src_seq_len),
            ).astype("int64"),
            "src_sequence_length": np.random.randint(
                1, src_seq_len, (iter_num * batch_size,)
            ).astype("int64"),
            "trg": np.random.randint(
                2,
                self.model_hparams["src_vocab_size"],
                (iter_num * batch_size, trg_seq_len),
            ).astype("int64"),
            "trg_sequence_length": np.random.randint(
                1, trg_seq_len, (iter_num * batch_size,)
            ).astype("int64"),
            "label": np.random.randint(
                2,
                self.model_hparams["src_vocab_size"],
                (iter_num * batch_size, trg_seq_len, 1),
            ).astype("int64"),
        }

        place = (
            core.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else core.CPUPlace()
        )
        self.exe = Executor(place)

    def test_mle_train(self):
        paddle.enable_static()
        self.model_hparams["decoding_strategy"] = "train_greedy"
        agent = SeqPGAgent(
            model_cls=Seq2SeqModel,
            alg_cls=MLE,
            model_hparams=self.model_hparams,
            alg_hparams={"lr": 0.001},
            executor=self.exe,
            main_program=fluid.Program(),
            startup_program=fluid.Program(),
            seed=123,
        )
        self.exe.run(agent.startup_program)
        for iter_idx in range(self.iter_num):
            reward, cost = agent.learn(
                {
                    "src": self.data["src"][
                        iter_idx
                        * self.batch_size : (iter_idx + 1)
                        * self.batch_size,
                        :,
                    ],
                    "src_sequence_length": self.data["src_sequence_length"][
                        iter_idx
                        * self.batch_size : (iter_idx + 1)
                        * self.batch_size
                    ],
                    "trg": self.data["trg"][
                        iter_idx
                        * self.batch_size : (iter_idx + 1)
                        * self.batch_size,
                        :,
                    ],
                    "trg_sequence_length": self.data["trg_sequence_length"][
                        iter_idx
                        * self.batch_size : (iter_idx + 1)
                        * self.batch_size
                    ],
                    "label": self.data["label"][
                        iter_idx
                        * self.batch_size : (iter_idx + 1)
                        * self.batch_size
                    ],
                },
                fetch_list=[agent.cost, agent.cost],
            )
            print(
                "iter_idx: %d, reward: %f, cost: %f"
                % (iter_idx, reward.mean(), cost)
            )

    def test_greedy_train(self):
        paddle.enable_static()
        self.model_hparams["decoding_strategy"] = "infer_greedy"
        agent = SeqPGAgent(
            model_cls=Seq2SeqModel,
            alg_cls=PolicyGradient,
            model_hparams=self.model_hparams,
            alg_hparams={"lr": 0.001},
            executor=self.exe,
            main_program=fluid.Program(),
            startup_program=fluid.Program(),
            seed=123,
        )
        self.exe.run(agent.startup_program)
        for iter_idx in range(self.iter_num):
            reward, cost = agent.learn(
                {
                    "src": self.data["src"][
                        iter_idx
                        * self.batch_size : (iter_idx + 1)
                        * self.batch_size,
                        :,
                    ],
                    "src_sequence_length": self.data["src_sequence_length"][
                        iter_idx
                        * self.batch_size : (iter_idx + 1)
                        * self.batch_size
                    ],
                },
                fetch_list=[agent.reward, agent.cost],
            )
            print(
                "iter_idx: %d, reward: %f, cost: %f"
                % (iter_idx, reward.mean(), cost)
            )

    def test_sample_train(self):
        paddle.enable_static()
        self.model_hparams["decoding_strategy"] = "infer_sample"
        agent = SeqPGAgent(
            model_cls=Seq2SeqModel,
            alg_cls=PolicyGradient,
            model_hparams=self.model_hparams,
            alg_hparams={"lr": 0.001},
            executor=self.exe,
            main_program=fluid.Program(),
            startup_program=fluid.Program(),
            seed=123,
        )
        self.exe.run(agent.startup_program)
        for iter_idx in range(self.iter_num):
            reward, cost = agent.learn(
                {
                    "src": self.data["src"][
                        iter_idx
                        * self.batch_size : (iter_idx + 1)
                        * self.batch_size,
                        :,
                    ],
                    "src_sequence_length": self.data["src_sequence_length"][
                        iter_idx
                        * self.batch_size : (iter_idx + 1)
                        * self.batch_size
                    ],
                },
                fetch_list=[agent.reward, agent.cost],
            )
            print(
                "iter_idx: %d, reward: %f, cost: %f"
                % (iter_idx, reward.mean(), cost)
            )

    def test_beam_search_infer(self):
        paddle.set_default_dtype("float32")
        paddle.enable_static()
        self.model_hparams["decoding_strategy"] = "beam_search"
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            source = fluid.data(name="src", shape=[None, None], dtype="int64")
            source_length = fluid.data(
                name="src_sequence_length", shape=[None], dtype="int64"
            )
            model = Seq2SeqModel(**self.model_hparams)
            output = model(source, source_length)

        self.exe.run(startup_program)
        for iter_idx in range(self.iter_num):
            trans_ids = self.exe.run(
                program=main_program,
                feed={
                    "src": self.data["src"][
                        iter_idx
                        * self.batch_size : (iter_idx + 1)
                        * self.batch_size,
                        :,
                    ],
                    "src_sequence_length": self.data["src_sequence_length"][
                        iter_idx
                        * self.batch_size : (iter_idx + 1)
                        * self.batch_size
                    ],
                },
                fetch_list=[output],
            )[0]

    def func_dynamic_basic_decoder(self):
        paddle.disable_static()
        src = paddle.to_tensor(np.random.randint(8, size=(8, 4)))
        src_length = paddle.to_tensor(np.random.randint(8, size=(8)))
        model = Seq2SeqModel(**self.model_hparams)
        probs, samples, sample_length = model(src, src_length)
        paddle.enable_static()

    def test_dynamic_basic_decoder(self):
        with _test_eager_guard():
            self.func_dynamic_basic_decoder()
        self.func_dynamic_basic_decoder()


class ModuleApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._np_rand_state = np.random.get_state()
        cls._py_rand_state = random.getstate()
        cls._random_seed = 123
        np.random.seed(cls._random_seed)
        random.seed(cls._random_seed)

        cls.model_cls = type(
            cls.__name__ + "Model",
            (Layer,),
            {
                "__init__": cls.model_init_wrapper(cls.model_init),
                "forward": cls.model_forward,
            },
        )

    @classmethod
    def tearDownClass(cls):
        np.random.set_state(cls._np_rand_state)
        random.setstate(cls._py_rand_state)

    @staticmethod
    def model_init_wrapper(func):
        def __impl__(self, *args, **kwargs):
            Layer.__init__(self)
            func(self, *args, **kwargs)

        return __impl__

    @staticmethod
    def model_init(model, *args, **kwargs):
        raise NotImplementedError(
            "model_init acts as `Model.__init__`, thus must implement it"
        )

    @staticmethod
    def model_forward(model, *args, **kwargs):
        return model.module(*args, **kwargs)

    def make_inputs(self):
        # TODO(guosheng): add default from `self.inputs`
        raise NotImplementedError(
            "model_inputs makes inputs for model, thus must implement it"
        )

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
        gen = paddle.seed(self._random_seed)
        paddle.framework.random._manual_program_seed(self._random_seed)
        scope = fluid.core.Scope()
        with fluid.scope_guard(scope):
            layer = (
                self.model_cls(**self.attrs)
                if isinstance(self.attrs, dict)
                else self.model_cls(*self.attrs)
            )
            model = Model(layer, inputs=self.make_inputs())
            model.prepare()
            if self.param_states:
                model.load(self.param_states, optim_state=None)
            return model.predict_batch(self.inputs)

    def check_output_with_place(self, place, mode="test"):
        dygraph_output = self._calc_output(place, mode, dygraph=True)
        stgraph_output = self._calc_output(place, mode, dygraph=False)
        expect_output = getattr(self, "outputs", None)
        for actual_t, expect_t in zip(dygraph_output, stgraph_output):
            np.testing.assert_allclose(actual_t, expect_t, rtol=1e-05, atol=0)
        if expect_output:
            for actual_t, expect_t in zip(dygraph_output, expect_output):
                np.testing.assert_allclose(
                    actual_t, expect_t, rtol=1e-05, atol=0
                )

    def check_output(self):
        devices = ["CPU", "GPU"] if fluid.is_compiled_with_cuda() else ["CPU"]
        for device in devices:
            place = set_device(device)
            self.check_output_with_place(place)


class TestBeamSearch(ModuleApiTest):
    def setUp(self):
        paddle.set_default_dtype("float64")
        shape = (8, 32)
        self.inputs = [
            np.random.random(shape).astype("float64"),
            np.random.random(shape).astype("float64"),
        ]
        self.outputs = None
        self.attrs = {
            "vocab_size": 100,
            "embed_dim": 32,
            "hidden_size": 32,
        }
        self.param_states = {}

    @staticmethod
    def model_init(
        self,
        vocab_size,
        embed_dim,
        hidden_size,
        bos_id=0,
        eos_id=1,
        beam_size=4,
        max_step_num=20,
    ):
        embedder = paddle.fluid.dygraph.Embedding(
            size=[vocab_size, embed_dim], dtype="float64"
        )
        output_layer = nn.Linear(hidden_size, vocab_size)
        cell = nn.LSTMCell(embed_dim, hidden_size)
        self.max_step_num = max_step_num
        self.beam_search_decoder = BeamSearchDecoder(
            cell,
            start_token=bos_id,
            end_token=eos_id,
            beam_size=beam_size,
            embedding_fn=embedder,
            output_fn=output_layer,
        )

    @staticmethod
    def model_forward(model, init_hidden, init_cell):
        return dynamic_decode(
            model.beam_search_decoder,
            [init_hidden, init_cell],
            max_step_num=model.max_step_num,
            impute_finished=True,
            is_test=True,
        )[0]

    def make_inputs(self):
        inputs = [
            Input([None, self.inputs[0].shape[-1]], "float64", "init_hidden"),
            Input([None, self.inputs[1].shape[-1]], "float64", "init_cell"),
        ]
        return inputs

    def func_check_output(self):
        self.setUp()
        self.make_inputs()
        self.check_output()

    def test_check_output(self):
        with _test_eager_guard():
            self.func_check_output()
        self.func_check_output()


if __name__ == '__main__':
    unittest.main()
