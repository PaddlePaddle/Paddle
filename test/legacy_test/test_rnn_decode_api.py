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

import collections
import random
import unittest

import numpy as np

import paddle
from paddle import Model, base, nn, set_device
from paddle.base import layers
from paddle.base.data_feeder import convert_dtype
from paddle.nn import (
    RNN,
    BeamSearchDecoder,
    Embedding,
    Layer,
    Linear,
    LSTMCell,
    SimpleRNNCell,
    dynamic_decode,
)
from paddle.static import InputSpec as Input

paddle.enable_static()


class PolicyGradient:
    """policy gradient"""

    def __init__(self, lr=None):
        self.lr = lr

    def learn(self, act_prob, action, reward, length=None):
        """
        update policy model self.model with policy gradient algorithm
        """
        self.reward = paddle.static.py_func(
            func=reward_func, x=[action, length], out=reward
        )
        neg_log_prob = paddle.nn.functional.cross_entropy(
            act_prob, action, reduction='none', use_softmax=False
        )
        cost = neg_log_prob * reward
        cost = (
            (paddle.sum(cost) / paddle.sum(length))
            if length is not None
            else paddle.mean(cost)
        )
        optimizer = paddle.optimizer.Adam(self.lr)
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
        loss = paddle.nn.functional.cross_entropy(
            input=probs,
            label=label,
            soft_label=False,
            reduction='none',
            use_softmax=False,
        )
        max_seq_len = paddle.shape(probs)[1]
        mask = paddle.static.nn.sequence_lod.sequence_mask(
            length, maxlen=max_seq_len, dtype="float32"
        )
        loss = loss * mask
        loss = paddle.mean(loss, axis=[0])
        loss = paddle.sum(loss)
        optimizer = paddle.optimizer.Adam(self.lr)
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
            base.Program() if main_program is None else main_program
        )
        self.startup_program = (
            base.Program() if startup_program is None else startup_program
        )
        if seed is not None:
            paddle.seed(seed)
        self.build_program(model_cls, alg_cls, model_hparams, alg_hparams)
        self.executor = executor

    def build_program(self, model_cls, alg_cls, model_hparams, alg_hparams):
        with base.program_guard(self.main_program, self.startup_program):
            source = paddle.static.data(
                name="src", shape=[None, None], dtype="int64"
            )
            source_length = paddle.static.data(
                name="src_sequence_length", shape=[None], dtype="int64"
            )
            # only for teacher-forcing MLE training
            target = paddle.static.data(
                name="trg", shape=[None, None], dtype="int64"
            )
            target_length = paddle.static.data(
                name="trg_sequence_length", shape=[None], dtype="int64"
            )
            label = paddle.static.data(
                name="label", shape=[None, None, 1], dtype="int64"
            )
            self.model = model_cls(**model_hparams)
            self.alg = alg_cls(**alg_hparams)
            self.probs, self.samples, self.sample_length = self.model(
                source, source_length, target, target_length
            )
            self.samples.stop_gradient = True
            self.reward = paddle.static.data(
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
            base.enable_dygraph(place)
        else:
            base.disable_dygraph()
        gen = paddle.seed(self._random_seed)
        if paddle.framework.use_pir_api():
            with paddle.pir_utils.OldIrGuard():
                paddle.framework.random._manual_program_seed(self._random_seed)
            paddle.framework.random._manual_program_seed(self._random_seed)
        else:
            paddle.framework.random._manual_program_seed(self._random_seed)
        scope = base.core.Scope()
        with base.scope_guard(scope):
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
        devices = ["CPU", "GPU"] if base.is_compiled_with_cuda() else ["CPU"]
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
        self.embedder = Embedding(vocab_size, embed_dim)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.cell = nn.LSTMCell(embed_dim, hidden_size)

        self.max_step_num = max_step_num
        self.beam_search_decoder = BeamSearchDecoder(
            self.cell,
            start_token=bos_id,
            end_token=eos_id,
            beam_size=beam_size,
            embedding_fn=self.embedder,
            output_fn=self.output_layer,
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

    def test_check_output(self):
        self.setUp()
        self.make_inputs()
        self.check_output()


class EncoderCell(SimpleRNNCell):
    def __init__(
        self,
        num_layers,
        input_size,
        hidden_size,
        dropout_prob=0.0,
        init_scale=0.1,
    ):
        super().__init__(input_size, hidden_size)
        self.dropout_prob = dropout_prob
        # use add_sublayer to add multi-layers
        self.lstm_cells = []
        for i in range(num_layers):
            self.lstm_cells.append(
                self.add_sublayer(
                    f"lstm_{i}",
                    LSTMCell(
                        input_size=input_size if i == 0 else hidden_size,
                        hidden_size=hidden_size,
                    ),
                )
            )

    def forward(self, step_input, states):
        new_states = []
        for i, lstm_cell in enumerate(self.lstm_cells):
            out, new_state = lstm_cell(step_input, states[i])
            step_input = (
                layers.dropout(
                    out,
                    self.dropout_prob,
                    dropout_implementation='upscale_in_train',
                )
                if self.dropout_prob > 0
                else out
            )
            new_states.append(new_state)
        return step_input, new_states

    @property
    def state_shape(self):
        return [cell.state_shape for cell in self.lstm_cells]


class Encoder(Layer):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_size,
        num_layers,
        dropout_prob=0.0,
        init_scale=0.1,
    ):
        super().__init__()
        self.embedder = Embedding(vocab_size, embed_dim)
        self.stack_lstm = RNN(
            EncoderCell(
                num_layers, embed_dim, hidden_size, dropout_prob, init_scale
            ),
            is_reverse=False,
            time_major=False,
        )

    def forward(self, sequence, sequence_length):
        inputs = self.embedder(sequence)
        encoder_output, encoder_state = self.stack_lstm(
            inputs, sequence_length=sequence_length
        )
        return encoder_output, encoder_state


DecoderCell = EncoderCell


class Decoder(Layer):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_size,
        num_layers,
        dropout_prob=0.0,
        init_scale=0.1,
    ):
        super().__init__()
        self.embedder = Embedding(vocab_size, embed_dim)
        self.stack_lstm = RNN(
            DecoderCell(
                num_layers, embed_dim, hidden_size, dropout_prob, init_scale
            ),
            is_reverse=False,
            time_major=False,
        )
        self.output_layer = Linear(hidden_size, vocab_size, bias_attr=False)

    def forward(self, target, decoder_initial_states):
        inputs = self.embedder(target)
        decoder_output, _ = self.stack_lstm(
            inputs, initial_states=decoder_initial_states
        )
        predict = self.output_layer(decoder_output)
        return predict


class TrainingHelper:
    def __init__(self, inputs, sequence_length, time_major=False):
        self.inputs = inputs
        self.sequence_length = sequence_length
        self.time_major = time_major
        self.inputs_ = paddle.utils.map_structure(
            lambda x: paddle.nn.functional.pad(
                x,
                pad=(
                    ([0, 1] + [0, 0] * (len(x.shape) - 1))
                    if time_major
                    else ([0, 0, 0, 1] + [0, 0] * (len(x.shape) - 2))
                ),
            ),
            self.inputs,
        )

    def initialize(self):
        init_finished = paddle.equal(
            self.sequence_length,
            paddle.full(
                shape=[1], dtype=self.sequence_length.dtype, fill_value=0
            ),
        )
        init_inputs = paddle.utils.map_structure(
            lambda x: x[0] if self.time_major else x[:, 0], self.inputs
        )
        return init_inputs, init_finished

    def sample(self, time, outputs, states):
        sample_ids = paddle.argmax(outputs, axis=-1)
        return sample_ids

    def next_inputs(self, time, outputs, states, sample_ids):
        time = (
            paddle.cast(time, "int32")
            if convert_dtype(time.dtype) not in ["int32"]
            else time
        )
        if self.sequence_length.dtype != time.dtype:
            self.sequence_length = paddle.cast(self.sequence_length, time.dtype)
        next_time = time + 1
        finished = paddle.less_equal(self.sequence_length, next_time)

        def _slice(x):
            axes = [0 if self.time_major else 1]
            return paddle.squeeze(
                paddle.slice(
                    x, axes=axes, starts=[next_time], ends=[next_time + 1]
                ),
                axis=axes,
            )

        next_inputs = paddle.utils.map_structure(_slice, self.inputs_)
        return finished, next_inputs, states


class BasicDecoder(paddle.nn.decode.Decoder):
    def __init__(self, cell, helper, output_fn=None):
        super().__init__()
        self.cell = cell
        self.helper = helper
        self.output_fn = output_fn

    def initialize(self, initial_cell_states):
        (initial_inputs, initial_finished) = self.helper.initialize()
        return initial_inputs, initial_cell_states, initial_finished

    class OutputWrapper(
        collections.namedtuple("OutputWrapper", ("cell_outputs", "sample_ids"))
    ):
        pass

    def step(self, time, inputs, states, **kwargs):
        cell_outputs, cell_states = self.cell(inputs, states, **kwargs)
        if self.output_fn is not None:
            cell_outputs = self.output_fn(cell_outputs)
        sample_ids = self.helper.sample(
            time=time, outputs=cell_outputs, states=cell_states
        )
        sample_ids.stop_gradient = True
        (finished, next_inputs, next_states) = self.helper.next_inputs(
            time=time,
            outputs=cell_outputs,
            states=cell_states,
            sample_ids=sample_ids,
        )
        outputs = self.OutputWrapper(cell_outputs, sample_ids)
        return (outputs, next_states, next_inputs, finished)


class BaseModel(Layer):
    def __init__(
        self,
        vocab_size=10,
        embed_dim=32,
        hidden_size=32,
        num_layers=1,
        dropout_prob=0.0,
        init_scale=0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.word_embedding = Embedding(vocab_size, embed_dim)
        self.encoder = Encoder(
            vocab_size,
            embed_dim,
            hidden_size,
            num_layers,
            dropout_prob,
            init_scale,
        )
        self.decoder = Decoder(
            vocab_size,
            embed_dim,
            hidden_size,
            num_layers,
            dropout_prob,
            init_scale,
        )

    def forward(self, src, src_length, trg, trg_length):
        encoder_output = self.encoder(src, src_length)
        trg_emb = self.decoder.embedder(trg)
        helper = TrainingHelper(inputs=trg_emb, sequence_length=trg_length)
        decoder = BasicDecoder(self.decoder.stack_lstm.cell, helper)
        (
            decoder_output,
            decoder_final_state,
            dec_seq_lengths,
        ) = dynamic_decode(
            decoder,
            inits=self.decoder.stack_lstm.cell.get_initial_states(
                encoder_output
            ),
            impute_finished=True,
            is_test=False,
            return_length=True,
        )
        logits, samples, sample_length = (
            decoder_output.cell_outputs,
            decoder_output.sample_ids,
            dec_seq_lengths,
        )
        return logits


class TestDynamicDecode(ModuleApiTest):
    def setUp(self):
        paddle.set_default_dtype("float64")
        shape = (1, 10)
        bs_shape = 1
        self.inputs = [
            np.random.randint(0, 10, size=shape).astype("int64"),
            np.random.randint(0, 10, size=bs_shape).astype("int64"),
            np.random.randint(0, 10, size=shape).astype("int64"),
            np.random.randint(0, 10, size=bs_shape).astype("int64"),
        ]
        self.outputs = None
        self.attrs = {
            "vocab_size": 10,
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
    ):
        self.model = BaseModel(
            vocab_size=vocab_size, embed_dim=embed_dim, hidden_size=hidden_size
        )

    @staticmethod
    def model_forward(model, src, src_length, trg, trg_length):
        return model.model(src, src_length, trg, trg_length)

    def make_inputs(self):
        inputs = [
            Input([None, None], "int64", "src"),
            Input([None], "int64", "src_length"),
            Input([None, None], "int64", "trg"),
            Input([None], "int64", "trg_length"),
        ]
        return inputs

    def test_check_output(self):
        self.setUp()
        self.make_inputs()
        self.check_output()


if __name__ == '__main__':
    unittest.main()
