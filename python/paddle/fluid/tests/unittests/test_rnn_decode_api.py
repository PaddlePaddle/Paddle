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
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.nn as nn
from paddle import Model, set_device
from paddle.fluid.dygraph import Layer
from paddle.fluid.framework import _test_eager_guard
from paddle.nn import BeamSearchDecoder, dynamic_decode
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
        loss = paddle.nn.functional.cross_entropy(
            input=probs,
            label=label,
            soft_label=False,
            reduction='none',
            use_softmax=False,
        )
        max_seq_len = paddle.shape(probs)[1]
        mask = layers.sequence_mask(length, maxlen=max_seq_len, dtype="float32")
        loss = loss * mask
        loss = paddle.mean(loss, axis=[0])
        loss = paddle.sum(loss)
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
        embedder = paddle.nn.Embedding(vocab_size, embed_dim)
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
