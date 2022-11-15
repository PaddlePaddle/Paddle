#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy as np
import paddle
import paddle.nn as nn
from paddle.fluid.framework import core, _non_static_mode, _test_eager_guard
from paddle.fluid.layer_helper import LayerHelper
from paddle import _legacy_C_ops

import sys
import tempfile

sys.path.append("./tokenizer")
from tokenizer.bert_tokenizer import BertTokenizer


def to_string_tensor(string_values, name):
    """
    Create the tensor that the value holds the list of string.
    NOTICE: The value will be holded in the cpu place.

    Args:
        string_values(list[string]): The value will be setted to the tensor.
        name(string): The name of the tensor.
    """
    tensor = paddle.Tensor(
        core.VarDesc.VarType.STRING,
        [],
        name,
        core.VarDesc.VarType.STRINGS,
        False,
    )
    tensor.value().set_string_list(string_values)
    return tensor


def to_map_tensor(string_dict, name):
    """
    Create the tensor that the value holds the map, the type of key is the string
    and the value is the int.
    NOTICE: The value will be holded in the cpu place.

    Args:
        string_dict(dict): The value will be setted to the tensor.
        name(string): The name of the tensor.
    """
    tensor = paddle.Tensor(
        core.VarDesc.VarType.RAW, [], name, core.VarDesc.VarType.VOCAB, True
    )
    tensor.value().set_vocab(string_dict)
    return tensor


class FasterTokenizer(nn.Layer):
    def __init__(self, vocab_dict):
        super().__init__()
        vocab_tensor = to_map_tensor(vocab_dict, "vocab")
        self.register_buffer("vocab", vocab_tensor, persistable=True)

    def forward(
        self,
        text,
        text_pair=None,
        do_lower_case=True,
        max_seq_len=-1,
        is_split_into_words=False,
        pad_to_max_seq_len=False,
    ):
        if _non_static_mode():
            input_ids, seg_ids = _legacy_C_ops.faster_tokenizer(
                self.vocab,
                text,
                text_pair,
                "do_lower_case",
                do_lower_case,
                "max_seq_len",
                max_seq_len,
                "pad_to_max_seq_len",
                pad_to_max_seq_len,
                "is_split_into_words",
                is_split_into_words,
            )
            return input_ids, seg_ids

        attrs = {
            "do_lower_case": do_lower_case,
            "max_seq_len": max_seq_len,
            "pad_to_max_seq_len": pad_to_max_seq_len,
            "is_split_into_words": is_split_into_words,
        }
        helper = LayerHelper("faster_tokenizer")
        input_ids = helper.create_variable_for_type_inference(dtype="int64")
        seg_ids = helper.create_variable_for_type_inference(dtype="int64")
        if text_pair is None:
            helper.append_op(
                type='faster_tokenizer',
                inputs={'Vocab': self.vocab, 'Text': text},
                outputs={'InputIds': input_ids, 'SegmentIds': seg_ids},
                attrs=attrs,
            )
        else:
            helper.append_op(
                type='faster_tokenizer',
                inputs={
                    'Vocab': self.vocab,
                    'Text': text,
                    'TextPair': text_pair,
                },
                outputs={'InputIds': input_ids, 'SegmentIds': seg_ids},
                attrs=attrs,
            )
        return input_ids, seg_ids


class Predictor:
    def __init__(self, model_dir):
        model_file = os.path.join(model_dir, "inference.pdmodel")
        params_file = os.path.join(model_dir, "inference.pdiparams")
        if not os.path.exists(model_file):
            raise ValueError("not find model file path {}".format(model_file))
        if not os.path.exists(params_file):
            raise ValueError("not find params file path {}".format(params_file))
        config = paddle.inference.Config(model_file, params_file)

        # fast_tokenizer op only support cpu.
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(10)

        config.switch_use_feed_fetch_ops(False)
        self.predictor = paddle.inference.create_predictor(config)
        self.input_handles = [
            self.predictor.get_input_handle(name)
            for name in self.predictor.get_input_names()
        ]
        self.output_handles = [
            self.predictor.get_output_handle(name)
            for name in self.predictor.get_output_names()
        ]

    def predict(self, data):

        self.input_handles[0].copy_from_cpu(data)
        self.predictor.run()
        input_ids = self.output_handles[0].copy_to_cpu()
        token_type_ids = self.output_handles[1].copy_to_cpu()
        return input_ids, token_type_ids


class TestBertTokenizerOp(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.save_path = os.path.join(self.temp_dir.name, "fast_tokenizer")
        self.param_path = os.path.join(self.save_path, "model.pdparams")
        self.inference_path = os.path.join(self.save_path, "inference")

    def tearDown(self):
        self.temp_dir.cleanup()

    def init_data(self):
        self.faster_tokenizer = FasterTokenizer(self.bert_tokenizer.vocab)
        self.text = [
            '选择珠江花园的原因就是方便，有电动扶梯直接到达海边，周围餐馆、食廊、商场、超市、摊位一应俱全。'
            '酒店装修一般，但还算整洁。 泳池在大堂的屋顶，因此很小，不过女儿倒是喜欢。 包的早餐是西式的，'
            '还算丰富。 服务吗，一般'
        ]
        self.text_pair = ['非常不错，服务很好，位于市中心区，交通方便，不过价格也高！']
        self.text_tensor = to_string_tensor(self.text, "text")
        self.text_pair_tensor = to_string_tensor(self.text_pair, "text_pair")
        self.texts = [
            '很好的地理位置，一蹋糊涂的服务，萧条的酒店。',
            ' 选择珠江花园的原因就是方便，有电动扶梯直接到达海边，周围餐馆、食廊、商场、超市、摊位一应俱全。酒店装修一般，'
            '但还算整洁。 泳池在大堂的屋顶，因此很小，不过女儿倒是喜欢。 包的早餐是西式的，还算丰富。 服务吗，一般',
            'Test bert tokenizer. The first text.',
        ]
        self.text_pairs = [
            '非常不错，服务很好，位于市中心区，交通方便，不过价格也高！',
            '房间太小。其他的都一般。。。。。。。。。',
            'Test bert tokenizer. The second text.',
        ]
        self.texts_tensor = to_string_tensor(self.texts, "texts")
        self.text_pairs_tensor = to_string_tensor(self.text_pairs, "text_pairs")

    def run_padding(self):
        self.init_data()
        self.max_seq_len = 128
        self.pad_to_max_seq_len = True
        self.is_split_into_words = False

        # case 1: only one text (batch_size = 1)
        input_ids, token_type_ids = self.faster_tokenizer(
            text=self.text_tensor,
            do_lower_case=self.bert_tokenizer.do_lower_case,
            max_seq_len=self.max_seq_len,
            pad_to_max_seq_len=self.pad_to_max_seq_len,
            is_split_into_words=self.is_split_into_words,
        )
        input_ids = input_ids.numpy()
        token_type_ids = token_type_ids.numpy()

        encoded_inputs = self.bert_tokenizer(
            text=self.text,
            max_seq_len=self.max_seq_len,
            pad_to_max_seq_len=self.pad_to_max_seq_len,
            is_split_into_words=self.is_split_into_words,
        )
        py_input_ids = np.array(encoded_inputs[0]["input_ids"]).reshape([1, -1])
        py_token_type_ids = np.array(
            encoded_inputs[0]["token_type_ids"]
        ).reshape([1, -1])
        np.testing.assert_allclose(input_ids, py_input_ids, rtol=0, atol=0.01)
        np.testing.assert_allclose(
            token_type_ids, py_token_type_ids, rtol=0, atol=0.01
        )

        # case 2: only one text and one text_pair (batch_size = 1)
        input_ids, token_type_ids = self.faster_tokenizer(
            text=self.text_tensor,
            text_pair=self.text_pair_tensor,
            do_lower_case=self.bert_tokenizer.do_lower_case,
            max_seq_len=self.max_seq_len,
            pad_to_max_seq_len=self.pad_to_max_seq_len,
            is_split_into_words=self.is_split_into_words,
        )
        input_ids = input_ids.numpy()
        token_type_ids = token_type_ids.numpy()

        encoded_inputs = self.bert_tokenizer(
            text=self.text,
            text_pair=self.text_pair,
            max_seq_len=self.max_seq_len,
            pad_to_max_seq_len=self.pad_to_max_seq_len,
            is_split_into_words=self.is_split_into_words,
        )
        py_input_ids = np.array(encoded_inputs[0]["input_ids"]).reshape([1, -1])
        py_token_type_ids = np.array(
            encoded_inputs[0]["token_type_ids"]
        ).reshape([1, -1])
        np.testing.assert_allclose(input_ids, py_input_ids, rtol=0, atol=0.01)
        np.testing.assert_allclose(
            token_type_ids, py_token_type_ids, rtol=0, atol=0.01
        )

        # case 3: only texts (batch_size = 3)
        input_ids, token_type_ids = self.faster_tokenizer(
            text=self.texts_tensor,
            do_lower_case=self.bert_tokenizer.do_lower_case,
            max_seq_len=self.max_seq_len,
            pad_to_max_seq_len=self.pad_to_max_seq_len,
            is_split_into_words=self.is_split_into_words,
        )
        input_ids = input_ids.numpy()
        token_type_ids = token_type_ids.numpy()

        encoded_inputs = self.bert_tokenizer(
            self.texts,
            max_seq_len=self.max_seq_len,
            pad_to_max_seq_len=self.pad_to_max_seq_len,
            is_split_into_words=self.is_split_into_words,
        )
        py_input_ids = [i["input_ids"] for i in encoded_inputs]
        py_token_type_ids = [i["token_type_ids"] for i in encoded_inputs]
        py_input_ids = np.array(py_input_ids).reshape([3, -1])
        py_token_type_ids = np.array(py_token_type_ids).reshape([3, -1])
        np.testing.assert_allclose(input_ids, py_input_ids, rtol=0, atol=0.01)
        np.testing.assert_allclose(
            token_type_ids, py_token_type_ids, rtol=0, atol=0.01
        )

        # case 4: texts and text pairs (batch_size = 3)
        input_ids, token_type_ids = self.faster_tokenizer(
            text=self.texts_tensor,
            text_pair=self.text_pairs_tensor,
            do_lower_case=self.bert_tokenizer.do_lower_case,
            max_seq_len=self.max_seq_len,
            pad_to_max_seq_len=self.pad_to_max_seq_len,
            is_split_into_words=self.is_split_into_words,
        )
        input_ids = input_ids.numpy()
        token_type_ids = token_type_ids.numpy()

        encoded_inputs = self.bert_tokenizer(
            self.texts,
            self.text_pairs,
            max_seq_len=self.max_seq_len,
            pad_to_max_seq_len=self.pad_to_max_seq_len,
            is_split_into_words=self.is_split_into_words,
        )
        py_input_ids = [i["input_ids"] for i in encoded_inputs]
        py_token_type_ids = [i["token_type_ids"] for i in encoded_inputs]
        py_input_ids = np.array(py_input_ids).reshape([3, -1])
        py_token_type_ids = np.array(py_token_type_ids).reshape([3, -1])
        np.testing.assert_allclose(input_ids, py_input_ids, rtol=0, atol=0.01)
        np.testing.assert_allclose(
            token_type_ids, py_token_type_ids, rtol=0, atol=0.01
        )

    def test_padding(self):
        with _test_eager_guard():
            self.run_padding()
        self.run_padding()

    def run_no_padding(self):
        self.init_data()
        self.max_seq_len = 128
        self.pad_to_max_seq_len = False
        self.is_split_into_words = False

        # case 1: only one text (batch_size = 1)
        input_ids, token_type_ids = self.faster_tokenizer(
            text=self.text_tensor,
            do_lower_case=self.bert_tokenizer.do_lower_case,
            max_seq_len=self.max_seq_len,
            pad_to_max_seq_len=self.pad_to_max_seq_len,
            is_split_into_words=self.is_split_into_words,
        )
        input_ids = input_ids.numpy()
        token_type_ids = token_type_ids.numpy()

        encoded_inputs = self.bert_tokenizer(
            self.text,
            max_seq_len=self.max_seq_len,
            pad_to_max_seq_len=self.pad_to_max_seq_len,
            is_split_into_words=self.is_split_into_words,
        )
        py_input_ids = np.array(encoded_inputs[0]["input_ids"]).reshape([1, -1])
        py_token_type_ids = np.array(
            encoded_inputs[0]["token_type_ids"]
        ).reshape([1, -1])
        np.testing.assert_allclose(input_ids, py_input_ids, rtol=0, atol=0.01)
        np.testing.assert_allclose(
            token_type_ids, py_token_type_ids, rtol=0, atol=0.01
        )

        # case 2: only one text and one text_pair (batch_size = 1)
        input_ids, token_type_ids = self.faster_tokenizer(
            self.text_tensor,
            self.text_pair_tensor,
            do_lower_case=self.bert_tokenizer.do_lower_case,
            max_seq_len=self.max_seq_len,
            pad_to_max_seq_len=self.pad_to_max_seq_len,
            is_split_into_words=self.is_split_into_words,
        )
        input_ids = input_ids.numpy()
        token_type_ids = token_type_ids.numpy()

        encoded_inputs = self.bert_tokenizer(
            self.text,
            self.text_pair,
            max_seq_len=self.max_seq_len,
            pad_to_max_seq_len=self.pad_to_max_seq_len,
            is_split_into_words=self.is_split_into_words,
        )
        py_input_ids = np.array(encoded_inputs[0]["input_ids"]).reshape([1, -1])
        py_token_type_ids = np.array(
            encoded_inputs[0]["token_type_ids"]
        ).reshape([1, -1])
        np.testing.assert_allclose(input_ids, py_input_ids, rtol=0, atol=0.01)
        np.testing.assert_allclose(
            token_type_ids, py_token_type_ids, rtol=0, atol=0.01
        )

    def test_no_padding(self):
        with _test_eager_guard():
            self.run_no_padding()
        self.run_no_padding()

    def run_is_split_into_words(self):
        self.init_data()
        self.is_split_into_words = True

        input_ids, token_type_ids = self.faster_tokenizer(
            self.text_tensor,
            do_lower_case=self.bert_tokenizer.do_lower_case,
            is_split_into_words=self.is_split_into_words,
        )
        input_ids = input_ids.numpy()
        token_type_ids = token_type_ids.numpy()
        encoded_inputs = self.bert_tokenizer(
            list(self.text[0]), is_split_into_words=self.is_split_into_words
        )
        py_input_ids = np.array(encoded_inputs["input_ids"]).reshape([1, -1])
        py_token_type_ids = np.array(encoded_inputs["token_type_ids"]).reshape(
            [1, -1]
        )
        np.testing.assert_allclose(input_ids, py_input_ids, rtol=0, atol=0.01)
        np.testing.assert_allclose(
            token_type_ids, py_token_type_ids, rtol=0, atol=0.01
        )

    def test_is_split_into_words(self):
        with _test_eager_guard():
            self.run_is_split_into_words()
        self.run_is_split_into_words()

    def test_inference(self):
        self.init_data()
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        paddle.save(self.faster_tokenizer.state_dict(), self.param_path)
        state_dict = paddle.load(self.param_path)
        self.faster_tokenizer.set_dict(state_dict)

        static_model = paddle.jit.to_static(
            self.faster_tokenizer,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None], dtype=core.VarDesc.VarType.STRINGS
                ),  # texts
            ],
        )
        # Save in static graph model.
        paddle.jit.save(static_model, self.inference_path)
        predictor = Predictor(self.save_path)
        input_ids, token_type_ids = predictor.predict(self.text)

        encoded_inputs = self.bert_tokenizer(self.text)
        py_input_ids = np.array(encoded_inputs[0]["input_ids"]).reshape([1, -1])
        py_token_type_ids = np.array(
            encoded_inputs[0]["token_type_ids"]
        ).reshape([1, -1])
        np.testing.assert_allclose(input_ids, py_input_ids, rtol=0, atol=0.01)
        np.testing.assert_allclose(
            token_type_ids, py_token_type_ids, rtol=0, atol=0.01
        )

    def test_feed_string_var(self):
        self.init_data()
        paddle.enable_static()
        x = paddle.static.data(
            name="x", shape=[-1], dtype=core.VarDesc.VarType.STRINGS
        )
        exe = paddle.static.Executor(paddle.framework.CPUPlace())
        exe.run(paddle.static.default_main_program(), feed={'x': self.text})
        paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
