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

from __future__ import print_function

import io
import unittest
import numpy as np
import paddle
from paddle.fluid.framework import core

import sys
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
    tensor = paddle.Tensor(core.VarDesc.VarType.STRINGS, [], name,
                           core.VarDesc.VarType.STRINGS, False)
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
    tensor = paddle.Tensor(core.VarDesc.VarType.WSTRING_MAP, [], name,
                           core.VarDesc.VarType.WSTRING_MAP, True)
    tensor.value().set_string_map(string_dict)
    return tensor


class TestBertTokenizerOp(unittest.TestCase):
    def setUp(self):
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.vocab_tensor = to_map_tensor(self.bert_tokenizer.vocab, "vocab")
        # self.set_attr()
        self.init_data()

    def set_attr(self):
        self.max_seq_len = 128
        self.pad_to_max_seq_len = True
        self.is_split_into_words = False

    def init_data(self):
        self.text = [
            '选择珠江花园的原因就是方便，有电动扶梯直接到达海边，周围餐馆、食廊、商场、超市、摊位一应俱全。'
            '酒店装修一般，但还算整洁。 泳池在大堂的屋顶，因此很小，不过女儿倒是喜欢。 包的早餐是西式的，还算丰富。 服务吗，一般'
        ]
        self.text_pair = ['非常不错，服务很好，位于市中心区，交通方便，不过价格也高！']
        self.text_tensor = to_string_tensor(self.text, "text")
        self.text_pair_tensor = to_string_tensor(self.text_pair, "text_pair")
        self.texts = [
            '很好的地理位置，一蹋糊涂的服务，萧条的酒店。',
            ' 选择珠江花园的原因就是方便，有电动扶梯直接到达海边，周围餐馆、食廊、商场、超市、摊位一应俱全。酒店装修一般，'
            '但还算整洁。 泳池在大堂的屋顶，因此很小，不过女儿倒是喜欢。 包的早餐是西式的，还算丰富。 服务吗，一般'
        ]
        self.text_pairs = [
            '非常不错，服务很好，位于市中心区，交通方便，不过价格也高！', '房间太小。其他的都一般。。。。。。。。。'
        ]
        self.texts_tensor = to_string_tensor(self.texts, "texts")
        self.text_pairs_tensor = to_string_tensor(self.text_pairs, "text_pairs")

        self.words = [" ".join(list(self.text[0]))]
        self.words_tensor = to_string_tensor(self.words, "words")

    def test_padding(self):
        paddle.disable_static()

        self.max_seq_len = 128
        self.pad_to_max_seq_len = True
        self.is_split_into_words = False

        # case 1: only one text (batch_size = 1)
        input_ids, token_type_ids = core.ops.bert_tokenizer(
            self.vocab_tensor, self.text_tensor, None, "max_seq_len",
            self.max_seq_len, "pad_to_max_seq_len", self.pad_to_max_seq_len,
            "is_split_into_words", self.is_split_into_words)
        input_ids = input_ids.numpy()
        token_type_ids = token_type_ids.numpy()

        encoded_inputs = self.bert_tokenizer(
            self.text,
            max_seq_len=self.max_seq_len,
            pad_to_max_seq_len=self.pad_to_max_seq_len,
            is_split_into_words=self.is_split_into_words)
        py_input_ids = np.array(encoded_inputs[0]["input_ids"]).reshape([1, -1])
        py_token_type_ids = np.array(encoded_inputs[0][
            "token_type_ids"]).reshape([1, -1])
        self.assertTrue(np.allclose(input_ids, py_input_ids, rtol=0, atol=0.01))
        self.assertTrue(
            np.allclose(
                token_type_ids, py_token_type_ids, rtol=0, atol=0.01))

        # case 2: only one text and one text_pair (batch_size = 1)
        input_ids, token_type_ids = core.ops.bert_tokenizer(
            self.vocab_tensor, self.text_tensor, self.text_pair_tensor,
            "max_seq_len", self.max_seq_len, "pad_to_max_seq_len",
            self.pad_to_max_seq_len, "is_split_into_words",
            self.is_split_into_words)
        input_ids = input_ids.numpy()
        token_type_ids = token_type_ids.numpy()

        encoded_inputs = self.bert_tokenizer(
            self.text,
            self.text_pair,
            max_seq_len=self.max_seq_len,
            pad_to_max_seq_len=self.pad_to_max_seq_len,
            is_split_into_words=self.is_split_into_words)
        py_input_ids = np.array(encoded_inputs[0]["input_ids"]).reshape([1, -1])
        py_token_type_ids = np.array(encoded_inputs[0][
            "token_type_ids"]).reshape([1, -1])
        self.assertTrue(np.allclose(input_ids, py_input_ids, rtol=0, atol=0.01))
        self.assertTrue(
            np.allclose(
                token_type_ids, py_token_type_ids, rtol=0, atol=0.01))

        # case 3: only texts (batch_size = 2)
        input_ids, token_type_ids = core.ops.bert_tokenizer(
            self.vocab_tensor, self.texts_tensor, None, "max_seq_len",
            self.max_seq_len, "pad_to_max_seq_len", self.pad_to_max_seq_len,
            "is_split_into_words", self.is_split_into_words)
        input_ids = input_ids.numpy()
        token_type_ids = token_type_ids.numpy()

        encoded_inputs = self.bert_tokenizer(
            self.texts,
            max_seq_len=self.max_seq_len,
            pad_to_max_seq_len=self.pad_to_max_seq_len,
            is_split_into_words=self.is_split_into_words)
        py_input_ids = [i["input_ids"] for i in encoded_inputs]
        py_token_type_ids = [i["token_type_ids"] for i in encoded_inputs]
        py_input_ids = np.array(py_input_ids).reshape([2, -1])
        py_token_type_ids = np.array(py_token_type_ids).reshape([2, -1])
        self.assertTrue(np.allclose(input_ids, py_input_ids, rtol=0, atol=0.01))
        self.assertTrue(
            np.allclose(
                token_type_ids, py_token_type_ids, rtol=0, atol=0.01))

        # case 4: texts and text pairs (batch_size = 2)
        input_ids, token_type_ids = core.ops.bert_tokenizer(
            self.vocab_tensor, self.texts_tensor, self.text_pairs_tensor,
            "max_seq_len", self.max_seq_len, "pad_to_max_seq_len",
            self.pad_to_max_seq_len, "is_split_into_words",
            self.is_split_into_words)
        input_ids = input_ids.numpy()
        token_type_ids = token_type_ids.numpy()

        encoded_inputs = self.bert_tokenizer(
            self.texts,
            self.text_pairs,
            max_seq_len=self.max_seq_len,
            pad_to_max_seq_len=self.pad_to_max_seq_len,
            is_split_into_words=self.is_split_into_words)
        py_input_ids = [i["input_ids"] for i in encoded_inputs]
        py_token_type_ids = [i["token_type_ids"] for i in encoded_inputs]
        py_input_ids = np.array(py_input_ids).reshape([2, -1])
        py_token_type_ids = np.array(py_token_type_ids).reshape([2, -1])
        self.assertTrue(np.allclose(input_ids, py_input_ids, rtol=0, atol=0.01))
        self.assertTrue(
            np.allclose(
                token_type_ids, py_token_type_ids, rtol=0, atol=0.01))

    def test_is_split_into_words(self):
        paddle.disable_static()

        self.max_seq_len = 128
        self.pad_to_max_seq_len = False
        self.is_split_into_words = True

        # case 1: only one text (batch_size = 1)
        input_ids, token_type_ids = core.ops.bert_tokenizer(
            self.vocab_tensor, self.text_tensor, None, "max_seq_len",
            self.max_seq_len, "pad_to_max_seq_len", self.pad_to_max_seq_len,
            "is_split_into_words", self.is_split_into_words)
        input_ids = input_ids.numpy()
        token_type_ids = token_type_ids.numpy()

        encoded_inputs = self.bert_tokenizer(
            self.text,
            max_seq_len=self.max_seq_len,
            pad_to_max_seq_len=self.pad_to_max_seq_len,
            is_split_into_words=self.is_split_into_words)
        py_input_ids = np.array(encoded_inputs[0]["input_ids"]).reshape([1, -1])
        py_token_type_ids = np.array(encoded_inputs[0][
            "token_type_ids"]).reshape([1, -1])
        self.assertTrue(np.allclose(input_ids, py_input_ids, rtol=0, atol=0.01))
        self.assertTrue(
            np.allclose(
                token_type_ids, py_token_type_ids, rtol=0, atol=0.01))

        # case 2: only one text and one text_pair (batch_size = 1)
        input_ids, token_type_ids = core.ops.bert_tokenizer(
            self.vocab_tensor, self.text_tensor, self.text_pair_tensor,
            "max_seq_len", self.max_seq_len, "pad_to_max_seq_len",
            self.pad_to_max_seq_len, "is_split_into_words",
            self.is_split_into_words)
        input_ids = input_ids.numpy()
        token_type_ids = token_type_ids.numpy()

        encoded_inputs = self.bert_tokenizer(
            self.text,
            self.text_pair,
            max_seq_len=self.max_seq_len,
            pad_to_max_seq_len=self.pad_to_max_seq_len,
            is_split_into_words=self.is_split_into_words)
        py_input_ids = np.array(encoded_inputs[0]["input_ids"]).reshape([1, -1])
        py_token_type_ids = np.array(encoded_inputs[0][
            "token_type_ids"]).reshape([1, -1])
        self.assertTrue(np.allclose(input_ids, py_input_ids, rtol=0, atol=0.01))
        self.assertTrue(
            np.allclose(
                token_type_ids, py_token_type_ids, rtol=0, atol=0.01))

    def test_no_padding(self):
        paddle.disable_static()

        self.max_seq_len = 128
        self.pad_to_max_seq_len = False
        self.is_split_into_words = False

        # case 1: only one text (batch_size = 1)
        input_ids, token_type_ids = core.ops.bert_tokenizer(
            self.vocab_tensor, self.text_tensor, None, "max_seq_len",
            self.max_seq_len, "pad_to_max_seq_len", self.pad_to_max_seq_len,
            "is_split_into_words", self.is_split_into_words)
        input_ids = input_ids.numpy()
        token_type_ids = token_type_ids.numpy()

        encoded_inputs = self.bert_tokenizer(
            self.text,
            max_seq_len=self.max_seq_len,
            pad_to_max_seq_len=self.pad_to_max_seq_len,
            is_split_into_words=self.is_split_into_words)
        py_input_ids = np.array(encoded_inputs[0]["input_ids"]).reshape([1, -1])
        py_token_type_ids = np.array(encoded_inputs[0][
            "token_type_ids"]).reshape([1, -1])
        self.assertTrue(np.allclose(input_ids, py_input_ids, rtol=0, atol=0.01))
        self.assertTrue(
            np.allclose(
                token_type_ids, py_token_type_ids, rtol=0, atol=0.01))

        # case 2: only one text and one text_pair (batch_size = 1)
        input_ids, token_type_ids = core.ops.bert_tokenizer(
            self.vocab_tensor, self.text_tensor, self.text_pair_tensor,
            "max_seq_len", self.max_seq_len, "pad_to_max_seq_len",
            self.pad_to_max_seq_len, "is_split_into_words",
            self.is_split_into_words)
        input_ids = input_ids.numpy()
        token_type_ids = token_type_ids.numpy()

        encoded_inputs = self.bert_tokenizer(
            self.text,
            self.text_pair,
            max_seq_len=self.max_seq_len,
            pad_to_max_seq_len=self.pad_to_max_seq_len,
            is_split_into_words=self.is_split_into_words)
        py_input_ids = np.array(encoded_inputs[0]["input_ids"]).reshape([1, -1])
        py_token_type_ids = np.array(encoded_inputs[0][
            "token_type_ids"]).reshape([1, -1])
        self.assertTrue(np.allclose(input_ids, py_input_ids, rtol=0, atol=0.01))
        self.assertTrue(
            np.allclose(
                token_type_ids, py_token_type_ids, rtol=0, atol=0.01))

    def test_is_split_into_words(self):
        paddle.disable_static()

        self.is_split_into_words = True

        input_ids, token_type_ids = core.ops.bert_tokenizer(
            self.vocab_tensor, self.text_tensor, None, "is_split_into_words",
            self.is_split_into_words)
        input_ids = input_ids.numpy()
        token_type_ids = token_type_ids.numpy()
        encoded_inputs = self.bert_tokenizer(
            list(self.text[0]), is_split_into_words=self.is_split_into_words)
        py_input_ids = np.array(encoded_inputs["input_ids"]).reshape([1, -1])
        py_token_type_ids = np.array(encoded_inputs["token_type_ids"]).reshape(
            [1, -1])
        self.assertTrue(np.allclose(input_ids, py_input_ids, rtol=0, atol=0.01))
        self.assertTrue(
            np.allclose(
                token_type_ids, py_token_type_ids, rtol=0, atol=0.01))


if __name__ == '__main__':
    unittest.main()
