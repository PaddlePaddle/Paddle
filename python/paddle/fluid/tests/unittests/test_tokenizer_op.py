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

import unittest

import paddle
import paddle.fluid.core as core

from paddlenlp.transformers import BertTokenizer

# from paddlenlp.ops import to_strings_tensor, to_map_tensor


def to_strings_tensor(string_values, name):
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
    Create the tensor that the value holds the map, the type of key is the string.
    NOTICE: The value will be holded in the cpu place. 
 
    Args:
        string_dict(dict): The value will be setted to the tensor.
        name(string): The name of the tensor.
    """
    tensor = paddle.Tensor(core.VarDesc.VarType.MAP, [], name,
                           core.VarDesc.VarType.MAP, True)
    tensor.value().set_string_map(string_dict)
    return tensor


class TestTokenizerDemo(unittest.TestCase):
    def test_string_input(self):
        paddle.set_device('gpu')

        text = ['今天天气不错', '大暴雨']
        text_tensor = to_strings_tensor(text, "text")
        pair = ["测试测试", "测试测试"]
        pair_tensor = to_strings_tensor(pair, "pair")

        t = BertTokenizer.from_pretrained('bert-base-chinese')
        vocab = t.vocab.token_to_idx
        vocab_tensor = to_map_tensor(vocab, "vocab")

        input_ids, seg_ids = core.ops.tokenizer(vocab_tensor, text_tensor,
                                                pair_tensor,
                                                "is_split_into_words", False)
        encoded_inputs = t(text, pair)
        print(input_ids)
        print(encoded_inputs[0]["input_ids"])
        print(encoded_inputs[1]["input_ids"])

        print(seg_ids)
        print(encoded_inputs[0]["token_type_ids"])
        print(encoded_inputs[1]["token_type_ids"])


if __name__ == '__main__':
    unittest.main()
