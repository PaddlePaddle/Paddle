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
from paddlenlp.ops import to_strings_tensor, to_map_tensor


class TestTokenizerDemo(unittest.TestCase):
    def test_string_input(self):
        paddle.set_device('cpu')

        text = ['今天天气不错', '大暴雨']
        text_tensor = to_strings_tensor(text, "text")
        pair = ["测试测试", "测试测试"]
        pair_tensor = to_strings_tensor(pair, "pair")

        t = BertTokenizer.from_pretrained('bert-base-chinese')
        vocab = t.vocab.token_to_idx
        vocab_tensor = to_map_tensor(vocab, "vocab")

        input_ids, seg_ids = core.ops.tokenizer(vocab_tensor, text_tensor, None,
                                                "max_seq_len", 10,
                                                "pad_to_max_seq_len", False)
        print(input_ids)
        print(seg_ids)


if __name__ == '__main__':
    unittest.main()
