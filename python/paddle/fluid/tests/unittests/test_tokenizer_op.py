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
from paddlenlp.ops import set_string_list, set_string_map


class TestTokenizerDemo(unittest.TestCase):
    def test_string_input(self):
        paddle.set_device('cpu')

        tokens = ['今天天气不错', '大暴雨']
        tokens_tensor = set_string_list(tokens, "demo_tokens")

        t = BertTokenizer.from_pretrained('bert-base-chinese')
        vocab = t.vocab.token_to_idx
        vocab_tensor = set_string_map(vocab, "demo_vocab")

        input_ids, seg_ids = core.ops.tokenizer(tokens_tensor, vocab_tensor)
        print(input_ids)
        print(seg_ids)


if __name__ == '__main__':
    unittest.main()
