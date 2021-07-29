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


class TestTokenizerDemo(unittest.TestCase):
    def test_string_input(self):
        paddle.set_device('cpu')

        tokens = ['今天天气不错', '大暴雨']
        tokens_tensor = paddle.Tensor(core.VarDesc.VarType.STRINGS, [], "demo_tokens",
                          core.VarDesc.VarType.STRINGS, False)
        tokens_tensor.value().set_string_list(tokens)

        t = BertTokenizer.from_pretrained('bert-base-chinese')
        vocab = t.vocab.token_to_idx

        vocab_tensor = paddle.Tensor(core.VarDesc.VarType.MAP, [], "demo_vocab",
                          core.VarDesc.VarType.MAP, False)
        vocab_tensor.value().set_string_map(vocab)

        output = core.ops.tokenizer(text=tokens_tensor, vocab=vocab_tensor)
        print(output)


if __name__ == '__main__':
    unittest.main()
