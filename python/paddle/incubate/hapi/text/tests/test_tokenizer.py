#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.incubate.hapi.text.data_utils import PreTrainedTokenizer, BertBasicTokenizer, WordpieceTokenizer, BertTokenizer, convert_to_unicode, whitespace_tokenize
import unittest
import chardet
MODEL_NAME_OR_PATH = ""
VOCAB_PATH = ""


class TestPreProcess(unittest.TestCase):
    def test_convert_to_unicode(self):
        text1 = "中国".encode("utf-8")
        text1 = convert_to_unicode(text1)
        text2 = "abc"
        text2 = convert_to_unicode(text2)
 
    def test_whitespace_tokenize(self):
        text1 = " I love PaddlePaddle  "
        tokens1 = whitespace_tokenize(text1)
        for token in tokens1:
            assert " " not in token
        text2 = ""
        tokens2 = whitespace_tokenize(text2)
        assert tokens2 == []

class TestPreTrainedTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = PreTrainedTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
 
    def test_tokens_to_ids(self):
        tokens = ["a", "an", "the"]
        converted_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
 
    def test_tokens_to_str(self):
        tokens = ["a", "an", "the"]
        string = self.tokenizer.convert_tokens_to_string(tokens)
        assert string == "aanthe"

    def test_save_pretrained_and_resources(self):
        SAVE = MODEL_DIR + "_1"
        self.tokenizer.save_pretrained(MODEL_NAME_OR_PATH)
        # delete file

    def test_load_save_vocabulary(self):
        vocab = load_vocabulary(VOCAB_PATH)
        save_vocabulary(vocab, MODEL_NAME_OR_PATH)


class TestBertBasicTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = PreTrainedTokenizer.from_pretrained(MODEL_NAME_OR_PATH)

    def test_tokenize(self):
        text = ""
        output_tokens = self.tokenizer.tokenize(text)


class TestWordpieceTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = PreTrainedTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
 
    def test_tokenize(self):
        text = ""
        output_tokens = self.tokenizer.tokenize(text)
        pass


class TestBertTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BertTokenizer(vocab_file)
 
    def test_vocab_size(self):
        vocab_size = self.tokenizer.vocab_size()

    def test_convert_tokens_to_string(self):
        tokens = []
        output_string = self.tokenizer.convert_tokens_to_string()


if __name__ == '__main__':
    unittest.main()