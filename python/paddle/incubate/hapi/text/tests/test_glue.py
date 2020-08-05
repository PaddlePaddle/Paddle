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

from paddle.incubate.hapi.text.glue import TSVDataset, GlueCoLA, GlueSST2, 
GlueMRPC, GlueSTSB, GlueQQP, GlueMNLI, GlueQNLI, GlueRTE, GlueWNLI

import unittest

"""
TODO: Downloads dataset.
"""
TSV_DATASET_PATH = "test_data/tsv.tsv"
GLUE_COLA_PATH = ""
GLUE_SST2_PATH = ""
GLUE_MRPC_PATH = ""
GLUE_STSB_PATH = ""
GLUE_QQP_PATH = ""
GLUE_MNLI_PATH = ""
GLUE_QNLI_PATH = ""
GLUE_RTE_PATH = ""
GLUE_WNLI_PATH = ""

class TestTSVDataset(unittest.TestCase):
	def setUp(self):
		self.dataset = TSVDataset(TSV_DATASET_PATH, num_discard_samples=1, field_indices=[0, 2], num_discard_samples=0,
								field_indices=None, allow_missing=False)

	def test_data_format(self):
		assert self.dataset[0] == ['a', 'Zhang']
		assert self.dataset[1] == ['b', 'Jia']

class TestGlueCoLA(unittest.TestCase):
	def setUp(self):
		self.cola_train = GlueCoLA('train', root=GLUE_COLA_PATH)
		self.cola_dev = GlueCoLA('dev', root=GLUE_COLA_PATH)
		self.cola_test = GlueCoLA('test', root=GLUE_COLA_PATH)

	def test_data_format(self):
		assert len(self.cola_train) == 1000
		assert len(self.cola_dev[0]) == 2

	def test_get_labels(self):
		assert self.cola_train.get_labels() == ["0", "1"]
		assert self.cola_dev.get_labels() == ["0", "1"]
		assert self.cola_test.get_labels() == ["0", "1"]

class TestGlueSST2(unittest.TestCase):
	def setUp(self):
		self.sst_train = GlueSST2('train', root=GLUE_SST2_PATH)
		self.sst_dev = GlueSST2('dev', root=GLUE_SST2_PATH)
		self.sst_test = GlueSST2('test', root=GLUE_SST2_PATH)

	def test_data_format(self):


	def test_get_labels(self):
		assert self.sst_train.get_labels() == ["0", "1"]
		assert self.sst_dev.get_labels() == ["0", "1"]
		assert self.sst_test.get_labels() == ["0", "1"]

class TestGlueMRPC(unittest.TestCase):
	def setUp(self):
		self.mrpc_train = GlueMRPC('train', root=GLUE_MRPC_PATH)
		self.mrpc_dev = GlueMRPC('dev', root=GLUE_MRPC_PATH)
		self.mrpc_test = GlueMRPC('test', root=GLUE_MRPC_PATH)

	def test_data_format(self):

	def test_get_labels(self):
		assert self.mrpc_train.get_labels() == ["0", "1"]
		assert self.mrpc_dev.get_labels() == ["0", "1"]
		assert self.mrpc_test.get_labels() == ["0", "1"]

class TestGlueSTSB(unittest.TestCase):
	def setUp(self):
		self.stsb_train = GlueSTSB('train', root=GLUE_STSB_PATH)
		self.stsb_dev = GlueSTSB('dev', root=GLUE_STSB_PATH)
		self.stsb_test = GlueSTSB('test', root=GLUE_STSB_PATH)

	def test_data_format(self):

	def test_get_labels(self):
		assert self.stsb_train.get_labels() == None
		assert self.stsb_dev.get_labels() == None
		assert self.stsb_test.get_labels() == None

class TestGlueQQP(unittest.TestCase):
	def setUp(self):
		self.qqp_train = GlueQQP('train', root=GLUE_QQP_PATH)
		self.qqp_dev = GlueQQP('dev', root=GLUE_QQP_PATH)
		self.qqp_test = GlueQQP('test', root=GLUE_QQP_PATH)
	def test_data_format(self):

	def test_get_labels(self):
		assert self.qqp_train.get_labels() == ["0", "1"]
		assert self.qqp_dev.get_labels() == ["0", "1"]
		assert self.qqp_test.get_labels() == ["0", "1"]

class TestGlueMNLI(unittest.TestCase):
	def setUp(self):
		self.mnli_train = GlueMNLI('train', root=GLUE_MNLI_PATH)
		self.mnli_dev_matched = GlueMNLI('dev_matched', root=GLUE_MNLI_PATH)
		self.mnli_dev_mismatched = GlueMNLI('dev_mismatched', root=GLUE_MNLI_PATH)
		self.mnli_test_matched = GlueMNLI('test_matched', root=GLUE_MNLI_PATH)
		self.mnli_test_mismatched = GlueMNLI('test_mismatched', root=GLUE_MNLI_PATH)

	def test_data_format(self):

	def test_get_labels(self):
		assert self.mnli_train.get_labels() == ["contradiction", "entailment", "neutral"]
		assert self.mnli_dev_matched.get_labels() == ["contradiction", "entailment", "neutral"]
		assert self.mnli_dev_mismatched.get_labels() == ["contradiction", "entailment", "neutral"]
		assert self.mnli_test_matched.get_labels() == ["contradiction", "entailment", "neutral"]
		assert self.mnli_test_mismatched.get_labels() == ["contradiction", "entailment", "neutral"]


class TestGlueQNLI(unittest.TestCase):
	def setUp(self):
		self.qnli_train = gluonnlp.data.GlueQNLI('train', root=GLUE_QNLI_PATH)
		self.qnli_dev = gluonnlp.data.GlueQNLI('dev', root=GLUE_QNLI_PATH)
		self.qnli_test = gluonnlp.data.GlueQNLI('test', root=GLUE_QNLI_PATH)

	def test_data_format(self):

	def test_get_labels(self):
		assert self.qnli_train.get_labels() == ["entailment", "not_entailment"]
		assert self.qnli_dev.get_labels() == ["entailment", "not_entailment"]
		assert self.qnli_test.get_labels() == ["entailment", "not_entailment"]

class GlueRTE(unittest.TestCase):
	def setUp(self):
		self.rte_train = gluonnlp.data.GlueRTE('train', root=GLUE_RTE_PATH)
		self.rte_dev = gluonnlp.data.GlueRTE('dev', root=GLUE_RTE_PATH)
		self.rte_test = gluonnlp.data.GlueRTE('test', root=GLUE_RTE_PATH)

	def test_data_format(self):

	def test_get_labels(self):
		assert self.rte_train.get_labels() == ["entailment", "not_entailment"]
		assert self.rte_dev.get_labels() == ["entailment", "not_entailment"]
		assert self.rte_test.get_labels() == ["entailment", "not_entailment"]

class GlueWNLI(unittest.TestCase):
	def setUp(self):
		self.wnli_train = gluonnlp.data.GlueWNLI('train', root=GLUE_WNLI_PATH)
		self.wnli_dev = gluonnlp.data.GlueWNLI('dev', root=GLUE_WNLI_PATH)
		self.wnli_test = gluonnlp.data.GlueWNLI('test', root=GLUE_WNLI_PATH)

	def test_data_format(self):

	def test_get_labels(self):
		assert self.wnli_train.get_labels() == ["0", "1"]
		assert self.wnli_dev.get_labels() == ["0", "1"]
		assert self.wnli_test.get_labels() == ["0", "1"]

if __name__ == '__main__':
	unittest.main()