#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from pass_test import PassTest

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core


class EmbEltwiseLayerNormFusePassTest(PassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            word_id = fluid.layers.data(
                name="word_id",
                shape=[1, 128, 1],
                dtype="int64",
                append_batch_size=False,
            )
            pos_id = fluid.layers.data(
                name="pos_id",
                shape=[1, 128, 1],
                dtype="int64",
                append_batch_size=False,
            )
            sent_id = fluid.layers.data(
                name="sent_id",
                shape=[1, 128, 1],
                dtype="int64",
                append_batch_size=False,
            )
            word_emb = fluid.layers.embedding(
                input=word_id, size=(128, 768), dtype='float32'
            )
            pos_emb = fluid.layers.embedding(
                input=pos_id, size=(128, 768), dtype='float32'
            )
            sent_emb = fluid.layers.embedding(
                input=sent_id, size=(128, 768), dtype='float32'
            )
            add1 = paddle.add(word_emb, pos_emb)
            add2 = paddle.add(add1, sent_emb)
            hidden1 = fluid.layers.layer_norm(input=add2, begin_norm_axis=2)

            id1 = fluid.layers.data(
                name="id1",
                shape=[1, 128, 1],
                dtype="int64",
                append_batch_size=False,
            )
            id2 = fluid.layers.data(
                name="id2",
                shape=[1, 128, 1],
                dtype="int64",
                append_batch_size=False,
            )
            id3 = fluid.layers.data(
                name="id3",
                shape=[1, 128, 1],
                dtype="int64",
                append_batch_size=False,
            )
            id4 = fluid.layers.data(
                name="id4",
                shape=[1, 128, 1],
                dtype="int64",
                append_batch_size=False,
            )
            emb1 = fluid.layers.embedding(
                input=id1, size=(128, 768), dtype='float32'
            )
            emb2 = fluid.layers.embedding(
                input=id2, size=(128, 768), dtype='float32'
            )
            emb3 = fluid.layers.embedding(
                input=id3, size=(128, 768), dtype='float32'
            )
            emb4 = fluid.layers.embedding(
                input=id4, size=(128, 768), dtype='float32'
            )
            add_1 = paddle.add(emb1, emb2)
            add_2 = paddle.add(add_1, emb3)
            add_3 = paddle.add(add_2, emb4)
            hidden_1 = fluid.layers.layer_norm(input=add_3, begin_norm_axis=2)

        self.feeds = {
            "word_id": np.random.randint(
                low=0, high=128, size=(1, 128, 1)
            ).astype("int64"),
            "pos_id": np.random.randint(
                low=0, high=128, size=(1, 128, 1)
            ).astype("int64"),
            "sent_id": np.random.randint(
                low=0, high=128, size=(1, 128, 1)
            ).astype("int64"),
            "id1": np.random.randint(low=0, high=128, size=(1, 128, 1)).astype(
                "int64"
            ),
            "id2": np.random.randint(low=0, high=128, size=(1, 128, 1)).astype(
                "int64"
            ),
            "id3": np.random.randint(low=0, high=128, size=(1, 128, 1)).astype(
                "int64"
            ),
            "id4": np.random.randint(low=0, high=128, size=(1, 128, 1)).astype(
                "int64"
            ),
        }
        self.fetch_list = [hidden1, hidden_1]
        self.pass_names = "embedding_eltwise_layernorm_fuse_pass"
        self.fused_op_type = "fused_embedding_eltwise_layernorm"
        self.num_fused_ops = 2

    def test_check_output(self):
        use_gpu_set = [True]
        if not core.is_compiled_with_cuda():
            return
        self.pass_attrs = {
            "embedding_eltwise_layernorm_fuse_pass": {"use_gpu": True}
        }
        place = fluid.CUDAPlace(0)
        self.check_output_with_place(place, startup_on_cpu=True)


if __name__ == "__main__":
    unittest.main()
