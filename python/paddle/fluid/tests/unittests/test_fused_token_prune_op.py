# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
from op_test import OpTest
from paddle.framework import core
# from paddle import _C_ops


# def fused_token_prune(attn, mask, x, factor):
#     tensor_shape = x.shape  #x.shape
#     B = tensor_shape[0]  #.numpy()[0]
#     N = tensor_shape[1]
#     C = tensor_shape[2]
#     nb_head = mask.shape[1]

#     mask = mask >= 0
#     attn *= mask

#     attn = np.sum(attn, axis=1)
#     attn_by = np.sum(attn, axis=1)  #shape is (B, N)
#     # print("===========")
#     # print("attn_by: ", attn_by)
#     inds = np.argsort(attn_by[:, 1:], axis=-1) + 1
#     inds = inds[:, ::-1]

#     cls_ind = np.zeros(tensor_shape[0], dtype=np.int64)
#     cls_ind = np.expand_dims(cls_ind, axis=1)
#     cls_inds = np.concatenate([cls_ind, inds], axis=1)
#     # print("=================")
#     # print("after argsort: ", cls_inds)

#     max_slimmed_seq_len = int(N * factor)
#     cls_inds = cls_inds[:, :max_slimmed_seq_len]

#     cls_inds = np.expand_dims(cls_inds, axis=-1)
#     cls_inds = np.tile(cls_inds, (1, 1, C))

#     slimmed_x = np.take_along_axis(x, cls_inds, axis=1)
#     return slimmed_x


class TestFusedTokenPruneOp(OpTest):
    def setUp(self):
        self.op_type = 'fused_token_prune'
        self.dtype = np.float32
        attn = [[1, 2], [3, 4]]
        attn = np.array(attn, dtype='float32')
        attn = np.expand_dims(attn, axis=0)
        self.attn = np.expand_dims(attn, axis=0) # [1,1,2,2] bsz = 1, nd_head=1, max_seq_len=2
        mask = [[1, 1], [-1, -1]]
        mask = np.array(mask, dtype='float32')
        mask = np.expand_dims(mask, axis=0)
        self.mask = np.expand_dims(mask, axis=0) # same as attn
        x = [[1, 2, 3], [4, 5, 6]]
        x  = np.array(x, dtype='float32')
        self.x = np.expand_dims(x, axis=0) # [1, 2, 3] bsz = 1, max_seq_len=2, c=3
        new_mask = [[1]]
        new_mask = np.array(new_mask, dtype='float32')
        new_mask = np.expand_dims(new_mask, axis=0) 
        self.new_mask = np.expand_dims(new_mask, axis=0)  #[1, 1, 1, 1]
        self.inputs = {'Attn': self.attn,
                        'Mask': self.mask,
                        'X': self.x,
                        'NewMask': self.new_mask}
        # self.factor = 0.5
        # self.attn = np.random.rand(16, 12, 128, 128)
        # self.mask = np.random.uniform(-1, 1, [16, 12, 128, 128])
        # self.x = np.random.rand(16, 128, 768)

        out_py = [[[1,2,3]]]
        out_py = np.array(out_py, dtype='float32')
        self.outputs = {'SlimmedX': out_py}


    # def test_fused_token_prune_op(self):
        
    #     attn = paddle.to_tensor(self.attn)
    #     mask = paddle.to_tensor(self.mask)
    #     x = paddle.to_tensor(self.x)
    #     new_mask = paddle.to_tensor(self.new_mask)
    #     out = _C_ops.fused_token_prune(attn, x, mask, new_mask)
    #     self.assertTrue(
    #         np.array_equal(out.numpy(), out_py),
    #         "paddle out: {}\n py out: {}\n".format(out.numpy(), out_py))

    def test_check_output(self):
        self.check_output_with_place(core.CUDAPlace(0))


if __name__ == "__main__":
    unittest.main()
