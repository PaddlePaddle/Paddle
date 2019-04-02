# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import importlib

fluid.core._set_eager_deletion_mode(0.0, 1.0, True)

from test_elementwise_add_op import *
from test_elementwise_sub_op import *
from test_concat_op import *
from test_gather_op import *
from test_gaussian_random_batch_size_like_op import *
from test_lod_reset_op import *
from test_scatter_op import *
from test_mean_op import *
from test_slice_op import *
from test_linear_chain_crf_op import *
from test_bilinear_interp_op import *
from test_nearest_interp_op import *
from test_sequence_concat import *
from test_seq_conv import *
from test_seq_pool import *
from test_sequence_expand_as import *
from test_sequence_expand import *
from test_sequence_pad_op import *
from test_sequence_unpad_op import *
from test_sequence_scatter_op import *
from test_sequence_slice_op import *
from test_pad2d_op import *
from test_fill_zeros_like2_op import *

if __name__ == '__main__':
    unittest.main()
