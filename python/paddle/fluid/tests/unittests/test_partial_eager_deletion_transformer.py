# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest
import paddle.fluid as fluid

os.environ['RECORDIO_FILENAME'] = './p_gc_transformer.wmt16.recordio'

fluid.core._set_eager_deletion_mode(0.0, 0.55, True)

from test_parallel_executor_transformer import TestTransformer

if __name__ == '__main__':
    unittest.main()
