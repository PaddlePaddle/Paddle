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
from ..layer_helper import LayerHelper
from ..fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype

# TODO: define searching & indexing functions of a tensor
# __all__ = ['argmax',
#            'argmin',
#            'argsort',
#            'has_inf',
#            'has_nan',
#            'masked_select',
#            'topk',
#            'where',
#            'index_select',
#            'nonzero',
#            'sort']

__all__ = ['index_sample']


def index_sample(x, index):
    """
    used for tdm infer
    """
    helper = LayerHelper("index_sample", **locals())
    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             'paddle.tensor.search.index_sample')
    check_variable_and_dtype(index, 'index', ['int32', 'int64'],
                             'paddle.tensor.search.index_sample')
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type='index_sample',
        inputs={'X': x,
                'Index': index},
        outputs={'Out': out})
    return out
