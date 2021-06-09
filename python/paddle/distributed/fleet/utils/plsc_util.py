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
"""PLSC: PaddlePaddle Large Scale Classification utils"""

from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import check_variable_and_dtype

__all__ = ['class_center_sample']


def class_center_sample(label, num_class, ratio=0.1, ignore_label=-1):
    """
    Give a label list, first get unique positive class center in label list,
    then randomly sample negative class centers with a given sampling ratio,
    and return sampled class center and remap the input label list
    using sampled class center. If `len(unique_label_list) < num_class * ratio`, then we
    sample randomly `num_class * ratio - len(unique_label_list)` class center from
    `[0, num_class) \ unique_label_list`. If `len(unique_label_list) >= num_class * ratio`,
    then sampled_class_center = unique_label_list.

    Args:
    	label: label list, each label in [0, num_class)
    	num_class: total number of classes
    	ratio: sample ratio over num class
    	ignore_label: ignore label to sample

    Return:
    	remaped_label: remap label using sampled class center
    	sampled_class: sampled class center from [0, num_class)

    Examples:
    	.. code-block:: python

    	# example 1
    	label = [-1, 0, -1, -1]
    	remaped_label, sampled_class = class_center_sample(label, num_class=5, ratio=0.4)
    	remaped_label == [-1, 0, -1, 1]
    	sampled_class == [0, 4]

    	# example 2
    	label = [0, -1, 3, 2]
    	remaped_label, sampled_class = class_center_sample(label, num_class=5, ratio=0.4)
    	remaped_label == [0, -1, 2, 1]
    	sampled_class == [0, 2, 3]

    	# example 3
    	label = [0, -1, 3, 0]
    	remaped_label, sampled_class = class_center_sample(label, num_class=5, ratio=0.4)
    	remaped_label == [0, -1, 1, 0]
    	sampled_class == [0, 3]
    """

    check_variable_and_dtype(label, 'label', ['int64', 'int'],
                             'class_center_sample')
    op_type = 'class_center_sample'
    helper = LayerHelper(op_type, **locals())
    out = helper.create_variable_for_type_inference(dtype=label.dtype)
    sampled_class = helper.create_variable_for_type_inference(dtype=label.dtype)
    seed = helper.main_program.random_seed
    helper.append_op(
        type=op_type,
        inputs={'X': label},
        outputs={'Out': out,
                 'SampledClass': sampled_class},
        attrs={
            'num_class': num_class,
            'ratio': ratio,
            'ignore_label': ignore_label,
            'seed': seed
        })
    return out, sampled_class
