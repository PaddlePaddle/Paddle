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

import numpy as np

from paddle.fluid.framework import static_only

# TODO: define loss functions of neural network
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.layers.layer_function_generator import templatedoc
from paddle.fluid.param_attr import ParamAttr
from paddle.nn.initializer import Assign

from ...fluid.data_feeder import check_variable_and_dtype

__all__ = []


# FIXME(wuyi): let docstring_checker.py understand @autodoc.
# For now, the comments in c++ use types like Tensor, but in python side
# the type is often "Variable", and arguments may vary.
@static_only
@templatedoc(op_type="nce")
def nce(
    input,
    label,
    num_total_classes,
    sample_weight=None,
    param_attr=None,
    bias_attr=None,
    num_neg_samples=None,
    name=None,
    sampler="uniform",
    custom_dist=None,
    seed=0,
    is_sparse=False,
):
    """
    :api_attr: Static Graph

    ${comment}

    Args:
        input (Tensor): Input tensor, 2-D tensor with shape [batch_size, dim],
            and data type is float32 or float64.
        label (Tensor): Input label, 2-D tensor with shape [batch_size, num_true_class],
            and data type is int64.
        num_total_classes (int):${num_total_classes_comment}.
        sample_weight (Tensor|None): A Tensor of shape [batch_size, 1]
            storing a weight for each sample. The default weight for each
            sample is 1.0.
        param_attr (ParamAttr|None): To specify the weight parameter attribute.
            Default: None, which means the default weight parameter property is
            used. See usage for details in :ref:`api_fluid_ParamAttr` .
        bias_attr (ParamAttr|None): To specify the bias parameter attribute.
            Default: None, which means the default bias parameter property is
            used. See usage for details in :ref:`api_fluid_ParamAttr` .
        num_neg_samples (int): ${num_neg_samples_comment}.
        name(str|None): For detailed information, please refer to
            :ref:`api_guide_Name` . Usually name is no need to set and None by default.
        sampler (str, optional): The sampler used to sample class from negative classes.
                       It can be 'uniform', 'log_uniform' or 'custom_dist'.
                       default: 'uniform'.
        custom_dist (nd.array|None): A numpy ndarray with size=num_total_classes.
                       It is used when sampler is set to 'custom_dist'.
                       custom_dist[i] is the probability of i-th class to be sampled.
                       default: None.
        seed (int, optional): The seed used in sampler. Default 0, means no random seed.
        is_sparse(bool, optional): The flag indicating whether to use sparse update,
            the weight@GRAD and bias@GRAD will be changed to SelectedRows. Default False.

    Returns:
        Tensor: The output nce loss.

    Examples:
        .. code-block:: python


            import paddle
            import numpy as np

            paddle.enable_static()

            window_size = 5
            words = []
            for i in range(window_size):
                words.append(paddle.static.data(
                    name='word_{0}'.format(i), shape=[-1, 1], dtype='int64'))

            dict_size = 10000
            label_word = int(window_size / 2) + 1

            embs = []
            for i in range(window_size):
                if i == label_word:
                    continue

                emb = paddle.static.nn.embedding(input=words[i], size=[dict_size, 32],
                                    param_attr='embed', is_sparse=True)
                embs.append(emb)

            embs = paddle.concat(x=embs, axis=1)                # concat from 4 * [(-1, 1, 32)] to (-1, 4, 32)
            embs = paddle.reshape(x=embs, shape=(-1, 4 * 32))   # reshape to (batch_size = -1, dim = 4*32)
            loss = paddle.static.nn.nce(input=embs, label=words[label_word],
                        num_total_classes=dict_size, param_attr='nce.w_0',
                        bias_attr='nce.b_0')

            #or use custom distribution
            dist = np.array([0.05,0.5,0.1,0.3,0.05])
            loss = paddle.static.nn.nce(input=embs, label=words[label_word],
                    num_total_classes=5, param_attr='nce.w_1',
                    bias_attr='nce.b_1',
                    num_neg_samples=3,
                    sampler="custom_dist",
                    custom_dist=dist)
    """
    helper = LayerHelper('nce', **locals())
    check_variable_and_dtype(input, 'input', ['float32', 'float64'], 'nce')
    check_variable_and_dtype(label, 'label', ['int64'], 'nce')

    if input.ndim != 2:
        raise ValueError(
            f'The rank of `input` must be 2, but received {input.ndim}.'
        )

    dim = input.shape[1]
    num_true_class = label.shape[1]
    w = helper.create_parameter(
        attr=helper.param_attr,
        shape=[num_total_classes, dim],
        is_bias=False,
        dtype=input.dtype,
    )
    inputs = {}
    if helper.bias_attr:
        b = helper.create_parameter(
            attr=helper.bias_attr,
            shape=[num_total_classes, 1],
            is_bias=True,
            dtype=input.dtype,
        )
        inputs['Bias'] = b
    cost = helper.create_variable_for_type_inference(dtype=input.dtype)
    sample_logits = helper.create_variable_for_type_inference(dtype=input.dtype)
    sample_labels = helper.create_variable_for_type_inference(dtype=label.dtype)

    inputs['Input'] = input
    inputs['Label'] = label
    inputs['Weight'] = w
    inputs['SampleWeight'] = sample_weight if sample_weight is not None else []

    if sampler == "uniform":
        sampler = 0
    elif sampler == "log_uniform":
        sampler = 1
    elif sampler == "custom_dist":
        assert custom_dist is not None

        custom_dist_len = num_total_classes
        alias_probs_ = [0] * custom_dist_len
        alias_ = [0] * custom_dist_len
        bigs = []
        littles = []
        for i in range(custom_dist_len):
            normal_prob = custom_dist[i] * custom_dist_len
            if normal_prob - 1.0 > 0:
                bigs.append((i, normal_prob))
            elif 1.0 - normal_prob > 0:
                littles.append((i, normal_prob))
            else:
                alias_probs_[i] = normal_prob
                alias_[i] = -1

        while len(bigs) and len(littles):
            big = bigs.pop(0)
            little = littles.pop(0)

            big_idx = big[0]
            big_prob = big[1]

            alias_probs_[little[0]] = little[1]
            alias_[little[0]] = big_idx
            big_left = big[1] + little[1] - 1
            if big_left - 1.0 > 0:
                bigs.append((big_idx, big_left))
            elif 1.0 - big_left > 0:
                littles.append((big_idx, big_left))
            else:
                alias_probs_[big_idx] = big_left
                alias_[big_idx] = -1

        if len(bigs):
            big = bigs.pop(0)
            alias_probs_[big[0]] = 1.0
            alias_[big[0]] = -1
        if len(littles):
            little = littles.pop(0)
            alias_probs_[little[0]] = 1.0
            alias_[little[0]] = -1

        def _init_by_numpy_array(numpy_array):
            ret = helper.create_parameter(
                attr=ParamAttr(),
                shape=numpy_array.shape,
                dtype=numpy_array.dtype,
                default_initializer=Assign(numpy_array),
            )
            ret.stop_gradient = True
            return ret

        inputs['CustomDistProbs'] = _init_by_numpy_array(
            np.array(custom_dist).astype('float32')
        )
        inputs['CustomDistAlias'] = _init_by_numpy_array(
            np.array(alias_).astype('int32')
        )
        inputs['CustomDistAliasProbs'] = _init_by_numpy_array(
            np.array(alias_probs_).astype('float32')
        )
        sampler = 2
    else:
        raise Exception("Unsupported sampler type.")

    if num_neg_samples is None:
        num_neg_samples = 10
    else:
        num_neg_samples = int(num_neg_samples)

    remote_prefetch = is_sparse
    print(
        "With sparse mode, if your models has only small parameter prefetch may cause speed down"
    )

    attrs = {
        'num_total_classes': int(num_total_classes),
        'num_neg_samples': num_neg_samples,
        'seed': seed,
        'sampler': sampler,
        'is_sparse': is_sparse,
        'remote_prefetch': remote_prefetch,
    }

    helper.append_op(
        type='nce',
        inputs=inputs,
        outputs={
            'Cost': cost,
            'SampleLogits': sample_logits,
            'SampleLabels': sample_labels,
        },
        attrs=attrs,
    )
    return cost / (num_neg_samples + 1)
