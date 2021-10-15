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

from ..nn import Layer
from ..fluid.framework import core, in_dygraph_mode
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import check_variable_and_dtype, check_type

__all__ = ['crf_decode', 'ViterbiDecoder']


def crf_decode(potentials,
               transition_params,
               sequence_length,
               include_start_end_tag=True,
               name=None):
    """
    Decode the highest scoring sequence of tags.
    Args:
        potentials (`Tensor` | `Varaiable`):  
            The unary emission tensor. Its dtype is float32 or float64 and has a shape of `[batch_size, sequence_length, num_tags]`.
        transition_params (`Tensor`| `Varaiable`): 
            The transition matrix.  Its dtype is float32 or float64 and has a shape of `[num_tags, num_tags]`.
        sequence_length (`Tensor`| `Varaiable`):  
            The input length tensor storing real length of each sequence for correctness. Its dtype is int64 and has a shape of `[batch_size]`.
        include_start_end_tag (`bool`, optional): 
                    If set to True, the last row and the last column of transitions will be considered as start tag,
                    the the penultimate row and the penultimate column of transitions will be considered as stop tag.
                    Else, all the rows and columns will be considered as the real tag. Defaults to ``True``.
        name (str|None) – A name for this layer(optional). If set None, the layer will be named automatically.
    Returns:
        scores(`Tensor`| `Varaiable`): 
            The scores tensor containing the score for the Viterbi sequence. Its dtype is float32 and has a shape of `[batch_size]`.
        paths(`Tensor`| `Varaiable`): 
            The paths tensor containing the highest scoring tag indices. Its dtype is int64 and has a shape of `[batch_size, sequence_length`].
    """
    if in_dygraph_mode():
        return core.ops.viterbi_decode(potentials, transition_params,
                                       sequence_length, 'include_start_end_tag',
                                       include_start_end_tag)
    check_variable_and_dtype(potentials, 'input', ['float32', 'float64'],
                             'viterbi_decode')
    check_variable_and_dtype(transition_params, 'transitions',
                             ['float32', 'float64'], 'viterbi_decode')
    check_variable_and_dtype(sequence_length, 'lengths', ['int64'],
                             'viterbi_decode')
    check_type(include_start_end_tag, 'include_start_end_tag', (bool, ),
               'viterbi_decode')

    helper = LayerHelper('viterbi_decode', **locals())
    attrs = {'include_start_end_tag': include_start_end_tag}
    scores = helper.create_variable_for_type_inference(potentials.dtype)
    path = helper.create_variable_for_type_inference('int64')
    helper.append_op(
        type='viterbi_decode',
        inputs={
            'Input': potentials,
            'Transition': transition_params,
            'Length': sequence_length
        },
        outputs={'Scores': scores,
                 'Path': path},
        attrs=attrs)
    return scores, path


class ViterbiDecoder(Layer):
    """ 
    ViterbiDecoder can decode the highest scoring sequence of tags, it should only be used at test time.
    Args:
        transitions (`Tensor`): 
            The transition matrix.  Its dtype is float32 and has a shape of `[num_tags, num_tags]`.
        include_start_end_tag (`bool`, optional): 
            If set to True, the last row and the last column of transitions will be considered as start tag,
            the the penultimate row and the penultimate column of transitions will be considered as stop tag.
            Else, all the rows and columns will be considered as the real tag. Defaults to ``True``.
    """

    def __init__(self, transitions, include_start_end_tag=True, name=None):
        super(ViterbiDecoder, self).__init__()
        self.transitions = transitions
        self.include_start_end_tag = include_start_end_tag
        self.name = name

    def forward(self, inputs, lengths):
        """
        Decode the highest scoring sequence of tags.
        Args:
            inputs (`Tensor`):  
                The unary emission tensor. Its dtype is float32 and has a shape of `[batch_size, sequence_length, num_tags]`.
            length (`Tensor`):  
                The input length tensor storing real length of each sequence for correctness. Its dtype is int64 and has a shape of `[batch_size]`.
        Returns:
            scores(`Tensor`): 
                The scores tensor containing the score for the Viterbi sequence. Its dtype is float32 and has a shape of `[batch_size]`.
            paths(`Tensor`): 
                The paths tensor containing the highest scoring tag indices. Its dtype is int64 and has a shape of `[batch_size, sequence_length`].
        """
        return crf_decode(inputs, self.transitions, lengths,
                          self.include_start_end_tag, self.name)
