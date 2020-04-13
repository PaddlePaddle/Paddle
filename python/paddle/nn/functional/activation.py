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

# TODO: define activation functions of neural network
__all__ = [
    #             'brelu',
    #            'elu',
    #            'erf',
    #            'gelu',
    #            'hard_shrink',
    #            'hard_sigmoid',
    #            'hard_swish',
    'hsigmoid',
    #            'leaky_relu',
    #            'logsigmoid',
    #            'maxout',
    #            'prelu',
    #            'relu',
    #            'relu6',
    #            'selu',
    #            'sigmoid',
    #            'soft_relu',
    #            'softmax',
    #            'softplus',
    #            'softshrink',
    #            'softsign',
    #            'swish',
    #            'tanh_shrink',
    #            'thresholded_relu',
    #            'log_softmax'
]

from ...fluid.layer_helper import LayerHelper


def hsigmoid(input,
             label,
             weight,
             bias,
             num_classes,
             path_table=None,
             path_code=None,
             is_sparse=False):
    """
    The hierarchical sigmoid organizes the classes into a complete binary tree to reduce the computational complexity
    and speed up the model training, especially the training of language model.
    Each leaf node of the complete binary tree represents a class(word) and each non-leaf node acts as a binary classifier.
    For each class(word), there's a unique path from root to itself, hsigmoid calculate the cost for each non-leaf node on
    the path, and sum them to get a total cost.
    Comparing to softmax, the OP can reduce the computational complexity from :math:`O(N)` to :math:`O(logN)`, where :math:`N`
    represents the number of classes or the size of word dict.

    The OP supports default tree and custom tree. For the default tree, you can refer to `Hierarchical Probabilistic Neural
    Network Language Model <http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf>`_. For the custom
    tree, you need to set :attr:`is_custom` to True, and do the following steps (take the language model as an example):

    1. Using a custom word dict to build a binary tree, each leaf node should be an word in the word dict.
    2. Creating a dict map word_id -> path that from the word to the root node, we call it path_table.
    3. Creating a dict map word_id -> code of path that from the word to the root node, we call it path_code.
       Code means the label of each binary classifier, 1 indicate true, 0 indicate false.
    4. Now, each word should has its path and code along the path, you can pass a batch of path and code related
       to the same batch of inputs.

    Parameters:
        input (Variable): A tensor with the shape [N, D], where N is the size of mini-batch,
            and D is the feature size. Its data type supports float32 and float64.
        label (Variable): A tensor contains the labels of training data. Its shape is [N, 1]
            and data type is int64.
        weight (Variable): A tensor with shape (num_classes - 1, D) if not using custom tree(path_code and path_table is None), or (num_classes, D) if using custom tree.
        bias (Variable): A tensor with shape (num_classes - 1, 1) if not using custom tree(path_code and path_table is None), or (num_classes, 1) if using custom tree.
        num_classes (int): The number of classes or the size of word dict, must be greater than 2.
            If the default tree is used (:attr:`is_custom` is set to False), :attr:`num_classes`
            should not be None. If the custom tree is used (:attr:`is_custom` is set to True),
            :attr:`num_classes` should be the number of non-leaf nodes, which indicates the num of
            classes using by the binary classifier.
        path_table (Variable, optional): A tensor that stores each batch of samples' path from leaf to root
            node, its shape is [N, L] and data type is int64, where L is the length of path. For each sample i,
            path_table[i] is a np.array like structure and each element in this array is the indexes in parent
            nodes' weight matrix. Default: None.
        path_code (Variable, optional): A tensor that stores each batch of samples' code of path from leaf
            to root node, its shape is [N, L] and data type is int64, which is the same as :attr:`path_table`.
            Each code of path is consisted with the code of nodes from leaf to root node. Default: None.
        is_sparse (bool, optional): Whether use sparse updating instead of dense updating, if it's True, the
            gradient of W and input will be sparse. Default: False.

    Returns:
        Variable: A tensor with the cost of hierarchical sigmoid, its shape is [N, 1] and data type is the same as :attr:`input`.

    Examples:

        .. code-block:: python

            from paddle import fluid, nn
            import paddle.fluid.dygraph as dg
            import paddle.nn.functional as F
            import numpy as np

            main = fluid.Program()
            start = fluid.Program()
            feature_size = 6
            num_classes = 8
            with fluid.unique_name.guard():
                with fluid.program_guard(main, start):
                    x = fluid.data("input", [-1, feature_size],
                                  dtype="float32")
                    label = fluid.data("labels", [-1, 1], dtype="int64")
                    w = fluid.data("weight", (num_classes -1, feature_size), dtype="float32")
                    b = fluid.data("bias", (num_classes -1, ), dtype="float32")
                    y = F.hsigmoid(x, label, w, b, num_classes)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(start)
            feed_dict = {
                "input": np.random.randn(4, feature_size).astype(np.float32),
                "labels": np.random.randint(0, num_classes, (4, 1)).astype(np.int64),
                "weight": np.random.randn(num_classes - 1, feature_size).astype(np.float32),
                "bias": np.random.randn(num_classes - 1, ).astype(np.float32),
            }
            y_np, = exe.run(main, feed=feed_dict, fetch_list=[y])
            print(y_np.shape)

          # (4, 1)
    """

    attrs = {
        "num_classes": num_classes,
        "is_sparse": is_sparse,
        "remote_prefetch": is_sparse
    }

    inputs = {
        "X": input,
        "W": weight,
        "Bias": bias,
        "PathTable": path_table,
        "PathCode": path_code,
        "Label": label
    }

    helper = LayerHelper('hierarchical_sigmoid', **locals())
    dtype = helper.input_dtype()

    out = helper.create_variable_for_type_inference(dtype)
    pre_out = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": out, "PreOut": pre_out, "W_Out": weight}

    helper.append_op(
        type="hierarchical_sigmoid",
        inputs=inputs,
        outputs=outputs,
        attrs=attrs)
    return out
