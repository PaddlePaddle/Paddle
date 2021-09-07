#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import docstring_checker
import pylint.testutils
import astroid
import pytest
import sys


class TestDocstring(pylint.testutils.CheckerTestCase):
    CHECKER_CLASS = docstring_checker.DocstringChecker

    def test_one_line(self):
        func_node = astroid.extract_node('''
        def test(): 
            """get 
            news.
            """
            if True:
                return 5
            return 5
        ''')

        self.checker.visit_functiondef(func_node)
        got = self.linter.release_messages()
        assert len(got) == 1
        assert 'W9001' == got[0][0]

    def test_one_line_1(self):
        func_node = astroid.extract_node('''
        def test(): 
            """get news"""
            if True:
                return 5
            return 5
        ''')

        self.checker.visit_functiondef(func_node)
        got = self.linter.release_messages()
        assert len(got) == 1
        assert 'W9002' == got[0][0]

    def test_args(self):
        func_node = astroid.extract_node('''
        def test(scale, mean): 
            """get news.
            Args:
                scale (int): scale is the number.
            """
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            mean=scale
        ''')

        self.checker.visit_functiondef(func_node)
        got = self.linter.release_messages()
        assert len(got) == 1
        assert 'W9003' == got[0][0]

    def test_missing(self):
        func_node = astroid.extract_node('''
        def test(): 
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            mean=scale
        ''')

        self.checker.visit_functiondef(func_node)
        got = self.linter.release_messages()
        assert len(got) == 1
        assert 'W9005' == got[0][0]

    def test_indent(self):
        func_node = astroid.extract_node('''
        def test(): 
            """ get get get get get get get get
              get get get get get get get get.
            """
            pass 
        ''')

        self.checker.visit_functiondef(func_node)
        got = self.linter.release_messages()
        assert len(got) == 1
        assert 'W9006' == got[0][0]

    def test_with_resturns(self):
        func_node = astroid.extract_node('''
        def test(): 
            """get news.
            Args:
                scale (int): scale is the number.
            """
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            return mean
        ''')

        self.checker.visit_functiondef(func_node)
        got = self.linter.release_messages()
        assert len(got) == 1
        assert 'W9007' == got[0][0]

    def test_with_raises(self):
        func_node = astroid.extract_node('''
        def test(): 
            """get news.
            Args:
                scale (int): scale is the number.
            """
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            mean=scale
            raise ValueError('A very specific bad thing happened.')
        ''')

        self.checker.visit_functiondef(func_node)
        got = self.linter.release_messages()
        assert len(got) == 1
        assert 'W9008' == got[0][0]

    def test_no_message(self):
        p = '''
def fc(input,
       size,
       num_flatten_dims=1,
       param_attr=None,
       bias_attr=None,
       act=None,
       name=None):
    """
    **Fully Connected Layer**
    The fully connected layer can take multiple tensors as its inputs. It
    creates a variable called weights for each input tensor, which represents
    a fully connected weight matrix from each input unit to each output unit.
    The fully connected layer multiplies each input tensor with its coresponding
    weight to produce an output Tensor. If multiple input tensors are given,
    the results of multiple multiplications will be sumed up. If bias_attr is
    not None, a bias variable will be created and added to the output. Finally,
    if activation is not None, it will be applied to the output as well.
    This process can be formulated as follows:

    Args:
        input (Variable|list of Variable): The input tensor(s) of this layer, and the dimension of
            the input tensor(s) is at least 2.
        size(int): The number of output units in this layer.
        num_flatten_dims (int, default 1): The fc layer can accept an input tensor with more than
            two dimensions. If this happens, the multidimensional tensor will first be flattened
            into a 2-dimensional matrix. The parameter `num_flatten_dims` determines how the input
            tensor is flattened: the first `num_flatten_dims` (inclusive, index starts from 1)
            dimensions will be flatten to form the first dimension of the final matrix (height of
            the matrix), and the rest `rank(X) - num_flatten_dims` dimensions are flattened to
            form the second dimension of the final matrix (width of the matrix). For example, suppose
            `X` is a 6-dimensional tensor with a shape [2, 3, 4, 5, 6], and `num_flatten_dims` = 3.
            Then, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] = [24, 30].
        param_attr (ParamAttr|list of ParamAttr, default None): The parameter attribute for learnable
            parameters/weights of this layer.
        bias_attr (ParamAttr|list of ParamAttr, default None): The parameter attribute for the bias
            of this layer. If it is set to None, no bias will be added to the output units.
        act (str, default None): Activation to be applied to the output of this layer.
        name (str, default None): The name of this layer.
    Returns:
        A tensor variable storing the transformation result.
    Raises:
        ValueError: If rank of the input tensor is less than 2.
    Examples:
        .. code-block:: python
            data = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
            fc = fluid.layers.fc(input=data, size=1000, act="tanh")
    """
    raise ValueError('A very specific bad thing happened.')
    size = 1
    size = 1
    size = 1
    size = 1
    size = 1
    size = 1
    size = 1
    size = 1
    size = 1
    size = 1
    size = 1
    size = 1
    size = 1
    return size
    '''

        func_node = astroid.extract_node(p)
        self.checker.visit_functiondef(func_node)
        got = self.linter.release_messages()
        assert len(got) == 0
