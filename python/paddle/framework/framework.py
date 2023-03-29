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

from paddle.fluid import core
from paddle.fluid.framework import Variable, _all_is_type

# TODO: define framework api
from paddle.fluid.layer_helper_base import LayerHelperBase

__all__ = []


def set_default_dtype(d):
    """
    Set default dtype. The default dtype is initially float32.

    Args:
        d(string|np.dtype): the dtype to make the default. It only
                            supports float16, bfloat16, float32 and float64.

    Returns:
        None.

    Examples:
        .. code-block:: python

            import paddle
            paddle.set_default_dtype("float32")

    """
    if isinstance(d, type):
        # This branch is for NumPy scalar types
        if d in [np.float16, np.float32, np.float64]:
            d = d.__name__
        else:
            raise TypeError(
                "set_default_dtype only supports [float16, float32, float64] "
                ", but received %s" % d.__name__
            )
    else:
        # This branch is for np.dtype and str
        if d in ['float16', 'float32', 'float64', 'bfloat16']:
            # NOTE(SigureMo): Since the np.dtype object is not an instance of
            # type, so it will not be handled by the previous branch. We need
            # to convert it to str here.
            d = str(d)
        else:
            raise TypeError(
                "set_default_dtype only supports [float16, float32, float64, bfloat16] "
                ", but received %s" % str(d)
            )

    LayerHelperBase.set_default_dtype(d)


def get_default_dtype():
    """
    Get the current default dtype. The default dtype is initially float32.

    Args:
        None.
    Returns:
        String, this global dtype only supports float16, float32, float64.

    Examples:
        .. code-block:: python

            import paddle
            paddle.get_default_dtype()
    """
    return LayerHelperBase.get_default_dtype()


def wrap_as_scalar(number):
    """Wrap a number(either python scalar or numpy scalar) as core.Scalar if
    it is not a scalar.


    Args:
        number (Number): number

    Returns:
        Scalar: A Scalar that contains the value.
    """
    if isinstance(number, core.Scalar):
        return number
    if isinstance(number, (bool, int, float, complex)):
        return core.Scalar(number)
    if isinstance(number, np.number):
        # it is a numpy scalar
        return core.Scalar(number.item())
    else:
        raise TypeError("Cannot wrap {} as core.Scalar".format(number))


def wrap_as_scalars(array):
    """This function is used to convert flat list, or numpy array(not
    necesarily flat) to list of core.Scalar, which correspond to
    std::vector<paddle::experimental::Scalar> in operator runtime.

    Args:
        array (List | np.ndarray): array of numbers

    Returns:
        List: list of core.Scalar, of which each element is a Scalar containing
          the corresponding value.
    """
    if isinstance(array, np.ndarray):
        array = array.ravel().tolist()
    return [wrap_as_scalar(item) for item in array]


def extract_plain_list(array):
    """extract value from a list of core.Scalar.

    Args:
        array (list): Scalars

    Returns:
        list: values extracted from the scalars.
    """
    return [item.value() for item in array]


def canonicalize_attrs(attrs, op_proto):
    """This function is used to canonicalize attributes(as a string->any dict)
    according to the type specification in the OpProto. This is especially
    important for operators that has any attributes of type Scalar or Scalars.

    Though various frontends of phi kernels & paddle operators can wrap variables
    of concrete types into Scalars(a tagged union of several numeric types) or
    vector of Scalars. Paddle operator requires strict type matching.

    Args:
        attrs (Dict[str, Any]): attribute dict intended to pass to an operator.
        op_proto (OpProto): Proto (signature) of the operator.

    Returns:
        Dict[str, Any]: canonicalized attributes.
    """
    canonicalized_attrs = attrs.copy()  # shallow copy is enough here
    for attr in op_proto.attrs:
        attr_name = attr.name
        type_index = attr.type
        if (attr_name not in attrs) or (attrs[attr_name] is None):
            continue

        attr_val = attrs[attr_name]

        # VAR and VARS should be skipped
        if isinstance(attr_val, Variable):
            continue
        if isinstance(attr_val, list) and _all_is_type(attr_val, Variable):
            continue

        # wrap
        if type_index == core.AttrType.SCALAR:
            canonicalized_attrs[attr_name] = core.Scalar(attr_val)
        elif type_index == core.AttrType.SCALARS:
            # it should be a list (or a numpy array)
            if len(attr_val) > 0:
                attr_val = np.array(attr_val).ravel().tolist()
                attr_val = [core.Scalar(x) for x in attr_val]
                canonicalized_attrs[attr_name] = attr_val

    return canonicalized_attrs
