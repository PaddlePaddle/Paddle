# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


from paddle.base.libpaddle import DataType

from . import OpResult

_already_patch_opresult = False


def create_tensor_with_batchsize(ref_var, value, dtype):
    assert isinstance(ref_var, OpResult)
    value = float(value)
    batch_dim = -1
    out_shape = []
    for i, d in enumerate(ref_var.shape):
        if d < 0:
            if batch_dim < 0:
                batch_dim = i
                out_shape.append(d)
            else:
                out_shape.append(1)
        else:
            out_shape.append(d)
    assert batch_dim != -1
    from paddle import _C_ops
    from paddle.framework import core

    out = _C_ops.full_batch_size_like(
        ref_var, out_shape, dtype, value, batch_dim, batch_dim, core.Place()
    )
    out.stop_gradient = True

    return out


def monkey_patch_opresult():
    def safe_get_dtype(var):
        try:
            dtype = var.dtype
        except:
            raise ValueError("Cannot get data type from var")
        return dtype

    _supported_int_dtype_ = [
        DataType.BOOL,
        DataType.UINT8,
        DataType.INT8,
        DataType.INT16,
        DataType.INT32,
        DataType.INT64,
    ]

    def _scalar_div_(var, value):
        return paddle.scale(var, 1.0 / value, 0.0)

    def _binary_creator_(
        method_name,
        python_api,
        reverse=False,
        scalar_method=None,
    ):
        def __impl__(self, other_var):
            # 1. scalar exists cases
            # we need combine the tensor.dtype and scalar.dtype, cast correct object
            if isinstance(other_var, float):
                # in all cases(+, -, *, /, **, //, %), we need cast tensor.dtype to float
                if self.dtype in _supported_int_dtype_:
                    paddle.cast(self, DataType.FLOAT32)
                # here use `scale` replace `elementwise` to get better performance
                # but only +, -, *, / can use this method
                if scalar_method is not None:
                    return scalar_method(self, other_var)
            elif isinstance(other_var, int):
                # in all cases(+, -, *, /, **, //, %), we can cast it to float
                # because the output tensor.dtype depend on the type of input tensor
                other_var = float(other_var)
                # division is a special case
                # NOTE(chenweihang): because we cast tensor to float32 instead float64,
                # the division result can only guarantee the numerical accuracy of 6 digits
                # after the decimal point. The result of numpy calculation is of float64 type,
                # so the calculation result here and the calculation result of numpy are
                # different after 6 decimal point. If necessary, we can also use float64 here.
                # torch's behavior here is consistent with ours
                if (
                    python_api == paddle.divide
                    and self.dtype in _supported_int_dtype_
                ):
                    paddle.cast(self, DataType.FLOAT32)
                # here use `scale` replace `elementwise` to get better performance
                # but only +, -, *, / can use this method
                if scalar_method is not None:
                    return scalar_method(self, other_var)
            else:
                # do nothing
                pass

            # 2. create OpResult for scalar
            lhs_dtype = safe_get_dtype(self)
            other_var_opresult = other_var
            if not isinstance(other_var, OpResult):
                if reverse:
                    for elem in self.shape:
                        if elem < 0:
                            other_var_opresult = create_tensor_with_batchsize(
                                self, other_var, lhs_dtype
                            )

                            break
                    else:
                        # when break is not triggered, enter the else branch
                        other_var_opresult = paddle.fill_constant(
                            self.shape,
                            lhs_dtype,
                            other_var,
                        )
                else:
                    # add fill_op to current_block
                    other_var_opresult = paddle.fill_constant(
                        [],
                        lhs_dtype,
                        other_var,
                    )

            # 3. unify right var type to left var
            rhs_dtype = safe_get_dtype(other_var_opresult)
            if lhs_dtype != rhs_dtype:
                other_var_opresult = paddle.cast(other_var_opresult, lhs_dtype)
            if reverse:
                tmp = self
                self = other_var_opresult
                other_var_opresult = tmp

            if (
                python_api == paddle.divide
            ) and self.dtype in _supported_int_dtype_:
                self = paddle.cast(self, DataType.FLOAT32)
                other_var = paddle.cast(other_var_opresult, DataType.FLOAT32)

            out = python_api(self, other_var_opresult)
            return out

        __impl__.__doc__ = """
            Args:
                self(OpResult): left hand OpResult
                other_var(OpResult|float|int): right hand OpResult

            Returns:
                OpResult
            """
        __impl__.__name__ = method_name
        return __impl__

    import paddle

    opresult_methods = [
        (
            '__div__',
            _binary_creator_(
                '__div__', paddle.tensor.divide, False, _scalar_div_
            ),
        ),
        (
            '__truediv__',
            _binary_creator_(
                '__truediv__', paddle.tensor.divide, False, _scalar_div_
            ),
        ),
        (
            '__rdiv__',
            _binary_creator_('__rdiv__', paddle.tensor.divide, True, None),
        ),
        (
            '__rtruediv__',
            _binary_creator_('__rtruediv__', paddle.tensor.divide, True, None),
        ),
    ]

    global _already_patch_opresult
    if not _already_patch_opresult:
        for method in opresult_methods:
            method_name = method[0]
            method_impl = method[1]
            setattr(OpResult, method_name, method_impl)

    else:
        # Handling Tensor Methods
        import paddle.tensor

        for method_name in paddle.tensor.tensor_method_func:
            if hasattr(OpResult, method_name):
                continue
            method_impl = getattr(paddle.tensor, method_name, None)
            if method_impl:
                setattr(OpResult, method_name, method_impl)

        # Handling __getitem__
        from ..base.variable_index import _getitem_static

        OpResult.__getitem__ = _getitem_static

    _already_patch_opresult = True
