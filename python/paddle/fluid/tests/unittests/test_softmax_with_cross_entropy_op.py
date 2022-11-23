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

import unittest
import numpy as np
import paddle
import paddle.fluid.core as core

from op_test import OpTest
from test_softmax_op import stable_softmax


def cross_entropy(softmax, label, soft_label, axis, ignore_index=-1):
    if soft_label:
        return (-label * np.log(softmax)).sum(axis=axis, keepdims=True)
    shape = softmax.shape
    axis %= len(shape)
    n = int(np.prod(shape[:axis]))
    axis_dim = shape[axis]
    remain = int(np.prod(shape[axis + 1:]))
    softmax_reshape = softmax.reshape((n, axis_dim, remain))
    label_reshape = label.reshape((n, 1, remain))
    result = np.zeros_like(label_reshape, dtype=softmax.dtype)
    for i in range(n):
        for j in range(remain):
            lbl = label_reshape[i, 0, j]
            if lbl != ignore_index:
                result[i, 0, j] -= np.log(softmax_reshape[i, lbl, j])
    return result.reshape(label.shape)


def python_api(logits,
               label,
               soft_label=False,
               use_softmax=True,
               numeric_stable_mode=True,
               ignore_index=-100,
               axis=-1):
    # here only can test paddle.nn.functional.softmax_with_cross_entropy,
    # the paddle.nn.functional.cross_entropy contains other math ops
    return paddle.nn.functional.softmax_with_cross_entropy(
        logits,
        label,
        soft_label=soft_label,
        ignore_index=ignore_index,
        numeric_stable_mode=numeric_stable_mode,
        return_softmax=use_softmax,
        axis=axis)


def python_core_api_without_softmax(logits,
                                    label,
                                    soft_label=False,
                                    use_softmax=False,
                                    numeric_stable_mode=True,
                                    ignore_index=-100,
                                    axis=-1):
    # the API paddle.nn.functional.softmax_with_cross_entropy cannot
    # set use_softmax=False, so add a core api manually
    assert use_softmax is False
    _, loss = paddle._C_ops.cross_entropy_with_softmax(logits, label,
                                                       soft_label, use_softmax,
                                                       numeric_stable_mode,
                                                       ignore_index, axis)
    return loss


class TestSoftmaxWithCrossEntropyOp(OpTest):
    """
    Test softmax with cross entropy operator with discreate one-hot labels.
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = False
        self.soft_label = False
        # explicilty use float32 for ROCm, as MIOpen does not yet support float64
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.axis = -1
        self.ignore_index = -1
        self.shape = [41, 37]
        self.use_softmax = True

    def hard_label_dtype(self):
        return "int64"

    def setUp(self):
        self.initParams()

        logits = getattr(
            self, "logits",
            np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype))
        softmax = np.apply_along_axis(stable_softmax, self.axis, logits)

        if self.soft_label:
            labels = np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype)
            labels /= np.sum(labels, axis=self.axis, keepdims=True)
        else:
            axis_dim = self.shape[self.axis]
            self.shape[self.axis] = 1
            labels = np.random.randint(0,
                                       axis_dim,
                                       self.shape,
                                       dtype=self.hard_label_dtype())

        loss = cross_entropy(softmax, labels, self.soft_label, self.axis,
                             self.ignore_index)

        if self.use_softmax == False:
            self.inputs = {"Logits": softmax, "Label": labels}
        else:
            self.inputs = {"Logits": logits, "Label": labels}

        self.outputs = {
            "Softmax": softmax.astype(self.dtype),
            "Loss": loss.astype(self.dtype)
        }
        self.attrs = {
            "numeric_stable_mode": self.numeric_stable_mode,
            "soft_label": self.soft_label,
            "ignore_index": self.ignore_index,
            "use_softmax": self.use_softmax,
        }

        if self.axis != -1:
            self.attrs['axis'] = self.axis

    def test_check_output(self):
        if self.python_api is not None:
            self.check_output(check_eager=True)
        self.check_output()

    def test_check_grad(self):
        if core.is_compiled_with_rocm():
            if self.python_api is not None:
                self.check_grad(["Logits"],
                                "Loss",
                                max_relative_error=5e-1,
                                check_eager=True)
            # HIP will have accuracy fail when using float32 in CPU place
            self.check_grad(["Logits"], "Loss", max_relative_error=5e-1)
        else:
            if self.python_api is not None:
                self.check_grad(["Logits"],
                                "Loss",
                                numeric_grad_delta=0.001,
                                check_eager=True)
            self.check_grad(["Logits"], "Loss", numeric_grad_delta=0.001)


class TestSoftmaxWithCrossEntropyOpInt32(TestSoftmaxWithCrossEntropyOp):

    def hard_label_dtype(self):
        return "int32"


class TestSoftmaxWithCrossEntropyOpInt16(TestSoftmaxWithCrossEntropyOp):

    def hard_label_dtype(self):
        return "int16"


class TestSoftmaxWithCrossEntropyOpInt8(TestSoftmaxWithCrossEntropyOp):

    def hard_label_dtype(self):
        return "int8"


class TestSoftmaxWithCrossEntropyOpUInt8(TestSoftmaxWithCrossEntropyOp):

    def hard_label_dtype(self):
        return "uint8"


class TestSoftmaxWithCrossEntropyOp_NotWithSoftmax_SoftLabel_1D(
        TestSoftmaxWithCrossEntropyOp):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_core_api_without_softmax
        self.python_out_sig = ["Loss"]
        self.numeric_stable_mode = True
        self.soft_label = True
        self.shape = [13, 8]
        self.axis = -1
        self.ignore_index = -1
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.use_softmax = False  #default is true, means "with softmax"


class TestSoftmaxWithCrossEntropyOp_NotWithSoftmax_HardLabel_1D(
        TestSoftmaxWithCrossEntropyOp):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_core_api_without_softmax
        self.python_out_sig = ["Loss"]
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [13, 8]
        self.axis = -1
        self.ignore_index = -1
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.use_softmax = False  #default is true, means "with softmax"


##############################################################################
#NotWithSoftmax_SoftLabel_2D start
##############################################################################
class TestSoftmaxWithCrossEntropyOp_NotWithSoftmax_SoftLabel_2D(
        TestSoftmaxWithCrossEntropyOp):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_core_api_without_softmax
        self.python_out_sig = ["Loss"]
        self.numeric_stable_mode = True
        self.soft_label = True
        self.shape = [3, 5, 7, 11]
        self.axis = -1
        self.ignore_index = -1
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.use_softmax = False  #default is true, means "with softmax"


class TestSoftmaxWithCrossEntropyOp_NotWithSoftmax_SoftLabel_2D_Axis2(
        TestSoftmaxWithCrossEntropyOp):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_core_api_without_softmax
        self.python_out_sig = ["Loss"]
        self.numeric_stable_mode = True
        self.soft_label = True
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.axis = 1
        self.ignore_index = -1
        self.shape = [3, 5, 7, 11]
        self.use_softmax = False  #default is true, means "with softmax"


class TestSoftmaxWithCrossEntropyOp_NotWithSoftmax_SoftLabel_2D_Axis3(
        TestSoftmaxWithCrossEntropyOp):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_core_api_without_softmax
        self.python_out_sig = ["Loss"]
        self.numeric_stable_mode = True
        self.soft_label = True
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.axis = 2
        self.ignore_index = -1
        self.shape = [3, 5, 7, 11]
        self.use_softmax = False  #default is true, means "with softmax"


class TestSoftmaxWithCrossEntropyOp_NotWithSoftmax_SoftLabel_2D_Axis4(
        TestSoftmaxWithCrossEntropyOp):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_core_api_without_softmax
        self.python_out_sig = ["Loss"]
        self.numeric_stable_mode = True
        self.soft_label = True
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.axis = 3
        self.ignore_index = -1
        self.shape = [3, 5, 7, 11]
        self.use_softmax = False  #default is true, means "with softmax"


##############################################################################
#NotWithSoftmax_SoftLabel_2D end
##############################################################################

##############################################################################
#NotWithSoftmax_HardLabel_2D start
##############################################################################


class TestSoftmaxWithCrossEntropyOp_NotWithSoftmax_HardLabel_2D(
        TestSoftmaxWithCrossEntropyOp):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_core_api_without_softmax
        self.python_out_sig = ["Loss"]
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.axis = -1
        self.ignore_index = -1
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.use_softmax = False  #default is true, means "with softmax"


class TestSoftmaxWithCrossEntropyOp_NotWithSoftmax_HardLabel_2D_Axis2(
        TestSoftmaxWithCrossEntropyOp):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_core_api_without_softmax
        self.python_out_sig = ["Loss"]
        self.numeric_stable_mode = True
        self.soft_label = False
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.axis = 1
        self.ignore_index = -1
        self.shape = [3, 5, 7, 11]
        self.use_softmax = False  #default is true, means "with softmax"


class TestSoftmaxWithCrossEntropyOp_NotWithSoftmax_HardLabel_2D_Axis3(
        TestSoftmaxWithCrossEntropyOp):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_core_api_without_softmax
        self.python_out_sig = ["Loss"]
        self.numeric_stable_mode = True
        self.soft_label = False
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.axis = 2
        self.ignore_index = -1
        self.shape = [3, 5, 7, 11]
        self.use_softmax = False  #default is true, means "with softmax"


class TestSoftmaxWithCrossEntropyOp_NotWithSoftmax_HardLabel_2D_Axis4(
        TestSoftmaxWithCrossEntropyOp):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_core_api_without_softmax
        self.python_out_sig = ["Loss"]
        self.numeric_stable_mode = True
        self.soft_label = False
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.axis = 3
        self.ignore_index = -1
        self.shape = [3, 5, 7, 11]
        self.use_softmax = False  #default is true, means "with softmax"


##############################################################################
#NotWithSoftmax_HardLabel_2D end
##############################################################################

##############################################################################
#NotWithSoftmax_HardLabel_2D_Ignore start
##############################################################################


class TestSoftmaxWithCrossEntropyOp_NotWithSoftmax_HardLabel_Ignore(
        TestSoftmaxWithCrossEntropyOp):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_core_api_without_softmax
        self.python_out_sig = ["Loss"]
        self.numeric_stable_mode = False
        self.soft_label = False
        self.shape = [13, 8]
        self.axis = -1
        self.ignore_index = 2
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.use_softmax = False  #default is true, means "with softmax"


class TestSoftmaxWithCrossEntropyOp_NotWithSoftmax_HardLabel_Ignore_Axis(
        TestSoftmaxWithCrossEntropyOp):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_core_api_without_softmax
        self.python_out_sig = ["Loss"]
        self.numeric_stable_mode = False
        self.soft_label = False
        self.shape = [13, 8]
        self.axis = 1
        self.ignore_index = 2
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.use_softmax = False  #default is true, means "with softmax"


class TestSoftmaxWithCrossEntropyOp_NotWithSoftmax_HardLabel_2D_Ignore(
        TestSoftmaxWithCrossEntropyOp):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_core_api_without_softmax
        self.python_out_sig = ["Loss"]
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.axis = -1
        self.ignore_index = 2
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.use_softmax = False  #default is true, means "with softmax"


class TestSoftmaxWithCrossEntropyOp_NotWithSoftmax_HardLabel_2D_Ignore_Axis3(
        TestSoftmaxWithCrossEntropyOp):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_core_api_without_softmax
        self.python_out_sig = ["Loss"]
        self.numeric_stable_mode = True
        self.soft_label = False
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.axis = 2
        self.ignore_index = 2
        self.shape = [3, 5, 7, 11]
        self.use_softmax = False  #default is true, means "with softmax"


##############################################################################
#NotWithSoftmax_HardLabel_2D_Ignore end
##############################################################################


class TestSoftmaxWithCrossEntropyOpNoCudnn(TestSoftmaxWithCrossEntropyOp):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.axis = -1
        self.ignore_index = -1
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.use_softmax = True


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSoftmaxWithCrossEntropyOpFp16(TestSoftmaxWithCrossEntropyOp):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = False
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.axis = -1
        self.ignore_index = -1
        self.dtype = np.float16

    def setUp(self):
        self.initParams()
        self.op_type = "softmax_with_cross_entropy"

        # NOTE: numpy float16 have very low accuracy, use float32 for numpy check.
        date_type = np.float32 if core.is_compiled_with_rocm() else np.float64
        logits = getattr(
            self, "logits",
            np.random.uniform(0.1, 1.0, self.shape).astype(date_type))
        softmax = np.apply_along_axis(stable_softmax, self.axis, logits)

        axis_dim = self.shape[self.axis]
        self.shape[self.axis] = 1
        labels = np.random.randint(0, axis_dim, self.shape, dtype="int64")

        loss = cross_entropy(softmax, labels, self.soft_label, self.axis)

        self.inputs = {"Logits": logits.astype(self.dtype), "Label": labels}
        self.outputs = {
            "Softmax": softmax.astype(self.dtype),
            "Loss": loss.astype(self.dtype)
        }
        self.attrs = {
            "numeric_stable_mode": self.numeric_stable_mode,
            "soft_label": self.soft_label,
        }
        if self.axis != -1:
            self.attrs['axis'] = self.axis

    def test_check_output(self):
        if self.python_api is not None:
            self.check_output(atol=1e-2, check_eager=True)
        self.check_output(atol=1e-2)

    def test_check_grad(self):
        if self.python_api is not None:
            self.check_grad(["Logits"],
                            "Loss",
                            max_relative_error=0.1,
                            check_eager=True)
        self.check_grad(["Logits"], "Loss", max_relative_error=0.1)


class TestSoftmaxWithCrossEntropyOpNoCudnnFp16(TestSoftmaxWithCrossEntropyOpFp16
                                               ):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.axis = -1
        self.ignore_index = -1
        self.dtype = np.float16

    def test_check_grad(self):
        if self.python_api is not None:
            self.check_grad(["Logits"],
                            "Loss",
                            max_relative_error=0.1,
                            check_eager=True)
        self.check_grad(["Logits"], "Loss", max_relative_error=0.1)


class TestSoftmaxWithCrossEntropyOp2(TestSoftmaxWithCrossEntropyOp):
    """
    Test softmax with cross entropy operator with soft labels.
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = True
        self.soft_label = True
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.axis = -1
        self.ignore_index = -1
        self.shape = [41, 37]
        self.use_softmax = True

    def test_check_output(self):
        if self.python_api is not None:
            self.check_output(check_eager=True)
        self.check_output()

    def test_check_grad(self):
        if core.is_compiled_with_rocm():
            # HIP will have accuracy fail when using float32 in CPU place
            if self.python_api is not None:
                self.check_grad(["Logits"],
                                "Loss",
                                max_relative_error=0.1,
                                check_eager=True)
            self.check_grad(["Logits"], "Loss", max_relative_error=0.1)
        else:
            if self.python_api is not None:
                self.check_grad(["Logits"], "Loss", check_eager=True)
            self.check_grad(["Logits"], "Loss")


class TestSoftmaxWithCrossEntropyOp3(TestSoftmaxWithCrossEntropyOp):
    """
    Test softmax with cross entropy operator with ignore_index.
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = False
        self.soft_label = False
        self.shape = [41, 37]
        self.ignore_index = 5
        self.axis = -1
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.use_softmax = True


class TestSoftmaxWithCrossEntropyOp3NoCudnn(TestSoftmaxWithCrossEntropyOp3):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.ignore_index = 4
        self.axis = -1
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.use_softmax = True


class TestSoftmaxWithCrossEntropyOpAxis1(TestSoftmaxWithCrossEntropyOp):
    """
    Test softmax with cross entropy operator with discreate one-hot labels.
    Given axis != -1
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = True
        self.soft_label = False
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.axis = 0
        self.ignore_index = -1
        self.shape = [3, 5, 7, 11]
        self.use_softmax = True


class TestSoftmaxWithCrossEntropyOpAxis2(TestSoftmaxWithCrossEntropyOp):
    """
    Test softmax with cross entropy operator with discreate one-hot labels.
    Given axis != -1
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = True
        self.soft_label = False
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.axis = 1
        self.ignore_index = -1
        self.shape = [3, 5, 7, 11]
        self.use_softmax = True


class TestSoftmaxWithCrossEntropyOpAxis3(TestSoftmaxWithCrossEntropyOp):
    """
    Test softmax with cross entropy operator with discreate one-hot labels.
    Given axis != -1
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = True
        self.soft_label = False
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.axis = 2
        self.ignore_index = -1
        self.shape = [3, 5, 7, 11]
        self.use_softmax = True


class TestSoftmaxWithCrossEntropyOpAxis4(TestSoftmaxWithCrossEntropyOp):
    """
    Test softmax with cross entropy operator with discreate one-hot labels.
    Given axis != -1
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = True
        self.soft_label = False
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.axis = 3
        self.ignore_index = -1
        self.shape = [3, 5, 7, 11]
        self.use_softmax = True


class TestSoftmaxWithCrossEntropyOpAxisDimEqualOne(TestSoftmaxWithCrossEntropyOp
                                                   ):
    """
    Test softmax with cross entropy operator with discreate one-hot labels.
    Given axis != -1
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = True
        self.soft_label = False
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.axis = -1
        self.ignore_index = -1
        self.shape = [3, 5, 7, 1]
        self.use_softmax = True


class TestSoftmaxWithCrossEntropyOpNoCudnnFp16Axis1(
        TestSoftmaxWithCrossEntropyOpNoCudnnFp16):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.axis = 0
        self.ignore_index = -1
        self.dtype = np.float16
        self.use_softmax = True


class TestSoftmaxWithCrossEntropyOpNoCudnnFp16Axis2(
        TestSoftmaxWithCrossEntropyOpNoCudnnFp16):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.axis = 1
        self.ignore_index = -1
        self.dtype = np.float16
        self.use_softmax = True


class TestSoftmaxWithCrossEntropyOpNoCudnnFp16Axis3(
        TestSoftmaxWithCrossEntropyOpNoCudnnFp16):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.axis = 2
        self.ignore_index = -1
        self.dtype = np.float16
        self.use_softmax = True


class TestSoftmaxWithCrossEntropyOpSoftLabelAxis1(TestSoftmaxWithCrossEntropyOp2
                                                  ):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = True
        self.soft_label = True
        self.shape = [3, 5, 7, 11]
        self.axis = 0
        self.ignore_index = -1
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.use_softmax = True


class TestSoftmaxWithCrossEntropyOpSoftLabelAxis2(TestSoftmaxWithCrossEntropyOp2
                                                  ):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = True
        self.soft_label = True
        self.shape = [3, 5, 7, 11]
        self.axis = 1
        self.ignore_index = -1
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.use_softmax = True


class TestSoftmaxWithCrossEntropyOpSoftLabelAxis3(TestSoftmaxWithCrossEntropyOp2
                                                  ):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = True
        self.soft_label = True
        self.shape = [3, 5, 7, 11]
        self.axis = 2
        self.ignore_index = -1
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.use_softmax = True


class TestSoftmaxWithCrossEntropyOpSoftLabelAxis4(TestSoftmaxWithCrossEntropyOp2
                                                  ):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = True
        self.soft_label = True
        self.shape = [3, 5, 7, 11]
        self.axis = 3
        self.ignore_index = -1
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.use_softmax = True


class TestSoftmaxWithCrossEntropyOpIgnoreIndexNoCudnnAxis1(
        TestSoftmaxWithCrossEntropyOp3):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.ignore_index = 1
        self.axis = 0
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.use_softmax = True


class TestSoftmaxWithCrossEntropyOpIgnoreIndexNoCudnnAxis2(
        TestSoftmaxWithCrossEntropyOp3):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.ignore_index = 0
        self.axis = 1
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.use_softmax = True


class TestSoftmaxWithCrossEntropyOpIgnoreIndexNoCudnnAxis3(
        TestSoftmaxWithCrossEntropyOp3):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.ignore_index = 3
        self.axis = 2
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.use_softmax = True


class TestSoftmaxWithCrossEntropyOpIgnoreIndexNoCudnnAxis4(
        TestSoftmaxWithCrossEntropyOp3):

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.ignore_index = 3
        self.axis = 3
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.use_softmax = True


class TestSoftmaxWithCrossEntropyOpBoundary0(TestSoftmaxWithCrossEntropyOp):
    """
    Test stable softmax with cross entropy operator will not product INF
    with small logits value.
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.axis = -1
        self.ignore_index = -1
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.logits = np.full(self.shape, -500.0).astype(self.dtype)
        self.use_softmax = True


class TestSoftmaxWithCrossEntropyOpBoundary1(TestSoftmaxWithCrossEntropyOp):
    """
    Test stable softmax with cross entropy operator will not product INF
    with small logits value.
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.python_api = python_api
        self.python_out_sig = ["Loss", "Softmax"]
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.axis = -1
        self.ignore_index = -1
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.logits = np.full(self.shape, 1000.0).astype(self.dtype)
        self.logits[:, :, 0, :] = -1000.0
        self.use_softmax = True


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
