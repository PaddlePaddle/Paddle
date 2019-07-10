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

from __future__ import print_function

from . import framework
import numpy as np
from .wrapped_decorator import signature_safe_contextmanager
from .core import VarDesc
from . import unique_name

__all__ = [
    'Constant', 'Uniform', 'Normal', 'TruncatedNormal', 'Xavier', 'Bilinear',
    'MSRA', 'force_init_on_cpu', 'init_on_cpu', 'ConstantInitializer',
    'UniformInitializer', 'NormalInitializer', 'TruncatedNormalInitializer',
    'XavierInitializer', 'BilinearInitializer', 'MSRAInitializer',
    'NumpyArrayInitializer'
]

_force_init_on_cpu_ = False


def force_init_on_cpu():
    """
    The flag of whether force to init variables on CPU.

    Returns:
        bool: the state if we should force init on CPU.

    Examples:

        .. code-block:: python

	    import paddle.fluid as fluid
        if fluid.initializer.force_init_on_cpu():
    		step = fluid.layers.create_global_var(
        	    shape=[2,3], value=1.0, dtype='float32')

    """
    return _force_init_on_cpu_


@signature_safe_contextmanager
def init_on_cpu():
    """
    Force the variable to be inited on CPU.

    Examples:
        .. code-block:: python

	    import paddle.fluid as fluid
        with fluid.initializer.init_on_cpu():
    		step = fluid.layers.create_global_var(
        	    shape=[2,3], value=1.0, dtype='float32')

    """
    global _force_init_on_cpu_

    pre_state = force_init_on_cpu()
    _force_init_on_cpu_ = True
    yield
    _force_init_on_cpu_ = pre_state


class Initializer(object):
    """Base class for variable initializers

    Defines the common interface of variable initializers.
    They add operations to the init program that are used
    to initialize variables. Users should not use this class
    directly, but need to use one of its implementations.
    """

    def __init__(self):
        pass

    def __call__(self, param, block):
        """Add corresponding initialization operations to the network
        """
        raise NotImplementedError()

    def _compute_fans(self, var):
        """Compute the fan_in and the fan_out for layers

        This method computes the fan_in and the fan_out
        for neural network layers, if not specified. It is
        not possible to perfectly estimate fan_in and fan_out.
        This method will estimate it correctly for matrix multiply and
        convolutions.

        Args:
            var: variable for which fan_in and fan_out have to be computed

        Returns:
            tuple of two integers (fan_in, fan_out)
        """
        shape = var.shape
        if not shape or len(shape) == 0:
            fan_in = fan_out = 1
        elif len(shape) == 1:
            fan_in = fan_out = shape[0]
        elif len(shape) == 2:
            # This is the case for simple matrix multiply
            fan_in = shape[0]
            fan_out = shape[1]
        else:
            # Assume this to be a convolutional kernel
            # In PaddlePaddle, the shape of the kernel is like:
            # [num_filters, num_filter_channels, ...] where the remaining
            # dimensions are the filter_size
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size

        return (fan_in, fan_out)


class ConstantInitializer(Initializer):
    """Implements the constant initializer

    Args:
        value (float): constant value to initialize the variable

    Examples:
        .. code-block:: python

    	    import paddle.fluid as fluid
            x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
	    fc = fluid.layers.fc(input=x, size=10,
    		param_attr=fluid.initializer.Constant(value=2.0))

    """

    def __init__(self, value=0.0, force_cpu=False):
        assert value is not None
        super(ConstantInitializer, self).__init__()
        self._value = value
        self._force_cpu = force_cpu

    def __call__(self, var, block):
        """Add constant initialization ops for a variable

        Args:
            var: Variable that needs to be initialized
            block: The block in which initialization ops
                   should be added

        Returns:
            the initialization op
        """
        assert isinstance(var, framework.Variable)
        assert isinstance(block, framework.Block)

        # to be compatible of fp16 initializers
        if var.dtype == VarDesc.VarType.FP16:
            out_dtype = VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['constant_init', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_dtype = var.dtype
            out_var = var

        # Initialization Ops should be prepended and not appended
        op = block._prepend_op(
            type="fill_constant",
            outputs={"Out": out_var},
            attrs={
                "shape": var.shape,
                "dtype": int(out_dtype),
                "value": float(self._value),
                'force_cpu': self._force_cpu or force_init_on_cpu()
            },
            stop_gradient=True)

        if var.dtype == VarDesc.VarType.FP16:
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype})

        if not framework.in_dygraph_mode():
            var.op = op
        return op


class UniformInitializer(Initializer):
    """Implements the random uniform distribution initializer

    Args:
        low (float): lower boundary of the uniform distribution
        high (float): upper boundary of the uniform distribution
        seed (int): random seed

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[1], dtype='float32')
            fc = fluid.layers.fc(input=x, size=10,
    		param_attr=fluid.initializer.Uniform(low=-0.5, high=0.5))
    """

    def __init__(self, low=-1.0, high=1.0, seed=0):
        assert low is not None
        assert high is not None
        assert high >= low
        assert seed is not None
        super(UniformInitializer, self).__init__()
        self._low = low
        self._high = high
        self._seed = seed

    def __call__(self, var, block):
        """Add uniform distribution initialization ops for a variable

        Args:
            var: Variable that needs to be initialized
            block: The block in which initialization ops
                   should be added

        Returns:
            the initialization op
        """
        assert isinstance(var, framework.Variable)
        assert isinstance(block, framework.Block)
        # Initialization Ops should be prepended and not appended
        if self._seed == 0:
            self._seed = block.program.random_seed

        # to be compatible of fp16 initializers
        if var.dtype == VarDesc.VarType.FP16:
            out_dtype = VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['uniform_random', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_dtype = var.dtype
            out_var = var

        op = block._prepend_op(
            type="uniform_random",
            outputs={"Out": out_var},
            attrs={
                "shape": var.shape,
                "dtype": out_dtype,
                "min": self._low,
                "max": self._high,
                "seed": self._seed
            },
            stop_gradient=True)

        if var.dtype == VarDesc.VarType.FP16:
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype})

        if not framework.in_dygraph_mode():
            var.op = op
        return op


class NormalInitializer(Initializer):
    """Implements the Random Normal(Gaussian) distribution initializer

    Args:
        loc (float): mean of the normal distribution
        scale (float): standard deviation of the normal distribution
        seed (int): random seed

    Examples:
        .. code-block:: python

	    import paddle.fluid as fluid
        x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
	    fc = fluid.layers.fc(input=x, size=10,
    		param_attr=fluid.initializer.Normal(loc=0.0, scale=2.0))

    """

    def __init__(self, loc=0.0, scale=1.0, seed=0):
        assert loc is not None
        assert scale is not None
        assert seed is not None
        super(NormalInitializer, self).__init__()
        self._mean = loc
        self._std_dev = scale
        self._seed = seed

    def __call__(self, var, block):
        """Add normal distribution initialization ops for a variable

        Args:
            var: Variable that needs to be initialized
            block: The block in which initialization ops
                   should be added

        Returns:
            the initialization op
        """
        assert isinstance(var, framework.Variable)
        assert isinstance(block, framework.Block)
        # Initialization Ops should be prepended and not appended
        if self._seed == 0:
            self._seed = block.program.random_seed

        # to be compatible of fp16 initalizers
        if var.dtype == VarDesc.VarType.FP16:
            out_dtype = VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['gaussian_random', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_dtype = var.dtype
            out_var = var

        op = block._prepend_op(
            type="gaussian_random",
            outputs={"Out": out_var},
            attrs={
                "shape": var.shape,
                "dtype": out_dtype,
                "mean": self._mean,
                "std": self._std_dev,
                "seed": self._seed,
                "use_mkldnn": False
            },
            stop_gradient=True)

        if var.dtype == VarDesc.VarType.FP16:
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype})
        if not framework.in_dygraph_mode():
            var.op = op
        return op


class TruncatedNormalInitializer(Initializer):
    """Implements the Random TruncatedNormal(Gaussian) distribution initializer

    Args:
        loc (float): mean of the normal distribution
        scale (float): standard deviation of the normal distribution
        seed (int): random seed

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[1], dtype='float32')
            fc = fluid.layers.fc(input=x, size=10,
                param_attr=fluid.initializer.TruncatedNormal(loc=0.0, scale=2.0))
    """

    def __init__(self, loc=0.0, scale=1.0, seed=0):
        assert loc is not None
        assert scale is not None
        assert seed is not None
        super(TruncatedNormalInitializer, self).__init__()
        self._mean = loc
        self._std_dev = scale
        self._seed = seed

    def __call__(self, var, block):
        """Add truncated normal distribution initialization ops for a variable

        Args:
            var: Variable that needs to be initialized
            block: The block in which initialization ops
                   should be added

        Returns:
            the initialization op
        """
        assert isinstance(var, framework.Variable)
        assert isinstance(block, framework.Block)
        # Initialization Ops should be prepended and not appended
        if self._seed == 0:
            self._seed = block.program.random_seed

        # to be compatible of fp16 initalizers
        if var.dtype == VarDesc.VarType.FP16:
            out_dtype = VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['truncated_gaussian_random', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_dtype = var.dtype
            out_var = var

        op = block._prepend_op(
            type="truncated_gaussian_random",
            outputs={"Out": out_var},
            attrs={
                "shape": var.shape,
                "dtype": out_dtype,
                "mean": self._mean,
                "std": self._std_dev,
                "seed": self._seed
            },
            stop_gradient=True)

        if var.dtype == VarDesc.VarType.FP16:
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype})
        if not framework.in_dygraph_mode():
            var.op = op
        return op


class XavierInitializer(Initializer):
    """
    This class implements the Xavier weight initializer from the paper
    `Understanding the difficulty of training deep feedforward neural
    networks <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_
    by Xavier Glorot and Yoshua Bengio.

    This initializer is designed to keep the scale of the gradients
    approximately same in all the layers. In case of Uniform distribution,
    the range is [-x, x], where

    .. math::

        x = \sqrt{\\frac{6.0}{fan\_in + fan\_out}}

    In case of Normal distribution, the mean is 0 and the standard deviation
    is

    .. math::

        \sqrt{\\frac{2.0}{fan\_in + fan\_out}}


    Args:
        uniform (bool): whether to use uniform or normal distribution
        fan_in (float): fan_in for Xavier initialization. If None, it is
                inferred from the variable.
        fan_out (float): fan_out for Xavier initialization. If None, it is
                 inferred from the variable.
        seed (int): random seed

    Note:
        It is recommended to set fan_in and fan_out to None for most cases.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            queries = fluid.layers.data(name='x', shape=[1], dtype='float32')
            fc = fluid.layers.fc(
                input=queries, size=10,
                param_attr=fluid.initializer.Xavier(uniform=False))

    """

    def __init__(self, uniform=True, fan_in=None, fan_out=None, seed=0):
        assert uniform is not None
        assert seed is not None
        super(XavierInitializer, self).__init__()
        self._uniform = uniform
        self._fan_in = fan_in
        self._fan_out = fan_out
        self._seed = seed

    def __call__(self, var, block):
        """Add xavier initialization ops for a variable

        Args:
            var: Variable that needs to be initialized
            block: The block in which initialization ops
                   should be added

        Returns:
            the initialization op
        """
        assert isinstance(var, framework.Variable)
        assert isinstance(block, framework.Block)
        f_in, f_out = self._compute_fans(var)

        # If fan_in and fan_out are passed, use them
        fan_in = f_in if self._fan_in is None else self._fan_in
        fan_out = f_out if self._fan_out is None else self._fan_out

        if self._seed == 0:
            self._seed = block.program.random_seed

        # to be compatible of fp16 initalizers
        if var.dtype == VarDesc.VarType.FP16:
            out_dtype = VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['xavier_init', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_dtype = var.dtype
            out_var = var

        if self._uniform:
            limit = np.sqrt(6.0 / float(fan_in + fan_out))
            op = block._prepend_op(
                type="uniform_random",
                outputs={"Out": out_var},
                attrs={
                    "shape": out_var.shape,
                    "dtype": out_dtype,
                    "min": -limit,
                    "max": limit,
                    "seed": self._seed
                },
                stop_gradient=True)

        else:
            std = np.sqrt(2.0 / float(fan_in + fan_out))
            op = block._prepend_op(
                type="gaussian_random",
                outputs={"Out": out_var},
                attrs={
                    "shape": out_var.shape,
                    "dtype": out_dtype,
                    "mean": 0.0,
                    "std": std,
                    "seed": self._seed
                },
                stop_gradient=True)

        if var.dtype == VarDesc.VarType.FP16:
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype})

        if not framework.in_dygraph_mode():
            var.op = op
        return op


class MSRAInitializer(Initializer):
    """Implements the MSRA initializer a.k.a. Kaiming Initializer

    This class implements the weight initialization from the paper
    `Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification <https://arxiv.org/abs/1502.01852>`_
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. This is a
    robust initialization method that particularly considers the rectifier
    nonlinearities. In case of Uniform distribution, the range is [-x, x], where

    .. math::

        x = \sqrt{\\frac{6.0}{fan\_in}}

    In case of Normal distribution, the mean is 0 and the standard deviation
    is

    .. math::

        \sqrt{\\frac{2.0}{fan\_in}}

    Args:
        uniform (bool): whether to use uniform or normal distribution
        fan_in (float): fan_in for MSRAInitializer. If None, it is\
        inferred from the variable.
        seed (int): random seed

    Note:
        It is recommended to set fan_in to None for most cases.

    Examples:
        .. code-block:: python
		
	    import paddle.fluid as fluid
        x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
	    fc = fluid.layers.fc(input=x, size=10,
    		param_attr=fluid.initializer.MSRA(uniform=False))

    """

    def __init__(self, uniform=True, fan_in=None, seed=0):
        """Constructor for MSRAInitializer
        """
        assert uniform is not None
        assert seed is not None
        super(MSRAInitializer, self).__init__()
        self._uniform = uniform
        self._fan_in = fan_in
        self._seed = seed

    def __call__(self, var, block):
        """Add MSRA initialization ops for a variable

        Args:
            var: Variable that needs to be initialized
            block: The block in which initialization ops
                   should be added

        Returns:
            the initialization op
        """
        assert isinstance(var, framework.Variable)
        assert isinstance(block, framework.Block)
        f_in, f_out = self._compute_fans(var)

        # If fan_in is passed, use it
        fan_in = f_in if self._fan_in is None else self._fan_in

        if self._seed == 0:
            self._seed = block.program.random_seed

        # to be compatible of fp16 initalizers
        if var.dtype == VarDesc.VarType.FP16:
            out_dtype = VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['masra_init', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_dtype = var.dtype
            out_var = var

        if self._uniform:
            limit = np.sqrt(6.0 / float(fan_in))
            op = block._prepend_op(
                type="uniform_random",
                outputs={"Out": out_var},
                attrs={
                    "shape": out_var.shape,
                    "dtype": int(out_dtype),
                    "min": -limit,
                    "max": limit,
                    "seed": self._seed
                },
                stop_gradient=True)

        else:
            std = np.sqrt(2.0 / float(fan_in))
            op = block._prepend_op(
                type="gaussian_random",
                outputs={"Out": out_var},
                attrs={
                    "shape": out_var.shape,
                    "dtype": int(out_dtype),
                    "mean": 0.0,
                    "std": std,
                    "seed": self._seed
                },
                stop_gradient=True)

        if var.dtype == VarDesc.VarType.FP16:
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype})

        if not framework.in_dygraph_mode():
            var.op = op
        return op


class BilinearInitializer(Initializer):
    """
    This initializer can be used in transposed convolution operator to
    act as upsampling. Users can upsample a feature map with shape of
    (B, C, H, W) by any integer factor. The usage is:

    Examples:

        .. code-block:: python

	    import paddle.fluid as fluid
        factor = 2
	    C = 2
	    w_attr = fluid.param_attr.ParamAttr(
		learning_rate=0., 
		regularizer=fluid.regularizer.L2Decay(0.),
                initializer=fluid.initializer.Bilinear())
	    x = fluid.layers.data(name="data", shape=[3, 32, 32], 
				  dtype="float32")
	    conv_up = fluid.layers.conv2d_transpose(
    		input=x,
    		num_filters=C,
    		output_size=None,
    		filter_size=2 * factor - factor % 2,
    		padding=int(math.ceil((factor - 1) / 2.)),
    		stride=factor,
    		groups=C,
    		param_attr=w_attr,
    		bias_attr=False)

    Where, `num_filters=C` and `groups=C` means this is channel-wise transposed
    convolution. The filter shape will be (C, 1, K, K) where K is `filer_size`,
    This initializer will set a (K, K) interpolation kernel for every channel
    of the filter identically. The resulting shape of the output feature map
    will be (B, C, factor * H, factor * W). Note that the learning rate and the
    weight decay are set to 0 in order to keep coefficient values of bilinear
    interpolation unchanged during training.

    """

    def __init__(self):
        """Constructor for BilinearInitializer.
        """
        super(BilinearInitializer, self).__init__()

    def __call__(self, var, block):
        """Add biliear initialization ops for a variable

        Args:
            var (Variable): Variable that needs to be initialized.
            block (Block): The block in which initialization ops should
                           be added.

        Returns:
            Operator: the initialization op

        Raises:
            ValueError: If type of `var` and `block` is not right.
                        If the shape of `var` size is not 4 and
                        var.shape[2] != var.shape[3].
        """
        if not isinstance(var, framework.Variable):
            raise ValueError("var must be framework.Variable.")

        if not isinstance(block, framework.Block):
            raise ValueError("block must be framework.Block.")

        shape = var.shape
        if len(shape) != 4:
            raise ValueError("the length of shape must be 4.")
        if shape[2] != shape[3]:
            raise ValueError("shape[2] must be equal to shape[3].")

        weight = np.zeros(np.prod(var.shape), dtype='float32')
        size = shape[3]
        # factor
        f = np.ceil(size / 2.)
        # center
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(np.prod(shape)):
            x = i % size
            y = (i / size) % size
            weight[i] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        weight = np.reshape(weight, shape)

        # to be compatible of fp16 initalizers
        if var.dtype == VarDesc.VarType.FP16:
            out_dtype = VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['bilinear_init', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_dtype = var.dtype
            out_var = var

        if out_dtype == VarDesc.VarType.FP32:
            value_name = "fp32_values"
            values = [float(v) for v in weight.flat]
        else:
            raise ValueError("Unsupported dtype %s", input.dtype)
        if np.prod(shape) > 1024 * 1024:
            raise ValueError("The size of input is too big. ")
        op = block.append_op(
            type='assign_value',
            outputs={'Out': [out_var]},
            attrs={
                'dtype': out_dtype,
                'shape': list(shape),
                value_name: values
            })

        if var.dtype == VarDesc.VarType.FP16:
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype})

        if not framework.in_dygraph_mode():
            var.op = op
        return op


class NumpyArrayInitializer(Initializer):
    """Init an parameter with an numpy array

    Args:
        value (numpy): numpy array to initialize the variable

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name="x", shape=[5], dtype='float32')
            fc = fluid.layers.fc(input=x, size=10,
                param_attr=fluid.initializer.NumpyArrayInitializer(numpy.array([1,2])))
    """

    def __init__(self, value):
        import numpy
        assert isinstance(value, numpy.ndarray)
        super(NumpyArrayInitializer, self).__init__()
        self._value = value

    def __call__(self, var, block):
        """Add constant initialization ops for a variable

        Args:
            var: Variable that needs to be initialized
            block: The block in which initialization ops
                   should be added

        Returns:
            the initialization op
        """
        assert isinstance(var, framework.Variable)
        assert isinstance(block, framework.Block)

        # to be compatible of fp16 initalizers
        if var.dtype == VarDesc.VarType.FP16:
            out_dtype = VarDesc.VarType.FP32
            np_value = self._value.astype("float32")
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['numpy_array_init', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_var = var
            out_dtype = var.dtype
            np_value = self._value

        # Initialization Ops should be prepended and not appended
        if out_dtype == VarDesc.VarType.FP32:
            value_name = "fp32_values"
            values = [float(v) for v in np_value.flat]
        elif out_dtype == VarDesc.VarType.INT32:
            value_name = "int32_values"
            values = [int(v) for v in np_value.flat]
        else:
            raise ValueError("Unsupported dtype %s", self._value.dtype)
        if self._value.size > 1024 * 1024 * 1024:
            raise ValueError("The size of input is too big. Please consider "
                             "saving it to file and 'load_op' to load it")
        op = block._prepend_op(
            type='assign_value',
            outputs={'Out': out_var},
            attrs={
                'dtype': out_dtype,
                'shape': list(self._value.shape),
                value_name: values
            },
            stop_gradient=True)

        if var.dtype == VarDesc.VarType.FP16:
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype})

        if not framework.in_dygraph_mode():
            var.op = op
        return op


# We short the class name, since users will use the initializer with the package
# name. The sample code:
#
# import paddle.fluid as fluid
#
# hidden = fluid.layers.fc(...,
#                          param_attr=ParamAttr(fluid.initializer.Xavier()))
#
# It is no need to add an `Initializer` as the class suffix
Constant = ConstantInitializer
Uniform = UniformInitializer
Normal = NormalInitializer
TruncatedNormal = TruncatedNormalInitializer
Xavier = XavierInitializer
MSRA = MSRAInitializer
Bilinear = BilinearInitializer
