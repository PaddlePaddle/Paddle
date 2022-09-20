# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
try:
    from paddle.version import full_version as __version__
    from paddle.version import commit as __git_commit__
    from paddle.cuda_env import *
except ImportError:
    import sys
    sys.stderr.write('''Warning with import paddle: you should not
     import paddle from the source directory; please install paddlepaddle*.whl firstly.'''
                     )

from .batch import batch  # noqa: F401
from .framework import monkey_patch_variable
from .framework import monkey_patch_math_varbase

monkey_patch_variable()
monkey_patch_math_varbase()

from .framework import disable_signal_handler  # noqa: F401
from .framework import get_flags  # noqa: F401
from .framework import set_flags  # noqa: F401

from .framework import disable_static  # noqa: F401
from .framework import enable_static  # noqa: F401
from .framework import in_dynamic_mode  # noqa: F401
from .fluid.dataset import *  # noqa: F401
from .fluid.lazy_init import LazyGuard  # noqa: F401

from .framework.dtype import iinfo  # noqa: F401
from .framework.dtype import dtype as dtype  # noqa: F401
from .framework.dtype import uint8  # noqa: F401
from .framework.dtype import int8  # noqa: F401
from .framework.dtype import int16  # noqa: F401
from .framework.dtype import int32  # noqa: F401
from .framework.dtype import int64  # noqa: F401
from .framework.dtype import float16  # noqa: F401
from .framework.dtype import float32  # noqa: F401
from .framework.dtype import float64  # noqa: F401
from .framework.dtype import bfloat16  # noqa: F401
from .framework.dtype import bool  # noqa: F401
from .framework.dtype import complex64  # noqa: F401
from .framework.dtype import complex128  # noqa: F401
if fluid.framework._in_eager_mode_:
    Tensor = framework.core.eager.Tensor
else:
    from .framework import VarBase as Tensor  # noqa: F401

Tensor.__qualname__ = 'Tensor'  # noqa: F401
import paddle.compat  # noqa: F401
import paddle.distributed  # noqa: F401
import paddle.sysconfig  # noqa: F401
import paddle.distribution  # noqa: F401
import paddle.nn  # noqa: F401
import paddle.distributed.fleet  # noqa: F401
import paddle.optimizer  # noqa: F401
import paddle.metric  # noqa: F401
import paddle.regularizer  # noqa: F401
import paddle.incubate  # noqa: F401
import paddle.autograd  # noqa: F401
import paddle.device  # noqa: F401

import paddle.jit  # noqa: F401
import paddle.amp  # noqa: F401
import paddle.dataset  # noqa: F401
import paddle.inference  # noqa: F401
import paddle.io  # noqa: F401
import paddle.onnx  # noqa: F401
import paddle.reader  # noqa: F401
import paddle.static  # noqa: F401
import paddle.vision  # noqa: F401
import paddle.geometric  # noqa: F401

from .tensor.attribute import is_complex  # noqa: F401
from .tensor.attribute import is_integer  # noqa: F401
from .tensor.attribute import rank  # noqa: F401
from .tensor.attribute import shape  # noqa: F401
from .tensor.attribute import real  # noqa: F401
from .tensor.attribute import imag  # noqa: F401
from .tensor.attribute import is_floating_point  # noqa: F401
from .tensor.creation import to_tensor  # noqa: F401
from .tensor.creation import diag  # noqa: F401
from .tensor.creation import diagflat  # noqa: F401
from .tensor.creation import eye  # noqa: F401
from .tensor.creation import linspace  # noqa: F401
from .tensor.creation import logspace  # noqa: F401
from .tensor.creation import ones  # noqa: F401
from .tensor.creation import ones_like  # noqa: F401
from .tensor.creation import zeros  # noqa: F401
from .tensor.creation import zeros_like  # noqa: F401
from .tensor.creation import arange  # noqa: F401
from .tensor.creation import full  # noqa: F401
from .tensor.creation import full_like  # noqa: F401
from .tensor.creation import triu  # noqa: F401
from .tensor.creation import tril  # noqa: F401
from .tensor.creation import meshgrid  # noqa: F401
from .tensor.creation import empty  # noqa: F401
from .tensor.creation import empty_like  # noqa: F401
from .tensor.creation import assign  # noqa: F401
from .tensor.creation import complex  # noqa: F401
from .tensor.creation import clone  # noqa: F401
from .tensor.creation import tril_indices  #noqa: F401
from .tensor.creation import triu_indices  #noqa: F401
from .tensor.linalg import matmul  # noqa: F401
from .tensor.linalg import dot  # noqa: F401
from .tensor.linalg import norm  # noqa: F401
from .tensor.linalg import transpose  # noqa: F401
from .tensor.linalg import dist  # noqa: F401
from .tensor.linalg import t  # noqa: F401
from .tensor.linalg import cross  # noqa: F401
from .tensor.linalg import cholesky  # noqa: F401
from .tensor.linalg import bmm  # noqa: F401
from .tensor.linalg import histogram  # noqa: F401
from .tensor.linalg import bincount  # noqa: F401
from .tensor.linalg import mv  # noqa: F401
from .tensor.logic import equal  # noqa: F401
from .tensor.linalg import eigvalsh  # noqa: F401
from .tensor.logic import greater_equal  # noqa: F401
from .tensor.logic import greater_than  # noqa: F401
from .tensor.logic import is_empty  # noqa: F401
from .tensor.logic import less_equal  # noqa: F401
from .tensor.logic import less_than  # noqa: F401
from .tensor.logic import logical_and  # noqa: F401
from .tensor.logic import logical_not  # noqa: F401
from .tensor.logic import logical_or  # noqa: F401
from .tensor.logic import logical_xor  # noqa: F401
from .tensor.logic import bitwise_and  # noqa: F401
from .tensor.logic import bitwise_not  # noqa: F401
from .tensor.logic import bitwise_or  # noqa: F401
from .tensor.logic import bitwise_xor  # noqa: F401
from .tensor.logic import not_equal  # noqa: F401
from .tensor.logic import allclose  # noqa: F401
from .tensor.logic import isclose  # noqa: F401
from .tensor.logic import equal_all  # noqa: F401
from .tensor.logic import is_tensor  # noqa: F401
from .tensor.manipulation import cast  # noqa: F401
from .tensor.manipulation import concat  # noqa: F401
from .tensor.manipulation import broadcast_tensors  # noqa: F401
from .tensor.manipulation import expand  # noqa: F401
from .tensor.manipulation import broadcast_to  # noqa: F401
from .tensor.manipulation import expand_as  # noqa: F401
from .tensor.manipulation import tile  # noqa: F401
from .tensor.manipulation import flatten  # noqa: F401
from .tensor.manipulation import gather  # noqa: F401
from .tensor.manipulation import gather_nd  # noqa: F401
from .tensor.manipulation import reshape  # noqa: F401
from .tensor.manipulation import reshape_  # noqa: F401
from .tensor.manipulation import flip as reverse  # noqa: F401
from .tensor.manipulation import scatter  # noqa: F401
from .tensor.manipulation import scatter_  # noqa: F401
from .tensor.manipulation import scatter_nd_add  # noqa: F401
from .tensor.manipulation import scatter_nd  # noqa: F401
from .tensor.manipulation import shard_index  # noqa: F401
from .tensor.manipulation import slice  # noqa: F401
from .tensor.manipulation import crop  # noqa: F401
from .tensor.manipulation import split  # noqa: F401
from .tensor.manipulation import vsplit  # noqa: F401
from .tensor.manipulation import squeeze  # noqa: F401
from .tensor.manipulation import squeeze_  # noqa: F401
from .tensor.manipulation import stack  # noqa: F401
from .tensor.manipulation import strided_slice  # noqa: F401
from .tensor.manipulation import unique  # noqa: F401
from .tensor.manipulation import unique_consecutive  # noqa: F401
from .tensor.manipulation import unsqueeze  # noqa: F401
from .tensor.manipulation import unsqueeze_  # noqa: F401
from .tensor.manipulation import unstack  # noqa: F401
from .tensor.manipulation import flip  # noqa: F401
from .tensor.manipulation import rot90  # noqa: F401
from .tensor.manipulation import unbind  # noqa: F401
from .tensor.manipulation import roll  # noqa: F401
from .tensor.manipulation import chunk  # noqa: F401
from .tensor.manipulation import tolist  # noqa: F401
from .tensor.manipulation import take_along_axis  # noqa: F401
from .tensor.manipulation import put_along_axis  # noqa: F401
from .tensor.manipulation import tensordot  # noqa: F401
from .tensor.manipulation import as_complex  # noqa: F401
from .tensor.manipulation import as_real  # noqa: F401
from .tensor.manipulation import moveaxis  # noqa: F401
from .tensor.manipulation import repeat_interleave  # noqa: F401
from .tensor.manipulation import index_add  # noqa: F401
from .tensor.manipulation import index_add_  # noqa: F401
from .tensor.math import abs  # noqa: F401
from .tensor.math import acos  # noqa: F401
from .tensor.math import asin  # noqa: F401
from .tensor.math import atan  # noqa: F401
from .tensor.math import atan2  # noqa: F401
from .tensor.math import ceil  # noqa: F401
from .tensor.math import cos  # noqa: F401
from .tensor.math import tan  # noqa: F401
from .tensor.math import cosh  # noqa: F401
from .tensor.math import cumsum  # noqa: F401
from .tensor.math import cumprod  # noqa: F401
from .tensor.math import logcumsumexp  # noqa: F401
from .tensor.math import logit  # noqa: F401
from .tensor.math import exp  # noqa: F401
from .tensor.math import expm1  # noqa: F401
from .tensor.math import floor  # noqa: F401
from .tensor.math import increment  # noqa: F401
from .tensor.math import log  # noqa: F401
from .tensor.math import log2  # noqa: F401
from .tensor.math import log10  # noqa: F401
from .tensor.math import multiplex  # noqa: F401
from .tensor.math import pow  # noqa: F401
from .tensor.math import reciprocal  # noqa: F401
from .tensor.math import all  # noqa: F401
from .tensor.math import any  # noqa: F401
from .tensor.math import round  # noqa: F401
from .tensor.math import rsqrt  # noqa: F401
from .tensor.math import scale  # noqa: F401
from .tensor.math import sign  # noqa: F401
from .tensor.math import sin  # noqa: F401
from .tensor.math import sinh  # noqa: F401
from .tensor.math import sqrt  # noqa: F401
from .tensor.math import square  # noqa: F401
from .tensor.math import stanh  # noqa: F401
from .tensor.math import sum  # noqa: F401
from .tensor.math import nansum  # noqa: F401
from .tensor.math import nanmean  # noqa: F401
from .tensor.math import count_nonzero  # noqa: F401
from .tensor.math import tanh  # noqa: F401
from .tensor.math import tanh_  # noqa: F401
from .tensor.math import add_n  # noqa: F401
from .tensor.math import max  # noqa: F401
from .tensor.math import maximum  # noqa: F401
from .tensor.math import amax  # noqa: F401
from .tensor.math import min  # noqa: F401
from .tensor.math import minimum  # noqa: F401
from .tensor.math import amin  # noqa: F401
from .tensor.math import mm  # noqa: F401
from .tensor.math import divide  # noqa: F401
from .tensor.math import floor_divide  # noqa: F401
from .tensor.math import remainder  # noqa: F401
from .tensor.math import remainder_  # noqa: F401
from .tensor.math import mod  # noqa: F401
from .tensor.math import floor_mod  # noqa: F401
from .tensor.math import multiply  # noqa: F401
from .tensor.math import renorm  # noqa: F401
from .tensor.math import add  # noqa: F401
from .tensor.math import subtract  # noqa: F401
from .tensor.math import logsumexp  # noqa: F401
from .tensor.math import inverse  # noqa: F401
from .tensor.math import log1p  # noqa: F401
from .tensor.math import erf  # noqa: F401
from .tensor.math import addmm  # noqa: F401
from .tensor.math import clip  # noqa: F401
from .tensor.math import trace  # noqa: F401
from .tensor.math import diagonal  # noqa: F401
from .tensor.math import kron  # noqa: F401
from .tensor.math import isfinite  # noqa: F401
from .tensor.math import isinf  # noqa: F401
from .tensor.math import isnan  # noqa: F401
from .tensor.math import prod  # noqa: F401
from .tensor.math import broadcast_shape  # noqa: F401
from .tensor.math import conj  # noqa: F401
from .tensor.math import trunc  # noqa: F401
from .tensor.math import digamma  # noqa: F401
from .tensor.math import neg  # noqa: F401
from .tensor.math import lgamma  # noqa: F401
from .tensor.math import acosh  # noqa: F401
from .tensor.math import asinh  # noqa: F401
from .tensor.math import atanh  # noqa: F401
from .tensor.math import lerp  # noqa: F401
from .tensor.math import erfinv  # noqa: F401
from .tensor.math import rad2deg  # noqa: F401
from .tensor.math import deg2rad  # noqa: F401
from .tensor.math import gcd  # noqa: F401
from .tensor.math import lcm  # noqa: F401
from .tensor.math import diff  # noqa: F401
from .tensor.math import angle  # noqa: F401
from .tensor.math import fmax  # noqa: F401
from .tensor.math import fmin  # noqa: F401
from .tensor.math import inner  # noqa: F401
from .tensor.math import outer  # noqa: F401
from .tensor.math import heaviside  # noqa: F401
from .tensor.math import frac  # noqa: F401
from .tensor.math import sgn  # noqa: F401
from .tensor.math import take  # noqa: F401

from .tensor.random import bernoulli  # noqa: F401
from .tensor.random import poisson  # noqa: F401
from .tensor.random import multinomial  # noqa: F401
from .tensor.random import standard_normal  # noqa: F401
from .tensor.random import normal  # noqa: F401
from .tensor.random import uniform  # noqa: F401
from .tensor.random import randn  # noqa: F401
from .tensor.random import rand  # noqa: F401
from .tensor.random import randint  # noqa: F401
from .tensor.random import randint_like  # noqa: F401
from .tensor.random import randperm  # noqa: F401
from .tensor.search import argmax  # noqa: F401
from .tensor.search import argmin  # noqa: F401
from .tensor.search import argsort  # noqa: F401
from .tensor.search import searchsorted  # noqa: F401
from .tensor.search import bucketize  # noqa: F401
from .tensor.search import masked_select  # noqa: F401
from .tensor.search import topk  # noqa: F401
from .tensor.search import where  # noqa: F401
from .tensor.search import index_select  # noqa: F401
from .tensor.search import nonzero  # noqa: F401
from .tensor.search import sort  # noqa: F401
from .tensor.search import kthvalue  # noqa: F401
from .tensor.search import mode  # noqa: F401

from .tensor.to_string import set_printoptions  # noqa: F401

from .tensor.einsum import einsum  # noqa: F401

from .framework.random import seed  # noqa: F401
from .framework.random import get_cuda_rng_state  # noqa: F401
from .framework.random import set_cuda_rng_state  # noqa: F401
from .framework import ParamAttr  # noqa: F401
from .framework import create_parameter  # noqa: F401
from .framework import CPUPlace  # noqa: F401
from .framework import IPUPlace  # noqa: F401
from .framework import CUDAPlace  # noqa: F401
from .framework import NPUPlace  # noqa: F401
from .framework import CUDAPinnedPlace  # noqa: F401
from .framework import MLUPlace  # noqa: F401
from .framework import CustomPlace  # noqa: F401

from .autograd import grad  # noqa: F401
from .autograd import no_grad  # noqa: F401
from .autograd import set_grad_enabled  # noqa: F401
from .autograd import is_grad_enabled  # noqa: F401
from .framework import save  # noqa: F401
from .framework import load  # noqa: F401
from .framework import DataParallel  # noqa: F401

from .framework import set_default_dtype  # noqa: F401
from .framework import get_default_dtype  # noqa: F401

from .tensor.search import index_sample  # noqa: F401
from .tensor.stat import mean  # noqa: F401
from .tensor.stat import std  # noqa: F401
from .tensor.stat import var  # noqa: F401
from .tensor.stat import numel  # noqa: F401
from .tensor.stat import median  # noqa: F401
from .tensor.stat import nanmedian  # noqa: F401
from .tensor.stat import quantile  # noqa: F401
from .tensor.stat import nanquantile  # noqa: F401
from .device import get_cudnn_version  # noqa: F401
from .device import set_device  # noqa: F401
from .device import get_device  # noqa: F401
from .device import is_compiled_with_xpu  # noqa: F401
from .device import is_compiled_with_npu  # noqa: F401
from .device import is_compiled_with_ipu  # noqa: F401
from .device import is_compiled_with_mlu  # noqa: F401
from .device import is_compiled_with_cinn  # noqa: F401
from .device import is_compiled_with_cuda  # noqa: F401
from .device import is_compiled_with_rocm  # noqa: F401
from .device import XPUPlace  # noqa: F401

# high-level api
from .hapi import Model  # noqa: F401
from . import callbacks  # noqa: F401
from .hapi import summary  # noqa: F401
from .hapi import flops  # noqa: F401
from . import hub  # noqa: F401
from . import linalg  # noqa: F401
from . import fft  # noqa: F401
from . import signal  # noqa: F401

import paddle.text  # noqa: F401
import paddle.vision  # noqa: F401

from .tensor.random import check_shape  # noqa: F401

# CINN has to set a flag to include a lib
if is_compiled_with_cinn():
    import os
    package_dir = os.path.dirname(os.path.abspath(__file__))
    runtime_include_dir = os.path.join(package_dir, "libs")
    cuh_file = os.path.join(runtime_include_dir, "cinn_cuda_runtime_source.cuh")
    if os.path.exists(cuh_file):
        os.environ.setdefault('runtime_include_dir', runtime_include_dir)

disable_static()

__all__ = [  # noqa
    'iinfo',
    'dtype',
    'uint8',
    'int8',
    'int16',
    'int32',
    'int64',
    'float16',
    'float32',
    'float64',
    'bfloat16',
    'bool',
    'complex64',
    'complex128',
    'addmm',
    'allclose',
    'isclose',
    't',
    'add',
    'subtract',
    'diag',
    'diagflat',
    'isnan',
    'scatter_nd_add',
    'unstack',
    'get_default_dtype',
    'save',
    'multinomial',
    'get_cuda_rng_state',
    'rank',
    'empty_like',
    'eye',
    'cumsum',
    'cumprod',
    'logcumsumexp',
    'logit',
    'LazyGuard',
    'sign',
    'is_empty',
    'equal',
    'equal_all',
    'is_tensor',
    'is_complex',
    'is_integer',
    'cross',
    'where',
    'log1p',
    'cos',
    'tan',
    'mean',
    'mode',
    'mv',
    'in_dynamic_mode',
    'min',
    'amin',
    'any',
    'slice',
    'normal',
    'logsumexp',
    'full',
    'unsqueeze',
    'unsqueeze_',
    'argmax',
    'Model',
    'summary',
    'flops',
    'sort',
    'searchsorted',
    'bucketize',
    'split',
    'vsplit',
    'logical_and',
    'full_like',
    'less_than',
    'kron',
    'clip',
    'Tensor',
    'crop',
    'ParamAttr',
    'stanh',
    'randint',
    'randint_like',
    'assign',
    'gather',
    'scale',
    'zeros',
    'rsqrt',
    'squeeze',
    'squeeze_',
    'to_tensor',
    'gather_nd',
    'isinf',
    'uniform',
    'floor_divide',
    'remainder',
    'floor_mod',
    'roll',
    'batch',
    'max',
    'amax',
    'logical_or',
    'bitwise_and',
    'bitwise_or',
    'bitwise_xor',
    'bitwise_not',
    'mm',
    'flip',
    'rot90',
    'bincount',
    'histogram',
    'multiplex',
    'CUDAPlace',
    'NPUPlace',
    'empty',
    'shape',
    'real',
    'imag',
    'is_floating_point',
    'complex',
    'reciprocal',
    'rand',
    'less_equal',
    'triu',
    'sin',
    'dist',
    'unbind',
    'meshgrid',
    'arange',
    'load',
    'numel',
    'median',
    'nanmedian',
    'quantile',
    'nanquantile',
    'no_grad',
    'set_grad_enabled',
    'is_grad_enabled',
    'mod',
    'abs',
    'tril',
    'pow',
    'zeros_like',
    'maximum',
    'topk',
    'index_select',
    'CPUPlace',
    'matmul',
    'seed',
    'acos',
    'logical_xor',
    'exp',
    'expm1',
    'bernoulli',
    'poisson',
    'sinh',
    'round',
    'DataParallel',
    'argmin',
    'prod',
    'broadcast_shape',
    'conj',
    'neg',
    'lgamma',
    'lerp',
    'erfinv',
    'inner',
    'outer',
    'square',
    'divide',
    'ceil',
    'atan',
    'atan2',
    'rad2deg',
    'deg2rad',
    'gcd',
    'lcm',
    'expand',
    'broadcast_to',
    'ones_like',
    'index_sample',
    'cast',
    'grad',
    'all',
    'ones',
    'not_equal',
    'sum',
    'nansum',
    'nanmean',
    'count_nonzero',
    'tile',
    'greater_equal',
    'isfinite',
    'create_parameter',
    'dot',
    'increment',
    'erf',
    'bmm',
    'chunk',
    'tolist',
    'tensordot',
    'greater_than',
    'shard_index',
    'argsort',
    'tanh',
    'tanh_',
    'transpose',
    'randn',
    'strided_slice',
    'unique',
    'unique_consecutive',
    'set_cuda_rng_state',
    'set_printoptions',
    'std',
    'flatten',
    'asin',
    'multiply',
    'disable_static',
    'masked_select',
    'var',
    'trace',
    'enable_static',
    'scatter_nd',
    'set_default_dtype',
    'disable_signal_handler',
    'expand_as',
    'stack',
    'sqrt',
    'randperm',
    'linspace',
    'logspace',
    'reshape',
    'reshape_',
    'reverse',
    'nonzero',
    'CUDAPinnedPlace',
    'logical_not',
    'add_n',
    'minimum',
    'scatter',
    'scatter_',
    'floor',
    'cosh',
    'log',
    'log2',
    'log10',
    'concat',
    'check_shape',
    'trunc',
    'frac',
    'digamma',
    'standard_normal',
    'diagonal',
    'broadcast_tensors',
    'einsum',
    'set_flags',
    'get_flags',
    'asinh',
    'acosh',
    'atanh',
    'as_complex',
    'as_real',
    'diff',
    'angle',
    'fmax',
    'fmin',
    'moveaxis',
    'repeat_interleave',
    'clone',
    'kthvalue',
    'renorm',
    'take_along_axis',
    'put_along_axis',
    'heaviside',
    'tril_indices',
    'index_add',
    "index_add_",
    'sgn',
    'triu_indices',
    'take',
]
