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
    from paddle.version import full_version as __version__  # noqa: F401
    from paddle.version import commit as __git_commit__  # noqa: F401
    from paddle.cuda_env import *  # noqa: F403
except ImportError:
    import sys

    sys.stderr.write(
        '''Warning with import paddle: you should not
     import paddle from the source directory; please install paddlepaddle*.whl firstly.'''
    )

from .batch import batch

# Do the *DUPLICATED* monkey-patch for the tensor object.
# We need remove the duplicated code here once we fix
# the illogical implement in the monkey-patch methods later.
from .framework import monkey_patch_variable
from .framework import monkey_patch_math_tensor

monkey_patch_variable()
monkey_patch_math_tensor()

from .framework import (
    disable_signal_handler,
    get_flags,
    set_flags,
    disable_static,
    enable_static,
    in_dynamic_mode,
)
from .base.dataset import *  # noqa: F403

from .framework.dtype import (
    iinfo,
    finfo,
    dtype,
    uint8,
    int8,
    int16,
    int32,
    int64,
    float16,
    float32,
    float64,
    bfloat16,
    bool,
    complex64,
    complex128,
)

Tensor = framework.core.eager.Tensor
Tensor.__qualname__ = 'Tensor'

import paddle.distributed.fleet  # noqa: F401
import paddle.optimizer  # noqa: F401
import paddle.metric  # noqa: F401
import paddle.regularizer  # noqa: F401
import paddle.incubate  # noqa: F401
import paddle.autograd  # noqa: F401
import paddle.device  # noqa: F401
import paddle.decomposition  # noqa: F401

import paddle.jit  # noqa: F401
import paddle.amp  # noqa: F401
import paddle.dataset  # noqa: F401
import paddle.inference  # noqa: F401
import paddle.io  # noqa: F401
import paddle.onnx  # noqa: F401
import paddle.reader  # noqa: F401
import paddle.static  # noqa: F401
import paddle.vision  # noqa: F401
import paddle.audio  # noqa: F401
import paddle.geometric  # noqa: F401
import paddle.sparse  # noqa: F401
import paddle.quantization  # noqa: F401

from .tensor.attribute import is_complex  # noqa: F401
from .tensor.attribute import is_integer  # noqa: F401
from .tensor.attribute import rank  # noqa: F401
from .tensor.attribute import shape  # noqa: F401
from .tensor.attribute import real  # noqa: F401
from .tensor.attribute import imag  # noqa: F401
from .tensor.attribute import is_floating_point  # noqa: F401
from .tensor.creation import create_parameter  # noqa: F401
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
from .tensor.creation import triu_  # noqa: F401
from .tensor.creation import tril  # noqa: F401
from .tensor.creation import tril_  # noqa: F401
from .tensor.creation import meshgrid  # noqa: F401
from .tensor.creation import empty  # noqa: F401
from .tensor.creation import empty_like  # noqa: F401
from .tensor.creation import assign  # noqa: F401
from .tensor.creation import complex  # noqa: F401
from .tensor.creation import clone  # noqa: F401
from .tensor.creation import tril_indices  # noqa: F401
from .tensor.creation import triu_indices  # noqa: F401
from .tensor.creation import polar  # noqa: F401
from .tensor.creation import geometric_  # noqa: F401
from .tensor.creation import cauchy_  # noqa: F401
from .tensor.linalg import matmul  # noqa: F401
from .tensor.linalg import dot  # noqa: F401
from .tensor.linalg import norm  # noqa: F401
from .tensor.linalg import transpose  # noqa: F401
from .tensor.linalg import transpose_  # noqa: F401
from .tensor.linalg import dist  # noqa: F401
from .tensor.linalg import t  # noqa: F401
from .tensor.linalg import t_  # noqa: F401
from .tensor.linalg import cdist  # noqa: F401
from .tensor.linalg import cross  # noqa: F401
from .tensor.linalg import cholesky  # noqa: F401
from .tensor.linalg import bmm  # noqa: F401
from .tensor.linalg import histogram  # noqa: F401
from .tensor.linalg import bincount  # noqa: F401
from .tensor.linalg import mv  # noqa: F401
from .tensor.logic import equal  # noqa: F401
from .tensor.logic import equal_  # noqa: F401
from .tensor.linalg import eigvalsh  # noqa: F401
from .tensor.logic import greater_equal  # noqa: F401
from .tensor.logic import greater_equal_  # noqa: F401
from .tensor.logic import greater_than  # noqa: F401
from .tensor.logic import greater_than_  # noqa: F401
from .tensor.logic import is_empty  # noqa: F401
from .tensor.logic import less_equal  # noqa: F401
from .tensor.logic import less_equal_  # noqa: F401
from .tensor.logic import less_than  # noqa: F401
from .tensor.logic import less_than_  # noqa: F401
from .tensor.logic import logical_and  # noqa: F401
from .tensor.logic import logical_and_  # noqa: F401
from .tensor.logic import logical_not  # noqa: F401
from .tensor.logic import logical_not_  # noqa: F401
from .tensor.logic import logical_or  # noqa: F401
from .tensor.logic import logical_or_  # noqa: F401
from .tensor.logic import logical_xor  # noqa: F401
from .tensor.logic import logical_xor_  # noqa: F401
from .tensor.logic import bitwise_and  # noqa: F401
from .tensor.logic import bitwise_and_  # noqa: F401
from .tensor.logic import bitwise_not  # noqa: F401
from .tensor.logic import bitwise_not_  # noqa: F401
from .tensor.logic import bitwise_or  # noqa: F401
from .tensor.logic import bitwise_or_  # noqa: F401
from .tensor.logic import bitwise_xor  # noqa: F401
from .tensor.logic import bitwise_xor_  # noqa: F401
from .tensor.logic import not_equal  # noqa: F401
from .tensor.logic import not_equal_  # noqa: F401
from .tensor.logic import allclose  # noqa: F401
from .tensor.logic import isclose  # noqa: F401
from .tensor.logic import equal_all  # noqa: F401
from .tensor.logic import is_tensor  # noqa: F401
from .tensor.manipulation import cast  # noqa: F401
from .tensor.manipulation import cast_  # noqa: F401
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
from .tensor.manipulation import index_put  # noqa: F401
from .tensor.manipulation import index_put_  # noqa: F401
from .tensor.manipulation import unflatten  # noqa: F401
from .tensor.manipulation import as_strided  # noqa: F401
from .tensor.manipulation import view  # noqa: F401
from .tensor.manipulation import view_as  # noqa: F401
from .tensor.manipulation import unfold  # noqa: F401
from .tensor.manipulation import masked_fill  # noqa: F401
from .tensor.manipulation import masked_fill_  # noqa: F401
from .tensor.math import abs  # noqa: F401
from .tensor.math import abs_  # noqa: F401
from .tensor.math import acos  # noqa: F401
from .tensor.math import acos_  # noqa: F401
from .tensor.math import asin  # noqa: F401
from .tensor.math import asin_  # noqa: F401
from .tensor.math import atan  # noqa: F401
from .tensor.math import atan_  # noqa: F401
from .tensor.math import atan2  # noqa: F401
from .tensor.math import ceil  # noqa: F401
from .tensor.math import cos  # noqa: F401
from .tensor.math import cos_  # noqa: F401
from .tensor.math import tan  # noqa: F401
from .tensor.math import tan_  # noqa: F401
from .tensor.math import cosh  # noqa: F401
from .tensor.math import cosh_  # noqa: F401
from .tensor.math import cumsum  # noqa: F401
from .tensor.math import cumsum_  # noqa: F401
from .tensor.math import cummax  # noqa: F401
from .tensor.math import cummin  # noqa: F401
from .tensor.math import cumprod  # noqa: F401
from .tensor.math import cumprod_  # noqa: F401
from .tensor.math import logcumsumexp  # noqa: F401
from .tensor.math import logit  # noqa: F401
from .tensor.math import logit_  # noqa: F401
from .tensor.math import exp  # noqa: F401
from .tensor.math import expm1  # noqa: F401
from .tensor.math import expm1_  # noqa: F401
from .tensor.math import floor  # noqa: F401
from .tensor.math import increment  # noqa: F401
from .tensor.math import log  # noqa: F401
from .tensor.math import log_  # noqa: F401
from .tensor.math import log2_  # noqa: F401
from .tensor.math import log2  # noqa: F401
from .tensor.math import log10  # noqa: F401
from .tensor.math import log10_  # noqa: F401
from .tensor.math import multiplex  # noqa: F401
from .tensor.math import pow  # noqa: F401
from .tensor.math import pow_  # noqa: F401
from .tensor.math import reciprocal  # noqa: F401
from .tensor.math import all  # noqa: F401
from .tensor.math import any  # noqa: F401
from .tensor.math import round  # noqa: F401
from .tensor.math import rsqrt  # noqa: F401
from .tensor.math import scale  # noqa: F401
from .tensor.math import sign  # noqa: F401
from .tensor.math import sin  # noqa: F401
from .tensor.math import sin_  # noqa: F401
from .tensor.math import sinh  # noqa: F401
from .tensor.math import sinh_  # noqa: F401
from .tensor.math import sqrt  # noqa: F401
from .tensor.math import square  # noqa: F401
from .tensor.math import square_  # noqa: F401
from .tensor.math import stanh  # noqa: F401
from .tensor.math import sum  # noqa: F401
from .tensor.math import nan_to_num  # noqa: F401
from .tensor.math import nan_to_num_  # noqa: F401
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
from .tensor.math import divide_  # noqa: F401
from .tensor.math import floor_divide  # noqa: F401
from .tensor.math import floor_divide_  # noqa: F401
from .tensor.math import remainder  # noqa: F401
from .tensor.math import remainder_  # noqa: F401
from .tensor.math import mod  # noqa: F401
from .tensor.math import mod_  # noqa: F401
from .tensor.math import floor_mod  # noqa: F401
from .tensor.math import floor_mod_  # noqa: F401
from .tensor.math import multiply  # noqa: F401
from .tensor.math import multiply_  # noqa: F401
from .tensor.math import renorm  # noqa: F401
from .tensor.math import renorm_  # noqa: F401
from .tensor.math import add  # noqa: F401
from .tensor.math import subtract  # noqa: F401
from .tensor.math import logsumexp  # noqa: F401
from .tensor.math import logaddexp  # noqa: F401
from .tensor.math import inverse  # noqa: F401
from .tensor.math import log1p  # noqa: F401
from .tensor.math import log1p_  # noqa: F401
from .tensor.math import erf  # noqa: F401
from .tensor.math import erf_  # noqa: F401
from .tensor.math import addmm  # noqa: F401
from .tensor.math import addmm_  # noqa: F401
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
from .tensor.math import trunc_  # noqa: F401
from .tensor.math import digamma  # noqa: F401
from .tensor.math import digamma_  # noqa: F401
from .tensor.math import neg  # noqa: F401
from .tensor.math import neg_  # noqa: F401
from .tensor.math import lgamma  # noqa: F401
from .tensor.math import lgamma_  # noqa: F401
from .tensor.math import acosh  # noqa: F401
from .tensor.math import acosh_  # noqa: F401
from .tensor.math import asinh  # noqa: F401
from .tensor.math import asinh_  # noqa: F401
from .tensor.math import atanh  # noqa: F401
from .tensor.math import atanh_  # noqa: F401
from .tensor.math import lerp  # noqa: F401
from .tensor.math import erfinv  # noqa: F401
from .tensor.math import rad2deg  # noqa: F401
from .tensor.math import deg2rad  # noqa: F401
from .tensor.math import gcd  # noqa: F401
from .tensor.math import gcd_  # noqa: F401
from .tensor.math import lcm  # noqa: F401
from .tensor.math import lcm_  # noqa: F401
from .tensor.math import diff  # noqa: F401
from .tensor.math import angle  # noqa: F401
from .tensor.math import fmax  # noqa: F401
from .tensor.math import fmin  # noqa: F401
from .tensor.math import inner  # noqa: F401
from .tensor.math import outer  # noqa: F401
from .tensor.math import heaviside  # noqa: F401
from .tensor.math import frac  # noqa: F401
from .tensor.math import frac_  # noqa: F401
from .tensor.math import sgn  # noqa: F401
from .tensor.math import take  # noqa: F401
from .tensor.math import frexp  # noqa: F401
from .tensor.math import ldexp  # noqa: F401
from .tensor.math import ldexp_  # noqa: F401
from .tensor.math import trapezoid  # noqa: F401
from .tensor.math import cumulative_trapezoid  # noqa: F401
from .tensor.math import vander  # noqa: F401
from .tensor.math import nextafter  # noqa: F401
from .tensor.math import i0  # noqa: F401
from .tensor.math import i0_  # noqa: F401
from .tensor.math import i0e  # noqa: F401
from .tensor.math import i1  # noqa: F401
from .tensor.math import i1e  # noqa: F401
from .tensor.math import polygamma  # noqa: F401
from .tensor.math import polygamma_  # noqa: F401

from .tensor.random import bernoulli  # noqa: F401
from .tensor.random import poisson  # noqa: F401
from .tensor.random import multinomial  # noqa: F401
from .tensor.random import standard_normal  # noqa: F401
from .tensor.random import normal  # noqa: F401
from .tensor.random import normal_  # noqa: F401
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
from .tensor.search import where_  # noqa: F401
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
from .framework.random import get_rng_state  # noqa: F401
from .framework.random import set_rng_state  # noqa: F401
from .framework import ParamAttr  # noqa: F401
from .framework import CPUPlace  # noqa: F401
from .framework import IPUPlace  # noqa: F401
from .framework import CUDAPlace  # noqa: F401
from .framework import CUDAPinnedPlace  # noqa: F401
from .framework import CustomPlace  # noqa: F401
from .framework import XPUPlace  # noqa: F401

from .autograd import grad  # noqa: F401
from .autograd import no_grad  # noqa: F401
from .autograd import enable_grad  # noqa:F401
from .autograd import set_grad_enabled  # noqa: F401
from .autograd import is_grad_enabled  # noqa: F401
from .framework import save  # noqa: F401
from .framework import load  # noqa: F401
from .distributed import DataParallel  # noqa: F401

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
from .device import is_compiled_with_ipu  # noqa: F401
from .device import is_compiled_with_cinn  # noqa: F401
from .device import is_compiled_with_cuda  # noqa: F401
from .device import is_compiled_with_rocm  # noqa: F401
from .device import is_compiled_with_custom_device  # noqa: F401
=======

from paddle import (  # noqa: F401
    distributed,
    sysconfig,
    distribution,
    nn,
    optimizer,
    metric,
    regularizer,
    incubate,
    autograd,
    device,
    decomposition,
    jit,
    amp,
    dataset,
    inference,
    io,
    onnx,
    reader,
    static,
    vision,
    audio,
    geometric,
    sparse,
    quantization,
)

from .tensor.attribute import (
    is_complex,
    is_integer,
    rank,
    shape,
    real,
    imag,
    is_floating_point,
)

from .tensor.creation import (
    create_parameter,
    to_tensor,
    diag,
    diagflat,
    eye,
    linspace,
    logspace,
    ones,
    ones_like,
    zeros,
    zeros_like,
    arange,
    full,
    full_like,
    triu,
    triu_,
    tril,
    tril_,
    meshgrid,
    empty,
    empty_like,
    assign,
    complex,
    clone,
    tril_indices,
    triu_indices,
    polar,
    geometric_,
    cauchy_,
)

from .tensor.linalg import (  # noqa: F401
    matmul,
    dot,
    norm,
    transpose,
    transpose_,
    dist,
    t,
    t_,
    cdist,
    cross,
    cholesky,
    bmm,
    histogram,
    bincount,
    mv,
    eigvalsh,
)

from .tensor.logic import (  # noqa: F401
    equal,
    equal_,
    greater_equal,
    greater_equal_,
    greater_than,
    greater_than_,
    is_empty,
    less_equal,
    less_equal_,
    less_than,
    less_than_,
    logical_and,
    logical_and_,
    logical_not,
    logical_not_,
    logical_or,
    logical_or_,
    logical_xor,
    logical_xor_,
    bitwise_and,
    bitwise_and_,
    bitwise_not,
    bitwise_not_,
    bitwise_or,
    bitwise_or_,
    bitwise_xor,
    bitwise_xor_,
    not_equal,
    not_equal_,
    allclose,
    isclose,
    equal_all,
    is_tensor,
)


from .tensor.manipulation import (  # noqa: F401
    cast,
    cast_,
    concat,
    broadcast_tensors,
    expand,
    broadcast_to,
    expand_as,
    tile,
    flatten,
    gather,
    gather_nd,
    reshape,
    reshape_,
    flip as reverse,
    scatter,
    scatter_,
    scatter_nd_add,
    scatter_nd,
    shard_index,
    slice,
    crop,
    split,
    vsplit,
    squeeze,
    squeeze_,
    stack,
    strided_slice,
    unique,
    unique_consecutive,
    unsqueeze,
    unsqueeze_,
    unstack,
    flip,
    rot90,
    unbind,
    roll,
    chunk,
    tolist,
    take_along_axis,
    put_along_axis,
    tensordot,
    as_complex,
    as_real,
    moveaxis,
    repeat_interleave,
    index_add,
    index_add_,
    index_put,
    index_put_,
    unflatten,
    as_strided,
    view,
    view_as,
    unfold,
)

from .tensor.math import (  # noqa: F401
    abs,
    abs_,
    acos,
    acos_,
    asin,
    asin_,
    atan,
    atan_,
    atan2,
    ceil,
    cos,
    cos_,
    tan,
    tan_,
    cosh,
    cosh_,
    cumsum,
    cumsum_,
    cummax,
    cummin,
    cumprod,
    cumprod_,
    logcumsumexp,
    logit,
    logit_,
    exp,
    expm1,
    expm1_,
    floor,
    increment,
    log,
    log_,
    log2_,
    log2,
    log10,
    log10_,
    multiplex,
    pow,
    pow_,
    reciprocal,
    all,
    any,
    round,
    rsqrt,
    scale,
    sign,
    sin,
    sin_,
    sinh,
    sinh_,
    sqrt,
    square,
    square_,
    stanh,
    sum,
    nan_to_num,
    nan_to_num_,
    nansum,
    nanmean,
    count_nonzero,
    tanh,
    tanh_,
    add_n,
    max,
    maximum,
    amax,
    min,
    minimum,
    amin,
    mm,
    divide,
    divide_,
    floor_divide,
    floor_divide_,
    remainder,
    remainder_,
    mod,
    mod_,
    floor_mod,
    floor_mod_,
    multiply,
    multiply_,
    renorm,
    renorm_,
    add,
    subtract,
    logsumexp,
    logaddexp,
    inverse,
    log1p,
    log1p_,
    erf,
    erf_,
    addmm,
    addmm_,
    clip,
    trace,
    diagonal,
    kron,
    isfinite,
    isinf,
    isnan,
    prod,
    broadcast_shape,
    conj,
    trunc,
    trunc_,
    digamma,
    digamma_,
    neg,
    neg_,
    lgamma,
    lgamma_,
    acosh,
    acosh_,
    asinh,
    asinh_,
    atanh,
    atanh_,
    lerp,
    erfinv,
    rad2deg,
    deg2rad,
    gcd,
    gcd_,
    lcm,
    lcm_,
    diff,
    angle,
    fmax,
    fmin,
    inner,
    outer,
    heaviside,
    frac,
    frac_,
    sgn,
    take,
    frexp,
    ldexp,
    ldexp_,
    trapezoid,
    cumulative_trapezoid,
    vander,
    nextafter,
    i0,
    i0_,
    i0e,
    i1,
    i1e,
    polygamma,
    polygamma_,
)

from .tensor.random import (
    bernoulli,
    poisson,
    multinomial,
    standard_normal,
    normal,
    normal_,
    uniform,
    randn,
    rand,
    randint,
    randint_like,
    randperm,
)
from .tensor.search import (
    argmax,
    argmin,
    argsort,
    searchsorted,
    bucketize,
    masked_select,
    topk,
    where,
    where_,
    index_select,
    nonzero,
    sort,
    kthvalue,
    mode,
)

from .tensor.to_string import set_printoptions

from .tensor.einsum import einsum

from .framework.random import (
    seed,
    get_cuda_rng_state,
    set_cuda_rng_state,
    get_rng_state,
    set_rng_state,
)
from .framework import (  # noqa: F401
    ParamAttr,
    CPUPlace,
    IPUPlace,
    CUDAPlace,
    CUDAPinnedPlace,
    CustomPlace,
    XPUPlace,
)

from .autograd import (
    grad,
    no_grad,
    enable_grad,
    set_grad_enabled,
    is_grad_enabled,
)
from .framework import (
    save,
    load,
)
from .distributed import DataParallel

from .framework import (
    set_default_dtype,
    get_default_dtype,
)

from .tensor.search import index_sample
from .tensor.stat import (
    mean,
    std,
    var,
    numel,
    median,
    nanmedian,
    quantile,
    nanquantile,
)
from .device import (  # noqa: F401
    get_cudnn_version,
    set_device,
    get_device,
    is_compiled_with_xpu,
    is_compiled_with_ipu,
    is_compiled_with_cinn,
    is_compiled_with_cuda,
    is_compiled_with_rocm,
    is_compiled_with_custom_device,
)

# high-level api
from . import (  # noqa: F401
    callbacks,
    hub,
    linalg,
    fft,
    signal,
    _pir_ops,
)
from .hapi import (
    Model,
    summary,
    flops,
)

import paddle.text  # noqa: F401
import paddle.vision  # noqa: F401

from .tensor.random import check_shape
from .nn.initializer.lazy_init import LazyGuard

# CINN has to set a flag to include a lib
if is_compiled_with_cinn():
    import os

    package_dir = os.path.dirname(os.path.abspath(__file__))
    runtime_include_dir = os.path.join(package_dir, "libs")
    cuh_file = os.path.join(runtime_include_dir, "cinn_cuda_runtime_source.cuh")
    if os.path.exists(cuh_file):
        os.environ.setdefault('runtime_include_dir', runtime_include_dir)

disable_static()

from .pir_utils import IrGuard

ir_change = IrGuard()
ir_change._switch_to_pir()

__all__ = [
    'iinfo',
    'finfo',
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
    'addmm_',
    'allclose',
    'isclose',
    't',
    't_',
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
    'get_rng_state',
    'rank',
    'empty_like',
    'eye',
    'cumsum',
    'cumsum_',
    'cummax',
    'cummin',
    'cumprod',
    'cumprod_',
    'logaddexp',
    'logcumsumexp',
    'logit',
    'logit_',
    'LazyGuard',
    'sign',
    'is_empty',
    'equal',
    'equal_',
    'equal_all',
    'is_tensor',
    'is_complex',
    'is_integer',
    'cross',
    'where',
    'where_',
    'log1p',
    'cos',
    'cos_',
    'tan',
    'tan_',
    'mean',
    'mode',
    'mv',
    'in_dynamic_mode',
    'min',
    'amin',
    'any',
    'slice',
    'normal',
    'normal_',
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
    'logical_and_',
    'full_like',
    'less_than',
    'less_than_',
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
    'floor_divide_',
    'remainder',
    'remainder_',
    'floor_mod',
    'floor_mod_',
    'roll',
    'batch',
    'max',
    'amax',
    'logical_or',
    'logical_or_',
    'bitwise_and',
    'bitwise_and_',
    'bitwise_or',
    'bitwise_or_',
    'bitwise_xor',
    'bitwise_xor_',
    'bitwise_not',
    'bitwise_not_',
    'mm',
    'flip',
    'rot90',
    'bincount',
    'histogram',
    'multiplex',
    'CUDAPlace',
    'empty',
    'shape',
    'real',
    'imag',
    'is_floating_point',
    'complex',
    'reciprocal',
    'rand',
    'less_equal',
    'less_equal_',
    'triu',
    'triu_',
    'sin',
    'sin_',
    'dist',
    'cdist',
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
    'enable_grad',
    'set_grad_enabled',
    'is_grad_enabled',
    'mod',
    'mod_',
    'abs',
    'abs_',
    'tril',
    'tril_',
    'pow',
    'pow_',
    'zeros_like',
    'maximum',
    'topk',
    'index_select',
    'CPUPlace',
    'matmul',
    'seed',
    'acos',
    'acos_',
    'logical_xor',
    'exp',
    'expm1',
    'expm1_',
    'bernoulli',
    'poisson',
    'sinh',
    'sinh_',
    'round',
    'DataParallel',
    'argmin',
    'prod',
    'broadcast_shape',
    'conj',
    'neg',
    'neg_',
    'lgamma',
    'lgamma_',
    'lerp',
    'erfinv',
    'inner',
    'outer',
    'square',
    'square_',
    'divide',
    'divide_',
    'ceil',
    'atan',
    'atan_',
    'atan2',
    'rad2deg',
    'deg2rad',
    'gcd',
    'gcd_',
    'lcm',
    'lcm_',
    'expand',
    'broadcast_to',
    'ones_like',
    'index_sample',
    'cast',
    'cast_',
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
    'greater_equal_',
    'isfinite',
    'create_parameter',
    'dot',
    'increment',
    'erf',
    'erf_',
    'bmm',
    'chunk',
    'tolist',
    'tensordot',
    'greater_than',
    'greater_than_',
    'shard_index',
    'argsort',
    'tanh',
    'tanh_',
    'transpose',
    'transpose_',
    'cauchy_',
    'geometric_',
    'randn',
    'strided_slice',
    'unique',
    'unique_consecutive',
    'set_cuda_rng_state',
    'set_rng_state',
    'set_printoptions',
    'std',
    'flatten',
    'asin',
    'multiply',
    'multiply_',
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
    'logical_not_',
    'add_n',
    'minimum',
    'scatter',
    'scatter_',
    'floor',
    'cosh',
    'log',
    'log_',
    'log2',
    'log2_',
    'log10',
    'log10_',
    'concat',
    'check_shape',
    'trunc',
    'trunc_',
    'frac',
    'frac_',
    'digamma',
    'digamma_',
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
    'renorm_',
    'take_along_axis',
    'put_along_axis',
    'nan_to_num',
    'nan_to_num_',
    'heaviside',
    'tril_indices',
    'index_add',
    "index_add_",
    "index_put",
    "index_put_",
    'sgn',
    'triu_indices',
    'take',
    'frexp',
    'ldexp',
    'ldexp_',
    'trapezoid',
    'cumulative_trapezoid',
    'polar',
    'vander',
    'unflatten',
    'as_strided',
    'view',
    'view_as',
    'unfold',
    'nextafter',
    'i0',
    'i0_',
    'i0e',
    'i1',
    'i1e',
    'polygamma',
    'polygamma_',
    'masked_fill',
    'masked_fill_',
]
