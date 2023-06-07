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

from .attribute import is_complex  # noqa: F401
from .attribute import is_integer  # noqa: F401
from .attribute import rank  # noqa: F401
from .attribute import shape  # noqa: F401
from .attribute import real  # noqa: F401
from .attribute import imag  # noqa: F401
from .attribute import is_floating_point  # noqa: F401
from .creation import create_parameter  # noqa: F401
from .creation import create_tensor  # noqa: F401
from .creation import to_tensor  # noqa: F401
from .creation import diag  # noqa: F401
from .creation import diagflat  # noqa: F401
from .creation import eye  # noqa: F401
from .creation import linspace  # noqa: F401
from .creation import fill_constant  # noqa: F401
from .creation import ones  # noqa: F401
from .creation import ones_like  # noqa: F401
from .creation import zeros  # noqa: F401
from .creation import zeros_like  # noqa: F401
from .creation import arange  # noqa: F401
from .creation import full  # noqa: F401
from .creation import full_like  # noqa: F401
from .creation import triu  # noqa: F401
from .creation import tril  # noqa: F401
from .creation import meshgrid  # noqa: F401
from .creation import empty  # noqa: F401
from .creation import empty_like  # noqa: F401
from .creation import complex  # noqa: F401
from .creation import polar  # noqa: F401
from .linalg import matmul  # noqa: F401
from .linalg import dot  # noqa: F401
from .linalg import cov  # noqa: F401
from .linalg import corrcoef  # noqa: F401
from .linalg import norm  # noqa: F401
from .linalg import cond  # noqa: F401
from .linalg import transpose  # noqa: F401
from .linalg import lstsq  # noqa: F401
from .linalg import dist  # noqa: F401
from .linalg import t  # noqa: F401
from .linalg import cross  # noqa: F401
from .linalg import cholesky  # noqa: F401
from .linalg import bmm  # noqa: F401
from .linalg import histogram  # noqa: F401
from .linalg import bincount  # noqa: F401
from .linalg import mv  # noqa: F401
from .linalg import eig  # noqa: F401
from .linalg import matrix_power  # noqa: F401
from .linalg import qr  # noqa: F401
from .linalg import eigvals  # noqa: F401
from .linalg import multi_dot  # noqa: F401
from .linalg import svd  # noqa: F401
from .linalg import eigvalsh  # noqa: F401
from .linalg import eigh  # noqa: F401
from .linalg import pinv  # noqa: F401
from .linalg import solve  # noqa: F401
from .linalg import cholesky_solve  # noqa: F401
from .linalg import lu  # noqa: F401
from .linalg import lu_unpack  # noqa: F401
from .linalg import cdist  # noqa: F401
from .logic import equal  # noqa: F401
from .logic import greater_equal  # noqa: F401
from .logic import greater_than  # noqa: F401
from .logic import is_empty  # noqa: F401
from .logic import less_equal  # noqa: F401
from .logic import less_than  # noqa: F401
from .logic import logical_and  # noqa: F401
from .logic import logical_not  # noqa: F401
from .logic import logical_or  # noqa: F401
from .logic import logical_xor  # noqa: F401
from .logic import bitwise_and  # noqa: F401
from .logic import bitwise_or  # noqa: F401
from .logic import bitwise_xor  # noqa: F401
from .logic import bitwise_not  # noqa: F401
from .logic import not_equal  # noqa: F401
from .logic import allclose  # noqa: F401
from .logic import isclose  # noqa: F401
from .logic import equal_all  # noqa: F401
from .logic import is_tensor  # noqa: F401
from .manipulation import cast  # noqa: F401
from .manipulation import concat  # noqa: F401
from .manipulation import expand  # noqa: F401
from .manipulation import broadcast_to  # noqa: F401
from .manipulation import broadcast_tensors  # noqa: F401
from .manipulation import expand_as  # noqa: F401
from .manipulation import tile  # noqa: F401
from .manipulation import flatten  # noqa: F401
from .manipulation import flatten_  # noqa: F401
from .manipulation import gather  # noqa: F401
from .manipulation import gather_nd  # noqa: F401
from .manipulation import reshape  # noqa: F401
from .manipulation import reshape_  # noqa: F401
from .manipulation import flip as reverse  # noqa: F401
from .manipulation import scatter  # noqa: F401
from .manipulation import scatter_  # noqa: F401
from .manipulation import scatter_nd_add  # noqa: F401
from .manipulation import scatter_nd  # noqa: F401
from .manipulation import shard_index  # noqa: F401
from .manipulation import slice  # noqa: F401
from .manipulation import split  # noqa: F401
from .manipulation import vsplit  # noqa: F401
from .manipulation import squeeze  # noqa: F401
from .manipulation import squeeze_  # noqa: F401
from .manipulation import stack  # noqa: F401
from .manipulation import strided_slice  # noqa: F401
from .manipulation import unique  # noqa: F401
from .manipulation import unique_consecutive  # noqa: F401
from .manipulation import unsqueeze  # noqa: F401
from .manipulation import unsqueeze_  # noqa: F401
from .manipulation import unstack  # noqa: F401
from .manipulation import flip  # noqa: F401
from .manipulation import rot90  # noqa: F401
from .manipulation import unbind  # noqa: F401
from .manipulation import roll  # noqa: F401
from .manipulation import chunk  # noqa: F401
from .manipulation import tensordot  # noqa: F401
from .manipulation import as_complex  # noqa: F401
from .manipulation import take_along_axis  # noqa: F401
from .manipulation import put_along_axis  # noqa: F401
from .manipulation import put_along_axis_  # noqa: F401
from .manipulation import as_real  # noqa: F401
from .manipulation import moveaxis  # noqa: F401
from .manipulation import repeat_interleave  # noqa: F401
from .manipulation import index_add  # noqa: F401
from .manipulation import index_add_  # noqa: F401
from .manipulation import index_put  # noqa: F401
from .manipulation import index_put_  # noqa: F401
from .manipulation import unflatten  # noqa: F401
from .math import abs  # noqa: F401
from .math import acos  # noqa: F401
from .math import asin  # noqa: F401
from .math import atan  # noqa: F401
from .math import ceil  # noqa: F401
from .math import ceil_  # noqa: F401
from .math import cos  # noqa: F401
from .math import tan  # noqa: F401
from .math import cosh  # noqa: F401
from .math import cumsum  # noqa: F401
from .math import cumprod  # noqa: F401
from .math import logcumsumexp  # noqa: F401
from .math import logit  # noqa: F401
from .math import exp  # noqa: F401
from .math import exp_  # noqa: F401
from .math import expm1  # noqa: F401
from .math import floor  # noqa: F401
from .math import floor_  # noqa: F401
from .math import increment  # noqa: F401
from .math import log  # noqa: F401
from .math import multiplex  # noqa: F401
from .math import pow  # noqa: F401
from .math import reciprocal  # noqa: F401
from .math import reciprocal_  # noqa: F401
from .math import round  # noqa: F401
from .math import round_  # noqa: F401
from .math import rsqrt  # noqa: F401
from .math import rsqrt_  # noqa: F401
from .math import scale  # noqa: F401
from .math import scale_  # noqa: F401
from .math import sign  # noqa: F401
from .math import sin  # noqa: F401
from .math import sinh  # noqa: F401
from .math import sqrt  # noqa: F401
from .math import sqrt_  # noqa: F401
from .math import square  # noqa: F401
from .math import stanh  # noqa: F401
from .math import sum  # noqa: F401
from .math import nan_to_num  # noqa: F401
from .math import nansum  # noqa: F401
from .math import nanmean  # noqa: F401
from .math import count_nonzero  # noqa: F401
from .math import tanh  # noqa: F401
from .math import tanh_  # noqa: F401
from .math import add_n  # noqa: F401
from .math import max  # noqa: F401
from .math import amax  # noqa: F401
from .math import maximum  # noqa: F401
from .math import min  # noqa: F401
from .math import amin  # noqa: F401
from .math import minimum  # noqa: F401
from .math import mm  # noqa: F401
from .math import divide  # noqa: F401
from .math import floor_divide  # noqa: F401
from .math import remainder  # noqa: F401
from .math import remainder_  # noqa: F401
from .math import mod  # noqa: F401
from .math import floor_mod  # noqa: F401
from .math import multiply  # noqa: F401
from .math import multiply_  # noqa: F401
from .math import add  # noqa: F401
from .math import add_  # noqa: F401
from .math import subtract  # noqa: F401
from .math import subtract_  # noqa: F401
from .math import atan2  # noqa: F401
from .math import logsumexp  # noqa: F401
from .math import logaddexp  # noqa: F401
from .math import inverse  # noqa: F401
from .math import log2  # noqa: F401
from .math import log10  # noqa: F401
from .math import log1p  # noqa: F401
from .math import erf  # noqa: F401
from .math import addmm  # noqa: F401
from .math import clip  # noqa: F401
from .math import clip_  # noqa: F401
from .math import trace  # noqa: F401
from .math import kron  # noqa: F401
from .math import isfinite  # noqa: F401
from .math import isinf  # noqa: F401
from .math import isnan  # noqa: F401
from .math import prod  # noqa: F401
from .math import all  # noqa: F401
from .math import any  # noqa: F401
from .math import broadcast_shape  # noqa: F401
from .math import conj  # noqa: F401
from .math import trunc  # noqa: F401
from .math import digamma  # noqa: F401
from .math import neg  # noqa: F401
from .math import lgamma  # noqa: F401
from .math import diagonal  # noqa: F401
from .math import acosh  # noqa: F401
from .math import asinh  # noqa: F401
from .math import atanh  # noqa: F401
from .math import lerp  # noqa: F401
from .math import lerp_  # noqa: F401
from .math import erfinv  # noqa: F401
from .math import erfinv_  # noqa: F401
from .math import rad2deg  # noqa: F401
from .math import deg2rad  # noqa: F401
from .math import gcd  # noqa: F401
from .math import lcm  # noqa: F401
from .math import diff  # noqa: F401
from .math import angle  # noqa: F401
from .math import fmax  # noqa: F401
from .math import fmin  # noqa: F401
from .math import inner  # noqa: F401
from .math import outer  # noqa: F401
from .math import heaviside  # noqa: F401
from .math import frac  # noqa: F401
from .math import sgn  # noqa: F401
from .math import take  # noqa: F401
from .math import frexp  # noqa: F401
from .math import ldexp  # noqa: F401
from .math import trapezoid  # noqa: F401
from .math import cumulative_trapezoid  # noqa: F401
from .math import sigmoid  # noqa: F401
from .math import sigmoid_  # noqa: F401
from .math import vander  # noqa: F401
from .math import nextafter  # noqa: F401
from .math import i0  # noqa: F401
from .math import i0e  # noqa: F401
from .math import i1  # noqa: F401
from .math import i1e  # noqa: F401
from .math import polygamma  # noqa: F401

from .random import multinomial  # noqa: F401
from .random import standard_normal  # noqa: F401
from .random import normal  # noqa: F401
from .random import uniform  # noqa: F401
from .random import uniform_  # noqa: F401
from .random import randn  # noqa: F401
from .random import rand  # noqa: F401
from .random import randint  # noqa: F401
from .random import randint_like  # noqa: F401
from .random import randperm  # noqa: F401
from .random import poisson  # noqa: F401
from .random import exponential_  # noqa: F401
from .search import argmax  # noqa: F401
from .search import argmin  # noqa: F401
from .search import argsort  # noqa: F401
from .search import searchsorted  # noqa: F401
from .search import bucketize  # noqa: F401
from .search import topk  # noqa: F401
from .search import where  # noqa: F401
from .search import index_select  # noqa: F401
from .search import nonzero  # noqa: F401
from .search import sort  # noqa: F401
from .search import index_sample  # noqa: F401
from .search import masked_select  # noqa: F401
from .search import kthvalue  # noqa: F401
from .search import mode  # noqa: F401

from .stat import mean  # noqa: F401
from .stat import std  # noqa: F401
from .stat import var  # noqa: F401
from .stat import numel  # noqa: F401
from .stat import median  # noqa: F401
from .stat import nanmedian  # noqa: F401
from .stat import quantile  # noqa: F401
from .stat import nanquantile  # noqa: F401

from .to_string import set_printoptions  # noqa: F401

from .array import array_length  # noqa: F401
from .array import array_read  # noqa: F401
from .array import array_write  # noqa: F401
from .array import create_array  # noqa: F401

from .einsum import einsum  # noqa: F401

# this list used in math_op_patch.py for _binary_creator_
tensor_method_func = [  # noqa
    'create_parameter',
    'create_tensor',
    'matmul',
    'dot',
    'cov',
    'corrcoef',
    'norm',
    'cond',
    'transpose',
    'lstsq',
    'dist',
    't',
    'cross',
    'cholesky',
    'bmm',
    'histogram',
    'bincount',
    'mv',
    'matrix_power',
    'qr',
    'eigvals',
    'eigvalsh',
    'abs',
    'acos',
    'all',
    'any',
    'asin',
    'atan',
    'ceil',
    'ceil_',
    'cos',
    'cosh',
    'cumsum',
    'cumprod',
    'logcumsumexp',
    'logit',
    'exp',
    'exp_',
    'expm1',
    'floor',
    'floor_',
    'increment',
    'logaddexp',
    'log',
    'log2',
    'log10',
    'logsumexp',
    'multiplex',
    'pow',
    'prod',
    'reciprocal',
    'reciprocal_',
    'round',
    'round_',
    'rsqrt',
    'rsqrt_',
    'scale',
    'scale_',
    'sign',
    'sin',
    'sinh',
    'sqrt',
    'sqrt_',
    'square',
    'stanh',
    'sum',
    'nan_to_num',
    'nansum',
    'nanmean',
    'count_nonzero',
    'tanh',
    'tanh_',
    'add_n',
    'max',
    'amax',
    'maximum',
    'min',
    'amin',
    'minimum',
    'fmax',
    'fmin',
    'mm',
    'inner',
    'outer',
    'divide',
    'floor_divide',
    'remainder',
    'remainder_',
    'mod',
    'floor_mod',
    'multiply',
    'multiply_',
    'add',
    'add_',
    'subtract',
    'subtract_',
    'atan',
    'logsumexp',
    'inverse',
    'log1p',
    'erf',
    'addmm',
    'clip',
    'clip_',
    'trace',
    'kron',
    'kthvalue',
    'isfinite',
    'isinf',
    'isnan',
    'broadcast_shape',
    'conj',
    'neg',
    'lgamma',
    'equal',
    'equal_all',
    'greater_equal',
    'greater_than',
    'is_empty',
    'less_equal',
    'less_than',
    'logical_and',
    'logical_not',
    'logical_or',
    'logical_xor',
    'not_equal',
    'allclose',
    'isclose',
    'is_tensor',
    'cast',
    'concat',
    'expand',
    'broadcast_to',
    'expand_as',
    'flatten',
    'flatten_',
    'gather',
    'gather_nd',
    'reshape',
    'reshape_',
    'reverse',
    'scatter',
    'scatter_',
    'scatter_nd_add',
    'scatter_nd',
    'shard_index',
    'slice',
    'split',
    'vsplit',
    'chunk',
    'tensordot',
    'squeeze',
    'squeeze_',
    'stack',
    'strided_slice',
    'transpose',
    'unique',
    'unique_consecutive',
    'unsqueeze',
    'unsqueeze_',
    'unstack',
    'flip',
    'rot90',
    'unbind',
    'roll',
    'tile',
    'argmax',
    'argmin',
    'argsort',
    'masked_select',
    'topk',
    'where',
    'index_select',
    'nonzero',
    'sort',
    'index_sample',
    'mean',
    'std',
    'var',
    'numel',
    'median',
    'nanmedian',
    'quantile',
    'nanquantile',
    'is_complex',
    'is_integer',
    'rank',
    'shape',
    'real',
    'imag',
    'is_floating_point',
    'digamma',
    'diagonal',
    'trunc',
    'frac',
    'bitwise_and',
    'bitwise_or',
    'bitwise_xor',
    'bitwise_not',
    'broadcast_tensors',
    'eig',
    'uniform_',
    'multi_dot',
    'solve',
    'cholesky_solve',
    'triangular_solve',
    'asinh',
    'atanh',
    'acosh',
    'lu',
    'lu_unpack',
    'cdist',
    'as_complex',
    'as_real',
    'rad2deg',
    'deg2rad',
    'gcd',
    'lcm',
    'diff',
    "mode",
    'lerp',
    'lerp_',
    'erfinv',
    'erfinv_',
    'angle',
    'moveaxis',
    'repeat_interleave',
    'take_along_axis',
    'put_along_axis',
    'put_along_axis_',
    'exponential_',
    'heaviside',
    'index_add',
    "index_add_",
    'index_put',
    'index_put_',
    'take',
    'bucketize',
    'sgn',
    'frexp',
    'ldexp',
    'trapezoid',
    'cumulative_trapezoid',
    'polar',
    'sigmoid',
    'sigmoid_',
    'vander',
    'nextafter',
    'unflatten',
    'i0',
    'i0e',
    'i1',
    'i1e',
    'polygamma',
]

# this list used in math_op_patch.py for magic_method bind
magic_method_func = [
    ('__and__', 'bitwise_and'),
    ('__or__', 'bitwise_or'),
    ('__xor__', 'bitwise_xor'),
    ('__invert__', 'bitwise_not'),
]
