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
from __future__ import print_function

#from .math import *
#from .creation import *
#from .linalg import *

# TODO: define alias in tensor and framework directory
# from .creation import create_tensor   #DEFINE_ALIAS
# from .creation import create_lod_tensor   #DEFINE_ALIAS
# from .creation import create_random_int_lod   #DEFINE_ALIAS
# from .creation import crop_tensor   #DEFINE_ALIAS
# from .creation import diag   #DEFINE_ALIAS
from .creation import eye  #DEFINE_ALIAS
# from .creation import fill_constant   #DEFINE_ALIAS
# from .creation import get__from_selected_rows   #DEFINE_ALIAS
from .creation import linspace  #DEFINE_ALIAS
# from .creation import ones   #DEFINE_ALIAS
# from .creation import ones_like   #DEFINE_ALIAS
# from .creation import range   #DEFINE_ALIAS
# from .creation import zeros   #DEFINE_ALIAS
# from .creation import zeros_like   #DEFINE_ALIAS
from .creation import arange  #DEFINE_ALIAS
# from .creation import eye   #DEFINE_ALIAS
from .creation import full  # DEFINE_ALIAS
# from .creation import linspace   #DEFINE_ALIAS
from .creation import full_like  #DEFINE_ALIAS
from .creation import triu  #DEFINE_ALIAS
from .creation import tril  #DEFINE_ALIAS
from .creation import meshgrid  #DEFINE_ALIAS
# from .stat import mean   #DEFINE_ALIAS
# from .stat import reduce_mean   #DEFINE_ALIAS
# from .stat import std   #DEFINE_ALIAS
# from .stat import var   #DEFINE_ALIAS
from .logic import equal  #DEFINE_ALIAS
# from .logic import greater_equal   #DEFINE_ALIAS
# from .logic import greater_than   #DEFINE_ALIAS
# from .logic import is_empty   #DEFINE_ALIAS
# from .logic import isfinite   #DEFINE_ALIAS
# from .logic import less_equal   #DEFINE_ALIAS
# from .logic import less_than   #DEFINE_ALIAS
# from .logic import logical_and   #DEFINE_ALIAS
# from .logic import logical_not   #DEFINE_ALIAS
# from .logic import logical_or   #DEFINE_ALIAS
# from .logic import logical_xor   #DEFINE_ALIAS
# from .logic import not_equal   #DEFINE_ALIAS
# from .logic import reduce_all   #DEFINE_ALIAS
# from .logic import reduce_any   #DEFINE_ALIAS
from .logic import allclose  #DEFINE_ALIAS
from .logic import elementwise_equal  #DEFINE_ALIAS
# from .logic import isnan   #DEFINE_ALIAS
# from . import Tensor   #DEFINE_ALIAS
# from . import LoDTensor   #DEFINE_ALIAS
# from . import LoDTensorArray   #DEFINE_ALIAS
# from .random import gaussin   #DEFINE_ALIAS
# from .random import uniform   #DEFINE_ALIAS
# from .random import shuffle   #DEFINE_ALIAS
from .random import randn  #DEFINE_ALIAS
# from .random import rand   #DEFINE_ALIAS
from .random import randint  #DEFINE_ALIAS
from .random import randperm

# from .attribute import rank   #DEFINE_ALIAS
# from .attribute import shape   #DEFINE_ALIAS
# from .io import save   #DEFINE_ALIAS
# from .io import load   #DEFINE_ALIAS
# from .linalg import cholesky   #DEFINE_ALIAS
# from .manipulation import cast   #DEFINE_ALIAS
# from .manipulation import concat   #DEFINE_ALIAS
# from .manipulation import expand   #DEFINE_ALIAS
# from .manipulation import expand_as   #DEFINE_ALIAS
# from .manipulation import flatten   #DEFINE_ALIAS
from .manipulation import gather  #DEFINE_ALIAS
# from .manipulation import gather_nd   #DEFINE_ALIAS
# from .manipulation import reshape   #DEFINE_ALIAS
# from .manipulation import reverse   #DEFINE_ALIAS
# from .manipulation import scatter   #DEFINE_ALIAS
# from .manipulation import scatter_nd_add   #DEFINE_ALIAS
# from .manipulation import scatter_nd   #DEFINE_ALIAS
# from .manipulation import shard_index   #DEFINE_ALIAS
# from .manipulation import slice   #DEFINE_ALIAS
from .manipulation import split  #DEFINE_ALIAS
from .manipulation import squeeze  #DEFINE_ALIAS
from .manipulation import stack  #DEFINE_ALIAS
# from .manipulation import strided_slice   #DEFINE_ALIAS
# from .manipulation import transpose   #DEFINE_ALIAS
# from .manipulation import unique   #DEFINE_ALIAS
# from .manipulation import unique_with_counts   #DEFINE_ALIAS
from .manipulation import unsqueeze  #DEFINE_ALIAS
# from .manipulation import unstack   #DEFINE_ALIAS
from .manipulation import flip  #DEFINE_ALIAS
# from .manipulation import unbind   #DEFINE_ALIAS
from .manipulation import roll  #DEFINE_ALIAS
