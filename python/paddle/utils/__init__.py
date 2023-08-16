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

from . import gast
from .deprecated import deprecated  # noqa: F401
from .lazy_import import try_import  # noqa: F401
from .op_version import OpLastCheckpointChecker  # noqa: F401
from .install_check import run_check  # noqa: F401
from . import unique_name  # noqa: F401
from ..fluid.framework import require_version  # noqa: F401

from . import download  # noqa: F401
from . import image_util  # noqa: F401
from . import cpp_extension  # noqa: F401
from . import dlpack
from . import layers_utils  # noqa: F401

from .layers_utils import convert_to_list  # noqa: F401
from .layers_utils import is_sequence  # noqa: F401
from .layers_utils import to_sequence  # noqa: F401
from .layers_utils import flatten  # noqa: F401
from .layers_utils import pack_sequence_as  # noqa: F401
from .layers_utils import map_structure  # noqa: F401
from .layers_utils import hold_mutable_vars  # noqa: F401
from .layers_utils import copy_mutable_vars  # noqa: F401
from .layers_utils import padding_to_same_structure  # noqa: F401
from .layers_utils import assert_same_structure  # noqa: F401
from .layers_utils import get_shape_tensor_inputs  # noqa: F401
from .layers_utils import convert_shape_to_list  # noqa: F401
from .layers_utils import check_shape  # noqa: F401
from .layers_utils import try_set_static_shape_tensor  # noqa: F401
from .layers_utils import try_get_constant_shape_from_tensor  # noqa: F401
from .layers_utils import get_inputs_outputs_in_block  # noqa: F401
from .layers_utils import _hash_with_id  # noqa: F401
from .layers_utils import _sorted  # noqa: F401
from .layers_utils import _yield_value  # noqa: F401
from .layers_utils import _yield_flat_nest  # noqa: F401
from .layers_utils import _sequence_like  # noqa: F401
from .layers_utils import _packed_nest_with_indices  # noqa: F401
from .layers_utils import _recursive_assert_same_structure  # noqa: F401
from .layers_utils import _is_symmetric_padding  # noqa: F401
from .layers_utils import _contain_var  # noqa: F401
from .layers_utils import _convert_to_tensor_list  # noqa: F401

__all__ = ['deprecated', 'run_check', 'require_version', 'try_import']  # noqa
