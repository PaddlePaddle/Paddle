# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import sys
from x86cpu import info as cpuinfo

load_noavx = False
if cpuinfo.supports_avx:
    try:
        from .core_avx import *
        from .core_avx import __doc__, __file__, __name__, __package__
        from .core_avx import __unittest_throw_exception__
        from .core_avx import _append_python_callable_object_and_return_id
        from .core_avx import _cleanup, _Scope
        from .core_avx import _get_use_default_grad_op_desc_maker_ops
        from .core_avx import _is_program_version_supported
        from .core_avx import _set_eager_deletion_mode
        from .core_avx import _set_fuse_parameter_group_size
        from .core_avx import _set_fuse_parameter_memory_size
    except ImportError as error:
        print('WARNING: Error importing avx core. You may not build with AVX, '
              'but AVX is supported on local machine, you could build paddle '
              'WITH_AVX=ON to get better performance.\n' +
              error.__class__.__name__)
        load_noavx = True
else:
    load_noavx = True

if load_noavx:
    try:
        from .core_noavx import *
        from .core_noavx import __doc__, __file__, __name__, __package__
        from .core_noavx import __unittest_throw_exception__
        from .core_noavx import _append_python_callable_object_and_return_id
        from .core_noavx import _cleanup, _Scope
        from .core_noavx import _get_use_default_grad_op_desc_maker_ops
        from .core_noavx import _is_program_version_supported
        from .core_noavx import _set_eager_deletion_mode
        from .core_noavx import _set_fuse_parameter_group_size
        from .core_noavx import _set_fuse_parameter_memory_size
    except ImportError as error:
        sys.exit("Error: Can not load core_noavx.* \n" +
                 error.__class__.__name__)
        load_noavx = True
