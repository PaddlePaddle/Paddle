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
import os
from x86cpu import info as cpuinfo

try:
    if os.name == 'nt':
        third_lib_path = os.path.abspath(os.path.dirname(
            __file__)) + os.sep + '..' + os.sep + 'libs'
        os.environ['path'] += ';' + third_lib_path
        sys.path.append(third_lib_path)

except ImportError as e:
    from .. import compat as cpt
    if os.name == 'nt':
        executable_path = os.path.abspath(os.path.dirname(sys.executable))
        raise ImportError(
            """NOTE: You may need to run \"set PATH=%s;%%PATH%%\"
        if you encounters \"DLL load failed\" errors. If you have python
        installed in other directory, replace \"%s\" with your own
        directory. The original error is: \n %s""" %
            (executable_path, executable_path, cpt.get_exception_message(e)))
    else:
        raise ImportError(
            """NOTE: You may need to run \"export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH\"
        if you encounters \"libmkldnn.so not found\" errors. If you have python
        installed in other directory, replace \"/usr/local/lib\" with your own
        directory. The original error is: \n""" + cpt.get_exception_message(e))
except Exception as e:
    raise e

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
    except ImportError:
        sys.stderr.write(
            'WARNING: Can not import avx core. You may not build with AVX, '
            'but AVX is supported on local machine, you could build paddle '
            'WITH_AVX=ON to get better performance. ')
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
        sys.exit("Error: Can not load core_noavx.* ." +
                 error.__class__.__name__)
        load_noavx = True
