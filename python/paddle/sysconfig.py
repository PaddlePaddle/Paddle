# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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

import os
import paddle
from pathlib import Path

__all__ = ['get_include', 'get_lib', 'get_compile_flags', 'get_link_flags']


def get_include():
    """
    Get the directory containing the PaddlePaddle C++ header files.
    Returns:
      The directory as string.

    Examples:
        .. code-block:: python

            import paddle
            include_dir = paddle.sysconfig.get_include()

    """
    import paddle
    return os.path.join(os.path.dirname(paddle.__file__), 'include')


def get_lib():
    """
    Get the directory containing the libpaddle_framework.
    Returns:
      The directory as string.

    Examples:
        .. code-block:: python

            import paddle
            include_dir = paddle.sysconfig.get_lib()

    """
    import paddle
    return os.path.join(os.path.dirname(paddle.__file__), 'libs')


def get_compile_flags():
    """Get the compilation flags for custom operators.
    Returns:
      The compilation flags.
      
    Examples:
      .. code-block:: python

          import paddle
          compiler_flags = paddle.sysconfig.get_compile_flags()
    """
    flags = []
    flags.append(f"-I{get_include()}")
    return flags


def get_link_flags():
    """Get the link flags for custom operators.
    
    Returns:
      The link flags.
      
      
    Examples:
      .. code-block:: python

          import paddle
          link_flags = paddle.sysconfig.get_link_flags()
    """
    flags = []
    flags.append("-L%s" % get_lib())
    core_so_path = Path(paddle.__file__).parent / 'fluid'
    flags.append("-L%s" % core_so_path)

    if paddle.fluid.core.has_avx_core:
        flags.append("-l:core_avx.so")
    else:
        flags.append("-l:core_noavx.so")
    return flags
