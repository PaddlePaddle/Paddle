# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import argparse
import glob
import inspect
import logging
import os
import platform
import re
import shutil
import sys
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, Callable, Literal

from typing_extensions import TypeAlias


def copy_libs(paddle_binary_dir):
    ext_suffix = (
        '.dll'
        if os.name == 'nt'
        else ('.dylib' if sys.platform == 'darwin' else '.so')
    )

    try:
        from env_dict import env_dict
    except ImportError:
        return

    if os.name != 'nt':
        package_data = {
            'paddle.base': [env_dict.get("FLUID_CORE_NAME") + '.so']
        }
    else:
        package_data = {
            'paddle.base': [
                env_dict.get("FLUID_CORE_NAME") + '.pyd',
                env_dict.get("FLUID_CORE_NAME") + '.lib',
            ]
        }
    package_data['paddle.base'] += [
        paddle_binary_dir + '/python/paddle/cost_model/static_op_benchmark.json'
    ]
    if 'develop' in sys.argv:
        package_dir = {'': 'python'}
    else:
        package_dir = {
            '': env_dict.get("PADDLE_BINARY_DIR") + '/python',
            'paddle.base.proto.profiler': env_dict.get("PADDLE_BINARY_DIR")
            + '/paddle/fluid/platform',
            'paddle.base.proto': env_dict.get("PADDLE_BINARY_DIR")
            + '/paddle/fluid/framework',
            'paddle.base': env_dict.get("PADDLE_BINARY_DIR")
            + '/python/paddle/base',
        }
    # put all thirdparty libraries in paddle.libs
    libs_path = paddle_binary_dir + '/python/paddle/libs'
    package_data['paddle.libs'] = []
    if env_dict.get("WITH_SHARED_PHI") == "ON":
        package_data['paddle.libs'] += [
            ('libphi' if os.name != 'nt' else 'phi') + ext_suffix
        ]
        shutil.copy(env_dict.get("PHI_LIB"), libs_path)

    if env_dict.get("WITH_SHARED_IR") == "ON":
        package_data['paddle.libs'] += [
            ('libpir' if os.name != 'nt' else 'pir') + ext_suffix
        ]
        shutil.copy(env_dict.get("IR_LIB"), libs_path)

    package_data['paddle.libs'] += [
        ('libwarpctc' if os.name != 'nt' else 'warpctc') + ext_suffix,
        ('libwarprnnt' if os.name != 'nt' else 'warprnnt') + ext_suffix,
    ]
    package_data['paddle.libs'] += [
        ('libcommon' if os.name != 'nt' else 'common') + ext_suffix,
    ]
    shutil.copy(env_dict.get("COMMON_LIB"), libs_path)
    shutil.copy(env_dict.get("WARPCTC_LIBRARIES"), libs_path)
    shutil.copy(env_dict.get("WARPRNNT_LIBRARIES"), libs_path)
    package_data['paddle.libs'] += [
        os.path.basename(env_dict.get("LAPACK_LIB")),
        os.path.basename(env_dict.get("BLAS_LIB")),
        os.path.basename(env_dict.get("GFORTRAN_LIB")),
        os.path.basename(env_dict.get("GNU_RT_LIB_1")),
    ]
    shutil.copy(env_dict.get("BLAS_LIB"), libs_path)
    shutil.copy(env_dict.get("LAPACK_LIB"), libs_path)
    shutil.copy(env_dict.get("GFORTRAN_LIB"), libs_path)
    shutil.copy(env_dict.get("GNU_RT_LIB_1"), libs_path)

    if not sys.platform.startswith("linux"):
        package_data['paddle.libs'] += [
            os.path.basename(env_dict.get("GNU_RT_LIB_2"))
        ]
        shutil.copy(env_dict.get("GNU_RT_LIB_2"), libs_path)
    if env_dict.get("WITH_MKL") == 'ON':
        shutil.copy(env_dict.get("MKLML_SHARED_LIB"), libs_path)
        shutil.copy(env_dict.get("MKLML_SHARED_IOMP_LIB"), libs_path)
        package_data['paddle.libs'] += [
            ('libmklml_intel' if os.name != 'nt' else 'mklml') + ext_suffix,
            ('libiomp5' if os.name != 'nt' else 'libiomp5md') + ext_suffix,
        ]
    else:
        if os.name == 'nt':
            # copy the openblas.dll
            shutil.copy(env_dict.get("OPENBLAS_SHARED_LIB"), libs_path)
            package_data['paddle.libs'] += ['openblas' + ext_suffix]
        elif (
            os.name == 'posix'
            and platform.machine() == 'aarch64'
            and env_dict.get("OPENBLAS_LIB").endswith('so')
        ):
            # copy the libopenblas.so on linux+aarch64
            # special: libpaddle.so without avx depends on 'libopenblas.so.0', not 'libopenblas.so'
            if os.path.exists(env_dict.get("OPENBLAS_LIB") + '.0'):
                shutil.copy(env_dict.get("OPENBLAS_LIB") + '.0', libs_path)
                package_data['paddle.libs'] += ['libopenblas.so.0']

    if env_dict.get("WITH_GPU") == 'ON':
        if len(env_dict.get("FLASHATTN_LIBRARIES", "")) > 1:
            package_data['paddle.libs'] += [
                os.path.basename(env_dict.get("FLASHATTN_LIBRARIES"))
            ]
            shutil.copy(env_dict.get("FLASHATTN_LIBRARIES"), libs_path)
    if env_dict.get("WITH_CINN") == 'ON':
        shutil.copy(
            env_dict.get("CINN_LIB_LOCATION")
            + '/'
            + env_dict.get("CINN_LIB_NAME"),
            libs_path,
        )
        shutil.copy(
            env_dict.get("CINN_INCLUDE_DIR")
            + '/paddle/cinn/runtime/cuda/cinn_cuda_runtime_source.cuh',
            libs_path,
        )
        package_data['paddle.libs'] += ['libcinnapi.so']
        package_data['paddle.libs'] += ['cinn_cuda_runtime_source.cuh']

        cinn_fp16_file = (
            env_dict.get("CINN_INCLUDE_DIR")
            + '/paddle/cinn/runtime/cuda/float16.h'
        )
        if os.path.exists(cinn_fp16_file):
            shutil.copy(cinn_fp16_file, libs_path)
            package_data['paddle.libs'] += ['float16.h']
        cinn_bf16_file = (
            env_dict.get("CINN_INCLUDE_DIR")
            + '/paddle/cinn/runtime/cuda/bfloat16.h'
        )
        if os.path.exists(cinn_bf16_file):
            shutil.copy(cinn_bf16_file, libs_path)
            package_data['paddle.libs'] += ['bfloat16.h']

        if env_dict.get("CMAKE_BUILD_TYPE") == 'Release' and os.name != 'nt':
            command = (
                "patchelf --set-rpath '$ORIGIN/../../nvidia/cuda_nvrtc/lib/:$ORIGIN/../../nvidia/cuda_runtime/lib/:$ORIGIN/../../nvidia/cublas/lib/:$ORIGIN/../../nvidia/cudnn/lib/:$ORIGIN/../../nvidia/curand/lib/:$ORIGIN/../../nvidia/cusolver/lib/:$ORIGIN/../../nvidia/nvtx/lib/:$ORIGIN/' %s/"
                % libs_path
                + env_dict.get("CINN_LIB_NAME")
            )
            if os.system(command) != 0:
                raise Exception(
                    'patch '
                    + libs_path
                    + '/'
                    + env_dict.get("CINN_LIB_NAME")
                    + ' failed',
                    'command: %s' % command,
                )
    if env_dict.get("WITH_PSLIB") == 'ON':
        shutil.copy(env_dict.get("PSLIB_LIB"), libs_path)
        shutil.copy(env_dict.get("JVM_LIB"), libs_path)
        if os.path.exists(env_dict.get("PSLIB_VERSION_PY")):
            shutil.copy(
                env_dict.get("PSLIB_VERSION_PY"),
                paddle_binary_dir
                + '/python/paddle/incubate/distributed/fleet/parameter_server/pslib/',
            )
        package_data['paddle.libs'] += ['libps' + ext_suffix]
        package_data['paddle.libs'] += ['libjvm' + ext_suffix]
    if env_dict.get("WITH_ONEDNN") == 'ON':
        if env_dict.get("CMAKE_BUILD_TYPE") == 'Release' and os.name != 'nt':
            # only change rpath in Release mode.
            # TODO(typhoonzero): use install_name_tool to patch mkl libs once
            # we can support mkl on mac.
            #
            # change rpath of libdnnl.so.1, add $ORIGIN/ to it.
            # The reason is that all thirdparty libraries in the same directory,
            # thus, libdnnl.so.1 will find libmklml_intel.so and libiomp5.so.
            command = "patchelf --set-rpath '$ORIGIN/' " + env_dict.get(
                "ONEDNN_SHARED_LIB"
            )
            if os.system(command) != 0:
                raise Exception(
                    "patch libdnnl.so failed, command: %s" % command
                )
        shutil.copy(env_dict.get("ONEDNN_SHARED_LIB"), libs_path)
        if os.name != 'nt':
            package_data['paddle.libs'] += ['libdnnl.so.3']
        else:
            package_data['paddle.libs'] += ['mkldnn.dll']

    if env_dict.get("WITH_ONNXRUNTIME") == 'ON':
        shutil.copy(env_dict.get("ONNXRUNTIME_SHARED_LIB"), libs_path)
        shutil.copy(env_dict.get("PADDLE2ONNX_LIB"), libs_path)
        if os.name == 'nt':
            package_data['paddle.libs'] += [
                'paddle2onnx.dll',
                'onnxruntime.dll',
            ]
        else:
            package_data['paddle.libs'] += [
                env_dict.get("PADDLE2ONNX_LIB_NAME"),
                env_dict.get("ONNXRUNTIME_LIB_NAME"),
            ]

    if env_dict.get("WITH_XPU") == 'ON':
        shutil.copy(env_dict.get("XPU_API_LIB"), libs_path)
        package_data['paddle.libs'] += [env_dict.get("XPU_API_LIB_NAME")]
        xpu_rt_lib_list = glob.glob(env_dict.get("XPU_RT_LIB") + '*')
        for xpu_rt_lib_file in xpu_rt_lib_list:
            shutil.copy(xpu_rt_lib_file, libs_path)
            package_data['paddle.libs'] += [os.path.basename(xpu_rt_lib_file)]

    if env_dict.get("WITH_XPU_BKCL") == 'ON':
        shutil.copy(env_dict.get("XPU_BKCL_LIB"), libs_path)
        package_data['paddle.libs'] += [env_dict.get("XPU_BKCL_LIB_NAME")]

    if env_dict.get("WITH_XPU_XFT") == 'ON':
        shutil.copy(env_dict.get("XPU_XFT_LIB"), libs_path)
        package_data['paddle.libs'] += [env_dict.get("XPU_XFT_LIB_NAME")]

    if env_dict.get("WITH_XPTI") == 'ON':
        shutil.copy(env_dict.get("XPU_XPTI_LIB"), libs_path)
        package_data['paddle.libs'] += [env_dict.get("XPU_XPTI_LIB_NAME")]

    if env_dict.get("WITH_XPU_XHPC") == 'ON':
        shutil.copy(env_dict.get("XPU_XBLAS_LIB"), libs_path)
        package_data['paddle.libs'] += [env_dict.get("XPU_XBLAS_LIB_NAME")]
        shutil.copy(env_dict.get("XPU_XFA_LIB"), libs_path)
        package_data['paddle.libs'] += [env_dict.get("XPU_XFA_LIB_NAME")]

    # remove unused paddle/libs/__init__.py
    if os.path.isfile(libs_path + '/__init__.py'):
        os.remove(libs_path + '/__init__.py')
    package_dir['paddle.libs'] = libs_path

    # change rpath of ${FLUID_CORE_NAME}.ext, add $ORIGIN/../libs/ to it.
    # The reason is that libwarpctc.ext, libwarprnnt.ext, libiomp5.ext etc are in paddle.libs, and
    # ${FLUID_CORE_NAME}.ext is in paddle.base, thus paddle/fluid/../libs will pointer to above libraries.
    # This operation will fix https://github.com/PaddlePaddle/Paddle/issues/3213
    if env_dict.get("CMAKE_BUILD_TYPE") == 'Release':
        if os.name != 'nt':
            # only change rpath in Release mode, since in Debug mode, ${FLUID_CORE_NAME}.xx is too large to be changed.
            if env_dict.get("APPLE") == "1":
                commands = [
                    "install_name_tool -id '@loader_path/../libs/' "
                    + env_dict.get("PADDLE_BINARY_DIR")
                    + '/python/paddle/base/'
                    + env_dict.get("FLUID_CORE_NAME")
                    + '.so'
                ]
                commands.append(
                    "install_name_tool -add_rpath '@loader_path/../libs/' "
                    + env_dict.get("PADDLE_BINARY_DIR")
                    + '/python/paddle/base/'
                    + env_dict.get("FLUID_CORE_NAME")
                    + '.so'
                )
                commands.append(
                    "install_name_tool -add_rpath '@loader_path/../libs/' "
                    + env_dict.get("PADDLE_BINARY_DIR")
                    + '/python/paddle/libs/'
                    + env_dict.get("COMMON_NAME")
                )
                if env_dict.get("WITH_SHARED_PHI") == "ON":
                    commands.append(
                        "install_name_tool -add_rpath '@loader_path' "
                        + env_dict.get("PADDLE_BINARY_DIR")
                        + '/python/paddle/libs/'
                        + env_dict.get("PHI_NAME")
                    )
                if env_dict.get("WITH_SHARED_IR") == "ON":
                    commands.append(
                        "install_name_tool -add_rpath '@loader_path' "
                        + env_dict.get("PADDLE_BINARY_DIR")
                        + '/python/paddle/libs/'
                        + env_dict.get("IR_NAME")
                    )
            else:
                commands = [
                    "patchelf --set-rpath '$ORIGIN/../../nvidia/cuda_runtime/lib:$ORIGIN/../../nvidia/cuda_nvrtc/lib:$ORIGIN/../../nvidia/cublas/lib:$ORIGIN/../../nvidia/cudnn/lib:$ORIGIN/../../nvidia/curand/lib:$ORIGIN/../../nvidia/cusparse/lib:$ORIGIN/../../nvidia/nvjitlink/lib:$ORIGIN/../../nvidia/cuda_cupti/lib:$ORIGIN/../../nvidia/cuda_runtime/lib:$ORIGIN/../../nvidia/cufft/lib:$ORIGIN/../../nvidia/cufft/lib:$ORIGIN/../../nvidia/cusolver/lib:$ORIGIN/../../nvidia/nccl/lib:$ORIGIN/../../nvidia/nvtx/lib:$ORIGIN/../libs/' "
                    + env_dict.get("PADDLE_BINARY_DIR")
                    + '/python/paddle/base/'
                    + env_dict.get("FLUID_CORE_NAME")
                    + '.so'
                ]
                if env_dict.get("WITH_SHARED_PHI") == "ON":
                    commands.append(
                        "patchelf --set-rpath '$ORIGIN/../../nvidia/cuda_runtime/lib:$ORIGIN:$ORIGIN/../libs' "
                        + env_dict.get("PADDLE_BINARY_DIR")
                        + '/python/paddle/libs/'
                        + env_dict.get("PHI_NAME")
                    )
                if env_dict.get("WITH_SHARED_IR") == "ON":
                    commands.append(
                        "patchelf --set-rpath '$ORIGIN:$ORIGIN/../libs' "
                        + env_dict.get("PADDLE_BINARY_DIR")
                        + '/python/paddle/libs/'
                        + env_dict.get("IR_NAME")
                    )
            # The sw_64 not support patchelf, so we just disable that.
            if platform.machine() != 'sw_64' and platform.machine() != 'mips64':
                for command in commands:
                    if os.system(command) != 0:
                        raise Exception(
                            'patch '
                            + env_dict.get("FLUID_CORE_NAME")
                            + '.%s failed' % ext_suffix,
                            'command: %s' % command,
                        )
    # type hints
    package_data['paddle'] = package_data.get('paddle', []) + ['py.typed']
    package_data['paddle.framework'] = package_data.get(
        'paddle.framework', []
    ) + ['*.pyi']
    package_data['paddle.base'] = package_data.get('paddle.base', []) + [
        '*.pyi'
    ]
    package_data['paddle.tensor'] = package_data.get('paddle.tensor', []) + [
        'tensor.pyi'
    ]

    return package_data, package_dir


logging.basicConfig(style="{", format="{message}", level=logging.INFO)
logger = logging.getLogger("Generating stub file for paddle.Tensor")
logger.setLevel(logging.INFO)

INDENT_SIZE = 4
INDENT = " " * INDENT_SIZE

MemberType: TypeAlias = Literal[
    "doc",
    "attribute",
    "method",
]


@dataclass
class Member:
    id: int
    name: str
    type: MemberType
    aliases: list[str]
    decorators: list[str]
    signature: str
    doc: str | None

    def add_alias(self, alias: str):
        self.aliases.append(alias)


@lru_cache
def _slot_pattern(slot_name: str) -> re.Pattern:
    return re.compile(
        r"(?P<indent> *)#\s*annotation:\s*\$\{" + slot_name + r"\}"
    )


class TensorGen:
    def __init__(self, template: str = ''):
        self._template = template

    def find_annotation_slot(self, slot_name: str) -> tuple[str, int, int]:
        pattern = _slot_pattern(slot_name)
        slot = []
        for mo in pattern.finditer(self._template):
            _indent = mo.group('indent')
            _start_index, _end_index = mo.span()
            slot.append((_indent, _start_index, _end_index))

        assert len(slot) == 1, self._template
        return slot[0]

    @property
    def tensor_docstring(self):
        return self.find_annotation_slot('tensor_docstring')

    @property
    def tensor_attributes(self):
        return self.find_annotation_slot('tensor_attributes')

    @property
    def tensor_methods(self):
        return self.find_annotation_slot('tensor_methods')

    @property
    def tensor_alias(self):
        return self.find_annotation_slot('tensor_alias')

    def find_apis(self, api_name: str) -> list[dict[str, tuple[str, int, int]]]:
        pattern = re.compile(
            r"(?P<indent> *)(?P<def_api>def "
            + api_name
            + r")(?P<signature>\(.*?\).*?:)(?P<docstring>.*?)(?P<ellipsis>\.{3})(?P<comment>[^\n]*#[^\n]*\n)?",
            re.DOTALL,
        )
        api = []
        for mo in pattern.finditer(self._template):
            _indent = mo.group('indent')
            _def_api = mo.group('def_api')
            _signature = mo.group('signature')
            _docstring = mo.group('docstring')
            _ellipsis = mo.group('ellipsis')
            _comment = mo.group('comment')
            _comment = '' if _comment is None else _comment

            _start_index, _end_index = mo.span()

            _start_indent = _start_index
            _end_indent = _start_indent + len(_indent)

            _start_def_api = _end_indent
            _end_def_api = _start_def_api + len(_def_api)

            _start_signature = _end_def_api
            _end_signature = _start_signature + len(_signature)

            _start_docstring = _end_signature
            _end_docstring = _start_docstring + len(_docstring)

            _start_ellipsis = _end_docstring
            _end_ellipsis = _start_ellipsis + len(_ellipsis)

            _start_comment = _end_ellipsis
            _end_comment = _start_comment + len(_comment)

            assert _end_index == _end_comment

            _api = {
                'indent': (_indent, _start_indent, _end_indent),
                'signature': (_signature, _start_signature, _end_signature),
                'docstring': (_docstring, _start_docstring, _end_docstring),
                'ellipsis': (_ellipsis, _start_ellipsis, _end_ellipsis),
                'comment': (_comment, _start_comment, _end_comment),
            }
            api.append(_api)

        return api

    def insert_template(self, code: str, start: int, end: int) -> None:
        self._template = self._template[:start] + code + self._template[end:]

    def add_method(self, func: Member):
        """
        1. insert docstring: tensor.prototype.pyi define the method without docstring
        2. insert method: tensor.prototype.pyi NOT define the method
        """
        methods = self.find_apis(func.name)
        if methods:
            # only use the last method
            method = methods[-1]
            # insert docstring if necessary
            if not method['docstring'][0].strip():
                doc = func.doc
                if doc:
                    comment = method['comment'][0]
                    doc_start = method['docstring'][1]
                    doc_end = method['docstring'][2]

                    api_indent = method['indent'][0]

                    assert len(api_indent) % INDENT_SIZE == 0

                    _indent = api_indent + INDENT

                    _doc = '\n'  # new line
                    _doc += f'{_indent}r"""\n'
                    _doc += with_indent(doc, len(_indent) // INDENT_SIZE)
                    _doc += "\n"
                    _doc += f'{_indent}"""\n'
                    _doc += f'{_indent}'

                    self.insert_template(comment + _doc, doc_start, doc_end)
        else:
            method_code = '\n'
            for decorator in func.decorators:
                method_code += f"@{decorator}\n"

            method_code += f"def {func.signature}:\n"
            if func.doc:
                method_code += f'{INDENT}r"""\n'
                method_code += with_indent(func.doc, 1)
                method_code += "\n"
                method_code += f'{INDENT}"""\n'
            method_code += f"{INDENT}...\n"

            _indent, _, _end_index = self.tensor_methods
            method_code = with_indent(method_code, len(_indent) // INDENT_SIZE)
            self.insert_template(method_code, _end_index, _end_index)

    def add_alias(self, alias: str, target: str):
        _indent, _, _end_index = self.tensor_alias
        aliases_code = "\n"
        aliases_code += f"{_indent}{alias} = {target}"
        self.insert_template(aliases_code, _end_index, _end_index)

    def add_attribute(self, name: str, type_: str):
        _indent, _, _end_index = self.tensor_attributes
        attr_code = "\n"
        attr_code += f"{_indent}{name}: {type_}"
        self.insert_template(attr_code, _end_index, _end_index)

    def add_doc(self, doc: str):
        _indent, _, _end_index = self.tensor_docstring
        docstring = "\n"
        docstring += 'r"""\n'
        docstring += doc
        docstring += "\n"
        docstring += '"""\n'
        docstring = with_indent(docstring, len(_indent) // INDENT_SIZE)
        self.insert_template(docstring, _end_index, _end_index)

    def codegen(self) -> str:
        header = (
            '# This file is auto generated by `tools/gen_tensor_stub.py`.\n\n'
        )
        return header + self._template


def is_inherited_member(name: str, cls: type) -> bool:
    """Check if the member is inherited from parent class"""

    if name in cls.__dict__:
        return False

    for base in cls.__bases__:
        if name in base.__dict__:
            return True

    return any(is_inherited_member(name, base) for base in cls.__bases__)


def is_property(member: Any) -> bool:
    """Check if the member is a property"""

    return isinstance(member, (property, cached_property))


def is_staticmethod(member: Any) -> bool:
    """Check if the member is a staticmethod"""

    return isinstance(member, staticmethod)


def is_classmethod(member: Any) -> bool:
    """Check if the member is a classmethod"""

    return isinstance(member, classmethod)


def process_lines(code: str, callback: Callable[[str], str]) -> str:
    lines = code.splitlines()
    end_with_newline = code.endswith("\n")
    processed_lines: list[str] = []
    for line in lines:
        processed_lines.append(callback(line))
    processed_code = "\n".join(processed_lines)
    if end_with_newline:
        processed_code += "\n"
    return processed_code


def with_indent(code: str, level: int) -> str:
    def add_indent_line(line: str) -> str:
        if not line:
            return line
        return INDENT + line

    def remove_indent_line(line: str) -> str:
        if not line:
            return line
        elif line.startswith(INDENT):
            return line[len(INDENT) :]
        else:
            return line

    if level == 0:
        return code
    elif level > 0:
        if level == 1:
            return process_lines(code, add_indent_line)
        code = process_lines(code, add_indent_line)
        return with_indent(code, level - 1)
    else:
        if level == -1:
            return process_lines(code, remove_indent_line)
        return with_indent(code, level - 1)


def func_sig_to_method_sig(func_sig: str) -> str:
    regex_func_sig = re.compile(
        r"^(?P<method_name>[_a-zA-Z0-9]+)\((?P<arg0>[^,*)]+(:.+)?)?(?P<rest_args>.*)?\)",
        re.DOTALL,
    )
    matched = regex_func_sig.search(func_sig)
    if matched is None:
        # TODO: resolve this case
        logging.warning(f"Cannot parse function signature: {func_sig}")
        return "_(self)"

    if matched.group('rest_args').startswith('*'):
        method_sig = regex_func_sig.sub(
            r"\g<method_name>(self, \g<rest_args>)", func_sig
        )

    else:
        method_sig = regex_func_sig.sub(
            r"\g<method_name>(self\g<rest_args>)", func_sig
        )

    return method_sig


def func_doc_to_method_doc(func_doc: str) -> str:
    # Iterate every line, insert the indent and remove document of the first argument
    method_doc = ""
    is_first_arg = False
    first_arg_offset = 0

    for line in func_doc.splitlines():
        current_line_offset = len(line) - len(line.lstrip())
        # Remove the first argument (self in Tensor method) from docstring
        if is_first_arg:
            if current_line_offset <= first_arg_offset:
                is_first_arg = False
            if not first_arg_offset:
                first_arg_offset = current_line_offset
            if is_first_arg:
                continue
        method_doc += f"{line}\n" if line else "\n"
        if line.lstrip().startswith("Args:"):
            is_first_arg = True

    return method_doc


def get_tensor_members():
    import paddle

    tensor_class = paddle.Tensor

    members: dict[int, Member] = {}
    for name, member in inspect.getmembers(tensor_class):
        member_id = id(member)
        member_doc = inspect.getdoc(member)
        member_doc_cleaned = (
            func_doc_to_method_doc(inspect.cleandoc(member_doc))
            if member_doc is not None
            else None
        )
        try:
            sig = inspect.signature(member)
            # TODO: classmethod
            member_signature = f"{name}{sig}"

        except (TypeError, ValueError):
            member_signature = f"{name}()"

        if is_inherited_member(name, tensor_class):
            continue

        # Filter out private members except magic methods
        if name.startswith("_") and not (
            name.startswith("__") and name.endswith("__")
        ):
            continue

        if member_id in members:
            members[member_id].add_alias(name)
            continue

        if name == '__doc__':
            members[member_id] = Member(
                member_id,
                name,
                "doc",
                [],
                [],
                "__doc__",
                member,
            )
        elif is_property(member) or inspect.isdatadescriptor(member):
            members[member_id] = Member(
                member_id,
                name,
                "method",
                [],
                ["property"],
                f"{name}(self)",
                member_doc_cleaned,
            )
        elif is_classmethod(member):
            members[member_id] = Member(
                member_id,
                name,
                "method",
                [],
                ["classmethod"],
                member_signature,
                member_doc_cleaned,
            )
        elif is_staticmethod(member):
            members[member_id] = Member(
                member_id,
                name,
                "method",
                [],
                ["staticmethod"],
                member_signature,
                member_doc_cleaned,
            )
        elif (
            inspect.isfunction(member)
            or inspect.ismethod(member)
            or inspect.ismethoddescriptor(member)
        ):
            members[member_id] = Member(
                member_id,
                name,
                "method",
                [],
                [],
                func_sig_to_method_sig(member_signature),
                member_doc_cleaned,
            )
        else:
            logging.debug(f"Skip unknown type of member: {name}, {member}")
    return members


def get_tensor_template(path: str) -> str:
    with open(path) as f:
        return ''.join(f.readlines())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        default="python/paddle/tensor/tensor.prototype.pyi",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="python/paddle/tensor/tensor.pyi",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--binary-dir",
        type=str,
        default="",
    )

    args = parser.parse_args()

    copy_libs(args.binary_dir)

    # Get members of Tensor
    tensor_members = get_tensor_members()
    logging.debug(f'total members in Tensor: {len(tensor_members)}')

    # Get tensor template
    tensor_template = get_tensor_template(args.input_file)

    # Generate the Tensor stub
    tensor_gen = TensorGen(tensor_template)

    for member in tensor_members.values():
        if member.type == "method":
            tensor_gen.add_method(member)
            for alias in member.aliases:
                tensor_gen.add_alias(alias, member.name)
        elif member.type == "attribute":
            tensor_gen.add_attribute(member.name, "Any")
        elif member.type == "doc":
            tensor_gen.add_doc(member.doc)

    # Write to target file
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(tensor_gen.codegen())


if __name__ == "__main__":
    main()
