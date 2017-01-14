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
    from paddle_api_config import *
    import os.path
    import platform

    system = platform.system().lower()
    is_osx = (system == 'darwin')
    is_win = (system == 'windows')
    is_lin = (system == 'linux')

    if is_lin:
        whole_start = "-Wl,--whole-archive"
        whole_end = "-Wl,--no-whole-archive"
    elif is_osx:
        whole_start = ""
        whole_end = ""

    LIB_DIRS = [
        "math", 'function', 'utils', 'parameter', "gserver", "api", "cuda",
        "pserver", "trainer"
    ]
    PARENT_LIB_DIRS = ['proto']

    class PaddleLDFlag(object):
        def __init__(self):
            self.paddle_build_dir = PADDLE_BUILD_DIR
            self.paddle_build_dir = os.path.abspath(self.paddle_build_dir)
            self.with_gpu = PaddleLDFlag.cmake_bool(WITH_GPU)
            self.protolib = PROTOBUF_LIBRARY
            self.zlib = ZLIB_LIBRARIES
            self.thread = CMAKE_THREAD_LIB
            self.dl_libs = CMAKE_DL_LIBS
            self.with_python = PaddleLDFlag.cmake_bool(WITH_PYTHON)
            self.python_libs = PYTHON_LIBRARIES

            self.glog_libs = GLOG_LIBRARIES

            self.with_coverage = PaddleLDFlag.cmake_bool(WITH_COVERALLS)
            self.gflags_libs = GFLAGS_LIBRARIES
            self.gflags_location = GFLAGS_LOCATION
            self.cblas_libs = CBLAS_LIBRARIES
            self.curt = CUDA_LIBRARIES

        def ldflag_str(self):
            return " ".join(
                [self.libs_dir_str(), self.parent_dir_str(), self.libs_str()])

        def libs_dir_str(self):
            libdirs = LIB_DIRS
            return " ".join(
                map(lambda x: "-L" + os.path.join(self.paddle_build_dir, x),
                    libdirs))

        def parent_dir_str(self):
            libdirs = PARENT_LIB_DIRS
            return " ".join(
                map(lambda x: "-L" + os.path.join(self.paddle_build_dir, '..', x),
                    libdirs))

        def libs_str(self):
            libs = [
                whole_start,
                "-lpaddle_gserver",
                "-lpaddle_function",
                whole_end,
                "-lpaddle_pserver",
                "-lpaddle_trainer_lib",
                "-lpaddle_network",
                '-lpaddle_parameter',
                "-lpaddle_math",
                '-lpaddle_utils',
                "-lpaddle_proto",
                "-lpaddle_cuda",
                "-lpaddle_api",
                self.normalize_flag(self.protolib),
                self.normalize_flag(self.glog_libs),
                self.normalize_flag(self.gflags_libs),
                self.normalize_flag(self.zlib),
                self.normalize_flag(self.thread),
                self.normalize_flag(self.dl_libs),
                self.normalize_flag(self.cblas_libs),
            ]

            if self.with_python:
                libs.append(self.normalize_flag(self.python_libs))
            if self.with_gpu:
                libs.append(self.normalize_flag(self.curt))
            if self.with_coverage:
                libs.append("-fprofile-arcs")
            return " ".join(filter(lambda l: len(l) != 0, libs))

        def normalize_flag(self, cmake_flag):
            """
            CMake flag string to ld flag
            :type cmake_flag: str
            """
            if ";" in cmake_flag:
                return " ".join(map(self.normalize_flag, cmake_flag.split(";")))
            if cmake_flag.startswith("/"):  # is a path
                return cmake_flag
            elif cmake_flag.startswith("-l"):  # normal link command
                return cmake_flag
            elif cmake_flag in [
                    "gflags-shared", "gflags-static", "gflags_nothreads-shared",
                    "gflags_nothreads-static"
            ]:  # special for gflags
                assert PaddleLDFlag.cmake_bool(self.gflags_location)
                return self.gflags_location
            elif len(cmake_flag) != 0:
                return "".join(["-l", cmake_flag])
            else:
                return ""

        @staticmethod
        def cmake_bool(cmake_str):
            """
            CMake bool string to bool
            :param cmake_str: cmake boolean string
            :type cmake_str: str
            :rtype: bool
            """
            if cmake_str in ["FALSE", "OFF", "NO"] or cmake_str.endswith(
                    "-NOTFOUND"):
                return False
            else:
                return True

        def c_flag(self):
            if self.with_coverage:
                return [
                    "-fprofile-arcs", "-ftest-coverage", "-O0", "-g",
                    "-std=c++11"
                ]
            else:
                return ["-std=c++11"]
except ImportError:

    class PaddleLDFlag(object):
        def ldflag_str(self):
            pass

        def c_flag(self):
            pass
