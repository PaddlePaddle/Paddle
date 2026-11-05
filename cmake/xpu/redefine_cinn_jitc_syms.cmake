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
#
# cmake/xpu/redefine_cinn_jitc_syms.cmake
# ---------------------------------------
# Run as a script: cmake -DOBJDIR=<CMakeFiles/<target>.dir> [-DOBJCOPY=<path>]
#                        -P redefine_cinn_jitc_syms.cmake
#
# Renames the undefined xpurtc::CompileContext destructor references inside
# CINN's compiled objects so they bind to the SONAME-isolated
# libxpujitc_xtdk.so (see cmake/xpu/isolate_xpujitc.py) instead of the XHPC
# libxpujitc.so. This is the object-side counterpart of the export rename done
# on the isolated shared library.
#
# Only compiler.cc.o and compiler_xpu.cc.o reference the destructor; all other
# xpurtc symbols CINN uses are XTDK-unique and resolve without renaming.
#
# Invoked as a PRE_LINK step from cmake/cinn.cmake so it runs after the objects
# are (re)compiled but before cinnapi/cinncore(_static) are linked/archived.
# objcopy --redefine-sym is a no-op when the old symbol is absent, so this is
# safe to re-run on incremental builds.

if(NOT DEFINED OBJDIR)
  message(FATAL_ERROR "redefine_cinn_jitc_syms.cmake: OBJDIR not set")
endif()

if(NOT DEFINED OBJCOPY OR OBJCOPY STREQUAL "")
  set(OBJCOPY "objcopy")
endif()

set(_objs "${OBJDIR}/paddle/cinn/backends/compiler.cc.o"
          "${OBJDIR}/paddle/cinn/backends/xpu/compiler_xpu.cc.o")

foreach(_obj ${_objs})
  if(EXISTS "${_obj}")
    execute_process(
      COMMAND
        "${OBJCOPY}" --redefine-sym
        _ZN6xpurtc14CompileContextD1Ev=_ZN6Xpurtc14CompileContextD1Ev
        --redefine-sym
        _ZN6xpurtc14CompileContextD2Ev=_ZN6Xpurtc14CompileContextD2Ev "${_obj}"
      RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
      message(
        FATAL_ERROR
          "redefine_cinn_jitc_syms.cmake: objcopy failed (rc=${_rc}) on ${_obj}"
      )
    endif()
    message(STATUS "isolated xpurtc dtor ref in ${_obj}")
  endif()
endforeach()
