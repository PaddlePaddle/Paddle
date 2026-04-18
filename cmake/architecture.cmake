# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

include_guard()

function(paddle_normalize_target_arch out_var)
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _processor)
  if(_processor STREQUAL "")
    string(TOLOWER "${CMAKE_HOST_SYSTEM_PROCESSOR}" _processor)
  endif()

  if(_processor MATCHES "^(x86_64|amd64)$")
    set(_arch "x86_64")
  elseif(_processor MATCHES "^(aarch64|arm64)$")
    set(_arch "aarch64")
  else()
    set(_arch "${_processor}")
  endif()

  set(${out_var}
      "${_arch}"
      PARENT_SCOPE)
endfunction()

function(paddle_get_system_library_arch_dir out_var)
  paddle_normalize_target_arch(_arch)
  set(_dir "/usr/lib/${_arch}-linux-gnu")
  if(EXISTS "${_dir}")
    set(${out_var}
        "${_dir}"
        PARENT_SCOPE)
  else()
    set(${out_var}
        ""
        PARENT_SCOPE)
  endif()
endfunction()

function(paddle_detect_cuda_target_dir out_var)
  set(_cuda_target_dir "")
  if(NOT CUDA_TOOLKIT_ROOT_DIR)
    set(${out_var}
        "${_cuda_target_dir}"
        PARENT_SCOPE)
    return()
  endif()

  set(_targets_root "${CUDA_TOOLKIT_ROOT_DIR}/targets")
  paddle_normalize_target_arch(_arch)
  set(_candidates)
  if(_arch STREQUAL "aarch64")
    list(APPEND _candidates "sbsa-linux" "aarch64-linux")
  elseif(_arch STREQUAL "x86_64")
    list(APPEND _candidates "x86_64-linux")
  elseif(NOT _arch STREQUAL "")
    list(APPEND _candidates "${_arch}-linux")
  endif()

  foreach(_candidate IN LISTS _candidates)
    if(EXISTS "${_targets_root}/${_candidate}")
      set(_cuda_target_dir "${_candidate}")
      break()
    endif()
  endforeach()

  if(_cuda_target_dir STREQUAL "" AND EXISTS "${_targets_root}")
    file(
      GLOB _detected_targets
      LIST_DIRECTORIES true
      "${_targets_root}/*")
    list(LENGTH _detected_targets _detected_target_count)
    if(_detected_target_count EQUAL 1)
      list(GET _detected_targets 0 _detected_target)
      get_filename_component(_cuda_target_dir "${_detected_target}" NAME)
    endif()
  endif()

  set(${out_var}
      "${_cuda_target_dir}"
      PARENT_SCOPE)
endfunction()

function(paddle_get_llvm_native_target out_var)
  paddle_normalize_target_arch(_arch)
  if(_arch STREQUAL "aarch64")
    set(_llvm_target "AArch64")
  elseif(_arch STREQUAL "x86_64")
    set(_llvm_target "X86")
  else()
    string(TOUPPER "${_arch}" _llvm_target)
  endif()
  set(${out_var}
      "${_llvm_target}"
      PARENT_SCOPE)
endfunction()
