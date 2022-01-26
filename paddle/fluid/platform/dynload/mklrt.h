/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <mkl_dfti.h>
#include <mutex>  // NOLINT

#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/pten/backends/dynload/port.h"

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag mklrt_dso_flag;
extern void* mklrt_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load mkldfti routine
 * via operator overloading.
 */
#define DYNAMIC_LOAD_MKLRT_WRAP(__name)                       \
  using DynLoad__##__name = pten::dynload::DynLoad__##__name; \
  extern DynLoad__##__name __name

// mkl_dfti.h has a macro that shadows the function with the same name
// un-defeine this macro so as to export that function
#undef DftiCreateDescriptor

#define MKLDFTI_ROUTINE_EACH(__macro) \
  __macro(DftiCreateDescriptor);      \
  __macro(DftiCreateDescriptor_s_1d); \
  __macro(DftiCreateDescriptor_d_1d); \
  __macro(DftiCreateDescriptor_s_md); \
  __macro(DftiCreateDescriptor_d_md); \
  __macro(DftiSetValue);              \
  __macro(DftiGetValue);              \
  __macro(DftiCommitDescriptor);      \
  __macro(DftiComputeForward);        \
  __macro(DftiComputeBackward);       \
  __macro(DftiFreeDescriptor);        \
  __macro(DftiErrorClass);            \
  __macro(DftiErrorMessage);

MKLDFTI_ROUTINE_EACH(DYNAMIC_LOAD_MKLRT_WRAP)

#undef DYNAMIC_LOAD_MKLRT_WRAP

// define another function to avoid naming conflict
DFTI_EXTERN MKL_LONG DftiCreateDescriptorX(DFTI_DESCRIPTOR_HANDLE* desc,
                                           enum DFTI_CONFIG_VALUE prec,
                                           enum DFTI_CONFIG_VALUE domain,
                                           MKL_LONG dim, MKL_LONG* sizes);

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
