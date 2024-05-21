// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <hip/hip_runtime.h>
#include <rocsparse.h>

#include <mutex>  // NOLINT
#include <type_traits>

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

namespace phi {
namespace dynload {
extern std::once_flag rocsparse_dso_flag;
extern void *rocsparse_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load rocsparse routine
 * via operator overloading.
 *
 * note: default dynamic linked libs
 */
#define DECLARE_DYNAMIC_LOAD_ROCSPARSE_WRAP(__name)                   \
  struct DynLoad__##__name {                                          \
    template <typename... Args>                                       \
    rocsparse_status operator()(Args... args) {                       \
      using rocsparse_func = decltype(&::__name);                     \
      std::call_once(rocsparse_dso_flag, []() {                       \
        rocsparse_dso_handle = phi::dynload::GetCusparseDsoHandle();  \
      });                                                             \
      static void *p_##__name = dlsym(rocsparse_dso_handle, #__name); \
      return reinterpret_cast<rocsparse_func>(p_##__name)(args...);   \
    }                                                                 \
  };                                                                  \
  extern DynLoad__##__name __name

#if defined(PADDLE_WITH_HIP)
#define ROCSPARSE_ROUTINE_EACH(__macro) \
  __macro(rocsparse_create_handle);     \
  __macro(rocsparse_destroy_handle);    \
  __macro(rocsparse_set_stream);        \
  __macro(rocsparse_csr2coo);

ROCSPARSE_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_ROCSPARSE_WRAP)

#if HIP_VERSION >= 402
#define ROCSPARSE_ROUTINE_EACH_R2(__macro) \
  __macro(rocsparse_create_coo_descr);     \
  __macro(rocsparse_create_csr_descr);     \
  __macro(rocsparse_destroy_spmat_descr);  \
  __macro(rocsparse_create_dnmat_descr);   \
  __macro(rocsparse_destroy_dnmat_descr);  \
  __macro(rocsparse_spmm);

ROCSPARSE_ROUTINE_EACH_R2(DECLARE_DYNAMIC_LOAD_ROCSPARSE_WRAP)
#endif

#if HIP_VERSION >= 403
#define ROCSPARSE_ROUTINE_EACH_R3(__macro) \
  __macro(rocsparse_sddmm_buffer_size);    \
  __macro(rocsparse_sddmm_preprocess);     \
  __macro(rocsparse_sddmm);

ROCSPARSE_ROUTINE_EACH_R3(DECLARE_DYNAMIC_LOAD_ROCSPARSE_WRAP)
#endif

#endif  // PADDLE_WITH_HIP

#undef DECLARE_DYNAMIC_LOAD_ROCSPARSE_WRAP
}  // namespace dynload
}  // namespace phi
