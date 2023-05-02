// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#if !defined(_WIN32)

#include <type_traits>

#define PD_CUSTOM_PHI_KERNEL_STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg) \
  _PD_CUSTOM_PHI_KERNEL_STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg)

#define _PD_CUSTOM_PHI_KERNEL_STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg)  \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

#define PD_DECLARE_CAPI(module_name)                             \
  PD_CUSTOM_PHI_KERNEL_STATIC_ASSERT_GLOBAL_NAMESPACE(           \
      PD_DECLARE_tp_kernel_ns_check_##module_name##_,            \
      "PD_DECLARE_KERNEL must be called in global namespace.");  \
  extern int TouchCAPISymbolFor##module_name##_();               \
  UNUSED static int __declare_capi_symbol_for_##module_name##_ = \
      TouchCAPISymbolFor##module_name##_()

#define PD_REGISTER_CAPI(module_name)                           \
  PD_CUSTOM_PHI_KERNEL_STATIC_ASSERT_GLOBAL_NAMESPACE(          \
      PD_DECLARE_tp_kernel_ns_check_##module_name##_,           \
      "PD_DECLARE_KERNEL must be called in global namespace."); \
  int TouchCAPISymbolFor##module_name##_() { return 0; }

#endif
