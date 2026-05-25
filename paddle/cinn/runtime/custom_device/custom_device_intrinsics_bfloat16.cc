// Copyright (c) 2026 CINN Authors. All Rights Reserved.
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
#ifdef CINN_WITH_CUSTOM_DEVICE
#include "paddle/cinn/backends/extern_func_jit_register.h"
#include "paddle/cinn/backends/function_prototype.h"
#include "paddle/cinn/common/bfloat16.h"
#include "paddle/cinn/common/type.h"
#include "paddle/cinn/runtime/custom_device/custom_device_backend_api.h"
#include "paddle/cinn/runtime/custom_device/custom_device_util.h"
#include "paddle/phi/backends/device_manager.h"

using cinn::common::bfloat16;
using cinn::runtime::custom_device::CustomBackendAPI;
using cinn_buffer_ptr_t = cinn_buffer_t *;
using cinn_int_ptr_t = int *;

namespace cinn {
namespace runtime {
namespace custom_device {
void ForceRegisterCustomDeviceIntrinsicsBFloat16() {
  auto dev_types = phi::DeviceManager::GetAllCustomDeviceTypes();
  std::string custom_device_name = "unknown_custom_device";
  int device_id = 0;
  if (!dev_types.empty()) {
    custom_device_name = dev_types[0];
    device_id = phi::DeviceManager::GetDevice(custom_device_name);
  }
  VLOG(0)
      << "Registering CINN Custom Device Intrinsics BFloat16 for Target Name: "
      << custom_device_name;

  cinn::common::Target target(
      cinn::common::Target::OS::Linux,
      cinn::common::CustomDeviceArch{custom_device_name, device_id},
      cinn::common::Target::Bit::k64,
      {cinn::common::Target::Feature::JIT},
      {});
  using cinn::backends::FunctionProto;

// bfloat16
#define REGISTER_EXTERN_FUNC_2_IN_1_BF16(func__)                             \
  REGISTER_EXTERN_SOURCE_FUNC_2_IN_1_OUT(cinn_custom_device_##func__##_bf16, \
                                         target,                             \
                                         bfloat16,                           \
                                         bfloat16,                           \
                                         bfloat16);

  REGISTER_EXTERN_FUNC_2_IN_1_BF16(pow)
  REGISTER_EXTERN_FUNC_2_IN_1_BF16(mod)

#undef REGISTER_EXTERN_FUNC_2_IN_1_BF16

#define REGISTER_EXTERN_FUNC_1_IN_1_BF16(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(        \
      cinn_custom_device_##func__##_bf16, target, bfloat16, bfloat16);

  REGISTER_EXTERN_FUNC_1_IN_1_BF16(ceil)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(floor)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(round)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(trunc)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(sin)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(cos)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(tan)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(exp)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(log)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(log2)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(log10)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(sqrt)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(rsqrt)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(cbrt)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(abs)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(erf)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(sinh)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(cosh)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(tanh)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(asin)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(acos)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(atan)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(asinh)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(acosh)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(atanh)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(sigmoid)

#undef REGISTER_EXTERN_FUNC_1_IN_1_BF16

#define REGISTER_EXTERN_FUNC_1_IN_1_BF16_OUT_BOOL(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(                 \
      cinn_custom_device_##func__##_bf16, target, bfloat16, bool);

  REGISTER_EXTERN_FUNC_1_IN_1_BF16_OUT_BOOL(isnan)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16_OUT_BOOL(isinf)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16_OUT_BOOL(isfinite)

#undef REGISTER_EXTERN_FUNC_1_IN_1_BF16_OUT_BOOL

#define REGISTER_CINN_CUSTOM_DEVICE_GT_NUM(TYPE_SUFFIX, TYPE)                 \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_custom_device_gt_num_##TYPE_SUFFIX, \
                                     target)                                  \
      .template SetRetType(cinn::common::type_of<int>())                      \
      .template AddInputType(cinn::common::type_of<cinn_buffer_ptr_t>())      \
      .template AddInputType(cinn::common::type_of<int>())                    \
      .template AddInputType<TYPE>()                                          \
      .template AddInputType(cinn::common::type_of<int>())                    \
      .template AddInputType(cinn::common::type_of<int>())                    \
      .End();

  REGISTER_CINN_CUSTOM_DEVICE_GT_NUM(bf16, bfloat16);

#undef REGISTER_CINN_CUSTOM_DEVICE_GT_NUM

#define REGISTER_CINN_CUSTOM_DEVICE_LT_NUM(TYPE_SUFFIX, TYPE)                 \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_custom_device_lt_num_##TYPE_SUFFIX, \
                                     target)                                  \
      .template SetRetType(cinn::common::type_of<int>())                      \
      .template AddInputType(cinn::common::type_of<cinn_buffer_ptr_t>())      \
      .template AddInputType(cinn::common::type_of<int>())                    \
      .template AddInputType<TYPE>()                                          \
      .template AddInputType(cinn::common::type_of<int>())                    \
      .template AddInputType(cinn::common::type_of<int>())                    \
      .End();

  REGISTER_CINN_CUSTOM_DEVICE_LT_NUM(bf16, bfloat16);

#undef REGISTER_CINN_CUSTOM_DEVICE_LT_NUM

#define REGISTER_CINN_CUSTOM_DEVICE_INDEX_ADD(TYPE_SUFFIX, TYPE)         \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(                                    \
      cinn_custom_device_index_add_##TYPE_SUFFIX, target)                \
      .template SetRetType<TYPE>()                                       \
      .template AddInputType<TYPE>()                                     \
      .template AddInputType(cinn::common::type_of<int>())               \
      .template AddInputType(cinn::common::type_of<cinn_buffer_ptr_t>()) \
      .template AddInputType(cinn::common::type_of<int>())               \
      .template AddInputType(cinn::common::type_of<int>())               \
      .template AddInputType(cinn::common::type_of<cinn_buffer_ptr_t>()) \
      .template AddInputType(cinn::common::type_of<int>())               \
      .End();

  REGISTER_CINN_CUSTOM_DEVICE_INDEX_ADD(bf16, bfloat16);

#undef REGISTER_CINN_CUSTOM_DEVICE_INDEX_ADD
}

}  // namespace custom_device
}  // namespace runtime
}  // namespace cinn
#endif  // CINN_WITH_CUSTOM_DEVICE
