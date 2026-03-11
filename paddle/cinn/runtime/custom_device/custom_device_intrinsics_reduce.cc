// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/cinn/common/float16.h"
#include "paddle/cinn/runtime/custom_device/custom_device_backend_api.h"
#include "paddle/phi/backends/device_manager.h"

#define CINN_CUSTOM_DEVICE_FP16

using cinn::common::float16;
using cinn::runtime::custom_device::CustomBackendAPI;

namespace cinn {
namespace runtime {
namespace custom_device {
void ForceRegisterCustomDeviceIntrinsicsReduce() {
  auto dev_types = phi::DeviceManager::GetAllCustomDeviceTypes();
  std::string custom_device_name = "unknown_custom_device";
  int device_id = 0;
  if (!dev_types.empty()) {
    custom_device_name = dev_types[0];
    device_id = phi::DeviceManager::GetDevice(custom_device_name);
  }
  VLOG(0)
      << "Force Registering Custom Device Intrinsics Reduce  for Target Name: "
      << custom_device_name;

  cinn::common::Target target(
      cinn::common::Target::OS::Linux,
      cinn::common::CustomDeviceArch{custom_device_name, device_id},
      cinn::common::Target::Bit::k64,
      {cinn::common::Target::Feature::JIT},
      {});

#define EXPAND_ARG_REDUCE_MACRO(MACRO, DTYPE, TYPE, ...) \
  MACRO(max_argidx_##DTYPE##_i32, TYPE, ##__VA_ARGS__)   \
  MACRO(min_argidx_##DTYPE##_i32, TYPE, ##__VA_ARGS__)   \
  MACRO(max_argidx_##DTYPE##_i64, TYPE, ##__VA_ARGS__)   \
  MACRO(min_argidx_##DTYPE##_i64, TYPE, ##__VA_ARGS__)

#define EXPAND_REDUCE_INT32_REGISTER_MACRO(MACRO, ...) \
  MACRO(sum_int32, int, ##__VA_ARGS__)                 \
  MACRO(prod_int32, int, ##__VA_ARGS__)                \
  MACRO(max_int32, int, ##__VA_ARGS__)                 \
  MACRO(min_int32, int, ##__VA_ARGS__)                 \
  EXPAND_ARG_REDUCE_MACRO(MACRO, i32, int, ##__VA_ARGS__)

#define EXPAND_REDUCE_INT64_REGISTER_MACRO(MACRO, ...) \
  MACRO(sum_int64, int64_t, ##__VA_ARGS__)             \
  MACRO(prod_int64, int64_t, ##__VA_ARGS__)            \
  MACRO(max_int64, int64_t, ##__VA_ARGS__)             \
  MACRO(min_int64, int64_t, ##__VA_ARGS__)             \
  EXPAND_ARG_REDUCE_MACRO(MACRO, i64, int64_t, ##__VA_ARGS__)

#define EXPAND_REDUCE_FP32_REGISTER_MACRO(MACRO, ...) \
  MACRO(sum_fp32, float, ##__VA_ARGS__)               \
  MACRO(prod_fp32, float, ##__VA_ARGS__)              \
  MACRO(max_fp32, float, ##__VA_ARGS__)               \
  MACRO(min_fp32, float, ##__VA_ARGS__)               \
  MACRO(sum_welford_fp32, float, ##__VA_ARGS__)       \
  EXPAND_ARG_REDUCE_MACRO(MACRO, fp32, float, ##__VA_ARGS__)

#define EXPAND_REDUCE_BOOL_REGISTER_MACRO(MACRO, ...) \
  MACRO(all, bool, ##__VA_ARGS__)                     \
  MACRO(any, bool, ##__VA_ARGS__)

#define EXPAND_REDUCE_FP64_REGISTER_MACRO(MACRO, ...) \
  MACRO(sum_fp64, double, ##__VA_ARGS__)              \
  MACRO(prod_fp64, double, ##__VA_ARGS__)             \
  MACRO(max_fp64, double, ##__VA_ARGS__)              \
  MACRO(min_fp64, double, ##__VA_ARGS__)              \
  MACRO(sum_welford_fp64, double, ##__VA_ARGS__)      \
  EXPAND_ARG_REDUCE_MACRO(MACRO, fp64, double, ##__VA_ARGS__)

#ifdef CINN_CUSTOM_DEVICE_BF16
#define EXPAND_REDUCE_BF16_REGISTER_MACRO(MACRO, ...) \
  MACRO(sum_bf16, bfloat16, ##__VA_ARGS__)            \
  MACRO(prod_bf16, bfloat16, ##__VA_ARGS__)           \
  MACRO(max_bf16, bfloat16, ##__VA_ARGS__)            \
  MACRO(min_bf16, bfloat16, ##__VA_ARGS__)
#endif

#ifdef CINN_CUSTOM_DEVICE_FP16
#define EXPAND_REDUCE_FP16_REGISTER_MACRO(MACRO, ...) \
  MACRO(sum_fp16, float16, ##__VA_ARGS__)             \
  MACRO(prod_fp16, float16, ##__VA_ARGS__)            \
  MACRO(max_fp16, float16, ##__VA_ARGS__)             \
  MACRO(min_fp16, float16, ##__VA_ARGS__)             \
  EXPAND_ARG_REDUCE_MACRO(MACRO, fp16, float16, ##__VA_ARGS__)
#endif

#define EXPAND_REDUCE_UINT8_REGISTER_MACRO(MACRO, ...) \
  EXPAND_ARG_REDUCE_MACRO(MACRO, u8, uint8_t, ##__VA_ARGS__)

#define EXPAND_REDUCE_INT16_REGISTER_MACRO(MACRO, ...) \
  EXPAND_ARG_REDUCE_MACRO(MACRO, i16, int16_t, ##__VA_ARGS__)

#define REGISTER_BLOCK_REDUCE_FUNC_IMPL(REDUCE_TYPE, DTYPE)                   \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_block_reduce_##REDUCE_TYPE, target) \
      .template SetRetType<DTYPE>()                                           \
      .template AddInputType<DTYPE>()                                         \
      .template AddInputType(cinn::common::type_of<cinn_buffer_t *>())        \
      .template AddInputType(cinn::common::type_of<bool>())                   \
      .End();

  EXPAND_REDUCE_INT32_REGISTER_MACRO(REGISTER_BLOCK_REDUCE_FUNC_IMPL)
  EXPAND_REDUCE_INT64_REGISTER_MACRO(REGISTER_BLOCK_REDUCE_FUNC_IMPL)
  EXPAND_REDUCE_FP32_REGISTER_MACRO(REGISTER_BLOCK_REDUCE_FUNC_IMPL)
  EXPAND_REDUCE_FP64_REGISTER_MACRO(REGISTER_BLOCK_REDUCE_FUNC_IMPL)
  EXPAND_REDUCE_BOOL_REGISTER_MACRO(REGISTER_BLOCK_REDUCE_FUNC_IMPL)
  EXPAND_REDUCE_UINT8_REGISTER_MACRO(REGISTER_BLOCK_REDUCE_FUNC_IMPL)
  EXPAND_REDUCE_INT16_REGISTER_MACRO(REGISTER_BLOCK_REDUCE_FUNC_IMPL)

#ifdef CINN_CUSTOM_DEVICE_BF16
  EXPAND_REDUCE_BF16_REGISTER_MACRO(REGISTER_BLOCK_REDUCE_FUNC_IMPL)
#endif

#ifdef CINN_CUSTOM_DEVICE_FP16
  EXPAND_REDUCE_FP16_REGISTER_MACRO(REGISTER_BLOCK_REDUCE_FUNC_IMPL)
#endif

#undef REGISTER_BLOCK_REDUCE_FUNC_IMPL

#define REGISTER_DISCRETE_REDUCE_FUNC_IMPL(REDUCE_TYPE, DTYPE)           \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_discrete_reduce_##REDUCE_TYPE, \
                                     target)                             \
      .template SetRetType<DTYPE>()                                      \
      .template AddInputType<DTYPE>()                                    \
      .template AddInputType(cinn::common::type_of<cinn_buffer_t *>())   \
      .End();

  EXPAND_REDUCE_INT32_REGISTER_MACRO(REGISTER_DISCRETE_REDUCE_FUNC_IMPL)
  EXPAND_REDUCE_INT64_REGISTER_MACRO(REGISTER_DISCRETE_REDUCE_FUNC_IMPL)
  EXPAND_REDUCE_FP32_REGISTER_MACRO(REGISTER_DISCRETE_REDUCE_FUNC_IMPL)
  EXPAND_REDUCE_FP64_REGISTER_MACRO(REGISTER_DISCRETE_REDUCE_FUNC_IMPL)
  EXPAND_REDUCE_BOOL_REGISTER_MACRO(REGISTER_DISCRETE_REDUCE_FUNC_IMPL)

#ifdef CINN_CUSTOM_DEVICE_BF16
  EXPAND_REDUCE_BF16_REGISTER_MACRO(REGISTER_DISCRETE_REDUCE_FUNC_IMPL)
#endif

#ifdef CINN_CUSTOM_DEVICE_FP16
  EXPAND_REDUCE_FP16_REGISTER_MACRO(REGISTER_DISCRETE_REDUCE_FUNC_IMPL)
#endif

#undef REGISTER_DISCRETE_REDUCE_FUNC_IMPL

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_grid_reduce_update_semaphore, target)
      .template SetRetType(cinn::common::type_of<bool>())
      .template AddInputType(cinn::common::type_of<int *>())
      .End();

#define REGISTER_BLOCK_SHUFFLE_FUNC_IMPL(REDUCE_TYPE, DTYPE)              \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(block_shuffle_##REDUCE_TYPE, target) \
      .template SetRetType<DTYPE>()                                       \
      .template AddInputType(cinn::common::type_of<cinn_buffer_t *>())    \
      .template AddInputType(cinn::common::type_of<int>())                \
      .End();

  EXPAND_REDUCE_INT32_REGISTER_MACRO(REGISTER_BLOCK_SHUFFLE_FUNC_IMPL)
  EXPAND_REDUCE_INT64_REGISTER_MACRO(REGISTER_BLOCK_SHUFFLE_FUNC_IMPL)
  EXPAND_REDUCE_FP32_REGISTER_MACRO(REGISTER_BLOCK_SHUFFLE_FUNC_IMPL)
  EXPAND_REDUCE_FP64_REGISTER_MACRO(REGISTER_BLOCK_SHUFFLE_FUNC_IMPL)
  EXPAND_REDUCE_BOOL_REGISTER_MACRO(REGISTER_BLOCK_SHUFFLE_FUNC_IMPL)

#ifdef CINN_CUSTOM_DEVICE_BF16
  EXPAND_REDUCE_BF16_REGISTER_MACRO(REGISTER_BLOCK_SHUFFLE_FUNC_IMPL)
#endif

#ifdef CINN_CUSTOM_DEVICE_FP16
  EXPAND_REDUCE_FP16_REGISTER_MACRO(REGISTER_BLOCK_SHUFFLE_FUNC_IMPL)
#endif

#undef REGISTER_BLOCK_SHUFFLE_FUNC_IMPL

#define REGISTER_GRID_REDUCE_FUNC_IMPL(REDUCE_TYPE, DTYPE)                   \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_grid_reduce_##REDUCE_TYPE, target) \
      .template SetRetType<DTYPE>()                                          \
      .template AddInputType(cinn::common::type_of<cinn_buffer_t *>())       \
      .template AddInputType(cinn::common::type_of<int>())                   \
      .End();
  EXPAND_REDUCE_INT32_REGISTER_MACRO(REGISTER_GRID_REDUCE_FUNC_IMPL)
  EXPAND_REDUCE_INT64_REGISTER_MACRO(REGISTER_GRID_REDUCE_FUNC_IMPL)
  EXPAND_REDUCE_FP32_REGISTER_MACRO(REGISTER_GRID_REDUCE_FUNC_IMPL)
  EXPAND_REDUCE_FP64_REGISTER_MACRO(REGISTER_GRID_REDUCE_FUNC_IMPL)
  EXPAND_REDUCE_BOOL_REGISTER_MACRO(REGISTER_GRID_REDUCE_FUNC_IMPL)

#ifdef CINN_CUSTOM_DEVICE_BF16
  EXPAND_REDUCE_BF16_REGISTER_MACRO(REGISTER_GRID_REDUCE_FUNC_IMPL)
#endif

#ifdef CINN_CUSTOM_DEVICE_FP16
  EXPAND_REDUCE_FP16_REGISTER_MACRO(REGISTER_GRID_REDUCE_FUNC_IMPL)
#endif

#undef REGISTER_GRID_REDUCE_FUNC_IMPL

#undef EXPAND_REDUCE_INT32_REGISTER_MACRO
#undef EXPAND_REDUCE_INT64_REGISTER_MACRO
#undef EXPAND_REDUCE_FP32_REGISTER_MACRO
#undef EXPAND_REDUCE_FP64_REGISTER_MACRO
#undef EXPAND_REDUCE_BOOL_REGISTER_MACRO

#ifdef CINN_CUSTOM_DEVICE_BF16
#undef EXPAND_REDUCE_BF16_REGISTER_MACRO
#endif

#ifdef CINN_CUSTOM_DEVICE_FP16
#undef EXPAND_REDUCE_FP16_REGISTER_MACRO
#endif
}

}  // namespace custom_device
}  // namespace runtime
}  // namespace cinn
#endif  // CINN_WITH_CUSTOM_DEVICE
