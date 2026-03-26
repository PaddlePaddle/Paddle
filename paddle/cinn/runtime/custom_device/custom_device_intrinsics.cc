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
#include "paddle/cinn/backends/llvm/runtime_symbol_registry.h"
#include "paddle/cinn/runtime/custom_device/custom_device_backend_api.h"
#include "paddle/cinn/runtime/custom_device/custom_device_util.h"
#include "paddle/phi/backends/device_manager.h"

namespace cinn {
namespace runtime {
namespace custom_device {

using cinn::backends::GlobalSymbolRegistry;
using cinn::runtime::custom_device::CustomBackendAPI;
using cinn_buffer_ptr_t = cinn_buffer_t *;
using cinn_int_ptr_t = int *;

void ForceRegisterCinnCustomDeviceHostAPI() {
  VLOG(0) << "Registering CINN Custom Device Host API...";

  GlobalSymbolRegistry::Global().RegisterFn(
      "backend_api.custom_device",
      reinterpret_cast<void *>(CustomBackendAPI::Global()));

  using cinn::runtime::custom_device::cinn_call_custom_device_kernel;
  REGISTER_EXTERN_FUNC_HELPER(cinn_call_custom_device_kernel,
                              cinn::common::DefaultHostTarget())
      .template SetRetType<void>()
      .template AddInputType<void *>()
      .template AddInputType<void *>()
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType<void *>()
      .End();

  using cinn::runtime::custom_device::infer_shape_set_value;
  REGISTER_EXTERN_FUNC_HELPER(infer_shape_set_value,
                              cinn::common::DefaultHostTarget())
      .template SetRetType<void>()
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int64_t>())
      .template AddInputType(cinn::common::type_of<int64_t **>())
      .End();
}

void ForceRegisterCinnCustomDeviceIntrinsics() {
  auto dev_types = phi::DeviceManager::GetAllCustomDeviceTypes();
  std::string custom_device_name = "unknown_custom_device";
  int device_id = 0;
  if (!dev_types.empty()) {
    custom_device_name = dev_types[0];
    device_id = phi::DeviceManager::GetDevice(custom_device_name);
  }

  VLOG(0) << "Registering CINN Custom Device Intrinsics for Target Name: "
          << custom_device_name;

  cinn::common::Target target(
      cinn::common::Target::OS::Linux,
      cinn::common::CustomDeviceArch{custom_device_name, device_id},
      cinn::common::Target::Bit::k64,
      {cinn::common::Target::Feature::JIT},
      {});

// bool for 1 input 1 output
#define REGISTER_EXTERN_FUNC_1_IN_1_OUT_BOOL(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(            \
      cinn_custom_device_##func__##_bool, target, bool, bool)

  REGISTER_EXTERN_FUNC_1_IN_1_OUT_BOOL(bitwise_not);

#undef REGISTER_EXTERN_FUNC_1_IN_1_OUT_BOOL

// bool for 2 input 1 output
#define REGISTER_EXTERN_FUNC_2_IN_1_OUT_BOOL(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_2_IN_1_OUT(            \
      cinn_custom_device_##func__##_bool, target, bool, bool, bool)

  REGISTER_EXTERN_FUNC_2_IN_1_OUT_BOOL(bitwise_and);
  REGISTER_EXTERN_FUNC_2_IN_1_OUT_BOOL(bitwise_or);
  REGISTER_EXTERN_FUNC_2_IN_1_OUT_BOOL(bitwise_xor);

#undef REGISTER_EXTERN_FUNC_2_IN_1_OUT_BOOL

// uint8 for 1 input 1 output
#define REGISTER_EXTERN_FUNC_1_IN_1_OUT_UINT8(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(             \
      cinn_custom_device_##func__##_uint8, target, uint8_t, uint8_t)

  REGISTER_EXTERN_FUNC_1_IN_1_OUT_UINT8(bitwise_not);

#undef REGISTER_EXTERN_FUNC_1_IN_1_OUT_UINT8

// uint8 for 2 input 1 output
#define REGISTER_EXTERN_FUNC_2_IN_1_OUT_UINT8(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_2_IN_1_OUT(             \
      cinn_custom_device_##func__##_uint8, target, uint8_t, uint8_t, uint8_t);

  REGISTER_EXTERN_FUNC_2_IN_1_OUT_UINT8(bitwise_and);
  REGISTER_EXTERN_FUNC_2_IN_1_OUT_UINT8(bitwise_or);
  REGISTER_EXTERN_FUNC_2_IN_1_OUT_UINT8(bitwise_xor);
  REGISTER_EXTERN_FUNC_2_IN_1_OUT_UINT8(logical_right_shift);

#undef REGISTER_EXTERN_FUNC_2_IN_1_OUT_UINT8

// int8 for 1 input 1 output
#define REGISTER_EXTERN_FUNC_1_IN_1_OUT_INT8(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(            \
      cinn_custom_device_##func__##_int8, target, int8_t, int8_t)

  REGISTER_EXTERN_FUNC_1_IN_1_OUT_INT8(bitwise_not);

#undef REGISTER_EXTERN_FUNC_1_IN_1_OUT_INT8

// int8 for 2 input 1 output
#define REGISTER_EXTERN_FUNC_2_IN_1_OUT_INT8(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_2_IN_1_OUT(            \
      cinn_custom_device_##func__##_int8, target, int8_t, int8_t, int8_t);

  REGISTER_EXTERN_FUNC_2_IN_1_OUT_INT8(bitwise_and);
  REGISTER_EXTERN_FUNC_2_IN_1_OUT_INT8(bitwise_or);
  REGISTER_EXTERN_FUNC_2_IN_1_OUT_INT8(bitwise_xor);
  REGISTER_EXTERN_FUNC_2_IN_1_OUT_INT8(logical_right_shift);

#undef REGISTER_EXTERN_FUNC_2_IN_1_OUT_INT8

// int16 for 1 input 1 output
#define REGISTER_EXTERN_FUNC_1_IN_1_OUT_INT16(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(             \
      cinn_custom_device_##func__##_int16, target, int16_t, int16_t)

  REGISTER_EXTERN_FUNC_1_IN_1_OUT_INT16(bitwise_not);

#undef REGISTER_EXTERN_FUNC_1_IN_1_OUT_INT16

// int16 for 2 input 1 output
#define REGISTER_EXTERN_FUNC_2_IN_1_OUT_INT16(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_2_IN_1_OUT(             \
      cinn_custom_device_##func__##_int16, target, int16_t, int16_t, int16_t);

  REGISTER_EXTERN_FUNC_2_IN_1_OUT_INT16(bitwise_and);
  REGISTER_EXTERN_FUNC_2_IN_1_OUT_INT16(bitwise_or);
  REGISTER_EXTERN_FUNC_2_IN_1_OUT_INT16(bitwise_xor);
  REGISTER_EXTERN_FUNC_2_IN_1_OUT_INT16(logical_right_shift);

#undef REGISTER_EXTERN_FUNC_2_IN_1_OUT_INT16

// float
#define REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(             \
      cinn_custom_device_##func__##_fp32, target, float, float);

  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(abs);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(exp);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(erf);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(sqrt);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(rsqrt);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(log);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(log2);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(log10);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(floor);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(ceil);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(round);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(trunc);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(cos);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(cosh);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(tan);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(sin);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(sinh);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(acos);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(acosh);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(asin);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(asinh);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(atan);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(atanh);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(tanh);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(cbrt);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(sigmoid);

#undef REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT

#define REGISTER_EXTERN_FUNC_1_IN_FLOAT_1_OUT_BOOL(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(                  \
      cinn_custom_device_##func__##_fp32, target, float, bool);

  REGISTER_EXTERN_FUNC_1_IN_FLOAT_1_OUT_BOOL(isnan);
  REGISTER_EXTERN_FUNC_1_IN_FLOAT_1_OUT_BOOL(isfinite);
  REGISTER_EXTERN_FUNC_1_IN_FLOAT_1_OUT_BOOL(isinf);

#undef REGISTER_EXTERN_FUNC_1_IN_FLOAT_1_OUT_BOOL

#define REGISTER_EXTERN_FUNC_2_IN_1_FLOAT(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_2_IN_1_OUT(         \
      cinn_custom_device_##func__##_fp32, target, float, float, float);

  REGISTER_EXTERN_FUNC_2_IN_1_FLOAT(pow)
  REGISTER_EXTERN_FUNC_2_IN_1_FLOAT(mod)

#undef REGISTER_EXTERN_FUNC_2_IN_1_FLOAT

  // double

#define REGISTER_EXTERN_FUNC_1_IN_1_FP64(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(        \
      cinn_custom_device_##func__##_fp64, target, double, double);

  REGISTER_EXTERN_FUNC_1_IN_1_FP64(abs);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(exp);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(erf);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(sqrt);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(rsqrt);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(log);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(log2);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(log10);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(floor);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(ceil);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(round);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(trunc);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(cos);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(cosh);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(tan);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(sin);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(sinh);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(acos);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(acosh);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(asin);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(asinh);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(atan);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(atanh);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(tanh);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(cbrt);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(sigmoid);

#undef REGISTER_EXTERN_FUNC_1_IN_1_FP64

#define REGISTER_EXTERN_FUNC_1_IN_FP64_1_OUT_BOOL(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(                 \
      cinn_custom_device_##func__##_fp64, target, double, bool);

  REGISTER_EXTERN_FUNC_1_IN_FP64_1_OUT_BOOL(isnan);
  REGISTER_EXTERN_FUNC_1_IN_FP64_1_OUT_BOOL(isfinite);
  REGISTER_EXTERN_FUNC_1_IN_FP64_1_OUT_BOOL(isinf);

#undef REGISTER_EXTERN_FUNC_1_IN_FP64_1_OUT_BOOL

#define REGISTER_EXTERN_FUNC_2_IN_1_FP64(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_2_IN_1_OUT(        \
      cinn_custom_device_##func__##_fp64, target, double, double, double);

  REGISTER_EXTERN_FUNC_2_IN_1_FP64(pow)
  REGISTER_EXTERN_FUNC_2_IN_1_FP64(mod)

#undef REGISTER_EXTERN_FUNC_2_IN_1_FP64

  // int32

#define REGISTER_EXTERN_FUNC_1_IN_1_INT32(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(         \
      cinn_custom_device_##func__##_int32, target, int, int);

  REGISTER_EXTERN_FUNC_1_IN_1_INT32(bitwise_not)
  REGISTER_EXTERN_FUNC_1_IN_1_INT32(clz)
  REGISTER_EXTERN_FUNC_1_IN_1_INT32(popc)
  REGISTER_EXTERN_FUNC_1_IN_1_INT32(trunc)

#undef REGISTER_EXTERN_FUNC_1_IN_1_INT32

#define REGISTER_EXTERN_FUNC_1_IN_1_INT64(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(         \
      cinn_custom_device_##func__##_int64, target, int64_t, int64_t);

  REGISTER_EXTERN_FUNC_1_IN_1_INT64(bitwise_not)
  REGISTER_EXTERN_FUNC_1_IN_1_INT64(clz)
  REGISTER_EXTERN_FUNC_1_IN_1_INT64(popc)
  REGISTER_EXTERN_FUNC_1_IN_1_INT64(trunc)
  REGISTER_EXTERN_FUNC_1_IN_1_INT64(abs)

#undef REGISTER_EXTERN_FUNC_1_IN_1_INT64

#define REGISTER_EXTERN_FUNC_2_IN_1_INT32(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_2_IN_1_OUT(         \
      cinn_custom_device_##func__##_int32, target, int, int, int);

  REGISTER_EXTERN_FUNC_2_IN_1_INT32(pow)
  REGISTER_EXTERN_FUNC_2_IN_1_INT32(left_shift)
  REGISTER_EXTERN_FUNC_2_IN_1_INT32(right_shift)
  REGISTER_EXTERN_FUNC_2_IN_1_INT32(bitwise_and)
  REGISTER_EXTERN_FUNC_2_IN_1_INT32(bitwise_or)
  REGISTER_EXTERN_FUNC_2_IN_1_INT32(bitwise_xor)
  REGISTER_EXTERN_FUNC_2_IN_1_INT32(logical_right_shift)
  REGISTER_EXTERN_FUNC_2_IN_1_INT32(mod)

#undef REGISTER_EXTERN_FUNC_2_IN_1_INT32

#define REGISTER_EXTERN_FUNC_2_IN_1_INT64(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_2_IN_1_OUT(         \
      cinn_custom_device_##func__##_int64, target, int64_t, int64_t, int64_t);

  REGISTER_EXTERN_FUNC_2_IN_1_INT64(pow)
  REGISTER_EXTERN_FUNC_2_IN_1_INT64(bitwise_and)
  REGISTER_EXTERN_FUNC_2_IN_1_INT64(bitwise_or)
  REGISTER_EXTERN_FUNC_2_IN_1_INT64(bitwise_xor)
  REGISTER_EXTERN_FUNC_2_IN_1_INT64(mod)
  REGISTER_EXTERN_FUNC_2_IN_1_INT64(logical_right_shift)

#undef REGISTER_EXTERN_FUNC_2_IN_1_INT64

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_custom_device_find_int, target)
      .template SetRetType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<cinn_buffer_ptr_t>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_custom_device_find_float, target)
      .template SetRetType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<cinn_buffer_ptr_t>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<float>())
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_custom_device_find_int_nd, target)
      .template SetRetType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<cinn_buffer_ptr_t>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_custom_device_find_float_nd, target)
      .template SetRetType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<cinn_buffer_ptr_t>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<float>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_custom_device_find_int_from, target)
      .template SetRetType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<cinn_buffer_ptr_t>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_custom_device_find_float_from, target)
      .template SetRetType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<cinn_buffer_ptr_t>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<float>())
      .template AddInputType(cinn::common::type_of<int>())
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_custom_device_next_smallest_int32,
                                     target)
      .template SetRetType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<cinn_buffer_ptr_t>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .End();

#define _REGISTER_CINN_NVGPU_LT_NUM(TYPE_SUFFIX, TYPE)                        \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_custom_device_lt_num_##TYPE_SUFFIX, \
                                     target)                                  \
      .template SetRetType(cinn::common::type_of<int>())                      \
      .template AddInputType(cinn::common::type_of<cinn_buffer_ptr_t>())      \
      .template AddInputType(cinn::common::type_of<int>())                    \
      .template AddInputType(cinn::common::type_of<TYPE>())                   \
      .template AddInputType(cinn::common::type_of<int>())                    \
      .template AddInputType(cinn::common::type_of<int>())                    \
      .End();

  _REGISTER_CINN_NVGPU_LT_NUM(fp32, float);
  _REGISTER_CINN_NVGPU_LT_NUM(fp64, double);
  _REGISTER_CINN_NVGPU_LT_NUM(uint8, uint8_t);
  _REGISTER_CINN_NVGPU_LT_NUM(int16, int16_t);

  _REGISTER_CINN_NVGPU_LT_NUM(int32, int);
  _REGISTER_CINN_NVGPU_LT_NUM(int64, int64_t);

#undef _REGISTER_CINN_NVGPU_LT_NUM

#define _REGISTER_CINN_NVGPU_GT_NUM(TYPE_SUFFIX, TYPE)                        \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_custom_device_gt_num_##TYPE_SUFFIX, \
                                     target)                                  \
      .template SetRetType(cinn::common::type_of<int>())                      \
      .template AddInputType(cinn::common::type_of<cinn_buffer_ptr_t>())      \
      .template AddInputType(cinn::common::type_of<int>())                    \
      .template AddInputType(cinn::common::type_of<TYPE>())                   \
      .template AddInputType(cinn::common::type_of<int>())                    \
      .template AddInputType(cinn::common::type_of<int>())                    \
      .End();

  _REGISTER_CINN_NVGPU_GT_NUM(fp32, float);
  _REGISTER_CINN_NVGPU_GT_NUM(fp64, double);
  _REGISTER_CINN_NVGPU_GT_NUM(uint8, uint8_t);
  _REGISTER_CINN_NVGPU_GT_NUM(int16, int16_t);
  _REGISTER_CINN_NVGPU_GT_NUM(int32, int);
  _REGISTER_CINN_NVGPU_GT_NUM(int64, int64_t);

#undef _REGISTER_CINN_NVGPU_GT_NUM

#define _REGISTER_CINN_NVGPU_INDEX_ADD(TYPE_SUFFIX, TYPE)                \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(                                    \
      cinn_custom_device_index_add_##TYPE_SUFFIX, target)                \
      .template SetRetType(cinn::common::type_of<TYPE>())                \
      .template AddInputType(cinn::common::type_of<TYPE>())              \
      .template AddInputType(cinn::common::type_of<int>())               \
      .template AddInputType(cinn::common::type_of<cinn_buffer_ptr_t>()) \
      .template AddInputType(cinn::common::type_of<int>())               \
      .template AddInputType(cinn::common::type_of<int>())               \
      .template AddInputType(cinn::common::type_of<cinn_buffer_ptr_t>()) \
      .template AddInputType(cinn::common::type_of<int>())               \
      .End();

  _REGISTER_CINN_NVGPU_INDEX_ADD(bool, bool);
  _REGISTER_CINN_NVGPU_INDEX_ADD(int8, int8_t);
  _REGISTER_CINN_NVGPU_INDEX_ADD(int32, int32_t);
  _REGISTER_CINN_NVGPU_INDEX_ADD(int64, int64_t);
  _REGISTER_CINN_NVGPU_INDEX_ADD(fp32, float);
  _REGISTER_CINN_NVGPU_INDEX_ADD(fp64, double);

#undef _REGISTER_CINN_NVGPU_INDEX_ADD

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_custom_device_resize_bilinear, target)
      .template SetRetType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<cinn_buffer_ptr_t>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_custom_device_resize_bicubic, target)
      .template SetRetType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<cinn_buffer_ptr_t>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .template AddInputType(cinn::common::type_of<int>())
      .End();
}

}  // namespace custom_device
}  // namespace runtime
}  // namespace cinn
#endif  // CINN_WITH_CUSTOM_DEVICE
