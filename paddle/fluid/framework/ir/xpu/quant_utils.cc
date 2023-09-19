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

#include "paddle/fluid/framework/ir/xpu/quant_utils.h"
#include <vector>
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/assign_kernel.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace paddle {
namespace framework {
namespace ir {

void Assign(const phi::DenseTensor& in, phi::DenseTensor* out) {
  auto* cpu_ctx = static_cast<phi::CPUContext*>(
      platform::DeviceContextPool::Instance().Get(phi::CPUPlace()));
  out->Resize(in.dims());
  out->set_type(in.dtype());
  out->set_layout(in.layout());

  paddle::experimental::CheckAndTrans2Contiguous(
      const_cast<phi::DenseTensor*>(&in));
  phi::AssignKernel(*cpu_ctx, in, out);
}

void Transpose2D(phi::DenseTensor* in, phi::DenseTensor* out) {
  paddle::experimental::CheckAndTrans2Contiguous(in);
  auto in_dims = in->dims();
  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      2,
      platform::errors::InvalidArgument(
          "In dims rank should be 2, but received in dims size is [%d].",
          in_dims.size()));

  phi::DenseTensor trans_tensor;
  phi::DenseTensor* out_ptr = out == nullptr ? &trans_tensor : out;
  out_ptr->Resize({in_dims[1], in_dims[0]});
  out_ptr->set_type(in->type());
  out_ptr->set_layout(in->layout());

  auto* cpu_ctx = static_cast<phi::CPUContext*>(
      platform::DeviceContextPool::Instance().Get(phi::CPUPlace()));
  std::vector<int> axis{1, 0};
  switch (in->dtype()) {
    case phi::DataType::FLOAT16:
      phi::TransposeKernel<phi::dtype::float16>(*cpu_ctx, *in, axis, out_ptr);
      break;
    case phi::DataType::FLOAT32:
      phi::TransposeKernel<float>(*cpu_ctx, *in, axis, out_ptr);
      break;
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Only support fp16 and fp32, but received dtype is %s.",
          phi::DataTypeToString(in->dtype())));
      break;
  }

  if (out == nullptr) {
    Assign(*out_ptr, in);
  }
}

void CastToInt32(phi::DenseTensor* in, phi::DenseTensor* out) {
  auto* cpu_ctx = static_cast<phi::CPUContext*>(
      platform::DeviceContextPool::Instance().Get(phi::CPUPlace()));

  phi::DenseTensor int32_tensor;
  phi::DenseTensor* out_ptr = out == nullptr ? &int32_tensor : out;
  out_ptr->Resize(in->dims());
  out_ptr->set_type(phi::DataType::INT32);
  out_ptr->set_layout(in->layout());

  switch (in->dtype()) {
    case phi::DataType::INT64:
      phi::CastKernel<int64_t>(*cpu_ctx, *in, phi::DataType::INT32, out_ptr);
      break;
    case phi::DataType::INT32:
      if (out == nullptr) {
        return;
      } else {
        phi::AssignKernel(*cpu_ctx, *in, out_ptr);
      }
      break;
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Only support int64 and int32, but received dtype is %s.",
          phi::DataTypeToString(in->dtype())));
      break;
  }

  if (out == nullptr) {
    Assign(*out_ptr, in);
  }
}

void CastToFp32(phi::DenseTensor* in, phi::DenseTensor* out) {
  auto* cpu_ctx = static_cast<phi::CPUContext*>(
      platform::DeviceContextPool::Instance().Get(phi::CPUPlace()));

  paddle::experimental::CheckAndTrans2Contiguous(in);

  phi::DenseTensor fp32_tensor;
  phi::DenseTensor* out_ptr = out == nullptr ? &fp32_tensor : out;
  out_ptr->Resize(in->dims());
  out_ptr->set_type(phi::DataType::FLOAT32);
  out_ptr->set_layout(in->layout());

  switch (in->dtype()) {
    case phi::DataType::FLOAT16:
      phi::CastKernel<phi::dtype::float16>(
          *cpu_ctx, *in, phi::DataType::FLOAT32, out_ptr);
      break;
    case phi::DataType::FLOAT32:
      if (out == nullptr) {
        return;
      } else {
        phi::AssignKernel(*cpu_ctx, *in, out_ptr);
      }
      break;
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Only support fp16 and fp32, but received dtype is %s.",
          phi::DataTypeToString(in->dtype())));
      break;
  }

  if (out == nullptr) {
    Assign(*out_ptr, in);
  }
}

static float FindMaxAbs(const float* data, int len) {
  float max_f = 0.0f;
  for (int i = 0; i < len; ++i) {
    float max = std::abs(data[i]);
    if (max > max_f) {
      max_f = max;
    }
  }
  return max_f;
}

static float IEEECompliance0(float f) {
  uint32_t* ptr = reinterpret_cast<uint32_t*>(&f);
  uint32_t sign = (*ptr) & 0x80000000;
  uint32_t uf = 0;
  // nan -> inf
  if (std::isnan(f)) {
    uf = (sign | 0x7F800000);
    float* ptr = reinterpret_cast<float*>(&uf);
    return *ptr;
  } else if (std::isnormal(f) || (std::isinf(f)) || (f == 0)) {
    return f;
  } else {
    // denormal -> +-0
    uf = 0x0;
    float* ptr = reinterpret_cast<float*>(&uf);
    return *ptr;
  }
}

static inline long RoundHalfToEven(const float src) {  // NOLINT
  long ret = llround(src);                             // NOLINT
  if (fabs(fabs(round(src) - src) - 0.5) > 0) {
    return ret;
  } else {
    if (abs(ret) % 2 == 0) {
      return ret;
    } else {
      return ret + (ret > 0 ? -1 : 1);
    }
  }
}

template <typename T, int RMAX>
static T Fp32ToIntx(const float f, float max) {
  max = IEEECompliance0(max);
  float input = IEEECompliance0(f);
  // +0 and -0 -> +0
  if (input == 0) {
    input = 0.0f;
  }

  float tmp = RMAX / max;
  if (std::isinf(tmp)) {
    uint32_t* ptr = reinterpret_cast<uint32_t*>(&input);
    if ((*ptr) >> 31 & 1) {
      return T(-RMAX);
    } else {
      return T(RMAX);
    }
  }

  tmp = input * tmp;
  if (std::isnan(tmp)) {
    return T(RMAX);
  }

  tmp = IEEECompliance0(tmp);
  // early check to avoid INF or big value get into convertor func.
  if (tmp > RMAX) {
    return T(RMAX);
  }
  if (tmp < -RMAX) {
    return T(-RMAX);
  }
  T ret = (T)RoundHalfToEven(tmp);
  if (ret > RMAX) {
    ret = T(RMAX);
  }
  if (ret < -RMAX) {
    ret = T(-RMAX);
  }
  return ret;
}

template <typename T>
static void QuantFP32ToIntX(const float* src_ptr,
                            T* dst_ptr,
                            float max_val,
                            int numel) {
  LOG(FATAL) << "Not support.";
}

template <>
void QuantFP32ToIntX<int16_t>(const float* src_ptr,
                              int16_t* dst_ptr,
                              float max_val,
                              int numel) {
  for (int i = 0; i < numel; i++) {
    dst_ptr[i] = Fp32ToIntx<int16_t, 32767>(src_ptr[i], max_val);
  }
}

template <>
void QuantFP32ToIntX<int8_t>(const float* src_ptr,
                             int8_t* dst_ptr,
                             float max_val,
                             int numel) {
  for (int i = 0; i < numel; i++) {
    dst_ptr[i] = Fp32ToIntx<int8_t, 127>(src_ptr[i], max_val);
  }
}

template <typename T>
void PrepareWeight(phi::DenseTensor* weight,
                   phi::DenseTensor* weight_max,
                   bool transpose) {
  // Convert fp16 to fp32
  phi::DenseTensor weight_fp32;
  CastToFp32(weight, &weight_fp32);

  // Transpose
  if (transpose) {
    Transpose2D(&weight_fp32);
  }

  // Find max
  int max_ptr_size = phi::backends::xpu::get_xpu_max_ptr_size(-1);
  int size = weight_fp32.numel();
  auto* weight_data = weight_fp32.data<float>();
  float max_val = FindMaxAbs(weight_data, size);
  std::vector<float> max_vec(max_ptr_size, max_val);
  weight_max->set_type(phi::DataType::FLOAT32);
  weight_max->Resize({max_ptr_size});
  auto* cpu_ctx = static_cast<phi::CPUContext*>(
      platform::DeviceContextPool::Instance().Get(phi::CPUPlace()));
  memcpy(cpu_ctx->Alloc<float>(weight_max),
         max_vec.data(),
         max_ptr_size * sizeof(float));

  // Quant
  weight->set_type(phi::CppTypeToDataType<T>::Type());
  weight->Resize(weight_fp32.dims());
  QuantFP32ToIntX(weight_data, cpu_ctx->Alloc<T>(weight), max_val, size);
}

template void PrepareWeight<int16_t>(phi::DenseTensor* weight,
                                     phi::DenseTensor* weight_max,
                                     bool transpose);
template void PrepareWeight<int8_t>(phi::DenseTensor* weight,
                                    phi::DenseTensor* weight_max,
                                    bool transpose);

}  // namespace ir
}  // namespace framework
}  // namespace paddle
