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
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace framework {
namespace ir {

template <typename T>
void Transpose2D(const phi::DenseTensor& in, phi::DenseTensor* out) {
  auto in_dims = in.dims();
  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      2,
      platform::errors::InvalidArgument(
          "In dims rank should be 2, but received in dims size is [%d].",
          in_dims.size()));
  out->Resize({in_dims[1], in_dims[0]});
  out->set_type(in.type());
  auto* dev_ctx = static_cast<phi::CPUContext*>(
      platform::DeviceContextPool::Instance().Get(phi::CPUPlace()));
  dev_ctx->Alloc<T>(out);
  std::vector<int> axis{1, 0};
  phi::funcs::Transpose<phi::CPUContext, T, 2> trans2d;
  trans2d(*dev_ctx, in, out, axis);
}

template void Transpose2D<float>(const phi::DenseTensor& in,
                                 phi::DenseTensor* out);

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

template <typename T>
void QuantWeight(phi::DenseTensor* weight,
                 phi::DenseTensor* weight_max,
                 bool transpose) {
  // Transpose
  auto* weight_data = weight->data<float>();
  phi::DenseTensor weight_trans;
  if (transpose) {
    Transpose2D<float>(*weight, &weight_trans);
    weight_data = weight_trans.data<float>();
    weight->Resize(weight_trans.dims());
  }
  // Find max
  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  const auto& dev_ctxs = pool.device_contexts();
  auto place = phi::XPUPlace();  // xpu:0
  for (auto it = dev_ctxs.begin(); it != dev_ctxs.end(); it++) {
    if (it->first.GetType() == phi::AllocationType::XPU) {  // maybe xpu:1
      place = it->first;
    }
  }
  phi::XPUContext* xpu_ctx = static_cast<phi::XPUContext*>(pool.Get(place));
  int max_ptr_size = xpu_ctx->x_context()->max_ptr_size();
  int size = weight->numel();
  float max_val = FindMaxAbs(weight_data, size);
  std::vector<float> max_vec(max_ptr_size, max_val);
  weight_max->set_type(paddle::experimental::CppTypeToDataType<float>::Type());
  weight_max->Resize({max_ptr_size});
  auto* dev_ctx = static_cast<phi::CPUContext*>(
      platform::DeviceContextPool::Instance().Get(phi::CPUPlace()));
  memcpy(dev_ctx->Alloc<float>(weight_max),
         max_vec.data(),
         max_ptr_size * sizeof(float));
  // Quant
  std::vector<T> quant_data(size);
  QuantFP32ToIntX(weight_data, quant_data.data(), max_val, size);
  weight->set_type(paddle::experimental::CppTypeToDataType<T>::Type());
  memcpy(dev_ctx->Alloc<T>(weight), quant_data.data(), size * sizeof(T));
}

template void QuantWeight<int16_t>(phi::DenseTensor* weight,
                                   phi::DenseTensor* weight_max,
                                   bool transpose);

}  // namespace ir
}  // namespace framework
}  // namespace paddle
