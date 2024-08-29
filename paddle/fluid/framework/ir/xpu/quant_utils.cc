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
#include "paddle/fluid/framework/ir/quantize_helper.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/kernels/assign_kernel.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace paddle {
namespace framework {
namespace ir {

void Assign(const phi::DenseTensor& in, phi::DenseTensor* out) {
  auto* cpu_ctx = static_cast<phi::CPUContext*>(
      phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
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
      common::errors::InvalidArgument(
          "In dims rank should be 2, but received in dims size is [%d].",
          in_dims.size()));

  phi::DenseTensor trans_tensor;
  phi::DenseTensor* out_ptr = out == nullptr ? &trans_tensor : out;
  out_ptr->Resize({in_dims[1], in_dims[0]});
  out_ptr->set_type(in->type());
  out_ptr->set_layout(in->layout());

  auto* cpu_ctx = static_cast<phi::CPUContext*>(
      phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
  std::vector<int> axis{1, 0};
  switch (in->dtype()) {
    case phi::DataType::FLOAT16:
      phi::TransposeKernel<phi::dtype::float16>(*cpu_ctx, *in, axis, out_ptr);
      break;
    case phi::DataType::FLOAT32:
      phi::TransposeKernel<float>(*cpu_ctx, *in, axis, out_ptr);
      break;
    case phi::DataType::INT16:
      phi::TransposeKernel<int16_t>(*cpu_ctx, *in, axis, out_ptr);
      break;
    case phi::DataType::INT8:
      phi::TransposeKernel<int8_t>(*cpu_ctx, *in, axis, out_ptr);
      break;
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "Only support fp16/fp32/int16/int8, but received dtype is %s.",
          phi::DataTypeToString(in->dtype())));
      break;
  }

  if (out == nullptr) {
    Assign(*out_ptr, in);
  }
}

void CastToInt32(phi::DenseTensor* in, phi::DenseTensor* out) {
  auto* cpu_ctx = static_cast<phi::CPUContext*>(
      phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));

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
      PADDLE_THROW(common::errors::InvalidArgument(
          "Only support int64 and int32, but received dtype is %s.",
          phi::DataTypeToString(in->dtype())));
      break;
  }

  if (out == nullptr) {
    Assign(*out_ptr, in);
  }
}
void CastTo(phi::DenseTensor* in, phi::DenseTensor* out, DataType out_dtype) {
  auto* cpu_ctx = static_cast<phi::CPUContext*>(
      phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));

  if (in->dtype() != phi::DataType::FLOAT16 &&
      in->dtype() != phi::DataType::FLOAT32) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Only support fp16 and fp32, but received dtype is %s.",
        phi::DataTypeToString(in->dtype())));
  }

  paddle::experimental::CheckAndTrans2Contiguous(in);
  phi::DenseTensor ori_tensor;
  phi::DenseTensor* out_ptr = out == nullptr ? &ori_tensor : out;
  out_ptr->Resize(in->dims());
  out_ptr->set_type(out_dtype);
  out_ptr->set_layout(in->layout());
  if (in->dtype() == out_dtype) {
    if (out == nullptr) {
      return;
    } else {
      phi::AssignKernel(*cpu_ctx, *in, out_ptr);
    }
  } else {
    if (in->dtype() == phi::DataType::FLOAT16) {
      phi::CastKernel<float16>(*cpu_ctx, *in, out_dtype, out_ptr);
    } else {
      phi::CastKernel<float>(*cpu_ctx, *in, out_dtype, out_ptr);
    }
    if (out == nullptr) {
      Assign(*out_ptr, in);
    }
  }
}

void CastToFp32(phi::DenseTensor* in, phi::DenseTensor* out) {
  CastTo(in, out, phi::DataType::FLOAT32);
}

void CastToFp16(phi::DenseTensor* in, phi::DenseTensor* out) {
  CastTo(in, out, phi::DataType::FLOAT16);
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
  PADDLE_THROW(common::errors::Unimplemented("Not support."));
}

template <>
void QuantFP32ToIntX<float>(const float* src_ptr,
                            float* dst_ptr,
                            float max_val,
                            int numel) {
  for (int i = 0; i < numel; i++) {
    dst_ptr[i] = static_cast<float>(src_ptr[i]);
  }
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

template <
    typename Tcpu,
    typename Txpu,
    typename std::enable_if<!std::is_same<Tcpu, float>::value, Tcpu>::type* ptr>
void ConvertWithQuant(phi::DenseTensor* weight,
                      phi::DenseTensor* weight_max,
                      phi::DenseTensor* scale_max,
                      bool transpose,
                      bool per_channel_quant) {
  std::stringstream ss;
  ss << "Not support for Tcpu is " << phi::CppTypeToDataType<Tcpu>::Type();
  PADDLE_THROW(common::errors::Fatal(ss.str()));
}

template <
    typename Tcpu,
    typename Txpu,
    typename std::enable_if<std::is_same<Tcpu, float>::value, Tcpu>::type* ptr>
void ConvertWithQuant(phi::DenseTensor* weight,
                      phi::DenseTensor* weight_max,
                      phi::DenseTensor* scale_max,
                      bool transpose,
                      bool per_channel_quant) {
  // Convert fp16 to fp32
  phi::DenseTensor weight_fp32;
  CastToFp32(weight, &weight_fp32);

  if (transpose) {  // (k, n) -> (n, k)
    Transpose2D(&weight_fp32);
  }

  auto* cpu_ctx = static_cast<phi::CPUContext*>(
      phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
  if (!per_channel_quant) {
    // Find max
    int max_ptr_size = phi::backends::xpu::get_xpu_max_ptr_size(-1);
    weight_max->set_type(phi::DataType::FLOAT32);
    weight_max->Resize({max_ptr_size});

    int size = weight_fp32.numel();
    auto* weight_fp32_data = weight_fp32.data<float>();
    float max_val = FindMaxAbs(weight_fp32_data, size);
    std::vector<float> max_vec(max_ptr_size, max_val);
    memcpy(cpu_ctx->Alloc<float>(weight_max),
           max_vec.data(),
           max_ptr_size * sizeof(float));
    // Quant
    weight->set_type(phi::CppTypeToDataType<Txpu>::Type());
    weight->Resize(weight_fp32.dims());
    QuantFP32ToIntX<Txpu>(
        weight_fp32_data, cpu_ctx->Alloc<Txpu>(weight), max_val, size);
  } else {
    std::vector<float> quant_scales{};
    auto GetQuantScales = [&](const float* weight_data,
                              int n,
                              int data_count) -> std::vector<float> {
      std::vector<float> scales;
      for (int i = 0; i < n; ++i) {
        float max_val = FindMaxAbs(weight_data + i * data_count, data_count);
        scales.push_back(max_val);
      }
      return scales;
    };

    int n = weight_fp32.dims()[0];
    int data_count = weight_fp32.numel() / n;
    auto* weight_fp32_data = weight_fp32.data<float>();
    quant_scales = GetQuantScales(weight_fp32_data, n, data_count);
    weight->set_type(phi::CppTypeToDataType<Txpu>::Type());
    weight->Resize(weight_fp32.dims());
    auto* weight_data = cpu_ctx->Alloc<Txpu>(weight);
    for (int i = 0; i < n; ++i) {
      QuantFP32ToIntX<Txpu>(weight_fp32_data + i * data_count,
                            weight_data + i * data_count,
                            quant_scales[i],
                            data_count);
    }
    int max_ptr_size = phi::backends::xpu::get_xpu_max_ptr_size(-1);
    // 1. Create weight_max tensor(all data is 1.0f)
    weight_max->set_type(phi::DataType::FLOAT32);
    weight_max->Resize({max_ptr_size});
    std::vector<float> ones_vec(max_ptr_size, 1.0);
    memcpy(cpu_ctx->Alloc<float>(weight_max),
           ones_vec.data(),
           max_ptr_size * sizeof(float));
    // 2. Create scale_max tensor
    scale_max->set_type(phi::DataType::FLOAT32);
    scale_max->Resize({static_cast<int64_t>(quant_scales.size())});
    memcpy(cpu_ctx->Alloc<float>(scale_max),
           quant_scales.data(),
           quant_scales.size() * sizeof(float));
  }
}

template <typename T>
void ConvertWithoutQuant(phi::DenseTensor* weight,
                         phi::DenseTensor* weight_max,
                         phi::DenseTensor* scale_max,
                         bool transpose,
                         const std::vector<float>& weight_scales) {
  if (transpose) {
    Transpose2D(weight);
  }
  bool per_tensor_quant = weight_scales.size() == 1;
  if (std::is_same<T, int8_t>::value || std::is_same<T, int16_t>::value) {
    PADDLE_ENFORCE_EQ(
        weight_scales.empty(),
        false,
        common::errors::InvalidArgument(
            "ConvertWithoutQuant is not allowed weight scales is empty!"));
    auto* cpu_ctx = static_cast<phi::CPUContext*>(
        phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
    if (per_tensor_quant) {
      int max_ptr_size = phi::backends::xpu::get_xpu_max_ptr_size(-1);
      std::vector<float> max_vec(max_ptr_size, weight_scales[0]);
      weight_max->set_type(phi::DataType::FLOAT32);
      weight_max->Resize({max_ptr_size});
      memcpy(cpu_ctx->Alloc<float>(weight_max),
             max_vec.data(),
             max_ptr_size * sizeof(float));
    } else {
      int max_ptr_size = phi::backends::xpu::get_xpu_max_ptr_size(-1);
      // 1. Create weight_max tensor(all data is 1.0f)
      weight_max->set_type(phi::DataType::FLOAT32);
      weight_max->Resize({max_ptr_size});
      std::vector<float> ones_vec(max_ptr_size, 1.0);
      memcpy(cpu_ctx->Alloc<float>(weight_max),
             ones_vec.data(),
             max_ptr_size * sizeof(float));
      // 2. Create scale_max tensor
      scale_max->set_type(phi::DataType::FLOAT32);
      scale_max->Resize({static_cast<int64_t>(weight_scales.size())});
      memcpy(cpu_ctx->Alloc<float>(scale_max),
             weight_scales.data(),
             weight_scales.size() * sizeof(float));
    }
  } else if (std::is_same<T, float>::value) {
    // Convert fp16 to fp32
    phi::DenseTensor weight_fp32;
    CastToFp32(weight, &weight_fp32);
    // Find max
    int max_ptr_size = phi::backends::xpu::get_xpu_max_ptr_size(-1);
    int size = weight_fp32.numel();
    auto* weight_data = weight_fp32.data<float>();
    float max_val = FindMaxAbs(weight_data, size);
    std::vector<float> max_vec(max_ptr_size, max_val);
    weight_max->set_type(phi::DataType::FLOAT32);
    weight_max->Resize({max_ptr_size});
    auto* cpu_ctx = static_cast<phi::CPUContext*>(
        phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
    memcpy(cpu_ctx->Alloc<float>(weight_max),
           max_vec.data(),
           max_ptr_size * sizeof(float));

    // Quant
    weight->set_type(phi::DataType::FLOAT32);
    weight->Resize(weight_fp32.dims());
    QuantFP32ToIntX<float>(
        weight_data, cpu_ctx->Alloc<float>(weight), max_val, size);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Only support float<->int31, int8<->int8 and int16<->int16 convert."));
  }
}

template void ConvertWithQuant<float, int16_t>(phi::DenseTensor* weight,
                                               phi::DenseTensor* weight_max,
                                               phi::DenseTensor* scale_max,
                                               bool transpose,
                                               bool per_channel_quant);

template void ConvertWithQuant<float, int8_t>(phi::DenseTensor* weight,
                                              phi::DenseTensor* weight_max,
                                              phi::DenseTensor* scale_max,
                                              bool transpose,
                                              bool per_channel_quant);

template void ConvertWithoutQuant<int8_t>(
    phi::DenseTensor* weight,
    phi::DenseTensor* weight_max,
    phi::DenseTensor* scale_max,
    bool transpose,
    const std::vector<float>& weight_scales);

template void ConvertWithoutQuant<float>(
    phi::DenseTensor* weight,
    phi::DenseTensor* weight_max,
    phi::DenseTensor* scale_max,
    bool transpose,
    const std::vector<float>& weight_scales);

bool IsPerTensorQuant(const std::vector<float>& weight_max) {
  bool per_tensor = true;
  PADDLE_ENFORCE_GT(
      weight_max.size(),
      0,
      common::errors::InvalidArgument(
          "Op's channel size: [%d] should great than zero", weight_max.size()));
  auto first = weight_max[0];
  for (size_t i = 1; i < weight_max.size(); ++i) {
    if (std::abs(first - weight_max[i]) > 1e-6) {
      per_tensor = false;
      break;
    }
  }
  return per_tensor;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
