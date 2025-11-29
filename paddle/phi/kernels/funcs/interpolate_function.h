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

#include "paddle/common/ddim.h"
#include "paddle/common/layout.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/phi/kernels/primitive/datamover_primitives.h"
#endif

namespace phi {
namespace funcs {

template <typename T>
inline T AreaPixelComputeScale(int64_t input_size,
                               int64_t output_size,
                               bool align_corners,
                               const T scale) {
  if (align_corners) {
    if (output_size > 1) {
      return static_cast<T>(input_size - 1) / (output_size - 1);
    }
  } else {
    if (scale > 0.) {
      return static_cast<T>(1.0) / scale;
    }
    if (output_size > 0) {
      return static_cast<T>(input_size) / output_size;
    }
  }
  return static_cast<T>(0);
}

template <typename T>
HOSTDEVICE inline T AreaPixelComputeSourceIndex(T scale,
                                                int64_t dst_index,
                                                bool align_corners,
                                                T align_type_value = 0.5) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    return scale * (dst_index + align_type_value) - align_type_value;
  }
}

template <typename T>
HOSTDEVICE inline T CubicConvolution1(T x, T A) {
  return ((A + 2) * x - (A + 3)) * x * x + 1;
}

template <typename T>
HOSTDEVICE inline T CubicConvolution2(T x, T A) {
  return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

template <typename T>
HOSTDEVICE inline void GetCubicUpsampleCoefficients(T coeffs[4], T t) {
  T A = static_cast<T>(-0.75);

  T x1 = t;
  coeffs[0] = CubicConvolution2<T>(x1 + 1.0, A);
  coeffs[1] = CubicConvolution1<T>(x1, A);

  // opposite coefficients
  T x2 = 1.0 - t;
  coeffs[2] = CubicConvolution1<T>(x2, A);
  coeffs[3] = CubicConvolution2<T>(x2 + 1.0, A);
}

inline void ExtractNCDWH(const DDim& dims,
                         const DataLayout& data_layout,
                         int64_t* N,
                         int64_t* C,
                         int64_t* D,
                         int64_t* H,
                         int64_t* W) {
  *N = dims[0];

  if (dims.size() == 3) {
    *C = data_layout == DataLayout::NCHW ? dims[1] : dims[2];
    *D = 1;
    *H = 1;
    *W = data_layout == DataLayout::NCHW ? dims[2] : dims[1];
  } else if (dims.size() == 4) {
    *C = data_layout == DataLayout::NCHW ? dims[1] : dims[3];
    *D = 1;
    *H = data_layout == DataLayout::NCHW ? dims[2] : dims[1];
    *W = data_layout == DataLayout::NCHW ? dims[3] : dims[2];
  } else {
    *C = data_layout == DataLayout::NCHW ? dims[1] : dims[4];
    *D = data_layout == DataLayout::NCHW ? dims[2] : dims[1];
    *H = data_layout == DataLayout::NCHW ? dims[3] : dims[2];
    *W = data_layout == DataLayout::NCHW ? dims[4] : dims[3];
  }
}

inline std::vector<int> get_new_shape(
    const std::vector<const DenseTensor*>& list_new_shape_tensor) {
  // get tensor from
  std::vector<int> vec_new_shape;
  auto& pool = phi::DeviceContextPool::Instance();
  for (size_t i = 0; i < list_new_shape_tensor.size(); ++i) {
    auto tensor = list_new_shape_tensor[i];
    phi::DeviceContext* dev_ctx = pool.Get(tensor->place());
    PADDLE_ENFORCE_EQ(tensor->dims() == common::make_ddim({1}) ||
                          tensor->dims() == common::make_ddim({}),
                      true,
                      errors::InvalidArgument(
                          "The shape of dimension tensor should be [1] or [],"
                          "but received d%.",
                          tensor->dims()));
    if (tensor->dtype() == phi::DataType::INT64) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
      if (tensor->place().GetType() == phi::AllocationType::CUSTOM) {
        DenseTensor temp;
        phi::Copy(*dev_ctx, *tensor, phi::CPUPlace(), true, &temp);
        vec_new_shape.push_back(static_cast<int64_t>(*temp.data<int64_t>()));
        continue;
      }
#endif
#ifdef PADDLE_WITH_XPU
      if (tensor->place().GetType() == phi::AllocationType::XPU) {
        DenseTensor temp;
        phi::Copy(*dev_ctx, *tensor, phi::CPUPlace(), true, &temp);
        vec_new_shape.push_back(static_cast<int64_t>(*temp.data<int64_t>()));
        continue;
      }
#endif
      if (tensor->place().GetType() == phi::AllocationType::GPU) {
        DenseTensor temp;
        phi::Copy(*dev_ctx, *tensor, phi::CPUPlace(), true, &temp);
        vec_new_shape.push_back(static_cast<int64_t>(*temp.data<int64_t>()));
      } else {
        vec_new_shape.push_back(static_cast<int64_t>(*tensor->data<int64_t>()));
      }
    } else if (tensor->dtype() == phi::DataType::INT32) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
      if (tensor->place().GetType() == phi::AllocationType::CUSTOM) {
        DenseTensor temp;
        phi::Copy(*dev_ctx, *tensor, phi::CPUPlace(), true, &temp);
        vec_new_shape.push_back(static_cast<int32_t>(*temp.data<int32_t>()));
        continue;
      }
#endif
#ifdef PADDLE_WITH_XPU
      if (tensor->place().GetType() == phi::AllocationType::XPU) {
        DenseTensor temp;
        phi::Copy(*dev_ctx, *tensor, phi::CPUPlace(), true, &temp);
        vec_new_shape.push_back(static_cast<int32_t>(*temp.data<int32_t>()));
        continue;
      }
#endif
      if (tensor->place().GetType() == phi::AllocationType::GPU) {
        DenseTensor temp;
        phi::Copy(*dev_ctx, *tensor, phi::CPUPlace(), true, &temp);
        vec_new_shape.push_back(static_cast<int32_t>(*temp.data<int32_t>()));
      } else {
        vec_new_shape.push_back(static_cast<int32_t>(*tensor->data<int32_t>()));
      }
    }
  }

  return vec_new_shape;
}

template <typename T>
inline std::vector<T> get_new_data_from_tensor(
    const DenseTensor* new_data_tensor) {
  std::vector<T> vec_new_data;
  auto* new_data = new_data_tensor->data<T>();
  DenseTensor cpu_starts_tensor;
  auto& pool = phi::DeviceContextPool::Instance();
  phi::DeviceContext* dev_ctx = pool.Get(new_data_tensor->place());
  if (new_data_tensor->place().GetType() == phi::AllocationType::GPU) {
    phi::Copy(
        *dev_ctx, *new_data_tensor, phi::CPUPlace(), true, &cpu_starts_tensor);
    new_data = cpu_starts_tensor.data<T>();
  }
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  if (new_data_tensor->place().GetType() == phi::AllocationType::CUSTOM) {
    phi::Copy(
        *dev_ctx, *new_data_tensor, phi::CPUPlace(), true, &cpu_starts_tensor);
    new_data = cpu_starts_tensor.data<T>();
  }
#endif
#ifdef PADDLE_WITH_XPU
  if (new_data_tensor->place().GetType() == phi::AllocationType::XPU) {
    phi::Copy(
        *dev_ctx, *new_data_tensor, phi::CPUPlace(), true, &cpu_starts_tensor);
    new_data = cpu_starts_tensor.data<T>();
  }
#endif
  vec_new_data = std::vector<T>(new_data, new_data + new_data_tensor->numel());
  return vec_new_data;
}

#if defined(__NVCC__) || defined(__HIPCC__)

struct FastDivModForInterpolate {
 public:
  FastDivMod<int64_t> channels_div;
  FastDivMod<int64_t> output_w_div;
  FastDivMod<int64_t> output_wc_div;

  explicit HOSTDEVICE FastDivModForInterpolate(const int64_t channels,
                                               const int64_t output_w,
                                               const int64_t output_wc)
      : channels_div(channels),
        output_w_div(output_w),
        output_wc_div(output_wc) {}
};

#endif

namespace antialias {

// taken from
// https://github.com/pytorch/pytorch/blob/a527e816935957a164d74dd7c5069310b2857695/
// aten/src/ATen/native/cuda/UpSample.cuh#L207-L305
struct BilinearFilterFunctor {
  template <typename T>
  HOSTDEVICE T operator()(T x) const {
    if (x < 0) {
      x = -x;
    }
    if (x < 1) {
      return 1 - x;
    }
    return 0;
  }

  static constexpr int size = 2;
};
struct BicubicFilterFunctor {
  template <typename T>
  HOSTDEVICE T operator()(T x) const {
    // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
    const T a = -0.5;
    if (x < 0) {
      x = -x;
    }
    if (x < 1) {
      return ((a + 2) * x - (a + 3)) * x * x + 1;
    }
    if (x < 2) {
      return (((x - 5) * x + 8) * x - 4) * a;
    }
    return 0;
  }

  static constexpr int size = 4;
};

}  // namespace antialias

}  // namespace funcs
}  // namespace phi
