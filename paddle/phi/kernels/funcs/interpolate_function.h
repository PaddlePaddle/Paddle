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

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/fluid/platform/fast_divmod.h"
#endif

namespace phi {
namespace funcs {

template <typename T>
HOSTDEVICE inline T CubicConvolution1(T x, T A) {
  return ((A + 2) * x - (A + 3)) * x * x + 1;
}

template <typename T>
HOSTDEVICE inline T CubicConvolution2(T x, T A) {
  return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

template <typename T>
HOSTDEVICE inline void get_cubic_upsample_coefficients(T coeffs[4], T t) {
  T A = -0.75;

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
                         int* N,
                         int* C,
                         int* D,
                         int* H,
                         int* W) {
  *N = dims[0];

  if (dims.size() == 3) {
    *C = data_layout == DataLayout::kNCHW ? dims[1] : dims[2];
    *D = 1;
    *H = 1;
    *W = data_layout == DataLayout::kNCHW ? dims[2] : dims[1];
  } else if (dims.size() == 4) {
    *C = data_layout == DataLayout::kNCHW ? dims[1] : dims[3];
    *D = 1;
    *H = data_layout == DataLayout::kNCHW ? dims[2] : dims[1];
    *W = data_layout == DataLayout::kNCHW ? dims[3] : dims[2];
  } else {
    *C = data_layout == DataLayout::kNCHW ? dims[1] : dims[4];
    *D = data_layout == DataLayout::kNCHW ? dims[2] : dims[1];
    *H = data_layout == DataLayout::kNCHW ? dims[3] : dims[2];
    *W = data_layout == DataLayout::kNCHW ? dims[4] : dims[3];
  }
}

inline std::vector<int> get_new_shape(
    const std::vector<const DenseTensor*>& list_new_shape_tensor) {
  // get tensor from
  std::vector<int> vec_new_shape;
  for (size_t i = 0; i < list_new_shape_tensor.size(); ++i) {
    auto tensor = list_new_shape_tensor[i];
    PADDLE_ENFORCE_EQ(
        tensor->dims(),
        phi::make_ddim({1}),
        errors::InvalidArgument("The shape of dimension tensor should be [1],"
                                "but received d%.",
                                tensor->dims()));
    if (paddle::platform::is_gpu_place(tensor->place())) {
      DenseTensor temp;
      paddle::framework::TensorCopySync(
          *tensor, paddle::platform::CPUPlace(), &temp);
      vec_new_shape.push_back(static_cast<int32_t>(*temp.data<int32_t>()));
    } else {
      vec_new_shape.push_back(static_cast<int32_t>(*tensor->data<int32_t>()));
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
  if (paddle::platform::is_gpu_place(new_data_tensor->place())) {
    paddle::framework::TensorCopySync(
        *new_data_tensor, paddle::platform::CPUPlace(), &cpu_starts_tensor);
    new_data = cpu_starts_tensor.data<T>();
  }
#ifdef PADDLE_WITH_ASCEND_CL
  if (paddle::platform::is_npu_place(new_data_tensor->place())) {
    paddle::framework::TensorCopySync(
        *new_data_tensor, paddle::platform::CPUPlace(), &cpu_starts_tensor);
    new_data = cpu_starts_tensor.data<T>();
  }
#endif
#ifdef PADDLE_WITH_XPU
  if (paddle::platform::is_xpu_place(new_data_tensor->place())) {
    paddle::framework::TensorCopySync(
        *new_data_tensor, paddle::platform::CPUPlace(), &cpu_starts_tensor);
    new_data = cpu_starts_tensor.data<T>();
  }
#endif
  vec_new_data = std::vector<T>(new_data, new_data + new_data_tensor->numel());
  return vec_new_data;
}

#if defined(__NVCC__) || defined(__HIPCC__)
using paddle::platform::FastDivMod;

struct FastDivModForInterpolate {
 public:
  FastDivMod channels_div;
  FastDivMod output_w_div;
  FastDivMod output_wc_div;

  explicit HOSTDEVICE FastDivModForInterpolate(const int channels,
                                               const int output_w,
                                               const int outout_wc)
      : channels_div(FastDivMod(channels)),
        output_w_div(FastDivMod(output_w)),
        output_wc_div(FastDivMod(outout_wc)) {}
};

#endif

}  // namespace funcs
}  // namespace phi
