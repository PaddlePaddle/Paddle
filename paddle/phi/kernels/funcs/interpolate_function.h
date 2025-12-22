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

#include <algorithm>
#include <cmath>

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
                               const double scale) {
  if (align_corners) {
    if (output_size > 1) {
      return static_cast<T>(input_size - 1) / (output_size - 1);
    }
  } else {
    if (scale > 0.) {
      return static_cast<T>(1.0 / scale);
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
                                                bool align_corners) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    T src_idx = scale * (dst_index + T(0.5)) - T(0.5);
    return src_idx;
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

// Helper function to compute interpolation kernel size
inline int ComputeInterpSize(float ratio, int filter_size) {
  float support =
      (ratio >= 1.0f) ? (filter_size * 0.5f) * ratio : filter_size * 0.5f;
  return 1 + 2 * static_cast<int>(ceilf(support));
}

// Structure to hold AA interpolation launch configuration
struct AAInterpLaunchConfig {
  int block_x;
  int block_y;
  int grid_x;
  int grid_y;
  int grid_z;
  int interp_height;
  int interp_width;
  size_t shmem_size;

  AAInterpLaunchConfig(int out_h,
                       int out_w,
                       int64_t nc,
                       float ratio_h,
                       float ratio_w,
                       int filter_size,
                       size_t element_size,
                       size_t max_shmem,
                       int max_grid_z,
                       int warp_size,
                       bool need_buffer = true) {
    interp_height = ComputeInterpSize(ratio_h, filter_size);
    interp_width = ComputeInterpSize(ratio_w, filter_size);

    // Start with default block size
    block_x = std::min(warp_size, 32);
    block_y = std::min(256 / block_x, 8);

    // Compute required shared memory
    auto compute_shmem = [&]() -> size_t {
      size_t weights_per_block = static_cast<size_t>(interp_width) * block_x +
                                 static_cast<size_t>(interp_height) * block_y;
      if (need_buffer) {
        weights_per_block +=
            static_cast<size_t>(interp_height) * block_y * block_x;
      }
      return weights_per_block * element_size;
    };

    shmem_size = compute_shmem();

    // Dynamically reduce block size if shared memory exceeds limit
    while (shmem_size > max_shmem && (block_x > 4 || block_y > 1)) {
      // Reduce block_y first as it has larger impact on buffer size
      if (block_y > 1) {
        block_y = std::max(1, block_y / 2);
      } else if (block_x > 4) {
        block_x = std::max(4, block_x / 2);
      }
      shmem_size = compute_shmem();
    }

    // Compute grid dimensions
    grid_x = (out_w + block_x - 1) / block_x;
    grid_y = (out_h + block_y - 1) / block_y;
    grid_z = std::min(static_cast<int>(nc), max_grid_z);
  }

  bool IsValid(size_t max_shmem) const { return shmem_size <= max_shmem; }
};

}  // namespace antialias

}  // namespace funcs
}  // namespace phi
