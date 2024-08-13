/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/kernels/autotune/cache.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/gpudnn/conv_gpudnn_info.h"

namespace phi {

using GPUDNNDataLayout = phi::backends::gpu::DataLayout;

template <typename T>
using ScalingParamType =
    typename phi::backends::gpu::CudnnDataType<T>::ScalingParamType;

enum class ConvKind { kForward = 1, kBackwardData = 2, kBackwardFilter = 3 };

static inline double ToMegaBytes(size_t bytes) {
  return static_cast<double>(bytes) / (1 << 20);
}

static inline bool UseFixedWorkspace() {
  return FLAGS_conv_workspace_size_limit >= 0;
}

static size_t CalcWorkspaceLimitInBytes(bool use_fixed_workspace) {
  if (!use_fixed_workspace) {
    int device_id = phi::backends::gpu::GetCurrentDeviceId();
    int64_t allocated =
        memory_utils::DeviceMemoryStatCurrentValue("Allocated", device_id);
    int64_t reserved =
        memory_utils::DeviceMemoryStatCurrentValue("Reserved", device_id);
    int64_t available = phi::backends::gpu::GpuAvailableMemToAlloc();
    VLOG(3) << "[memory] allocated=" << ToMegaBytes(allocated)
            << " MB, reserved=" << ToMegaBytes(reserved)
            << " MB, available_to_alloc=" << ToMegaBytes(available) << " MB.";
    return std::max(available, reserved - allocated);
  } else {
    return FLAGS_conv_workspace_size_limit * 1024 * 1024;
  }
}

// The container of SearchAlgorithm::Find() result.
template <typename AlgoT>
struct SearchResult {
  SearchResult() {}

  explicit SearchResult(AlgoT a) : algo(a) {}
  explicit SearchResult(AlgoT a, float t, size_t size)
      : algo(a), time(t), workspace_size(size) {}

  AlgoT algo = static_cast<AlgoT>(0);
  float time = -1.f;
  size_t workspace_size = 0;
  bool exhaustive_search = false;
};

template <typename T>
static std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
  out << "[";
  bool is_first = true;
  for (auto const& tmp : v) {
    if (is_first) {
      out << tmp;
      is_first = false;
    } else {
      out << ", " << tmp;
    }
  }
  out << "]";
  return out;
}

// As the container of conv relevant descriptors.
template <typename HandleT, typename DataT>
struct ConvArgsBase {
  HandleT handle;
  phi::backends::gpu::TensorDescriptor idesc;
  phi::backends::gpu::TensorDescriptor odesc;
  phi::backends::gpu::FilterDescriptor wdesc;
  phi::backends::gpu::ConvolutionDescriptor cdesc;

  const phi::DenseTensor* x = nullptr;
  const phi::DenseTensor* w = nullptr;
  const phi::DenseTensor* o = nullptr;

  DataT cudnn_dtype;

  // strides
  std::vector<int> s;
  // paddings
  std::vector<int> p;
  // dilations
  std::vector<int> d;

  // groups
  int group;

  // data foramt
  GPUDNNDataLayout data_layout;

  ConvArgsBase(const HandleT& h,
               const phi::DenseTensor* x,
               const phi::DenseTensor* w,
               const phi::DenseTensor* o,
               const std::vector<int> s,
               const std::vector<int> p,
               const std::vector<int> d,
               DataT dtype,
               int g,
               GPUDNNDataLayout layout)
      : handle(h),
        x(x),
        w(w),
        o(o),
        s(s),
        p(p),
        d(d),
        cudnn_dtype(dtype),
        group(g),
        data_layout(layout) {}

  template <typename T>
  phi::autotune::ConvCacheKey ConvertToConvCacheKey() const {
    auto x_shape = common::vectorize(x->dims());
    auto w_shape = common::vectorize(w->dims());
    VLOG(10) << "[ConvArgs] x_dims=" << x_shape << ", w_dims=" << w_shape
             << ", strides=" << s << ", paddings=" << p << ", dilations=" << d
             << ", data=" << phi::CppTypeToDataType<T>::Type()
             << ", group=" << group
             << ", data layout=" << static_cast<int64_t>(data_layout);

    return phi::autotune::ConvCacheKey(x_shape,
                                       w_shape,
                                       p,
                                       s,
                                       d,
                                       phi::CppTypeToDataType<T>::Type(),
                                       group,
                                       static_cast<int64_t>(data_layout));
  }
};

static inline void GetNCDHW(const phi::DDim& dims,
                            const GPUDNNDataLayout& layout,
                            int* N,
                            int* C,
                            int* D,
                            int* H,
                            int* W) {
  *N = dims[0];
  *C = layout == GPUDNNDataLayout::kNCHW ? dims[1] : dims[dims.size() - 1];
  int i = layout == GPUDNNDataLayout::kNCHW ? 0 : 1;
  if (dims.size() == 5) {
    *D = dims[2 - i];
    *H = dims[3 - i];
    *W = dims[4 - i];
  } else {
    *D = 1;
    *H = dims[2 - i];
    *W = dims[3 - i];
  }
}

template <typename DeviceContext, typename T, size_t D>
static void RemovePaddingSlice(const phi::GPUContext& context,
                               const phi::DenseTensor* input,
                               phi::DenseTensor* out,
                               const std::vector<int>& starts,
                               const std::vector<int>& axes) {
  auto& place = *context.eigen_device();
  auto in_dims = input->dims();
  auto new_out_dims = out->dims();
  auto offsets = Eigen::DSizes<Eigen::DenseIndex, D>();
  auto extents = Eigen::DSizes<Eigen::DenseIndex, D>();
  for (size_t i = 0; i < D; ++i) {
    offsets[i] = 0;
    extents[i] = new_out_dims[i];
  }

  for (size_t i = 0; i < axes.size(); ++i) {
    int start = starts[i];
    if (start < 0) {
      start = (start + in_dims[axes[i]]);
    }
    start = std::max(start, 0);
    offsets[axes[i]] = start;
  }

  auto in_t =
      phi::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(*input);
  auto out_t = phi::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
      *out, new_out_dims);

  phi::funcs::EigenSlice<std::decay_t<decltype(place)>, T, D>::Eval(
      place, out_t, in_t, offsets, extents);
}

}  // namespace phi
