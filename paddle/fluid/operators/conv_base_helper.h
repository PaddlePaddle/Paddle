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

#include "paddle/fluid/framework/conv_search_cache.h"
#include "paddle/fluid/operators/conv_cudnn_op_cache.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/autotune/cache.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = platform::DataLayout;
using framework::AlgorithmsCache;
using framework::ConvSearchCache;

template <typename T>
using ScalingParamType = typename platform::CudnnDataType<T>::ScalingParamType;

// As the container of searchAlgorithm::Find() result.
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
  platform::TensorDescriptor idesc, odesc;
  platform::FilterDescriptor wdesc;
  platform::ConvolutionDescriptor cdesc;
  const framework::Tensor *x, *w, *o;
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
  DataLayout data_layout;

  ConvArgsBase(const framework::Tensor* x,
               const framework::Tensor* w,
               const framework::Tensor* o,
               const std::vector<int> s,
               const std::vector<int> p,
               const std::vector<int> d,
               DataT dtype,
               int g,
               DataLayout layout)
      : x(x),
        w(w),
        o(o),
        s(s),
        p(p),
        d(d),
        cudnn_dtype(dtype),
        group(g),
        data_layout(layout) {}

  template <typename T>
  phi::autotune::ConvCacheKey Convert2ConvCacheKey() const {
    auto x_shape = phi::vectorize(x->dims());
    auto w_shape = phi::vectorize(w->dims());
    VLOG(10) << "[ConvArgs] x_dims=" << x_shape << ", w_dims=" << w_shape
             << ", strides=" << s << ", paddings=" << p << ", dilations=" << d
             << ", data=" << paddle::experimental::CppTypeToDataType<T>::Type()
             << ", group=" << group
             << ", data layout=" << static_cast<int64_t>(data_layout);

    return phi::autotune::ConvCacheKey(
        x_shape,
        w_shape,
        p,
        s,
        d,
        paddle::experimental::CppTypeToDataType<T>::Type(),
        group,
        static_cast<int64_t>(data_layout));
  }
};

static inline void GetNCDHW(const framework::DDim& dims,
                            const DataLayout& layout,
                            int* N,
                            int* C,
                            int* D,
                            int* H,
                            int* W) {
  *N = dims[0];
  *C = layout == DataLayout::kNCHW ? dims[1] : dims[dims.size() - 1];
  int i = layout == DataLayout::kNCHW ? 0 : 1;
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

}  // namespace operators
}  // namespace paddle
