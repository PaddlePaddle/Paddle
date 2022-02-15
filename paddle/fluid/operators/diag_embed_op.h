// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

template <typename T>
struct DiagEmbedFunctor {
  DiagEmbedFunctor(const T* input, int64_t numel, const int64_t* dim,
                   int64_t offset, int64_t dims_size, T* output,
                   const int64_t* strides)
      : input_(input),
        numel_(numel),
        dim_(dim),
        offset_(offset),
        dims_size_(dims_size),
        output_(output),
        strides_(strides) {}

  HOSTDEVICE void operator()(size_t idx) const {
    int64_t position = 0;
    auto numel = numel_;
    int64_t num = idx;
    for (int64_t i = 0; i < dims_size_; i++) {
      numel = numel / dim_[i];
      position += num / numel * strides_[i];
      num = num % numel;
    }
    output_[position + offset_] = input_[idx];
  }

  const T* input_;
  int64_t numel_;
  const int64_t* dim_;
  int64_t offset_;
  int64_t dims_size_;
  T* output_;
  const int64_t* strides_;
};

template <typename DeviceContext, typename T>
class DiagEmbedKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    auto* out = context.Output<framework::Tensor>("Out");

    const int64_t offset = context.Attr<int>("offset");
    const int64_t dim1 = context.Attr<int>("dim1");
    const int64_t dim2 = context.Attr<int>("dim2");
    auto* input_data = input->data<T>();

    T* out_data = out->mutable_data<T>(context.GetPlace());
    pten::funcs::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    set_zero(dev_ctx, out, static_cast<T>(0.0));

    auto out_dims = out->dims();
    int dim1_ = dim1 < 0 ? out_dims.size() + dim1 : dim1;
    int dim2_ = dim2 < 0 ? out_dims.size() + dim2 : dim2;
    auto stride = framework::stride(out_dims);
    int64_t diag_size;
    int64_t storage_offset = 0;
    if (offset >= 0) {
      int64_t dim = out_dims[dim2_] - offset;
      diag_size = std::max<int64_t>(std::min(out_dims[dim1_], dim), 0);
    } else {
      int64_t dim = out_dims[dim1_] + offset;
      diag_size = std::max<int64_t>(std::min(dim, out_dims[dim2_]), 0);
    }
    if (diag_size == 0) {
      // skip
    } else if (offset >= 0) {
      storage_offset += offset * stride[dim2_];
    } else {
      storage_offset -= offset * stride[dim1_];
    }
    auto strides = vectorize(stride);
    strides.erase(strides.begin() + std::max(dim1_, dim2_));
    strides.erase(strides.begin() + std::min(dim1_, dim2_));
    strides.push_back(stride[dim1_] + stride[dim2_]);
    const auto dims = vectorize(input->dims());

#if defined(__NVCC__) || defined(__HIPCC__)
    thrust::device_vector<int64_t> dims_vec(dims);
    const int64_t* dims_arr = thrust::raw_pointer_cast(dims_vec.data());
    thrust::device_vector<int64_t> strides_vec(strides);
    const int64_t* strides_arr = thrust::raw_pointer_cast(strides_vec.data());
#else
    const int64_t* dims_arr = dims.data();
    const int64_t* strides_arr = strides.data();
#endif

    platform::ForRange<DeviceContext> for_range(dev_ctx, input->numel());
    DiagEmbedFunctor<T> functor(input_data, input->numel(), dims_arr,
                                storage_offset, dims.size(), out_data,
                                strides_arr);
    for_range(functor);
  }
};
}  // namespace operators
}  // namespace paddle
