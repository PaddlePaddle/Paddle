/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <iostream>
#include <chrono>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template<typename T>
inline void SliceTensor(const int64_t* start, const int64_t* end,
    const std::vector<std::pair<int64_t, int64_t>>& sorted_axes,
                                const Tensor& in, T* out_data) {
  // acquire the biggest selected axe, for optimizing
  int last_axe = sorted_axes[sorted_axes.size() - 1].first;

  // descent dimension of input tensor to 2-D matrix
  const auto& dims = in.dims();
  auto axe_dims = framework::slice_ddim(dims, 0, last_axe + 1);
  auto axe_stride = framework::stride(axe_dims);
  int64_t col_numel = 0l;
  if (last_axe + 1 < dims.size()) {  // exists default axe
    col_numel = framework::product(
                    slice_ddim(dims, last_axe + 1, dims.size()));
  } else {
    col_numel = 1;
  }
  size_t col_size = col_numel * sizeof(T);
  const T* matrix_data = in.data<T>();

  // for output
  size_t out_offset = 0ul;
  auto copy_one_column = [&col_numel, &matrix_data, &out_data,
                          &out_offset, &col_size](int64_t cur_row) {
    size_t in_offset = cur_row * col_numel;
    memcpy(out_data + out_offset, matrix_data + in_offset, col_size);
    out_offset += col_numel;
  };

  std::function<void(int, int, int)> recursive_slice_with_start;
  recursive_slice_with_start = [&recursive_slice_with_start,
                                &dims, &last_axe, &sorted_axes,
                                &axe_dims, &start, &end,
                                &copy_one_column, &axe_stride](
                                int dim, int axe, int cur_row) {
    if (dim > last_axe) {
      copy_one_column(cur_row);
      return;
    }

    int s = 0, e = dims[dim];
    if (sorted_axes[axe].first == dim) {
      s = start[sorted_axes[axe].second];
      e = end[sorted_axes[axe].second];
      axe++;
    }

    for (int i = s; i < e; ++i) {
      recursive_slice_with_start(dim + 1, axe, cur_row + i * axe_stride[dim]);
    }
  };

  recursive_slice_with_start(/* first dim */0, /* first axe */0, 0);

  return;
}

template <typename DeviceContext, typename T>
class SliceOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int64_t* start = const_cast<int64_t*>(
                     ctx.Input<Tensor>("Starts")->data<int64_t>());
    int64_t* end = const_cast<int64_t*>(
                        ctx.Input<Tensor>("Ends")->data<int64_t>());

    auto* in = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    T* out_data = out->mutable_data<T>(ctx.GetPlace());

    auto* axes = ctx.Input<Tensor>("Axes");
    std::vector<std::pair<int64_t, int64_t>> axes_vec;
    int64_t axe_len = ctx.Input<Tensor>("Starts")->dims()[0];
    auto& dims = in->dims();
    if (axes != nullptr) {
      // the axe sequnce in axes may not be in order, sort it firstly
      const int64_t* axes_data = axes->data<int64_t>();
      for (int64_t i = 0; i < axe_len; i++) {
        axes_vec.push_back(std::pair<int64_t, int64_t>(axes_data[i], i));
      }

      std::sort(
          axes_vec.begin(), axes_vec.end(),
          [](const std::pair<int64_t, size_t>& l,
                             const std::pair<int64_t, size_t>& r) {
            return l.first < r.first;
          });
    } else {
      for (int64_t i = 0; i < axe_len; i++) {
        axes_vec.push_back(std::pair<int64_t, int64_t>(i, i));
      }
    }

    int last_axe = axes_vec[axes_vec.size() - 1].first;

    // the biggest axe must be less than the rank of input
    PADDLE_ENFORCE_GT(dims.size(), last_axe);
    for (int64_t i = 0; i < static_cast<int64_t>(axes_vec.size()); ++i) {
      if (i > 0) {
        // can not contains duplicate axes
        PADDLE_ENFORCE_GT(axes_vec[i].first, axes_vec[i - 1].first);
      }
    }

    framework::DDim out_dims(dims);
    for (int64_t i = 0, j = 0; i <= last_axe; ++i) {
      if (axes_vec[j].first == i) {
        int64_t ai = axes_vec[j].second;
        if (start[ai] < 0) start[ai] += dims[i];
        if (start[ai] > dims[i]) start[ai] = dims[i];
        if (end[ai] < 0) end[ai] += dims[i];
        if (end[ai] > dims[i]) end[ai] = dims[i];

        framework::set(out_dims, i, end[ai] - start[ai]);
        ++j;
      }
    }

    SliceTensor(start, end, axes_vec, *in, out_data);
    out->Resize(out_dims);
  }
};

template <typename DeviceContext, typename T>
class SliceGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
  }
};

}  // namespace operators
}  // namespace paddle
