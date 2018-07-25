/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SliceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    int rank = ctx.Input<framework::Tensor>("Input")->dims().size();
    switch (rank) {
      case 1:
        SliceCompute<1>(ctx);
        break;
      case 2:
        SliceCompute<2>(ctx);
        break;
      case 3:
        SliceCompute<3>(ctx);
        break;
      case 4:
        SliceCompute<4>(ctx);
        break;
      case 5:
        SliceCompute<5>(ctx);
        break;
      case 6:
        SliceCompute<6>(ctx);
        break;
    }
  }

 private:
  template <size_t D>
  void SliceCompute(const framework::ExecutionContext &context) const {
    auto &place =
        *context.template device_context<DeviceContext>().eigen_device();
    auto in = context.Input<framework::Tensor>("Input");
    auto out = context.Output<framework::Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());
    auto out_dims = out->dims();
    auto in_dims = in->dims();
    auto axes = context.Attr<std::vector<int>>("axes");
    auto starts = context.Attr<std::vector<int>>("starts");

    auto offsets = Eigen::array<int, D>();
    auto extents = Eigen::array<int, D>();
    for (size_t i = 0; i < D; ++i) {
      offsets[i] = 0;
      extents[i] = out_dims[i];
    }
    int start;
    for (size_t i = 0; i < axes.size(); ++i) {
      start = starts[i];
      if (start < 0) {
        start = (start + in_dims[axes[i]]);
      }
      start = std::max(start, 0);
      offsets[axes[i]] = start;
    }
    auto in_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *in);
    auto out_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *out);
    out_t.device(place) = in_t.slice(offsets, extents);
  }
};

template <typename DeviceContext, typename T, typename GradFunctor>
class SliceGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto dx = ctx.Output<framework::Tensor>(framework::GradVarName("Input"));
    auto dy = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto dx_dims = dx->dims();
    auto dy_dims = dy->dims();
    std::vector<int> axes = ctx.Attr<std::vector<int>>("axes");
    std::vector<int> starts = ctx.Attr<std::vector<int>>("starts");

#define CALL_SLICE_OP_GRAD_FUNCTOR(rank)                               \
  do {                                                                 \
    framework::Dim<rank> dx_strides##rank;                             \
    framework::Dim<rank> dy_strides##rank;                             \
    GetStrides(dx_dims, &dx_strides##rank);                            \
    GetStrides(dy_dims, &dy_strides##rank);                            \
    GradFunctor functor##rank;                                         \
    functor##rank(ctx.template device_context<DeviceContext>(),        \
                  dx->mutable_data<T>(ctx.GetPlace()), dx->numel(),    \
                  dx_strides##rank, dy->data<T>(), dy->numel(),        \
                  dy_strides##rank,                                    \
                  GetIndexOffset(GetFullStarts(dx_dims, axes, starts), \
                                 dx_strides##rank));                   \
  } while (0)

    switch (dx_dims.size()) {
      case 1:
        CALL_SLICE_OP_GRAD_FUNCTOR(1);
        break;
      case 2:
        CALL_SLICE_OP_GRAD_FUNCTOR(2);
        break;
      case 3:
        CALL_SLICE_OP_GRAD_FUNCTOR(3);
        break;
      case 4:
        CALL_SLICE_OP_GRAD_FUNCTOR(4);
        break;
      case 5:
        CALL_SLICE_OP_GRAD_FUNCTOR(5);
        break;
      case 6:
        CALL_SLICE_OP_GRAD_FUNCTOR(6);
        break;
      default:
        PADDLE_THROW("slice_op does not support rank >= 6");
        break;
    }

#undef CALL_SLICE_OP_GRAD_FUNCTOR
  }

 private:
  template <int Rank>
  static void GetStrides(const framework::DDim &dim,
                         framework::Dim<Rank> *strides) {
    auto sz = static_cast<int>(dim.size());
    (*strides)[sz - 1] = 1;
    for (int i = sz - 2; i >= 0; --i) {
      (*strides)[i] = (*strides)[i + 1] * dim[i + 1];
    }
  }

  static std::vector<int64_t> GetFullStarts(const framework::DDim &dx_dims,
                                            const std::vector<int> &axes,
                                            const std::vector<int> &starts) {
    std::vector<int64_t> full_starts(dx_dims.size(), 0);
    for (size_t i = 0; i < axes.size(); ++i) {
      full_starts[axes[i]] =
          (starts[i] >= 0 ? starts[i] : starts[i] + dx_dims[axes[i]]);
    }
    return full_starts;
  }

  template <int Rank>
  static int64_t GetIndexOffset(const std::vector<int64_t> &full_starts,
                                const framework::Dim<Rank> &dx_strides) {
    int64_t offset = 0;
    for (int i = 0; i < Rank; ++i) offset += full_starts[i] * dx_strides[i];
    return offset;
  }
};

}  // namespace operators
}  // namespace paddle
