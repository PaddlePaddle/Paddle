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
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SliceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
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
  void SliceCompute(const framework::ExecutionContext& context) const {
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    auto in = context.Input<framework::Tensor>("Input");
    auto out = context.Output<framework::Tensor>("Out");
    auto out_dims = out->dims();
    auto in_dims = in->dims();

    // resize out_dims
    auto decrease_axis = context.Attr<std::vector<int>>("decrease_axis");
    if (decrease_axis.size() > 0) {
      if (decrease_axis.size() == (size_t)in_dims.size()) {
        std::vector<int> vec_origin_out_shape(decrease_axis.size(), 1);
        out->Resize(framework::make_ddim(vec_origin_out_shape));
      } else {
        std::vector<int> vec_origin_out_shape(
            out_dims.size() + decrease_axis.size(), -1);

        for (size_t i = 0; i < decrease_axis.size(); ++i) {
          vec_origin_out_shape[decrease_axis[i]] = 1;
        }

        int index = 0;
        for (size_t i = 0; i < vec_origin_out_shape.size(); ++i) {
          if (vec_origin_out_shape[i] == -1) {
            vec_origin_out_shape[i] = out_dims[index];
            ++index;
          }
        }

        out->Resize(framework::make_ddim(vec_origin_out_shape));
      }
    }

    out->mutable_data<T>(context.GetPlace());
    auto axes = context.Attr<std::vector<int>>("axes");
    auto starts = context.Attr<std::vector<int>>("starts");

    auto new_out_dims = out->dims();
    auto offsets = Eigen::array<int, D>();
    auto extents = Eigen::array<int, D>();
    for (size_t i = 0; i < D; ++i) {
      offsets[i] = 0;
      extents[i] = new_out_dims[i];
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
            *out, new_out_dims);
    out_t.device(place) = in_t.slice(offsets, extents);

    out->Resize(out_dims);
  }
};

template <typename DeviceContext, typename T>
class SliceGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    size_t rank = ctx.Input<framework::Tensor>("Input")->dims().size();
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
  void SliceCompute(const framework::ExecutionContext& context) const {
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    auto* d_out =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* d_input =
        context.Output<framework::Tensor>(framework::GradVarName("Input"));
    d_input->mutable_data<T>(context.GetPlace());
    auto out_dims = d_out->dims();
    auto in_dims = d_input->dims();
    auto axes = context.Attr<std::vector<int>>("axes");
    auto starts = context.Attr<std::vector<int>>("starts");

    auto decrease_axis = context.Attr<std::vector<int>>("decrease_axis");
    if (decrease_axis.size() > 0) {
      if (decrease_axis.size() == (size_t)in_dims.size()) {
        // all dims decrease
        std::vector<int> vec_origin_out_shape(decrease_axis.size(), 1);
        out_dims = framework::make_ddim(vec_origin_out_shape);
      } else {
        std::vector<int> vec_origin_out_shape(
            out_dims.size() + decrease_axis.size(), -1);

        for (size_t i = 0; i < decrease_axis.size(); ++i) {
          vec_origin_out_shape[decrease_axis[i]] = 1;
        }

        int index = 0;
        for (size_t i = 0; i < vec_origin_out_shape.size(); ++i) {
          if (vec_origin_out_shape[i] == -1) {
            vec_origin_out_shape[i] = out_dims[index];
            ++index;
          }
        }

        out_dims = framework::make_ddim(vec_origin_out_shape);
      }
    }

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
    Eigen::array<std::pair<int, int>, D> paddings;
    for (size_t i = 0; i < paddings.size(); ++i) {
      paddings[i].first = offsets[i];
      paddings[i].second = (in_dims[i] - out_dims[i]) - offsets[i];
    }
    auto d_in_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *d_input);
    auto d_out_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *d_out, out_dims);
    d_in_t.device(place) = d_out_t.pad(paddings, 0);
  }
};
}  // namespace operators
}  // namespace paddle
