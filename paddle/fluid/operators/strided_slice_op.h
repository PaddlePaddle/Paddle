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
#include <cstdlib>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class StridedSliceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int rank = ctx.Input<framework::Tensor>("Input")->dims().size();
    switch (rank) {
      case 1:
        StridedSliceCompute<1>(ctx);
        break;
      case 2:
        StridedSliceCompute<2>(ctx);
        break;
      case 3:
        StridedSliceCompute<3>(ctx);
        break;
      case 4:
        StridedSliceCompute<4>(ctx);
        break;
      case 5:
        StridedSliceCompute<5>(ctx);
        break;
      case 6:
        StridedSliceCompute<6>(ctx);
        break;
    }
  }

 private:
  template <size_t D>
  void StridedSliceCompute(const framework::ExecutionContext& context) const {
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    auto in = context.Input<framework::Tensor>("Input");
    auto out = context.Output<framework::Tensor>("Out");
    auto out_dims = out->dims();
    auto in_dims = in->dims();

    auto begin = context.Attr<std::vector<int>>("begin");
    auto end = context.Attr<std::vector<int>>("end");
    auto stride = context.Attr<std::vector<int>>("stride");

    auto begin_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
    auto end_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
    auto stride_indices = Eigen::DSizes<Eigen::DenseIndex, D>();

    Eigen::array<bool, D> reverse_axis;
    for (size_t axis = 0; axis < D; axis++) {
      int axis_size = in_dims[axis];
      // stride must not be zero
      begin_indices[axis] = begin[axis];
      end_indices[axis] = end[axis];
      if (begin[axis] < 0) {
        begin_indices[axis] = (begin[axis] + axis_size) % axis_size;
      }

      if (end[axis] < 0) {
        end_indices[axis] = (end[axis] + axis_size) % axis_size;
      }
      if (stride[axis] < 0) {
        reverse_axis[axis] = true;
        stride_indices[axis] = -stride[axis];
        int tmp;
        tmp = end_indices[axis];
        end_indices[axis] = begin_indices[axis];
        begin_indices[axis] = tmp;
      } else {
        reverse_axis[axis] = false;
        stride_indices[axis] = stride[axis];
      }
    }
    framework::Tensor reverse_input;
    reverse_input.mutable_data<T>(in_dims, context.GetPlace());

    out->mutable_data<T>(context.GetPlace());
    auto in_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *in);
    auto reverse_in_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            reverse_input);
    auto out_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *out, out_dims);
    reverse_in_t.device(place) = in_t.reverse(reverse_axis);
    out_t.device(place) =
        reverse_in_t.stridedSlice(begin_indices, end_indices, stride_indices);
  }
};

template <typename DeviceContext, typename T>
class StridedSliceGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    size_t rank = ctx.Input<framework::Tensor>("Input")->dims().size();
    switch (rank) {
      case 1:
        StridedSliceGradCompute<1>(ctx);
        break;
      case 2:
        StridedSliceGradCompute<2>(ctx);
        break;
      case 3:
        StridedSliceGradCompute<3>(ctx);
        break;
      case 4:
        StridedSliceGradCompute<4>(ctx);
        break;
      case 5:
        StridedSliceGradCompute<5>(ctx);
        break;
      case 6:
        StridedSliceGradCompute<6>(ctx);
        break;
    }
  }

 private:
  template <size_t D>
  void StridedSliceGradCompute(
      const framework::ExecutionContext& context) const {
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    auto* d_input =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* d_out =
        context.Output<framework::Tensor>(framework::GradVarName("Input"));
    d_out->mutable_data<T>(context.GetPlace());

    auto& dev_ctx = context.template device_context<DeviceContext>();
    math::SetConstant<DeviceContext, T> set_zero;
    set_zero(dev_ctx, d_out, static_cast<T>(0));
    auto out_dims = d_out->dims();
    auto in_dims = d_input->dims();
    auto begin = context.Attr<std::vector<int>>("begin");
    auto end = context.Attr<std::vector<int>>("end");
    auto stride = context.Attr<std::vector<int>>("stride");

    auto begin_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
    auto end_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
    auto stride_indices = Eigen::DSizes<Eigen::DenseIndex, D>();

    Eigen::array<bool, D> reverse_axis;
    for (size_t axis = 0; axis < D; axis++) {
      int axis_size = in_dims[axis];
      // stride must not be zero
      begin_indices[axis] = begin[axis];
      end_indices[axis] = end[axis];
      if (begin[axis] < 0) {
        begin_indices[axis] = (begin[axis] + axis_size) % axis_size;
      }

      if (end[axis] < 0) {
        end_indices[axis] = (end[axis] + axis_size) % axis_size;
      }
      if (stride[axis] < 0) {
        reverse_axis[axis] = true;
        stride_indices[axis] = -stride[axis];
        int tmp;
        tmp = end_indices[axis];
        end_indices[axis] = begin_indices[axis];
        begin_indices[axis] = tmp;
      } else {
        reverse_axis[axis] = false;
        stride_indices[axis] = stride[axis];
      }
    }
    framework::Tensor reverse_input;
    reverse_input.mutable_data<T>(in_dims, context.GetPlace());

    auto in_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *d_input);
    auto reverse_in_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            reverse_input);
    auto out_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *d_out, out_dims);

    reverse_in_t.device(place) = in_t.reverse(reverse_axis);
    out_t.stridedSlice(begin_indices, end_indices, stride_indices)
        .device(place) = reverse_in_t;

    VLOG(0) << "I am here";
  }
};
}  // namespace operators
}  // namespace paddle
