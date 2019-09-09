/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

static void StridedSliceFunctor(int* begin, int* end, int* stride,
                                int* reverse_axis, const framework::DDim dims,
                                const int size) {
  for (size_t axis = 0; axis < size; axis++) {
    int axis_size = dims[axis];
    // stride must not be zero
    if (begin[axis] < 0) {
      begin[axis] = begin[axis] + axis_size;
    }

    if (end[axis] < 0) {
      end[axis] = end[axis] + axis_size;
    }
    if (stride[axis] < 0) {
      reverse_axis[axis] = 1;
      stride[axis] = -stride[axis];
      if (begin[axis] > end[axis]) {
        // swap the reverse
        begin[axis] = begin[axis] + 1;
        end[axis] = end[axis] + 1;
      }
      std::swap(begin[axis], end[axis]);
    } else {
      reverse_axis[axis] = 0;
      stride[axis] = stride[axis];
    }
  }
}
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

    std::vector<int> reverse_vector(begin.size(), 0);
    StridedSliceFunctor(begin.data(), end.data(), stride.data(),
                        reverse_vector.data(), in_dims, begin.size());
    for (size_t axis = 0; axis < D; axis++) {
      begin_indices[axis] = begin[axis];
      end_indices[axis] = end[axis];
      stride_indices[axis] = stride[axis];
      reverse_axis[axis] = (reverse_vector[axis] == 1) ? true : false;
    }

    framework::Tensor tmp;
    tmp.mutable_data<T>(out_dims, context.GetPlace());

    out->mutable_data<T>(context.GetPlace());
    auto in_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *in);
    auto tmp_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            tmp);
    auto out_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *out, out_dims);
    tmp_t.device(place) =
        in_t.stridedSlice(begin_indices, end_indices, stride_indices);
    out_t.device(place) = tmp_t.reverse(reverse_axis);
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
    std::vector<int> reverse_vector(begin.size(), 0);

    StridedSliceFunctor(begin.data(), end.data(), stride.data(),
                        reverse_vector.data(), out_dims, begin.size());
    for (size_t axis = 0; axis < D; axis++) {
      begin_indices[axis] = begin[axis];
      end_indices[axis] = end[axis];
      stride_indices[axis] = stride[axis];
      reverse_axis[axis] = (reverse_vector[axis] == 1) ? true : false;
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
  }
};
}  // namespace operators
}  // namespace paddle
