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
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;

using Array1 = Eigen::DSizes<int64_t, 1>;
using Array2 = Eigen::DSizes<int64_t, 2>;
using Array3 = Eigen::DSizes<int64_t, 3>;
using Array4 = Eigen::DSizes<int64_t, 4>;

/**
 *Return a tensor with evenly spaced numbers over a specified interval.
 */
template <typename DeviceContext, typename T>
struct Linspace {
  void operator()(T start, T end, int count, bool align_corners,
                  framework::Tensor* numbers,
                  const framework::ExecutionContext& ctx);
};

template <typename DeviceContext, typename T>
inline void GetIdxMap(int n, int h, int w, bool align_corners, Tensor* grid,
                      const framework::ExecutionContext& ctx) {
  auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
  grid->mutable_data<T>({n, h, w, 3}, ctx.GetPlace());
  auto grid_t = EigenTensor<T, 4>::From(*grid);
  // Get indexes of height with shape [height, width, 1]
  Tensor h_idx;
  Linspace<DeviceContext, T> linspace;
  linspace((T)-1, (T)1, h, align_corners, &h_idx, ctx);
  auto h_idx_t = EigenTensor<T, 1>::From(h_idx);
  // Get indexes of width with shape [height, width, 1]
  Tensor w_idx;
  linspace((T)-1, (T)1, w, align_corners, &w_idx, ctx);
  auto w_idx_t = EigenTensor<T, 1>::From(w_idx);
  // Get constant ones tensor with shape [height, width, 1]
  Tensor ones;
  ones.mutable_data<T>({h, w, 1}, ctx.GetPlace());

  phi::funcs::SetConstant<DeviceContext, T>()(
      ctx.template device_context<DeviceContext>(), &ones, static_cast<T>(1));
  auto ones_t = EigenTensor<T, 3>::From(ones);
  // Get grid tensor with shape [n, h, w, 3] by concatenating h_idx, w_idx and
  // ones
  Tensor w_idx_map;
  w_idx_map.mutable_data<T>({h, w, 1}, ctx.GetPlace());
  auto w_idx_map_t = EigenTensor<T, 3>::From(w_idx_map);
  Tensor h_idx_map;
  h_idx_map.mutable_data<T>({h, w, 1}, ctx.GetPlace());
  auto h_idx_map_t = EigenTensor<T, 3>::From(h_idx_map);
  Tensor w_h_idx_map;
  w_h_idx_map.mutable_data<T>({h, w, 2}, ctx.GetPlace());
  auto w_h_idx_map_t = EigenTensor<T, 3>::From(w_h_idx_map);
  Tensor w_h_one_idx_map;
  w_h_one_idx_map.mutable_data<T>({h, w, 3}, ctx.GetPlace());
  auto w_h_one_idx_map_t = EigenTensor<T, 3>::From(w_h_one_idx_map);
  w_idx_map_t.device(place) = w_idx_t.reshape(Array2(1, w))
                                  .broadcast(Array2(h, 1))
                                  .reshape(Array3(h, w, 1));
  h_idx_map_t.device(place) = h_idx_t.reshape(Array2(1, h))
                                  .broadcast(Array2(w, 1))
                                  .shuffle(Array2(1, 0))
                                  .reshape(Array3(h, w, 1));

  w_h_idx_map_t.device(place) = w_idx_map_t.concatenate(h_idx_map_t, 2);
  w_h_one_idx_map_t.device(place) = w_h_idx_map_t.concatenate(ones_t, 2);
  grid_t.device(place) = w_h_one_idx_map_t.reshape(Array4(1, h, w, 3))
                             .broadcast(Array4(n, 1, 1, 1));
}

template <typename DeviceContext, typename T>
class AffineGridOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* theta = ctx.Input<Tensor>("Theta");
    int n = theta->dims()[0];
    auto size_attr = ctx.Attr<std::vector<int>>("output_shape");
    auto align_corners = ctx.Attr<bool>("align_corners");
    int h = 0;
    int w = 0;
    if (size_attr.size() == 0) {
      auto* output_shape = ctx.Input<Tensor>("OutputShape");
      Tensor h_sizes;
      framework::TensorCopy(*output_shape, platform::CPUPlace(), &h_sizes);
      const int* h_size_data = h_sizes.data<int>();
      h = h_size_data[2];
      w = h_size_data[3];
    } else {
      h = size_attr[2];
      w = size_attr[3];
    }
    auto* output = ctx.Output<Tensor>("Output");
    output->mutable_data<T>({n, h, w, 2}, ctx.GetPlace());
    phi::funcs::SetConstant<DeviceContext, T>()(
        ctx.template device_context<DeviceContext>(), output,
        static_cast<T>(0));
    Tensor grid;
    GetIdxMap<DeviceContext, T>(n, h, w, align_corners, &grid, ctx);
    // output = grid * theta.T
    // TODO(wanghaoshuang): Refine batched matrix multiply
    auto blas = phi::funcs::GetBlas<DeviceContext, T>(ctx);
    for (int i = 0; i < n; ++i) {
      Tensor sliced_grid = grid.Slice(i, i + 1).Resize(
          {static_cast<int64_t>(h) * static_cast<int64_t>(w), 3});
      Tensor sliced_theta = theta->Slice(i, i + 1).Resize({2, 3});
      Tensor sliced_out = output->Slice(i, i + 1).Resize(
          {static_cast<int64_t>(h) * static_cast<int64_t>(w), 2});
      blas.MatMul(sliced_grid, false, sliced_theta, true, T(1), &sliced_out,
                  T(0));
    }
  }
};

template <typename DeviceContext, typename T>
class AffineGridGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto output_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));
    auto theta_grad = ctx.Output<Tensor>(framework::GradVarName("Theta"));
    int n = output_grad->dims()[0];
    auto size_attr = ctx.Attr<std::vector<int>>("output_shape");
    auto align_corners = ctx.Attr<bool>("align_corners");
    int h = 0;
    int w = 0;
    if (size_attr.size() == 0) {
      auto* output_shape = ctx.Input<Tensor>("OutputShape");
      Tensor h_sizes;
      framework::TensorCopy(*output_shape, platform::CPUPlace(), &h_sizes);
      const int* h_size_data = h_sizes.data<int>();
      h = h_size_data[2];
      w = h_size_data[3];
    } else {
      h = size_attr[2];
      w = size_attr[3];
    }
    theta_grad->mutable_data<T>({n, 2, 3}, ctx.GetPlace());
    phi::funcs::SetConstant<DeviceContext, T>()(
        ctx.template device_context<DeviceContext>(), theta_grad,
        static_cast<T>(0));
    Tensor grid;
    GetIdxMap<DeviceContext, T>(n, h, w, align_corners, &grid, ctx);
    // output = grid * theta.T
    // TODO(wanghaoshuang): Refine batched matrix multiply
    auto blas = phi::funcs::GetBlas<DeviceContext, T>(ctx);
    for (int i = 0; i < n; ++i) {
      Tensor sliced_grid = grid.Slice(i, i + 1).Resize(
          {static_cast<int64_t>(h) * static_cast<int64_t>(w), 3});
      Tensor sliced_out_grad = output_grad->Slice(i, i + 1).Resize(
          {static_cast<int64_t>(h) * static_cast<int64_t>(w), 2});
      Tensor sliced_theta_grad = theta_grad->Slice(i, i + 1).Resize({2, 3});
      blas.MatMul(sliced_out_grad, true, sliced_grid, false, T(1),
                  &sliced_theta_grad, T(0));
    }
  }
};

}  // namespace operators
}  // namespace paddle
