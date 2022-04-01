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
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class TopkKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // Get the top k elements of each row of input tensor
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    auto* indices = ctx.Output<Tensor>("Indices");

    size_t k = static_cast<int>(ctx.Attr<int>("k"));
    auto* k_t = ctx.Input<Tensor>("K");
    if (k_t) {
      k = k_t->data<int>()[0];
      framework::DDim output_dims = output->dims();
      output_dims[output_dims.size() - 1] = k;
      output->Resize(output_dims);
      indices->Resize(output_dims);
    }

    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    int64_t* indices_data = indices->mutable_data<int64_t>(ctx.GetPlace());

    // reshape input to a flattern matrix(like flat_inner_dims)
    framework::DDim inputdims = input->dims();
    const size_t row =
        phi::product(phi::slice_ddim(inputdims, 0, inputdims.size() - 1));
    const size_t col = inputdims[inputdims.size() - 1];
    Eigen::DSizes<int, 2> flat2dims(row, col);
// NOTE: eigen shape doesn't affect paddle tensor.
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
    for (size_t i = 0; i < row; i++) {
      std::vector<std::pair<T, size_t>> vec;
      vec.reserve(col);
      // 1D vector
      if (inputdims.size() == 1) {
        auto eg_input = framework::EigenVector<T>::Flatten(*input);
        for (size_t j = 0; j < col; j++) {
          vec.push_back(std::pair<T, size_t>(eg_input(j), j));
        }
      } else {
        auto eg_input =
            framework::EigenMatrix<T>::Reshape(*input, inputdims.size() - 1);
        for (size_t j = 0; j < col; j++) {
          vec.push_back(std::pair<T, size_t>(eg_input(i, j), j));
        }
      }

      std::partial_sort(
          vec.begin(), vec.begin() + k, vec.end(),
          [](const std::pair<T, size_t>& l, const std::pair<T, size_t>& r) {
            return l.first > r.first;
          });
      for (size_t j = 0; j < k; j++) {
        output_data[i * k + j] = vec[j].first;
        indices_data[i * k + j] = int64_t(vec[j].second);
      }
    }
  }
};

template <typename DeviceContext, typename T>
class TopkGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* out_grad = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* indices = context.Input<Tensor>("Indices");
    auto* x_grad = context.Output<Tensor>(framework::GradVarName("X"));

    T* x_grad_data = x_grad->mutable_data<T>(context.GetPlace());
    const T* out_grad_data = out_grad->data<T>();
    const int64_t* indices_data = indices->data<int64_t>();
    size_t k = indices->dims()[indices->dims().size() - 1];

    framework::DDim xdims = x->dims();
    const size_t row =
        phi::product(phi::slice_ddim(xdims, 0, xdims.size() - 1));
    const size_t col = xdims[xdims.size() - 1];

    memset(x_grad_data, 0, row * col * sizeof(T));

    for (size_t i = 0; i < row; ++i) {
      for (size_t j = 0; j < k; ++j) {
        size_t idx = indices_data[i * k + j];
        x_grad_data[i * col + idx] = out_grad_data[i * k + j];
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
