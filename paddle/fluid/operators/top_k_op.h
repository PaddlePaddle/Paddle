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

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename DeviceContext, typename T>
class TopkKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // Get the top k elements of each row of input tensor
    // FIXME: only deal with matrix(2d tensor).
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    auto* indices = ctx.Output<Tensor>("Indices");
    // k is determined by Attr
    const size_t k = static_cast<int>(ctx.Attr<int>("k"));

    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    int64_t* indices_data = indices->mutable_data<int64_t>(ctx.GetPlace());

    auto eg_input = EigenMatrix<T>::From(*input);

    // reshape input to a flattern matrix(like flat_inner_dims)
    framework::DDim inputdims = input->dims();
    const size_t row = framework::product(
        framework::slice_ddim(inputdims, 0, inputdims.size() - 1));
    const size_t col = inputdims[inputdims.size() - 1];
    Eigen::DSizes<int, 2> flat2dims(row, col);
    // NOTE: eigen shape doesn't affect paddle tensor.
    eg_input.reshape(flat2dims);

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
    for (size_t i = 0; i < row; i++) {
      std::vector<std::pair<T, size_t>> vec;
      for (size_t j = 0; j < col; j++) {
        vec.push_back(std::pair<T, size_t>(eg_input(i, j), j));
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

}  // namespace operators
}  // namespace paddle
