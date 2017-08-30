/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T, typename AttrType = int>
class TopkKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // Get the top k elements of each row of input tensor
    // FIXME: only deal with matrix(2d tensor).
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    auto* indices = ctx.Output<Tensor>("Indices");
    // k is determined by Attr
    const unsigned int k =
        static_cast<AttrType>(ctx.op_.GetAttr<AttrType>("k"));

    output->mutable_data<T>(ctx.GetPlace());
    indices->mutable_data<T>(ctx.GetPlace());

    auto X = EigenMatrix<T>::From(*input);
    auto Out = EigenMatrix<T>::From(*output);
    auto Indices = EigenMatrix<T>::From(*indices);

    // reshape input to a flattern matrix(like flat_inner_dims)
    framework::DDim inputdims = input->dims();
    const unsigned int row = framework::product(
        framework::slice_ddim(inputdims, 0, inputdims.size() - 1));
    const unsigned int col = inputdims[inputdims.size() - 1];
    Eigen::DSizes<int, 2> flat2dims(row, col);
    X.reshape(flat2dims);

    for (size_t i = 0; i < row; i++) {
      // TODO(typhoonzero): make this more efficient
      std::vector<std::pair<T, size_t>> vec;
      for (size_t j = 0; j < col; j++) {
        vec.push_back(std::pair<T, size_t>(X(i, j), j));
      }

      std::partial_sort(
          vec.begin(), vec.begin() + k, vec.end(),
          [](const std::pair<T, size_t>& l, const std::pair<T, size_t>& r) {
            return l.first > r.first;
          });
      for (size_t j = 0; j < k; j++) {
        Out(i, j) = vec[j].first;
        Indices(i, j) = vec[j].second;
      }
    }
    // FIXME: Resize back to the original input shape
  }
};

}  // namespace operators
}  // namespace paddle
