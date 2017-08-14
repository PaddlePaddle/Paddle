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
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
class RowWiseAddKernel : public framework::OpKernel {
 public:
  void Compute(framework::ExecutionContext context) const override {
    auto out = context.Output<Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    auto input = EigenMatrix<T>::From(*context.Input<Tensor>("X"));
    auto bias = EigenVector<T>::From(*context.Input<Tensor>("b"));
    auto output = EigenMatrix<T>::From(*out);

    const int bias_size = bias.dimension(0);
    const int rest_size = input.size() / bias_size;
    Eigen::DSizes<int, 1> one_d(input.size());
    Eigen::DSizes<int, 1> bcast(rest_size);
    output.reshape(one_d).device(context.GetEigenDevice<Place>()) =
        input.reshape(one_d) + bias.broadcast(bcast).reshape(one_d);
  }
};

}  // namespace operators
}  // namespace paddle
