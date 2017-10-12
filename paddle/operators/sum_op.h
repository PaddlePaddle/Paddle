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

template <typename Place, typename T>
class SumKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto ins = context.MultiInput<Tensor>("X");
    auto* out = context.Output<Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    auto place = context.GetEigenDevice<Place>();
    auto result = EigenVector<T>::Flatten(*out);

    int N = ins.size();
    auto in = EigenVector<T>::Flatten(*(ins[0]));
    result.device(place) = in;
    for (int i = 1; i < N; i++) {
      auto in = EigenVector<T>::Flatten(*(ins[i]));
      result.device(place) = result + in;
    }
  }
};

}  // namespace operators
}  // namespace paddle
