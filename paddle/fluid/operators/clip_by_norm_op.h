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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename DeviceContext, typename T>
class ClipByNormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto max_norm = context.Attr<T>("max_norm");
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Output<Tensor>("Out");
    output->mutable_data<T>(context.GetPlace());

    auto x = EigenVector<T>::Flatten(*input);
    auto out = EigenVector<T>::Flatten(*output);
    auto x_norm = x.square().sum().sqrt();
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();

    auto temp = (x_norm <= max_norm).template cast<T>().eval();
    auto scaling = temp + (static_cast<T>(1) - temp) * max_norm / x_norm;
    Eigen::array<int, 1> one_dim{{1}};
    Eigen::DSizes<int, 1> m_dsize(input->numel());
    out.device(place) = x * scaling.reshape(one_dim).broadcast(m_dsize);
  }
};

}  // namespace operators
}  // namespace paddle
