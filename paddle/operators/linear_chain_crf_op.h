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

using framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
class LinearChainCrfOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override;

 protected:
  T ForwardOneSequence(const platform::DeviceContext& ctx,
                       const Tensor& emission, Tensor& emission_row_max,
                       Tensor& emission_exps, const Tensor& trans_weights,
                       Tensor& trans_weight_exps, const Tensor& label,
                       Tensor& a) const;

 private:
  T NormalizeL1(T* x, size_t len) const;
};

template <typename Place, typename T>
class LinearChainCrfGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override;
};

}  // namespace operators
}  // namespace paddle
