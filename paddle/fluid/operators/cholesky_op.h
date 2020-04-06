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

#include "Eigen/Cholesky"
#include "Eigen/Core"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class CholeskyCPUKernel : public framework::OpKernel<T> {
 public:
  // different with EigenMatrix in framework/eigen.h
  using EigenMatrix =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using InputMatrixMap = Eigen::Map<const EigenMatrix>;
  using OutputMatrixMap = Eigen::Map<EigenMatrix>;
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");

    bool upper = context.Attr<bool>("upper");
    auto& dims = x->dims();
    int batch_count = 1;
    for (int i = 0; i < dims.size() - 2; i++) {
      batch_count *= dims[i];
    }
    auto m = dims[dims.size() - 1];

    const auto* x_data = x->data<T>();
    auto* out_data = out->mutable_data<T>(context.GetPlace());
    // Cholesky decomposition for each matrix, maybe can use multi threads
    for (int i = 0; i < batch_count; i++) {
      auto input = InputMatrixMap(x_data + i * m * m, m, m);
      auto output = OutputMatrixMap(out_data + i * m * m, m, m);
      if (upper) {
        Eigen::LLT<
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
            Eigen::UpLoType::Upper>
            llt_decomposition(input);
        PADDLE_ENFORCE_EQ(
            llt_decomposition.info(), Eigen::Success,
            "Cholesky decomposition was not successful. The input matrice "
            "might not be not be positive definite.")
        output = llt_decomposition.matrixU();
      } else {
        Eigen::LLT<
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
            Eigen::UpLoType::Lower>
            llt_decomposition(input);
        PADDLE_ENFORCE_EQ(
            llt_decomposition.info(), Eigen::Success,
            "Cholesky decomposition was not successful. The input matrice "
            "might not be not be positive definite.")
        output = llt_decomposition.matrixL();
      }
    }
  }
};

template <typename T>
class CholeskyGradCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

}  // namespace operators
}  // namespace paddle
