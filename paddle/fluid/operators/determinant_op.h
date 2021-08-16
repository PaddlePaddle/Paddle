// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <Eigen/Dense>
#include <Eigen/LU>
#include <algorithm>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;
using Tensor = framework::Tensor;

template <typename T>
Eigen::MatrixXf Tensor2EigenMatrix(const Tensor& tensor) {
  Eigen::MatrixXf matrix;
  std::vector<T> vector;
  framework::TensorToVector(tensor, &vector);
  for (auto it = vector.begin(); it != vector.end(); it++) {
    matrix << *it;
  }
  return matrix;
}

template <typename T>
class DeterminantKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    // const T* input_data = input->data<T>();
    auto input_dim = vectorize(input->dims());
    auto input_dim_size = input_dim.size();

    auto* output = context.Output<framework::Tensor>("Out");
    // T* output_data = output->mutable_data<T>(context.GetPlace());
    auto output_dim = vectorize(output->dims());

    int batch_count = 1;
    for (int i = 0; i < input->dims().size() - 2; i++) {
      batch_count *= input_dim[i];
    }
    VLOG(2) << "input dim:" << input->dims();
    auto m = input_dim[input_dim_size - 1];  // square matrix length
    std::vector<T> output_vector;
    for (int i = 0; i < batch_count; i++) {
      Tensor slice_input = input->Slice(i * m * m, (i + 1) * m * m);
      Eigen::MatrixXf mat = Tensor2EigenMatrix<T>(slice_input);
      auto det = mat.determinant();
      output_vector.push_back(det);
      VLOG(3) << "det value:" << det;
    }
    framework::TensorFromVector(output_vector, output);
  }
};

}  // namespace operators
}  // namespace paddle
