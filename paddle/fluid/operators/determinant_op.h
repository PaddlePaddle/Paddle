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
#include <cmath>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T>
struct DeterminantFunctor {
  void operator()(const Tensor& input, int rank, int batch_count,
                  Tensor* output) {
    std::vector<T> input_vec;
    std::vector<float> output_vec;
    framework::TensorToVector(input, &input_vec);
    for (int i = 0; i < batch_count; ++i) {  // maybe can be parallel
      auto begin_idx = input_vec.begin() + i * rank * rank;
      auto end_idx = input_vec.begin() + (i + 1) * rank * rank;
      std::vector<T> sub_vec(begin_idx,
                             end_idx);  // get every square matrix data
      Eigen::MatrixXf matrix(rank, rank);
      for (int i = 0; i < rank; ++i) {
        for (int j = 0; j < rank; ++j) {
          matrix(i, j) = sub_vec[rank * i + j];
        }
      }
      VLOG(2) << "det value: " << matrix.determinant();
      VLOG(2) << "matrix val: " << matrix;
      output_vec.push_back(matrix.determinant());
    }
    framework::TensorFromVector(output_vec, output);
  }
};

template <typename T>
struct SlogDeterminantFunctor {
  void operator()(const Tensor& input, int rank, int batch_count,
                  Tensor* output) {
    std::vector<T> input_vec;
    std::vector<float> sin_vec;
    std::vector<float> log_vec;
    std::vector<float> output_vec;
    framework::TensorToVector(input, &input_vec);
    for (int i = 0; i < batch_count; ++i) {  // maybe can be parallel
      auto begin_idx = input_vec.begin() + i * rank * rank;
      auto end_idx = input_vec.begin() + (i + 1) * rank * rank;
      std::vector<T> sub_vec(begin_idx,
                             end_idx);  // get every square matrix data
      Eigen::MatrixXf matrix(rank, rank);
      for (int i = 0; i < rank; ++i) {
        for (int j = 0; j < rank; ++j) {
          matrix(i, j) = sub_vec[rank * i + j];
        }
      }
      VLOG(2) << "det value: " << matrix.determinant();
      VLOG(2) << "matrix val: " << matrix;
      sin_vec.push_back(sin(matrix.determinant()));
      log_vec.push_back(log(matrix.determinant()));
    }
    // merge sin_vec and log_vec as final output_vec
    output_vec.insert(output_vec.end(), sin_vec.begin(), sin_vec.end());
    output_vec.insert(output_vec.end(), log_vec.begin(), log_vec.end());
    framework::TensorFromVector(output_vec, output);
  }
};

template <typename T>
class DeterminantKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    // const T* input_data = input->data<T>();
    auto input_dim = vectorize(input->dims());
    auto input_dim_size = input_dim.size();
    auto* output = context.Output<framework::Tensor>("Out");
    // auto output_dim = vectorize(output->dims());

    int batch_count = 1;
    for (int i = 0; i < input->dims().size() - 2; i++) {
      batch_count *= input_dim[i];
    }
    VLOG(2) << "input dim:" << input->dims();
    auto rank = input_dim[input_dim_size - 1];  // square matrix length
    DeterminantFunctor<T>()(*input, rank, batch_count, output);
  }
};

template <typename T>
class DeterminantGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* dout =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    const T* dout_data = dout->data<T>();
    auto dout_dim = vectorize(dout->dims());

    auto* dx =
        context.Output<framework::Tensor>(framework::GradVarName("Input"));
    T* dx_data = dx->mutable_data<T>(context.GetPlace());

    int64_t numel = dx->numel();
    for (int64_t idx = 0; idx < numel; idx++) {
      dx_data[idx] = static_cast<T>(1);
    }
  }
};

template <typename T>
class SlogDeterminantKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    // const T* input_data = input->data<T>();
    auto input_dim = vectorize(input->dims());
    auto input_dim_size = input_dim.size();
    auto* output = context.Output<framework::Tensor>("Out");
    // auto output_dim = vectorize(output->dims());

    int batch_count = 1;
    for (int i = 0; i < input->dims().size() - 2; i++) {
      batch_count *= input_dim[i];
    }
    VLOG(2) << "input dim:" << input->dims();
    auto rank = input_dim[input_dim_size - 1];  // square matrix length
    SlogDeterminantFunctor<T>()(*input, rank, batch_count, output);
  }
};

}  // namespace operators
}  // namespace paddle
