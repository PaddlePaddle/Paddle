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
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T>
T sign(T val) {
  return static_cast<T>(T(0) < val) - (val < T(0));
}

template <typename T>
class EigenMatrix {};

template <>
class EigenMatrix<float> {
 public:
  using MatrixType = Eigen::MatrixXf;
};

template <>
class EigenMatrix<double> {
 public:
  using MatrixType = Eigen::MatrixXd;
};

inline int64_t GetBatchCount(framework::DDim dims) {
  int64_t batch_count = 1;
  auto dim_size = dims.size();
  PADDLE_ENFORCE_GT(dim_size, 2,
                    platform::errors::InvalidArgument(
                        "To get the number of batch square matrices, "
                        "the size of dimension should greater than 2.",
                        dim_size));

  // Cumulative multiplying each dimension until the last 2 to get the batch
  // count,
  // for example a tensor with shape [3,3,3,3], the batch count of matrices is
  // 9.
  for (int i = 0; i < dims.size() - 2; i++) {
    batch_count *= dims[i];
  }

  return batch_count;
}

template <typename T>
struct DeterminantFunctor {
  void operator()(const Tensor& input, const framework::ExecutionContext ctx,
                  int rank, int64_t batch_count, Tensor* output) {
    std::vector<T> input_vec;
    std::vector<T> output_vec;
    framework::TensorToVector(input, ctx.device_context(), &input_vec);
    for (int i = 0; i < batch_count; ++i) {  // maybe can be parallel
      auto begin_iter = input_vec.begin() + i * rank * rank;
      auto end_iter = input_vec.begin() + (i + 1) * rank * rank;
      std::vector<T> sub_vec(begin_iter,
                             end_iter);  // get every square matrix data
      Eigen::MatrixXf matrix(rank, rank);
      for (int i = 0; i < rank; ++i) {
        for (int j = 0; j < rank; ++j) {
          matrix(i, j) = sub_vec[rank * i + j];
        }
      }
      output_vec.push_back(matrix.determinant());
    }
    framework::TensorFromVector(output_vec, output);
  }
};
template <typename DeviceContext, typename T>
class DeterminantKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    auto input_dim = vectorize(input->dims());
    auto input_dim_size = input_dim.size();
    auto* output = context.Output<framework::Tensor>("Out");

    auto batch_count = GetBatchCount(input->dims());
    VLOG(2) << "input dim:" << input->dims();
    PADDLE_ENFORCE_GE(
        input_dim_size, 2,
        platform::errors::InvalidArgument(
            "the input matrix dimension size should greater than 2."));
    PADDLE_ENFORCE_EQ(input_dim[input_dim_size - 1],
                      input_dim[input_dim_size - 2],
                      platform::errors::InvalidArgument(
                          "the input matrix should be square matrix."));
    auto rank = input_dim[input_dim_size - 1];  // square matrix length
    DeterminantFunctor<T>()(*input, context, rank, batch_count, output);
    if (input_dim_size > 2) {
      auto output_dims =
          framework::slice_ddim(input->dims(), 0, input_dim_size - 2);
      output->Resize(output_dims);
    }
    VLOG(2) << "output dim:" << output->dims();
  }
};

template <typename DeviceContext, typename T>
class DeterminantGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Not support DeterminantGrad at this time."));
  }
};

template <typename T>
struct SlogDeterminantFunctor {
  void operator()(const Tensor& input, const framework::ExecutionContext ctx,
                  int rank, int batch_count, Tensor* output) {
    std::vector<T> input_vec;
    std::vector<T> sign_vec;
    std::vector<T> log_vec;
    std::vector<T> output_vec;
    framework::TensorToVector(input, ctx.device_context(), &input_vec);
    for (int i = 0; i < batch_count; ++i) {  // maybe can be parallel
      auto begin_iter = input_vec.begin() + i * rank * rank;
      auto end_iter = input_vec.begin() + (i + 1) * rank * rank;
      std::vector<T> sub_vec(begin_iter,
                             end_iter);  // get every square matrix data
      typename EigenMatrix<T>::MatrixType matrix(rank, rank);
      for (int i = 0; i < rank; ++i) {
        for (int j = 0; j < rank; ++j) {
          matrix(i, j) = sub_vec[rank * i + j];
        }
      }
      VLOG(2) << "det value: " << matrix.determinant();
      VLOG(2) << "matrix val: " << matrix;
      auto det_val = matrix.determinant();
      sign_vec.push_back(sign(det_val));
      det_val >= 0
          ? log_vec.push_back(log(det_val))
          : log_vec.push_back(log(
                abs(det_val)));  // for computing log value of a negative value.
    }
    // merge sign_vec and log_vec as final output_vec
    output_vec.insert(output_vec.end(), sign_vec.begin(), sign_vec.end());
    output_vec.insert(output_vec.end(), log_vec.begin(), log_vec.end());
    framework::TensorFromVector(output_vec, output);
  }
};

template <typename DeviceContext, typename T>
class SlogDeterminantKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    auto input_dim = vectorize(input->dims());
    auto input_dim_size = input_dim.size();
    auto* output = context.Output<framework::Tensor>("Out");

    auto batch_count = GetBatchCount(input->dims());
    VLOG(2) << "input dim:" << input->dims();
    PADDLE_ENFORCE_GE(
        input_dim_size, 2,
        platform::errors::InvalidArgument(
            "the input matrix dimension size should greater than 2."));
    PADDLE_ENFORCE_EQ(input_dim[input_dim_size - 1],
                      input_dim[input_dim_size - 2],
                      platform::errors::InvalidArgument(
                          "the input matrix should be square matrix."));
    auto rank = input_dim[input_dim_size - 1];  // square matrix length
    SlogDeterminantFunctor<T>()(*input, context, rank, batch_count, output);
    std::vector<int> output_dim_vec(input_dim.begin(), input_dim.end() - 2);
    output_dim_vec.insert(output_dim_vec.begin(),
                          2);  // make the output dims as same as numpy
    auto output_dims = framework::make_ddim(output_dim_vec);
    output->Resize(output_dims);
    VLOG(2) << "output dim:" << output->dims();
  }
};

template <typename DeviceContext, typename T>
class SlogDeterminantGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Not support SlogDeterminantGrad at this time."));
  }
};

}  // namespace operators
}  // namespace paddle
