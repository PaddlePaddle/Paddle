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
#include "paddle/fluid/operators/elementwise/elementwise_mul_op.h"
#include "paddle/fluid/operators/math/matrix_inverse.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/operators/unsqueeze_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T>
struct DeterminantFunctor {
  void operator()(const Tensor& input, const framework::ExecutionContext ctx,
                  int rank, int batch_count, Tensor* output) {
    std::vector<T> input_vec;
    std::vector<float> output_vec;
    framework::TensorToVector(input, ctx.device_context(), &input_vec);
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
      output_vec.push_back(matrix.determinant());
    }
    framework::TensorFromVector(output_vec, output);
  }
};
template <typename T>
class DeterminantKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    auto input_dim = vectorize(input->dims());
    auto input_dim_size = input_dim.size();
    auto* output = context.Output<framework::Tensor>("Out");

    int batch_count = 1;
    for (int i = 0; i < input->dims().size() - 2; i++) {
      batch_count *= input_dim[i];
    }

    VLOG(2) << "input dim:" << input->dims();
    auto rank = input_dim[input_dim_size - 1];  // square matrix length
    DeterminantFunctor<T>()(*input, context, rank, batch_count, output);
    auto output_dims =
        framework::slice_ddim(input->dims(), 0, input_dim_size - 2);
    output->Resize(output_dims);
    VLOG(2) << "output dim:" << output->dims();
  }
};

template <typename DeviceContext, typename T>
struct ElementwiseMulFunctor {
  void operator()(const framework::ExecutionContext& ctx,
                  const framework::Tensor* x, const framework::Tensor* y,
                  framework::Tensor* z) {
    default_elementwise_mul<DeviceContext, T>(ctx, x, y, z);
  }
};

template <typename DeviceContext, typename T>
class DeterminantGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& dev_ctx = context.template device_context<DeviceContext>();
    const auto* input = context.Input<framework::Tensor>("Input");
    const auto* det = context.Input<framework::Tensor>("Out");
    const auto* grad =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* ddet =
        context.Output<framework::Tensor>(framework::GradVarName("Input"));

    // let |A| = Determinant(A)
    // In pytorch:
    // d|A| = det(A).unsqueeze(-1).unsqueeze(-2) * inverse(A).transpose(-2, -1)
    // In tensorflow:
    // d|A| = det(A).reshape(tf.concat([det(A).shape, [1, 1]], 0)) * inv(A,
    // adjoint=True)
    // And ref to https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
    // we set d|A| = unsqueeze(dA * |A|, [-1, -2]) * inverse(A).transpose(-2,
    // -1)

    // First: inverse(A)
    framework::Tensor inverse_A;
    // A must be square matrices!
    inverse_A.Resize(input->dims());
    inverse_A.mutable_data<T>(context.GetPlace());

    math::MatrixInverseFunctor<DeviceContext, T> mat_inv;
    mat_inv(dev_ctx, *input, &inverse_A);

    VLOG(3) << "inverse(A) dims: " << inverse_A.dims();

    // Second: inverse(A).transpose(-2, -1)
    framework::Tensor transpose_inverse_A;
    // A must be square matrix! Transpose the last two dimension not change
    // the shape
    transpose_inverse_A.Resize(inverse_A.dims());
    transpose_inverse_A.mutable_data<T>(context.GetPlace());

    auto inverse_A_rank = inverse_A.dims().size();
    std::vector<int> trans_axis(inverse_A_rank);
    for (int i = 0; i < inverse_A_rank; ++i) {
      trans_axis[i] = i;
    }
    // Transpose the last two dimension
    trans_axis[inverse_A_rank - 1] = inverse_A_rank - 2;
    trans_axis[inverse_A_rank - 2] = inverse_A_rank - 1;
    TransCompute<DeviceContext, T>(inverse_A_rank, dev_ctx, inverse_A,
                                   &transpose_inverse_A, trans_axis);

    // Third: dA * |A|
    ElementwiseMulFunctor<DeviceContext, T> elem_mul;
    framework::Tensor mul_dA_detA;
    mul_dA_detA.Resize(grad->dims());
    mul_dA_detA.mutable_data<T>(context.GetPlace());
    elem_mul(context, grad, det, &mul_dA_detA);

    // Fourth: unsqueeze(dA * |A|, [-1, -2])
    auto unsqueeze_dims = UnsqueezeKernel<DeviceContext, T>::GetOutputShape(
        {-1, -2}, mul_dA_detA.dims());
    mul_dA_detA.Resize(unsqueeze_dims);

    VLOG(3) << "unsqueezed(dA * |A|) dims: " << mul_dA_detA.dims();

    // Finally: unsqueeze(dA * |A|) * inverse(A)
    ddet->Resize(transpose_inverse_A.dims());
    ddet->mutable_data<T>(context.GetPlace());
    elem_mul(context, &mul_dA_detA, &transpose_inverse_A, ddet);

    VLOG(3) << "d|A| dims: " << ddet->dims();
  }
};

template <typename T>
T sign(T val) {
  return static_cast<T>(T(0) < val) - (val < T(0));
}
template <typename T>
struct SlogDeterminantFunctor {
  void operator()(const Tensor& input, const framework::ExecutionContext ctx,
                  int rank, int batch_count, Tensor* output) {
    std::vector<T> input_vec;
    std::vector<float> sign_vec;
    std::vector<float> log_vec;
    std::vector<float> output_vec;
    framework::TensorToVector(input, ctx.device_context(), &input_vec);
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
      auto det_val = matrix.determinant();
      sign_vec.push_back(sign(det_val));
      det_val >= 0
          ? log_vec.push_back(log(det_val))
          : log_vec.push_back(
                -1.0 *
                log(abs(
                    det_val)));  // for computing log value of a negative value.
    }
    // merge sign_vec and log_vec as final output_vec
    output_vec.insert(output_vec.end(), sign_vec.begin(), sign_vec.end());
    output_vec.insert(output_vec.end(), log_vec.begin(), log_vec.end());
    framework::TensorFromVector(output_vec, output);
  }
};

template <typename T>
class SlogDeterminantKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    auto input_dim = vectorize(input->dims());
    auto input_dim_size = input_dim.size();
    auto* output = context.Output<framework::Tensor>("Out");

    int batch_count = 1;
    for (int i = 0; i < input->dims().size() - 2; i++) {
      batch_count *= input_dim[i];
    }
    VLOG(2) << "input dim:" << input->dims();
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

}  // namespace operators
}  // namespace paddle
