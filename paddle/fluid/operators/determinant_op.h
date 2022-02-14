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
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/operators/math/matrix_inverse.h"
#include "paddle/fluid/operators/svd_helper.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/for_range.h"

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

inline int64_t GetBatchCount(const framework::DDim dims) {
  int64_t batch_count = 1;
  auto dim_size = dims.size();
  PADDLE_ENFORCE_GE(
      dim_size, 2,
      platform::errors::InvalidArgument(
          "the input matrix dimension size should greater than 2."));

  // Cumulative multiplying each dimension until the last 2 to get the batch
  // count,
  // for example a tensor with shape [3,3,3,3], the batch count of matrices is
  // 9.
  for (int64_t i = 0; i < dims.size() - 2; i++) {
    batch_count *= dims[i];
  }

  return batch_count;
}

template <typename T>
struct DeterminantFunctor {
  void operator()(const Tensor& input, const framework::ExecutionContext ctx,
                  int64_t rank, int64_t batch_count, Tensor* output) {
    std::vector<T> input_vec;
    std::vector<T> output_vec;
    framework::TensorToVector(input, ctx.device_context(), &input_vec);
    for (int64_t i = 0; i < batch_count; ++i) {  // maybe can be parallel
      auto begin_iter = input_vec.begin() + i * rank * rank;
      auto end_iter = input_vec.begin() + (i + 1) * rank * rank;
      std::vector<T> sub_vec(begin_iter,
                             end_iter);  // get every square matrix data
      typename EigenMatrix<T>::MatrixType matrix(rank, rank);
      for (int64_t i = 0; i < rank; ++i) {
        for (int64_t j = 0; j < rank; ++j) {
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
    auto output_dims =
        framework::slice_ddim(input->dims(), 0, input_dim_size - 2);
    if (input_dim_size > 2) {
      output->Resize(output_dims);
    } else {
      // when input is a two-dimension matrix, The det value is a number.
      output->Resize({1});
    }
    VLOG(2) << "output dim:" << output->dims();
  }
};

template <typename T>
struct FoundZeroFunctor {
  FoundZeroFunctor(const T* x, int64_t numel, bool* res)
      : x_(x), numel_(numel), res_(res) {}
  HOSTDEVICE void operator()(size_t idx) const {
    if (*res_ || idx >= static_cast<size_t>(numel_)) {
      // founded zero number
      return;
    }
    *res_ = (x_[idx] == static_cast<T>(0));
  }
  const T* x_;
  int64_t numel_;
  bool* res_;
};

template <typename DeviceContext, typename T>
inline bool CheckMatrixInvertible(const framework::ExecutionContext& ctx,
                                  const framework::Tensor* det) {
  auto& dev_ctx = ctx.template device_context<DeviceContext>();
  auto numel = det->numel();

  framework::Tensor dev_tensor;
  auto* data = dev_tensor.mutable_data<bool>({1}, ctx.GetPlace());

  // set false
  pten::funcs::SetConstant<DeviceContext, bool> zero;
  zero(dev_ctx, &dev_tensor, false);

  // find whether zero
  platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
  FoundZeroFunctor<T> functor(det->data<T>(), numel, data);
  for_range(functor);

  // copy to host
  dev_ctx.Wait();
  framework::Tensor cpu_tensor;
  framework::TensorCopy(dev_tensor, platform::CPUPlace(), &cpu_tensor);

  // if founded zero, the matrix is not invertible
  // else the matrix is invertible
  auto* res = cpu_tensor.data<bool>();
  return !(*res);
}

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

    auto input_dims_size = input->dims().size();
    if (input_dims_size > 2) {
      PADDLE_ENFORCE_EQ(
          grad->dims().size() + 2, input_dims_size,
          platform::errors::InvalidArgument(
              "The grad tensor of det dims size should 2 less than"
              " input tensor's, but here differ %d",
              input_dims_size - grad->dims().size()));
    } else if (input_dims_size == 2) {
      // input dims size 2 and grad dims size 1 is possible
      PADDLE_ENFORCE_EQ(
          grad->dims().size(), 1,
          platform::errors::InvalidArgument(
              "The grad tensor of det dims size should 2 less than"
              " input tensor's, but here differ %d",
              input_dims_size - grad->dims().size()));
    } else {
      // checked in forward, pass
    }

    // Check Whether the matrix is invertible
    // (matrix A not invertible) == (det(A)=0)
    if (!CheckMatrixInvertible<DeviceContext, T>(context, det)) {
      // The matrix is not invertible
      VLOG(3) << "The input matrix not invertible!";
      ddet->Resize(input->dims());
      ddet->mutable_data<T>(context.GetPlace());
      pten::funcs::SetConstant<DeviceContext, T> zero;
      zero(dev_ctx, ddet, static_cast<T>(0.0f));
      return;
    }

    // The matrix is invertible
    // let |A| = Determinant(A)
    // Ref to https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
    // we set d|A| = unsqueeze(dA * |A|, [-1, -2]) * inverse(A).transpose(-2,
    // -1)

    math::DeviceIndependenceTensorOperations<DeviceContext, T> helper(context);

    // First: inverse(A)
    framework::Tensor inverse_A;
    // A must be square matrices!
    inverse_A.Resize(input->dims());
    inverse_A.mutable_data<T>(context.GetPlace());

    math::MatrixInverseFunctor<DeviceContext, T> mat_inv;
    mat_inv(dev_ctx, *input, &inverse_A);

    VLOG(3) << "inverse(A) dims: " << inverse_A.dims();

    // Second: inverse(A).transpose(-2, -1)
    framework::Tensor transpose_inverse_A = helper.Transpose(inverse_A);
    VLOG(3) << "(dA * |A|).transpose(-2, -1) dims: "
            << transpose_inverse_A.dims();

    // Third: dA * |A|
    auto mul_dA_detA = helper.Mul(*grad, *det);
    VLOG(3) << "dA * |A| dims: " << mul_dA_detA.dims();

    // Fourth: unsqueeze(dA * |A|, [-1, -2])
    auto unsqueeze1 = helper.Unsqueeze(mul_dA_detA, -1);
    auto unsqueeze2 = helper.Unsqueeze(unsqueeze1, -2);
    VLOG(3) << "unsqueezed(dA * |A|) dims: " << unsqueeze2.dims();

    // Finally: unsqueeze(dA * |A|) * inverse(A)
    auto res = helper.Mul(unsqueeze2, transpose_inverse_A);

    VLOG(3) << "unsqueeze(dA * |A|) * inverse(A) dims: " << res.dims();

    framework::TensorCopy(res, context.GetPlace(), ddet);

    ddet->Resize(input->dims());
    VLOG(3) << "d|A| dims: " << ddet->dims();
  }
};

template <typename T>
struct SlogDeterminantFunctor {
  void operator()(const Tensor& input, const framework::ExecutionContext ctx,
                  int64_t rank, int64_t batch_count, Tensor* output) {
    std::vector<T> input_vec;
    std::vector<T> sign_vec;
    std::vector<T> log_vec;
    std::vector<T> output_vec;
    framework::TensorToVector(input, ctx.device_context(), &input_vec);
    for (int64_t i = 0; i < batch_count; ++i) {  // maybe can be parallel
      auto begin_iter = input_vec.begin() + i * rank * rank;
      auto end_iter = input_vec.begin() + (i + 1) * rank * rank;
      std::vector<T> sub_vec(begin_iter,
                             end_iter);  // get every square matrix data
      typename EigenMatrix<T>::MatrixType matrix(rank, rank);
      for (int64_t i = 0; i < rank; ++i) {
        for (int64_t j = 0; j < rank; ++j) {
          matrix(i, j) = sub_vec[rank * i + j];
        }
      }
      VLOG(2) << "det value: " << matrix.determinant();
      VLOG(2) << "matrix val: " << matrix;
      auto det_val = matrix.determinant();
      sign_vec.push_back(sign(det_val));
      det_val >= 0
          ? log_vec.push_back(std::log(det_val))
          : log_vec.push_back(std::log(std::abs(
                det_val)));  // for computing log value of a negative value.
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
    if (input_dim.size() == static_cast<size_t>(2)) {
      // when input is a two-dimension matrix, The det value is a number.
      output_dim_vec = {1};
    }
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
    auto& dev_ctx = context.template device_context<DeviceContext>();
    const auto* input = context.Input<framework::Tensor>("Input");
    const auto* slogdet = context.Input<framework::Tensor>("Out");
    const auto* grad =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dslogdet =
        context.Output<framework::Tensor>(framework::GradVarName("Input"));

    PADDLE_ENFORCE_EQ(grad->dims()[0], 2,
                      platform::errors::InvalidArgument(
                          "The grad tensor of SlogDet should contain two"
                          " grad: sign and absslogdet, but here %ld.",
                          grad->dims()[0]));
    if (input->dims().size() > 2) {
      PADDLE_ENFORCE_EQ(
          grad->dims().size() + 1, input->dims().size(),
          platform::errors::InvalidArgument(
              "The grad tensor of slogdet dims size should 1 less than"
              " input tensor's, but here differ %d",
              input->dims().size() - grad->dims().size()));
    }

    // Check Whether the matrix is invertible
    // (matrix A not invertible) == (absslogdet(A)=0)
    auto slogdet_vec = slogdet->Split(1, 0);
    auto absslogdet_val = slogdet_vec[0];
    if (!CheckMatrixInvertible<DeviceContext, T>(context, &absslogdet_val)) {
      // The matrix is not invertible
      VLOG(3) << "The input matrix not invertible!";
      dslogdet->Resize(input->dims());
      dslogdet->mutable_data<T>(context.GetPlace());
      pten::funcs::SetConstant<DeviceContext, T> zero;
      zero(dev_ctx, dslogdet, std::numeric_limits<T>::quiet_NaN());
      return;
    }

    // The matrix is invertible
    // let sl|A| = SlogDeterminant(A)
    // Ref to https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
    // we set dsl|A| = unsqueeze(dslA, [-1, -2]) *
    // inverse(A).conj().transpose(-2, -1)

    math::DeviceIndependenceTensorOperations<DeviceContext, T> helper(context);

    // First: inverse(A)
    framework::Tensor inverse_A;
    // A must be square matrices!
    inverse_A.Resize(input->dims());
    inverse_A.mutable_data<T>(context.GetPlace());

    math::MatrixInverseFunctor<DeviceContext, T> mat_inv;
    mat_inv(dev_ctx, *input, &inverse_A);

    VLOG(3) << "inverse(A) dims: " << inverse_A.dims();

    // Second: inverse(A).conj()
    framework::Tensor conj_inverse_A;
    conj_inverse_A.Resize(inverse_A.dims());
    auto numel = input->numel();
    auto* conj_data = conj_inverse_A.mutable_data<T>(context.GetPlace(),
                                                     size_t(numel * sizeof(T)));

    platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
    math::ConjFunctor<T> functor(inverse_A.data<T>(), numel, conj_data);
    for_range(functor);

    VLOG(3) << "inverse(A).conj() dims: " << conj_inverse_A.dims();

    // Third: inverse(A).conj().transpose(-2, -1)
    framework::Tensor transpose_inverse_A = helper.Transpose(conj_inverse_A);
    VLOG(3) << "inverse(A).conj().transpose(-2, -1) dims: "
            << transpose_inverse_A.dims();

    // Fourth: split grad value to [sign_grad, absslogdet_grad]
    auto grad_vec = grad->Split(1, 0);
    auto det_grad = grad_vec[1];

    // remmove useless first dimension
    int det_grad_size = det_grad.dims().size();
    std::vector<int> det_grad_vec;
    for (int i = 1; i < det_grad_size; ++i) {
      det_grad_vec.emplace_back(det_grad.dims()[i]);
    }
    det_grad.Resize(det_grad.dims().reshape(det_grad_vec));

    // Fifth: unsqueeze(dslA, [-1, -2])
    auto unsqueeze1 = helper.Unsqueeze(det_grad, -1);
    auto unsqueeze2 = helper.Unsqueeze(unsqueeze1, -2);
    VLOG(3) << "unsqueezed(dslA, [-1, -2]) dims: " << unsqueeze2.dims();

    // Finally: unsqueeze(dslA) * inverse(A)
    auto res = helper.Mul(unsqueeze2, transpose_inverse_A);
    VLOG(3) << "unsqueeze(dslA) * inverse(A) dims: " << res.dims();

    framework::TensorCopy(res, context.GetPlace(), dslogdet);
    dslogdet->Resize(input->dims());
    VLOG(3) << "dsl|A| dims: " << dslogdet->dims();
  }
};

}  // namespace operators
}  // namespace paddle
