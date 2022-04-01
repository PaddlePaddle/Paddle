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
#include "paddle/fluid/platform/for_range.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/diag_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/matrix_inverse.h"
#include "paddle/phi/kernels/funcs/unsqueeze.h"
#include "paddle/phi/kernels/impl/determinant_grad_kernel_impl.h"
#include "paddle/phi/kernels/impl/determinant_kernel_impl.h"
#include "paddle/phi/kernels/matmul_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T>
T sign(T val) {
  return static_cast<T>(T(0) < val) - (val < T(0));
}

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
      typename phi::detail::EigenMatrix<T>::MatrixType matrix(rank, rank);
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

    auto batch_count = phi::detail::GetBatchCount(input->dims());
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
    auto output_dims = phi::make_ddim(output_dim_vec);
    output->Resize(output_dims);
    VLOG(2) << "output dim:" << output->dims();
  }
};

template <typename DeviceContext, typename T>
class SlogDeterminantGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& orig_dev_ctx = context.template device_context<DeviceContext>();
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

    auto& dev_ctx = static_cast<
        const typename framework::ConvertToPhiContext<DeviceContext>::TYPE&>(
        orig_dev_ctx);

    // Check Whether the matrix is invertible
    // (matrix A not invertible) == (absslogdet(A)=0)
    auto slogdet_vec = slogdet->Split(1, 0);
    auto absslogdet_val = slogdet_vec[0];
    if (!phi::detail::CheckMatrixInvertible<
            T, typename framework::ConvertToPhiContext<DeviceContext>::TYPE>(
            dev_ctx, &absslogdet_val)) {
      // The matrix is not invertible
      VLOG(3) << "The input matrix not invertible!";
      dslogdet->Resize(input->dims());
      phi::Full<T>(dev_ctx, phi::vectorize(input->dims()),
                   std::numeric_limits<T>::quiet_NaN(), dslogdet);
      return;
    }

    // The matrix is invertible
    // let sl|A| = SlogDeterminant(A)
    // Ref to https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
    // we set dsl|A| = unsqueeze(dslA, [-1, -2]) *
    // inverse(A).conj().transpose(-2, -1)

    // First: inverse(A)
    framework::Tensor inverse_A;
    // A must be square matrices!
    inverse_A.Resize(input->dims());
    inverse_A.mutable_data<T>(context.GetPlace());

    phi::funcs::MatrixInverseFunctor<DeviceContext, T> mat_inv;
    mat_inv(orig_dev_ctx, *input, &inverse_A);

    VLOG(3) << "inverse(A) dims: " << inverse_A.dims();

    // Second: inverse(A).conj()
    auto conj_inverse_A = phi::Conj<T>(dev_ctx, inverse_A);

    VLOG(3) << "inverse(A).conj() dims: " << conj_inverse_A.dims();

    // Third: inverse(A).conj().transpose(-2, -1)
    framework::Tensor transpose_inverse_A =
        phi::TransposeLast2Dim<T>(dev_ctx, conj_inverse_A);
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
    auto unsqueeze1 = phi::funcs::Unsqueeze(det_grad, -1);
    auto unsqueeze2 = phi::funcs::Unsqueeze(unsqueeze1, -2);
    VLOG(3) << "unsqueezed(dslA, [-1, -2]) dims: " << unsqueeze2.dims();

    // Finally: unsqueeze(dslA) * inverse(A)
    auto res = phi::Multiply<T>(dev_ctx, unsqueeze2, transpose_inverse_A);
    VLOG(3) << "unsqueeze(dslA) * inverse(A) dims: " << res.dims();

    framework::TensorCopy(res, context.GetPlace(), dslogdet);
    dslogdet->Resize(input->dims());
    VLOG(3) << "dsl|A| dims: " << dslogdet->dims();
  }
};

}  // namespace operators
}  // namespace paddle
