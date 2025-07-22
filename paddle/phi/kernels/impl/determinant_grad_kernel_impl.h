// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "glog/logging.h"
#include "paddle/phi/common/amp_type_traits.h"

#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/determinant_grad_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/matrix_inverse.h"
#include "paddle/phi/kernels/funcs/unsqueeze.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {
namespace detail {

// epsilon_ should be smaller if linalg.det achieves higher precision
template <typename T>
struct FoundZeroEpsilon {
  // default for float16
  static constexpr T value() { return static_cast<T>(1e-3f); }
};

template <>
struct FoundZeroEpsilon<float> {
  static constexpr float value() { return 1e-5f; }
};

template <>
struct FoundZeroEpsilon<double> {
  static constexpr double value() { return 1e-12; }
};

template <typename T>
struct FoundZeroFunctor {
  using RealType = phi::dtype::Real<T>;

  FoundZeroFunctor(const T* x, int64_t numel, bool* res)
      : x_(x), numel_(numel), res_(res) {}

  HOSTDEVICE void operator()(size_t idx) const {
    if (*res_ || idx >= static_cast<size_t>(numel_)) {
      // found a singular matrix
      return;
    }
    if (abs(x_[idx]) < FoundZeroEpsilon<RealType>::value()) {
      *res_ = true;
    }
  }

 private:
  const T* x_;
  int64_t numel_;
  bool* res_;
};

template <typename T, typename Context>
inline bool CheckMatrixInvertible(const Context& dev_ctx,
                                  const DenseTensor* det) {
  auto numel = det->numel();

  DenseTensor dev_tensor = phi::Empty<bool, Context>(dev_ctx, {1});

  // set false
  phi::funcs::SetConstant<Context, bool> zero;
  zero(dev_ctx, &dev_tensor, false);

  // find whether zero
  phi::funcs::ForRange<Context> for_range(dev_ctx, numel);
  FoundZeroFunctor<T> functor(det->data<T>(), numel, dev_tensor.data<bool>());
  for_range(functor);

  // copy to host
  DenseTensor cpu_tensor;
  phi::Copy<Context>(dev_ctx, dev_tensor, phi::CPUPlace(), false, &cpu_tensor);

  // if founded zero, the matrix is not invertible
  // else the matrix is invertible
  auto* res = cpu_tensor.data<bool>();
  return !(*res);
}

}  // namespace detail

template <typename T, typename Context>
void DeterminantGradKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& out,
                           const DenseTensor& out_grad,
                           DenseTensor* x_grad) {
  if (x_grad && x_grad->numel() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    return;
  }
  auto input_dims_size = x.dims().size();
  if (input_dims_size > 2) {
    PADDLE_ENFORCE_EQ(
        out_grad.dims().size() + 2,
        input_dims_size,
        common::errors::InvalidArgument(
            "The grad tensor of det dims size should be 2 less than"
            " input tensor's, but here differ %d",
            input_dims_size - out_grad.dims().size()));
  } else if (input_dims_size == 2) {
    // input dims size 2 and grad dims size 0 is possible
    PADDLE_ENFORCE_EQ(
        out_grad.dims().size(),
        0,
        common::errors::InvalidArgument(
            "The grad tensor of det dims size should be 2 less than"
            " input tensor's, but here differ %d",
            input_dims_size - out_grad.dims().size()));
  } else {
    // checked in forward, pass
  }

  // Check Whether the matrix is invertible
  // (matrix A not invertible) == (det(A)=0)
  if (!detail::CheckMatrixInvertible<T, Context>(dev_ctx, &out)) {
    // The matrix is not invertible
    VLOG(3) << "The input matrix not invertible!";
    x_grad->Resize(x.dims());
    phi::Full<T>(
        dev_ctx, common::vectorize(x.dims()), static_cast<T>(0.0f), x_grad);
    return;
  }

  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // The matrix is invertible
  // let |A| = Determinant(A)
  // Ref to https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
  // we set d|A| = unsqueeze(dA * |A|.conj(), [-1, -2]) *
  // inverse(A).conj().transpose(-2, -1)

  // First: inverse(A)
  DenseTensor transpose_inverse_A;
  {
    DenseTensor inverse_A;
    // A must be square matrices!
    inverse_A.Resize(x.dims());
    dev_ctx.template Alloc<MPType>(&inverse_A);

    phi::funcs::MatrixInverseFunctor<Context, MPType> mat_inv;
    if constexpr (!std::is_same_v<MPType, T>) {
      auto x_mp = phi::Cast<T, Context>(
          dev_ctx, x, phi::CppTypeToDataType<MPType>::Type());
      mat_inv(dev_ctx, x_mp, &inverse_A);
    } else {
      mat_inv(dev_ctx, x, &inverse_A);
    }

    auto conj_inverse_A = phi::Conj<MPType>(dev_ctx, inverse_A);
    VLOG(3) << "inverse(A).conj() dims: " << conj_inverse_A.dims();

    // Second: inverse(A).conj().transpose(-2, -1)
    transpose_inverse_A =
        phi::TransposeLast2Dim<MPType>(dev_ctx, conj_inverse_A);
    VLOG(3) << "(dA * |A|).transpose(-2, -1) dims: "
            << transpose_inverse_A.dims();
  }

  DenseTensor mul_unsqueezed;
  {
    DenseTensor mul_dA_detA;
    // Third: dA * |A|.conj()
    if constexpr (!std::is_same_v<MPType, T>) {
      auto out_mp = phi::Cast<T, Context>(
          dev_ctx, out, phi::CppTypeToDataType<MPType>::Type());
      auto out_grad_mp = phi::Cast<T, Context>(
          dev_ctx, out_grad, phi::CppTypeToDataType<MPType>::Type());

      auto conj_out_mp = phi::Conj<MPType>(dev_ctx, out_mp);
      mul_dA_detA = phi::Multiply<MPType>(dev_ctx, out_grad_mp, conj_out_mp);
    } else {
      auto conj_out = phi::Conj<T>(dev_ctx, out);
      mul_dA_detA = phi::Multiply<T>(dev_ctx, out_grad, conj_out);
    }
    VLOG(3) << "dA * |A| dims: " << mul_dA_detA.dims();

    // Fourth: unsqueeze(dA * |A|, [-1, -2])
    auto unsqueeze1 = phi::funcs::Unsqueeze(mul_dA_detA, -1);
    mul_unsqueezed = phi::funcs::Unsqueeze(unsqueeze1, -2);
    VLOG(3) << "unsqueezed(dA * |A|) dims: " << mul_unsqueezed.dims();
  }

  // Finally: unsqueeze(dA * |A|) * inverse(A)
  auto res_mp =
      phi::Multiply<MPType>(dev_ctx, mul_unsqueezed, transpose_inverse_A);
  VLOG(3) << "unsqueeze(dA * |A|) * inverse(A) dims: " << res_mp.dims();

  x_grad->Resize(x.dims());
  VLOG(3) << "d|A| dims: " << x_grad->dims();

  phi::Copy(dev_ctx, res_mp, dev_ctx.GetPlace(), false, x_grad);
}

}  // namespace phi
