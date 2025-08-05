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

#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/determinant_grad_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/lu_kernel.h"
#include "paddle/phi/kernels/lu_solve_kernel.h"

namespace phi {
namespace detail {

template <typename T>
struct DetZeroEpsilon {
  // default for float16
  static constexpr T value() { return static_cast<T>(1e-3f); }
};

template <>
struct DetZeroEpsilon<float> {
  static constexpr float value() { return 1e-5f; }
};

template <>
struct DetZeroEpsilon<double> {
  static constexpr double value() { return 1e-12; }
};

template <typename T>
struct PerturbEpsilon {
  // default for float16
  static constexpr T value() { return static_cast<T>(1e-3f); }
};

template <>
struct PerturbEpsilon<float> {
  static constexpr float value() { return 1e-5f; }
};

template <>
struct PerturbEpsilon<double> {
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
    if (abs(x_[idx]) < DetZeroEpsilon<RealType>::value()) {
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

template <typename T>
struct FusedPerturbAndDetFunctor {
  using RealType = phi::dtype::Real<T>;

  FusedPerturbAndDetFunctor(int64_t m,
                            const int* infos,
                            const T* grad_data,
                            T* lu_data,
                            const int* pivots_data,
                            T* out_k_data)
      : m_(m),
        infos_(infos),
        grad_data_(grad_data),
        lu_data_(lu_data),
        pivots_data_(pivots_data),
        out_k_data_(out_k_data) {}

  HOSTDEVICE void operator()(int64_t batch_id) const {
    const int64_t batch_offset = batch_id * m_ * m_;
    T* lu_batch = lu_data_ + batch_offset;

    // infos[i] != 0 indicates a singular matrix.
    if (infos_[batch_id] != 0) {
      constexpr RealType perturb_epsilon =
          detail::PerturbEpsilon<RealType>::value();
      for (int64_t i = 0; i < m_; ++i) {
        T& diag_val = lu_batch[i * m_ + i];
        constexpr RealType zero_epsilon =
            detail::DetZeroEpsilon<RealType>::value();
        if (std::abs(static_cast<RealType>(diag_val)) < zero_epsilon) {
          diag_val += static_cast<T>(perturb_epsilon);
        }
      }
    }

    const int* pivots_batch = pivots_data_ + batch_id * m_;

    T det_val = static_cast<T>(1);
    int64_t swaps = 0;
    for (int64_t i = 0; i < m_; ++i) {
      det_val *= lu_batch[i * m_ + i];
      if (pivots_batch[i] != (i + 1)) {
        swaps++;
      }
    }
    if (swaps % 2 != 0) {
      det_val = -det_val;
    }

    // k = grad_out * conj(det(A_perturbed))
    if constexpr (std::is_same_v<T, phi::dtype::complex<float>> ||
                  std::is_same_v<T, phi::dtype::complex<double>>) {
      out_k_data_[batch_id] = grad_data_[batch_id] * phi::dtype::conj(det_val);
    } else {
      out_k_data_[batch_id] = grad_data_[batch_id] * det_val;
    }
  }

  int64_t m_;
  const int* infos_;
  const T* grad_data_;
  T* lu_data_;
  const int* pivots_data_;
  T* out_k_data_;
};

template <typename Context, typename T>
struct MatrixIdentityFunctor {
  MatrixIdentityFunctor(int64_t m, T* output) : m_(m), output_(output) {}

  HOSTDEVICE void operator()(int64_t index) const {
    const int64_t col = index % m_;
    const int64_t global_row = index / m_;
    const int64_t row = global_row % m_;
    output_[index] = (row == col) ? static_cast<T>(1) : static_cast<T>(0);
  }

  const int64_t m_;
  T* output_;
};

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

  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  auto x_mp = x;
  auto out_grad_mp = out_grad;

  if constexpr (!std::is_same_v<MPType, T>) {
    x_mp = phi::Cast<T, Context>(
        dev_ctx, x, phi::CppTypeToDataType<MPType>::Type());
    out_grad_mp = phi::Cast<T, Context>(
        dev_ctx, out_grad, phi::CppTypeToDataType<MPType>::Type());
  }

  const auto& x_dims = x.dims();
  const int64_t m = x_dims[x_dims.size() - 1];
  const int64_t batch_count = x.numel() / (m * m);

  // LU decomposition
  DenseTensor lu_data, pivots, infos;
  lu_data.Resize(x_dims);
  pivots.Resize(common::slice_ddim(x_dims, 0, x_dims.size() - 1));
  infos.Resize({batch_count});
  LUKernel<MPType, Context>(
      dev_ctx, x_mp, /*pivot=*/true, &lu_data, &pivots, &infos);

  // perturb LU to avoid singularity
  DenseTensor k;
  k.Resize({batch_count});
  dev_ctx.template Alloc<MPType>(&k);

  funcs::ForRange<Context> for_range_batch(dev_ctx, batch_count);
  detail::FusedPerturbAndDetFunctor<MPType> fused_functor(
      m,
      infos.data<int>(),
      out_grad_mp.data<MPType>(),
      lu_data.data<MPType>(),
      pivots.data<int>(),
      k.data<MPType>());
  for_range_batch(fused_functor);

  std::vector<int64_t> k_dims = common::vectorize(x_dims);
  k_dims[k_dims.size() - 1] = 1;
  k_dims[k_dims.size() - 2] = 1;
  k.Resize(common::make_ddim(k_dims));

  DenseTensor I;
  I.Resize(x_dims);
  dev_ctx.template Alloc<MPType>(&I);
  funcs::ForRange<Context> for_range(dev_ctx, I.numel());
  detail::MatrixIdentityFunctor<Context, MPType> identity_functor(
      m, I.data<MPType>());
  for_range(identity_functor);

  DenseTensor a_inv_h;
  a_inv_h.Resize(x_dims);
  LuSolveKernel<MPType, Context>(
      dev_ctx, I, lu_data, pivots, /*trans=*/"C", &a_inv_h);

  auto grad_mp = phi::Multiply<MPType, Context>(dev_ctx, a_inv_h, k);

  x_grad->Resize(x_dims);
  phi::Copy(dev_ctx, grad_mp, dev_ctx.GetPlace(), false, x_grad);
}

}  // namespace phi
