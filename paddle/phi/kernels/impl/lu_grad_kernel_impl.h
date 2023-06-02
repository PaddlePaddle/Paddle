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

#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/triangular_solve_kernel.h"

#include "paddle/phi/kernels/impl/lu_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void LUGradKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& out,
                  const DenseTensor& pivots,
                  const DenseTensor& out_grad,
                  bool pivot UNUSED,
                  DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);

  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);

  auto xdims = x.dims();
  int xrank = xdims.size();
  int64_t m = xdims[xrank - 2];
  int64_t n = xdims[xrank - 1];
  int64_t k = std::min(m, n);

  DenseTensor L, U, L_narrow, U_narrow, L_narrow_mH, U_narrow_mH, grad_narrow;
  LU_Unpack<Context, T>(dev_ctx, &out, &L, &U);

  Tensor_narrow<Context, T>(dev_ctx, &L, &L_narrow, 0, k, 0, k);
  Tensor_narrow<Context, T>(dev_ctx, &U, &U_narrow, 0, k, 0, k);
  Tensor_narrow<Context, T>(dev_ctx, &out_grad, &grad_narrow, 0, k, 0, k);
  auto graddims = grad_narrow.dims();

  Tensor_Conj<Context, T>(dev_ctx, L_narrow, &L_narrow_mH);
  Tensor_Conj<Context, T>(dev_ctx, U_narrow, &U_narrow_mH);
  L_narrow_mH = Transpose2DTo6D<Context, T>(dev_ctx, L_narrow_mH);
  U_narrow_mH = Transpose2DTo6D<Context, T>(dev_ctx, U_narrow_mH);

  auto LmHdims = L_narrow_mH.dims();
  auto UmHdims = U_narrow_mH.dims();

  DenseTensor phi_L, phi_U, phi, psi;
  phi_L.Resize(LmHdims);
  dev_ctx.template Alloc<T>(&phi_L);
  phi_U.Resize(UmHdims);
  dev_ctx.template Alloc<T>(&phi_U);
  auto mat_dim_l = phi::funcs::CreateMatrixDescriptor(LmHdims, 0, false);
  auto mat_dim_u = phi::funcs::CreateMatrixDescriptor(UmHdims, 0, false);
  auto mat_dim_g = phi::funcs::CreateMatrixDescriptor(graddims, 0, false);
  blas.MatMul(L_narrow_mH,
              mat_dim_l,
              grad_narrow,
              mat_dim_g,
              static_cast<T>(1),
              &phi_L,
              static_cast<T>(0));

  blas.MatMul(grad_narrow,
              mat_dim_g,
              U_narrow_mH,
              mat_dim_u,
              static_cast<T>(1),
              &phi_U,
              static_cast<T>(0));

  auto phil_rank = LmHdims.size();
  auto phiu_rank = UmHdims.size();
  phi::funcs::ForRange<Context> l_for_range(dev_ctx, phi_L.numel());
  phi::funcs::TrilTriuCompute<T> tril_computer(phi_L.data<T>(),
                                               -1,
                                               true,
                                               LmHdims[phil_rank - 2],
                                               LmHdims[phil_rank - 1],
                                               phi_L.data<T>());
  l_for_range(tril_computer);

  phi::funcs::ForRange<Context> u_for_range(dev_ctx, phi_U.numel());
  phi::funcs::TrilTriuCompute<T> triu_computer(phi_U.data<T>(),
                                               0,
                                               false,
                                               UmHdims[phiu_rank - 2],
                                               UmHdims[phiu_rank - 1],
                                               phi_U.data<T>());
  u_for_range(triu_computer);

  Tensor_Add<Context, T>(dev_ctx, phi_L, phi_U, &phi);
  psi.Resize(xdims);
  dev_ctx.template Alloc<T>(&psi);
  phi::funcs::SetConstant<Context, T> setter;
  setter(dev_ctx, &psi, static_cast<T>(0));

  std::vector<int64_t> axes = {xrank - 2, xrank - 1};
  std::vector<int64_t> slice_starts(2, 0);
  std::vector<int64_t> slice_ends(2, 0);
  auto valuedims = vectorize(xdims);

  DenseTensor Pmat;
  Unpack_Pivot<Context, T>(dev_ctx, pivots, &Pmat, m, k);

  if (m <= n) {
    if (k < n) {
      DenseTensor U_complement, U_grad_complement, phi_complement,
          phi_complement_l;
      Tensor_narrow<Context, T>(dev_ctx, &U, &U_complement, 0, k, k, n);
      Tensor_narrow<Context, T>(
          dev_ctx, &out_grad, &U_grad_complement, 0, k, k, n);
      DenseTensor U_complement_mH =
          Transpose2DTo6D<Context, T>(dev_ctx, U_complement);

      Tensor_Conj<Context, T>(dev_ctx, U_complement_mH, &U_complement_mH);

      auto mat_dim_g = phi::funcs::CreateMatrixDescriptor(
          U_grad_complement.dims(), 0, false);
      auto mat_dim_u =
          phi::funcs::CreateMatrixDescriptor(U_complement_mH.dims(), 0, false);
      auto phidims = UmHdims;
      phidims[UmHdims.size() - 2] = k;
      phidims[UmHdims.size() - 1] = k;
      phi_complement.Resize(phidims);
      dev_ctx.template Alloc<T>(&phi_complement);
      blas.MatMul(U_grad_complement,
                  mat_dim_g,
                  U_complement_mH,
                  mat_dim_u,
                  static_cast<T>(1),
                  &phi_complement,
                  static_cast<T>(0));

      phi_complement_l.Resize(phidims);
      dev_ctx.template Alloc<T>(&phi_complement_l);
      const auto H = phidims[phidims.size() - 2];
      const auto W = phidims[phidims.size() - 1];
      phi::funcs::ForRange<Context> x_for_range(dev_ctx,
                                                phi_complement.numel());
      phi::funcs::TrilTriuCompute<T> tril_computer(
          phi_complement.data<T>(), -1, true, H, W, phi_complement_l.data<T>());
      x_for_range(tril_computer);

      Tensor_Sub<Context, T>(dev_ctx, phi, phi_complement_l, &phi);

      slice_starts[0] = 0;
      slice_starts[1] = k;
      slice_ends[0] = k;
      slice_ends[1] = n;
      valuedims[xrank - 2] = k;
      valuedims[xrank - 1] = n - k;
      SetValueCompute_dispatch<Context, T>(dev_ctx,
                                           &psi,
                                           &U_grad_complement,
                                           &psi,
                                           axes,
                                           &slice_starts,
                                           &slice_ends,
                                           valuedims,
                                           xrank);
    }

    DenseTensor psi_principal, phi_mH, psi_tmp;
    Tensor_Conj<Context, T>(dev_ctx, phi, &phi_mH);
    phi_mH = Transpose2DTo6D<Context, T>(dev_ctx, phi_mH);

    phi::TriangularSolveKernel<T, Context>(
        dev_ctx, U_narrow, phi_mH, true, false, false, &psi_principal);

    Tensor_Conj<Context, T>(dev_ctx, psi_principal, &psi_principal);
    psi_principal = Transpose2DTo6D<Context, T>(dev_ctx, psi_principal);
    slice_starts[0] = 0;
    slice_starts[1] = 0;
    slice_ends[0] = k;
    slice_ends[1] = k;
    valuedims[xrank - 2] = k;
    valuedims[xrank - 1] = k;

    SetValueCompute_dispatch<Context, T>(dev_ctx,
                                         &psi,
                                         &psi_principal,
                                         &psi,
                                         axes,
                                         &slice_starts,
                                         &slice_ends,
                                         valuedims,
                                         xrank);

    phi::TriangularSolveKernel<T, Context>(
        dev_ctx, L_narrow_mH, psi, true, false, true, &psi_tmp);

    auto mat_dim_p = phi::funcs::CreateMatrixDescriptor(Pmat.dims(), 0, false);
    auto mat_dim_b =
        phi::funcs::CreateMatrixDescriptor(psi_tmp.dims(), 0, false);
    blas.MatMul(Pmat,
                mat_dim_p,
                psi_tmp,
                mat_dim_b,
                static_cast<T>(1),
                x_grad,
                static_cast<T>(0));
  } else {
    DenseTensor L_complement, L_grad_complement, phi_complement,
        phi_complement_u;
    Tensor_narrow<Context, T>(dev_ctx, &L, &L_complement, k, m, 0, k);
    Tensor_narrow<Context, T>(
        dev_ctx, &out_grad, &L_grad_complement, k, m, 0, k);
    DenseTensor L_complement_mH =
        Transpose2DTo6D<Context, T>(dev_ctx, L_complement);
    Tensor_Conj<Context, T>(dev_ctx, L_complement_mH, &L_complement_mH);

    auto mat_dim_g =
        phi::funcs::CreateMatrixDescriptor(L_grad_complement.dims(), 0, false);
    auto mat_dim_u =
        phi::funcs::CreateMatrixDescriptor(L_complement_mH.dims(), 0, false);
    auto phidims = LmHdims;
    phidims[LmHdims.size() - 2] = k;
    phidims[LmHdims.size() - 1] = k;
    phi_complement.Resize(phidims);
    dev_ctx.template Alloc<T>(&phi_complement);
    blas.MatMul(L_complement_mH,
                mat_dim_u,
                L_grad_complement,
                mat_dim_g,
                static_cast<T>(1),
                &phi_complement,
                static_cast<T>(0));

    phi_complement_u.Resize(phidims);
    dev_ctx.template Alloc<T>(&phi_complement_u);
    const auto H = phidims[phidims.size() - 2];
    const auto W = phidims[phidims.size() - 1];
    phi::funcs::ForRange<Context> x_for_range(dev_ctx, phi_complement.numel());
    phi::funcs::TrilTriuCompute<T> triu_computer(
        phi_complement.data<T>(), 0, false, H, W, phi_complement_u.data<T>());
    x_for_range(triu_computer);

    Tensor_Sub<Context, T>(dev_ctx, phi, phi_complement_u, &phi);

    slice_starts[0] = k;
    slice_starts[1] = 0;
    slice_ends[0] = m;
    slice_ends[1] = k;
    valuedims[xrank - 2] = m - k;
    valuedims[xrank - 1] = k;
    SetValueCompute_dispatch<Context, T>(dev_ctx,
                                         &psi,
                                         &L_grad_complement,
                                         &psi,
                                         axes,
                                         &slice_starts,
                                         &slice_ends,
                                         valuedims,
                                         xrank);
    DenseTensor psi_principal, phi_mH, psi_tmp, U_narrow_mH;

    phi::TriangularSolveKernel<T, Context>(
        dev_ctx, L_narrow_mH, phi, true, false, true, &psi_principal);

    slice_starts[0] = 0;
    slice_starts[1] = 0;
    slice_ends[0] = k;
    slice_ends[1] = k;
    valuedims[xrank - 2] = k;
    valuedims[xrank - 1] = k;

    SetValueCompute_dispatch<Context, T>(dev_ctx,
                                         &psi,
                                         &psi_principal,
                                         &psi,
                                         axes,
                                         &slice_starts,
                                         &slice_ends,
                                         valuedims,
                                         xrank);

    psi_tmp.Resize(psi.dims());
    dev_ctx.template Alloc<T>(&psi_tmp);
    auto mat_dim_p = phi::funcs::CreateMatrixDescriptor(Pmat.dims(), 0, false);
    auto mat_dim_b = phi::funcs::CreateMatrixDescriptor(psi.dims(), 0, false);
    blas.MatMul(Pmat,
                mat_dim_p,
                psi,
                mat_dim_b,
                static_cast<T>(1),
                &psi_tmp,
                static_cast<T>(0));
    psi_tmp = Transpose2DTo6D<Context, T>(dev_ctx, psi_tmp);

    Tensor_Conj<Context, T>(dev_ctx, U_narrow, &U_narrow_mH);
    phi::TriangularSolveKernel<T, Context>(
        dev_ctx, U_narrow_mH, psi_tmp, true, false, false, &psi);
    *x_grad = Transpose2DTo6D<Context, T>(dev_ctx, psi);
  }
}

}  // namespace phi
