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

/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace phi {

template <typename Context, typename T>
inline DenseTensor MatMul(const Context& ctx,
                          const DenseTensor& matrix_a,
                          const DenseTensor& matrix_b,
                          const phi::DDim& a_dim,
                          const phi::DDim& b_dim) {
  auto blas = phi::funcs::GetBlas<Context, T>(ctx);

  DenseTensor matrix_c;
  phi::DDim c_dim = phi::make_ddim({a_dim[0], b_dim[1]});
  matrix_c.Resize(c_dim);
  ctx.template Alloc<T>(&matrix_c);

  auto mat_dim_a = phi::funcs::CreateMatrixDescriptor(a_dim, 0, false);
  auto mat_dim_b = phi::funcs::CreateMatrixDescriptor(b_dim, 0, false);
  const T alpha = static_cast<T>(1.0);
  blas.MatMul(matrix_a.data<T>(),
              mat_dim_a,
              matrix_b.data<T>(),
              mat_dim_b,
              alpha,
              matrix_c.data<T>(),
              T(0));
  return matrix_c;
}

/**
 * @brief Recursively calculate matrix multiplication according to the optimal
 * order
 * Let k = order[i,j], then ins[i...j] = ins[i...k] * ins[k+1 ...j]
 *
 * @param
 * ins: the input tensors
 * ins_dims: the shape of ins after reshape
 * order: the optimal order
 * i: the left of sub chain
 * j: the righe of sub chain
 * save_result: set true by backward
 * results: save the intermediate result during backward
 */
template <typename Context, typename T>
inline DenseTensor MatChainMul(const Context& ctx,
                               const std::vector<const DenseTensor*>& ins,
                               const std::vector<phi::DDim>& ins_dims,
                               const std::vector<uint64_t>& order,
                               const uint64_t i,
                               const uint64_t j,
                               const bool save_result,
                               std::vector<DenseTensor>* results) {
  if (i == j) {
    return *ins[i];
  }

  const auto A = MatChainMul<Context, T>(ctx,
                                         ins,
                                         ins_dims,
                                         order,
                                         i,
                                         order[i * ins.size() + j],
                                         save_result,
                                         results);
  phi::DDim a_dim = A.dims();
  if (i == order[i * ins.size() + j]) {
    a_dim = ins_dims[i];
  }

  const auto B = MatChainMul<Context, T>(ctx,
                                         ins,
                                         ins_dims,
                                         order,
                                         order[i * ins.size() + j] + 1,
                                         j,
                                         save_result,
                                         results);
  phi::DDim b_dim = B.dims();
  if (j == order[i * ins.size() + j] + 1) {
    b_dim = ins_dims[j];
  }

  auto result = MatMul<Context, T>(ctx, A, B, a_dim, b_dim);
  if (save_result) {
    (*results)[i * ins.size() + j] = result;
  }
  return result;
}

/**
 * @brief get the optimal order
 */
template <typename Context, typename T>
std::vector<uint64_t> GetOrder(const std::vector<const DenseTensor*>& ins,
                               const std::vector<phi::DDim>& ins_dims) {
  auto n = ins.size();
  // p: save the ins shape, the ins[i] shape is (p[i], p[i+1])
  std::vector<uint64_t> p(n + 1);
  for (uint64_t i = 0; i < n; i++) {
    p[i] = ins_dims[i][0];
  }
  p[n] = ins_dims[n - 1][1];

  // m[i, j]: save the lowest cost for multiplying ins[i...j]
  std::vector<uint64_t> m(n * n, 0);
  // define ins[i...j] means multiplying matrices from ins[i] to ins[j]
  // order[i, j] = k, this means that ins[i...k] and ins[k...j] fist and then
  // multiply the resulting matrices is the optimal order for ins[i...j]
  std::vector<uint64_t> order(n * n);
  for (uint64_t l = 1; l < n; l++) {
    for (uint64_t i = 0; i < n - l; i++) {
      auto j = i + l;
      m[i * n + j] = 0xffffffff;
      for (uint64_t k = i; k < j; k++) {
        uint64_t q =
            m[i * n + k] + m[(k + 1) * n + j] + p[i] * p[k + 1] * p[j + 1];
        if (q < m[i * n + j]) {
          m[i * n + j] = q;
          order[i * n + j] = k;
        }
      }
    }
  }
  return order;
}

template <typename Context, typename T>
static inline DenseTensor MultiDotMatChainOrder(
    const Context& ctx,
    const std::vector<const DenseTensor*>& ins,
    const std::vector<phi::DDim>& ins_dims,
    const bool save_result,
    std::vector<DenseTensor>* results) {
  auto order = GetOrder<Context, T>(ins, ins_dims);
  return MatChainMul<Context, T>(
      ctx, ins, ins_dims, order, 0, ins.size() - 1, save_result, results);
}

template <typename Context, typename T>
inline void GetDims(const std::vector<const DenseTensor*>& ins,
                    std::vector<phi::DDim>* ins_dims) {
  const auto n = ins.size();
  for (size_t i = 0; i < n; i++) {
    (*ins_dims)[i] = ins[i]->dims();
    if (i == 0 && (*ins_dims)[i].size() == 1) {
      (*ins_dims)[i] = phi::make_ddim({1, (*ins_dims)[i][0]});
    } else if (i == n - 1 && (*ins_dims)[i].size() == 1) {
      (*ins_dims)[i] = phi::make_ddim({(*ins_dims)[i][0], 1});
    }
  }
}

template <typename T, typename Context>
void MultiDotKernel(const Context& ctx,
                    const std::vector<const DenseTensor*>& x,
                    DenseTensor* out) {
  auto ins = x;
  ctx.template Alloc<T>(out);

  auto blas = phi::funcs::GetBlas<Context, T>(ctx);

  auto n = ins.size();
  std::vector<phi::DDim> ins_dims(n);
  GetDims<Context, T>(ins, &ins_dims);

  const T scale = static_cast<T>(1.0);
  if (n == 2) {
    auto mat_dim_a = phi::funcs::CreateMatrixDescriptor(ins_dims[0], 0, false);
    auto mat_dim_b = phi::funcs::CreateMatrixDescriptor(ins_dims[1], 0, false);
    blas.MatMul(*ins[0], mat_dim_a, *ins[1], mat_dim_b, scale, out, T(0));
  } else if (n == 3) {
    const auto Ma = ins_dims[0][0];
    const auto Ka = ins_dims[0][1];
    const auto Nb = ins_dims[1][1];
    const auto Nc = ins_dims[2][1];
    const uint64_t cost1 = Ma * Nb * (Ka + Nc);
    const uint64_t cost2 = Ka * Nc * (Nb + Ma);
    auto mat_dim_a = phi::funcs::CreateMatrixDescriptor(ins_dims[0], 0, false);
    auto mat_dim_b = phi::funcs::CreateMatrixDescriptor(ins_dims[1], 0, false);
    auto mat_dim_c = phi::funcs::CreateMatrixDescriptor(ins_dims[2], 0, false);
    if (cost1 < cost2) {
      DenseTensor tmp_out;
      phi::DDim tmp_dim = phi::make_ddim({Ma, Nb});
      tmp_out.Resize(tmp_dim);
      ctx.template Alloc<T>(&tmp_out);
      blas.MatMul(
          *ins[0], mat_dim_a, *ins[1], mat_dim_b, scale, &tmp_out, T(0));
      auto mat_dim_tmp = phi::funcs::CreateMatrixDescriptor(tmp_dim, 0, false);
      blas.MatMul(tmp_out, mat_dim_tmp, *ins[2], mat_dim_c, scale, out, T(0));
    } else {
      DenseTensor tmp_out;
      phi::DDim tmp_dim = phi::make_ddim({Ka, Nc});
      tmp_out.Resize(tmp_dim);
      ctx.template Alloc<T>(&tmp_out);
      std::cout << tmp_out << std::endl;
      blas.MatMul(
          *ins[1], mat_dim_b, *ins[2], mat_dim_c, scale, &tmp_out, T(0));
      auto mat_dim_tmp = phi::funcs::CreateMatrixDescriptor(tmp_dim, 0, false);
      blas.MatMul(*ins[0], mat_dim_a, tmp_out, mat_dim_tmp, scale, out, T(0));
    }
  } else {
    std::vector<DenseTensor> results;
    const auto tmp =
        MultiDotMatChainOrder<Context, T>(ctx, ins, ins_dims, false, &results);
    auto out_dim = out->dims();
    *out = tmp;
    out->Resize(out_dim);
  }
}

/**
 * @brief calculate dA and dB
 * dA = dout * transpose(B)
 * dB = transpose(A) * dout
 */
template <typename Context, typename T>
void CalcGrad(const Context& ctx,
              const DenseTensor& dout,
              const DenseTensor& A,
              const DenseTensor& B,
              const phi::DDim& dout_dim,
              const phi::DDim& a_dim,
              const phi::DDim& b_dim,
              DenseTensor* dA,
              DenseTensor* dB) {
  auto mat_dim_dout = phi::funcs::CreateMatrixDescriptor(dout_dim, 0, false);
  auto mat_dim_a = phi::funcs::CreateMatrixDescriptor(a_dim, 0, true);
  auto mat_dim_b = phi::funcs::CreateMatrixDescriptor(b_dim, 0, true);
  T alpha = static_cast<T>(1.0);
  auto blas = phi::funcs::GetBlas<Context, T>(ctx);
  blas.MatMul(A, mat_dim_a, dout, mat_dim_dout, alpha, dB, T(0));
  blas.MatMul(dout, mat_dim_dout, B, mat_dim_b, alpha, dA, T(0));
}

/**
 * @brief calculate multi matrix multiplication grad by a chain order
 * @param
 * dout: the grad of multi matrix multiplication out
 * dx: the out grad of inputs
 * ins: the input tensors
 * ins_dims: the shape of ins after reshape
 * order: the optimal order
 * i: the left of sub chain
 * j: the righe of sub chain
 * results: the intermediate result of farward
 */
template <typename Context, typename T>
void MatChainMulGrad(const Context& ctx,
                     const DenseTensor& dout,
                     std::vector<DenseTensor*>* dx,
                     const std::vector<const DenseTensor*>& ins,
                     const phi::DDim& dout_dim,
                     const std::vector<phi::DDim>& ins_dims,
                     const std::vector<uint64_t>& order,
                     const uint64_t i,
                     const uint64_t j,
                     const std::vector<DenseTensor>& results) {
  if (i == j) {
    *((*dx)[i]) = dout;
    return;
  }

  const auto n = ins.size();
  const auto right = order[i * n + j];
  const auto left = order[i * n + j] + 1;
  // get the multi result of left sub chain
  const auto* A = &results[i * n + right];
  phi::DDim a_dim = A->dims();
  if (i == right) {
    A = ins[i];
    a_dim = ins_dims[i];
  }
  // get the multi result of right sub chain
  const auto* B = &results[left * n + j];
  phi::DDim b_dim = B->dims();
  if (left == j) {
    B = ins[j];
    b_dim = ins_dims[j];
  }
  DenseTensor dA, dB;
  dA.Resize({dout_dim[0], b_dim[0]});
  dB.Resize({a_dim[1], dout_dim[1]});
  ctx.template Alloc<T>(&dA);
  ctx.template Alloc<T>(&dB);

  CalcGrad<Context, T>(ctx, dout, *A, *B, dout_dim, a_dim, b_dim, &dA, &dB);
  MatChainMulGrad<Context, T>(
      ctx, dA, dx, ins, dA.dims(), ins_dims, order, i, right, results);
  MatChainMulGrad<Context, T>(
      ctx, dB, dx, ins, dB.dims(), ins_dims, order, left, j, results);
}

template <typename Context, typename T>
void MultiDotGradMatChainOrder(const Context& ctx,
                               const DenseTensor& dout,
                               const std::vector<const DenseTensor*>& ins,
                               const phi::DDim& dout_dim,
                               const std::vector<phi::DDim>& ins_dims,
                               std::vector<DenseTensor*>* dx) {
  auto order = GetOrder<Context, T>(ins, ins_dims);
  auto n = ins.size();
  std::vector<DenseTensor> results(n * n);
  MatChainMul<Context, T>(ctx, ins, ins_dims, order, 0, n - 1, true, &results);
  MatChainMulGrad<Context, T>(
      ctx, dout, dx, ins, dout_dim, ins_dims, order, 0, n - 1, results);
}

template <typename T, typename Context>
void MultiDotGradKernel(const Context& ctx,
                        const DenseTensor& out_grad,
                        const std::vector<const DenseTensor*>& x,
                        std::vector<DenseTensor*> x_grad) {
  auto ins = x;
  auto dout = out_grad;
  auto dx = x_grad;

  auto blas = phi::funcs::GetBlas<Context, T>(ctx);

  const auto n = ins.size();
  for (size_t i = 0; i < n; i++) {
    ctx.template Alloc<T>(dx[i]);
  }

  std::vector<phi::DDim> ins_dims(n);
  GetDims<Context, T>(ins, &ins_dims);

  phi::DDim dout_dim = dout.dims();
  if (ins[0]->dims().size() == 1 && ins[n - 1]->dims().size() == 1) {
    dout_dim = phi::make_ddim({1, 1});
  } else if (ins[0]->dims().size() == 1) {
    if (dout_dim.size() == 1) {
      dout_dim = phi::make_ddim({1, dout_dim[0]});
    }
  } else if (ins[n - 1]->dims().size() == 1) {
    if (dout_dim.size() == 1) {
      dout_dim = phi::make_ddim({dout_dim[0], 1});
    }
  }

  T alpha = static_cast<T>(1);
  auto mat_dim_dout = phi::funcs::CreateMatrixDescriptor(dout_dim, 0, false);
  if (n == 2) {
    CalcGrad<Context, T>(ctx,
                         dout,
                         *ins[0],
                         *ins[1],
                         dout_dim,
                         ins_dims[0],
                         ins_dims[1],
                         dx[0],
                         dx[1]);
  } else if (n == 3) {
    const auto Ma = ins_dims[0][0];
    const auto Ka = ins_dims[0][1];
    const auto Nb = ins_dims[1][1];
    const auto Nc = ins_dims[2][1];
    const uint64_t cost1 = Ma * Nb * (Ka + Nc);
    const uint64_t cost2 = Ka * Nc * (Nb + Ma);
    auto mat_dim_a = phi::funcs::CreateMatrixDescriptor(ins_dims[0], 0, false);
    auto mat_dim_b = phi::funcs::CreateMatrixDescriptor(ins_dims[1], 0, false);
    auto mat_dim_c = phi::funcs::CreateMatrixDescriptor(ins_dims[2], 0, false);
    if (cost1 < cost2) {
      DenseTensor tmp_out, tmp_dout;
      tmp_out.Resize({Ma, Nb});
      ctx.template Alloc<T>(&tmp_out);
      tmp_dout.Resize({mat_dim_dout.height_, Nb});
      ctx.template Alloc<T>(&tmp_dout);
      blas.MatMul(
          *ins[0], mat_dim_a, *ins[1], mat_dim_b, alpha, &tmp_out, T(0));
      CalcGrad<Context, T>(ctx,
                           dout,
                           tmp_out,
                           *ins[2],
                           dout_dim,
                           tmp_out.dims(),
                           ins_dims[2],
                           &tmp_dout,
                           dx[2]);
      CalcGrad<Context, T>(ctx,
                           tmp_dout,
                           *ins[0],
                           *ins[1],
                           tmp_dout.dims(),
                           ins_dims[0],
                           ins_dims[1],
                           dx[0],
                           dx[1]);
    } else {
      DenseTensor tmp_out, tmp_dout;
      tmp_out.Resize({Ka, Nc});
      ctx.template Alloc<T>(&tmp_out);
      tmp_dout.Resize({Ka, mat_dim_dout.width_});
      ctx.template Alloc<T>(&tmp_dout);
      blas.MatMul(
          *ins[1], mat_dim_b, *ins[2], mat_dim_c, alpha, &tmp_out, T(0));
      CalcGrad<Context, T>(ctx,
                           dout,
                           *ins[0],
                           tmp_out,
                           dout_dim,
                           ins_dims[0],
                           tmp_dout.dims(),
                           dx[0],
                           &tmp_dout);
      CalcGrad<Context, T>(ctx,
                           tmp_dout,
                           *ins[1],
                           *ins[2],
                           tmp_dout.dims(),
                           ins_dims[1],
                           ins_dims[2],
                           dx[1],
                           dx[2]);
    }
  } else {
    MultiDotGradMatChainOrder<Context, T>(
        ctx, dout, ins, dout_dim, ins_dims, &dx);
    if (ins[n - 1]->dims().size() == 1) {
      dx[n - 1]->Resize({dx[n - 1]->dims()[0]});
    }
  }
}

}  // namespace phi
