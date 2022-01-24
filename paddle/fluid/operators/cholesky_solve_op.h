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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/lapack_function.h"
#include "paddle/fluid/operators/solve_op.h"
#include "paddle/fluid/operators/svd_helper.h"
#include "paddle/fluid/operators/triangular_solve_op.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/pten/kernels/math_kernel.h"

namespace paddle {
namespace operators {  // namespace operators

template <typename DeviceContext, typename T>
class CholeskySolveFunctor {
 public:
  void operator()(const platform::DeviceContext &dev_ctx, bool upper, int n,
                  int nrhs, T *Adata, int lda, T *Bdata, int *devInfo);
};

template <typename T>
class CholeskySolveFunctor<paddle::platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext &dev_ctx, bool upper, int n,
                  int nrhs, T *Adata, int lda, T *Bdata, int *devInfo) {
    char uplo = upper ? 'U' : 'L';
    math::lapackCholeskySolve<T>(uplo, n, nrhs, Adata, lda, Bdata, lda,
                                 devInfo);
  }
};

template <typename DeviceContext, typename T>
void cholesky_solve_fn(const paddle::framework::ExecutionContext &ctx,
                       const framework::Tensor &uin,
                       const framework::Tensor &bin, framework::Tensor *out,
                       bool upper) {
  const auto &dev_ctx = ctx.template device_context<DeviceContext>();
  // framework::Tensor broadcast
  std::vector<int64_t> u_bst_dims_vec;
  std::vector<int64_t> b_bst_dims_vec;
  std::tie(u_bst_dims_vec, b_bst_dims_vec) = get_broadcast_dims(uin, bin);
  framework::Tensor u_bst(uin.type());
  TensorExpand<T, DeviceContext>(dev_ctx, uin, &u_bst, u_bst_dims_vec);

  framework::Tensor b_bst(bin.type());
  TensorExpand<T, DeviceContext>(dev_ctx, bin, &b_bst, b_bst_dims_vec);

  math::DeviceIndependenceTensorOperations<DeviceContext, T> helper(ctx);

  // calculate u's conjugate for complex
  framework::Tensor u_conj(u_bst.type());
  platform::ForRange<DeviceContext> u_for_range(dev_ctx, u_bst.numel());
  math::ConjFunctor<T> u_functor(
      u_bst.data<T>(), u_bst.numel(),
      u_conj.mutable_data<T>(u_bst.dims(), dev_ctx.GetPlace()));
  u_for_range(u_functor);
  u_conj = helper.Transpose(u_conj);

  // calculate b's conjugate for complex
  framework::Tensor b_conj(b_bst.type());
  platform::ForRange<DeviceContext> b_for_range(dev_ctx, b_bst.numel());
  math::ConjFunctor<T> b_functor(
      b_bst.data<T>(), b_bst.numel(),
      b_conj.mutable_data<T>(b_bst.dims(), dev_ctx.GetPlace()));
  b_for_range(b_functor);
  b_conj = helper.Transpose(b_conj);

  auto ut_data = u_conj.mutable_data<T>(dev_ctx.GetPlace());
  auto uindims = u_bst.dims();
  auto bindims = b_bst.dims();
  int uinrank = uindims.size();
  int binrank = bindims.size();

  int n = uindims[uinrank - 2];
  int nrhs = bindims[binrank - 1];
  int ldab = std::max(1, n);

  // framework::Tensor out_copy(b_conj.type());
  // out_copy.Resize(b_conj.dims());
  framework::TensorCopy(b_conj, dev_ctx.GetPlace(), out);
  T *out_data = out->mutable_data<T>(dev_ctx.GetPlace());

  auto info_dims = slice_ddim(bindims, 0, binrank - 2);
  auto batchsize = product(info_dims);

  framework::Tensor tmp;
  std::vector<int> tmpdim(1, batchsize);
  tmp.Resize(framework::make_ddim(tmpdim));
  int *info = tmp.mutable_data<int>(dev_ctx.GetPlace());

  CholeskySolveFunctor<DeviceContext, T> functor;
  for (int b = 0; b < batchsize; b++) {
    auto uin_data_item = &ut_data[b * n * n];
    auto out_data_item = &out_data[b * n * nrhs];
    auto info_item = &info[b];
    functor(dev_ctx, upper, n, nrhs, uin_data_item, ldab, out_data_item,
            info_item);
  }

  // calculate out's conjugate for complex
  platform::ForRange<DeviceContext> out_for_range(dev_ctx, out->numel());
  math::ConjFunctor<T> out_functor(
      out->data<T>(), out->numel(),
      out->mutable_data<T>(out->dims(), dev_ctx.GetPlace()));
  out_for_range(out_functor);
  *out = helper.Transpose(*out);
}

template <typename DeviceContext, typename T>
class CholeskySolveKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext &ctx) const override {
    auto *uin = ctx.Input<framework::Tensor>("Y");
    auto *bin = ctx.Input<framework::Tensor>("X");
    auto *out = ctx.Output<framework::Tensor>("Out");
    auto upper = ctx.Attr<bool>("upper");
    cholesky_solve_fn<DeviceContext, T>(ctx, *uin, *bin, out, upper);
  }
};

template <typename DeviceContext, typename T>
class CholeskySolveGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *bin = ctx.Input<framework::Tensor>("X");
    auto *uin = ctx.Input<framework::Tensor>("Y");
    auto *out = ctx.Input<framework::Tensor>("Out");
    auto *dout = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto *db = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto *du = ctx.Output<framework::Tensor>(framework::GradVarName("Y"));
    auto upper = ctx.Attr<bool>("upper");

    const auto &dev_ctx = ctx.template device_context<DeviceContext>();
    math::DeviceIndependenceTensorOperations<DeviceContext, T> helper(ctx);

    std::vector<int64_t> u_bst_dims_vec;
    std::vector<int64_t> b_bst_dims_vec;
    std::tie(u_bst_dims_vec, b_bst_dims_vec) = get_broadcast_dims(*uin, *bin);
    framework::Tensor u_bst(uin->type());
    TensorExpand<T, DeviceContext>(dev_ctx, *uin, &u_bst, u_bst_dims_vec);

    framework::Tensor db_bst(bin->type());
    TensorExpand<T, DeviceContext>(dev_ctx, *bin, &db_bst, b_bst_dims_vec);

    if (dout) {
      db->mutable_data<T>(dev_ctx.GetPlace());
      cholesky_solve_fn<DeviceContext, T>(ctx, u_bst, *dout, &db_bst, upper);

      if (db_bst.dims() == db->dims()) {
        framework::TensorCopy(db_bst, dev_ctx.GetPlace(), dev_ctx, db);
      } else {
        MatrixReduceSumFunctor<DeviceContext, T> functor;
        functor(db_bst, db, ctx);
        db->Resize(bin->dims());
      }

      auto blas = math::GetBlas<DeviceContext, T>(ctx);

      // calculate out's conjugate for complex
      framework::Tensor out_conj(out->type());
      platform::ForRange<DeviceContext> out_for_range(dev_ctx, out->numel());
      math::ConjFunctor<T> out_functor(
          out->data<T>(), out->numel(),
          out_conj.mutable_data<T>(out->dims(), dev_ctx.GetPlace()));
      out_for_range(out_functor);
      out_conj = helper.Transpose(out_conj);

      framework::Tensor commonterm(out->type());
      auto outdims = out_conj.dims();
      auto dbdims = db_bst.dims();
      auto mat_dim_a = math::CreateMatrixDescriptor(outdims, 0, false);
      auto mat_dim_b = math::CreateMatrixDescriptor(dbdims, 0, false);
      auto cmtdim = outdims;
      cmtdim[cmtdim.size() - 2] = dbdims[dbdims.size() - 2];
      commonterm.Resize(cmtdim);
      commonterm.mutable_data<T>(dev_ctx.GetPlace());
      blas.MatMul(db_bst, mat_dim_b, out_conj, mat_dim_a, static_cast<T>(1),
                  &commonterm, static_cast<T>(0));

      // calculate commonterm's conjugate for complex
      framework::Tensor commonterm_conj(commonterm.type());
      platform::ForRange<DeviceContext> commonterm_for_range(
          dev_ctx, commonterm.numel());
      math::ConjFunctor<T> commonterm_functor(
          commonterm.data<T>(), commonterm.numel(),
          commonterm_conj.mutable_data<T>(commonterm.dims(),
                                          dev_ctx.GetPlace()));
      commonterm_for_range(commonterm_functor);
      commonterm_conj = helper.Transpose(commonterm_conj);

      pten::AddKernel<T>(dev_ctx, commonterm, commonterm_conj, -1, &commonterm);

      auto mat_dim_u = math::CreateMatrixDescriptor(u_bst.dims(), 0, false);
      auto mat_dim_c =
          math::CreateMatrixDescriptor(commonterm.dims(), 0, false);

      Tensor du_bst(uin->type());
      // get upper or lower triangular
      du_bst.Resize(u_bst.dims());
      du_bst.mutable_data<T>(dev_ctx.GetPlace());
      if (upper) {
        blas.MatMul(u_bst, mat_dim_u, commonterm, mat_dim_c, static_cast<T>(-1),
                    &du_bst, static_cast<T>(0));
      } else {
        blas.MatMul(commonterm, mat_dim_c, u_bst, mat_dim_u, static_cast<T>(-1),
                    &du_bst, static_cast<T>(0));
      }

      const auto &udims = u_bst.dims();
      const auto H = udims[udims.size() - 2];
      const auto W = udims[udims.size() - 1];
      platform::ForRange<DeviceContext> x_for_range(dev_ctx, u_bst.numel());
      TrilTriuCompute<T> tril_triu_computer(du_bst.data<T>(), 0, !upper, H, W,
                                            u_bst.data<T>());
      x_for_range(tril_triu_computer);

      du->mutable_data<T>(dev_ctx.GetPlace());
      if (u_bst.dims() == du->dims()) {
        framework::TensorCopy(u_bst, dev_ctx.GetPlace(), dev_ctx, du);
      } else {
        MatrixReduceSumFunctor<DeviceContext, T> functor;
        functor(u_bst, du, ctx);
        du->Resize(uin->dims());
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
