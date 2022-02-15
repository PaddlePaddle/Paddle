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

#include <memory>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/matrix_inverse.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
struct IdentityMatrixFunctor {
  IdentityMatrixFunctor(const int m, T* output) : m_(m), output_(output) {}

  HOSTDEVICE void operator()(size_t index) const {
    const int row = index / m_ % m_;
    const int col = index % m_;
    output_[index] = col == row ? static_cast<T>(1) : static_cast<T>(0);
  }

  const int m_;
  T* output_;
};

template <typename DeviceContext, typename T>
void MatrixPowerFunction(const Tensor* X, const int n, Tensor* Out,
                         const paddle::framework::ExecutionContext& ctx) {
  const auto& x_dims = X->dims();
  const int x_ndim = x_dims.size();
  T* out_data = Out->mutable_data<T>(ctx.GetPlace());

  auto& dev_ctx = ctx.template device_context<DeviceContext>();
  platform::ForRange<DeviceContext> for_range(dev_ctx, X->numel());

  if (n == 0) {
    // Out = Identity Matrix
    IdentityMatrixFunctor<T> functor(x_dims[x_ndim - 1], out_data);
    for_range(functor);
    return;
  }

  auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);

  Tensor new_x = ctx.AllocateTmpTensor<T, DeviceContext>(X->dims(), dev_ctx);
  int new_n = n;
  if (n > 0) {
    // newX = X
    framework::TensorCopy(*X, ctx.GetPlace(), dev_ctx, &new_x);
  } else {
    // newX = X^{-1}, n = -n
    math::MatrixInverseFunctor<DeviceContext, T> mat_inv;
    mat_inv(dev_ctx, *X, &new_x);
    new_n = -n;
  }

  if (new_n == 1) {
    framework::TensorCopy(new_x, ctx.GetPlace(), dev_ctx, Out);
    return;
  }

  auto no_trans_desc = math::CreateMatrixDescriptor(x_dims, 0, false);

  if (new_n == 2) {
    // Out = newX * newX
    Out->mutable_data<T>(ctx.GetPlace());
    blas.MatMul(new_x, no_trans_desc, new_x, no_trans_desc, static_cast<T>(1),
                Out, static_cast<T>(0));
    return;
  } else if (new_n == 3) {
    // Out = (newX * newX) * newX
    // Note: C[i] matrices in MatMul must not overlap, i.e. the individual
    // gemm operations must be computable independently; otherwise,
    // undefined behavior is expected.
    Tensor temp = ctx.AllocateTmpTensor<T, DeviceContext>(X->dims(), dev_ctx);
    blas.MatMul(new_x, no_trans_desc, new_x, no_trans_desc, static_cast<T>(1),
                &temp, static_cast<T>(0));
    blas.MatMul(temp, no_trans_desc, new_x, no_trans_desc, static_cast<T>(1),
                Out, static_cast<T>(0));
    return;
  } else if (new_n == 4) {
    // Out = (newX * newX) * (newX * newX)
    Tensor temp = ctx.AllocateTmpTensor<T, DeviceContext>(X->dims(), dev_ctx);
    blas.MatMul(new_x, no_trans_desc, new_x, no_trans_desc, static_cast<T>(1),
                &temp, static_cast<T>(0));
    blas.MatMul(temp, no_trans_desc, temp, no_trans_desc, static_cast<T>(1),
                Out, static_cast<T>(0));
    return;
  }

  // Calculate Out = newX^{n} for abs(n) > 4 with time complexity as O(logN)
  int bit = 0;
  Tensor z = Tensor(X->dtype());
  bool out_inited = false;
  Tensor temp_out = ctx.AllocateTmpTensor<T, DeviceContext>(X->dims(), dev_ctx);
  Tensor temp_z = ctx.AllocateTmpTensor<T, DeviceContext>(X->dims(), dev_ctx);
  while (new_n > 0) {
    bit = new_n & 0x1;
    new_n >>= 1;
    if (z.IsInitialized()) {
      blas.MatMul(z, no_trans_desc, z, no_trans_desc, static_cast<T>(1),
                  &temp_z, static_cast<T>(0));
      framework::TensorCopy(temp_z, ctx.GetPlace(), dev_ctx, &z);
    } else {
      z = ctx.AllocateTmpTensor<T, DeviceContext>(X->dims(), dev_ctx);
      framework::TensorCopy(new_x, ctx.GetPlace(), dev_ctx, &z);
    }
    if (bit == 1) {
      if (out_inited == true) {
        blas.MatMul(*Out, no_trans_desc, z, no_trans_desc, static_cast<T>(1),
                    &temp_out, static_cast<T>(0));
        framework::TensorCopy(temp_out, ctx.GetPlace(), dev_ctx, Out);
      } else {
        framework::TensorCopy(z, ctx.GetPlace(), dev_ctx, Out);
        out_inited = true;
      }
    }
  }
  return;
}

template <typename DeviceContext, typename T>
class MatrixPowerKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    const Tensor* X = ctx.Input<Tensor>("X");
    Tensor* Out = ctx.Output<Tensor>("Out");
    int n = ctx.Attr<int>("n");

    const auto& x_dims = X->dims();
    const int x_ndim = x_dims.size();
    PADDLE_ENFORCE_EQ(
        x_dims[x_ndim - 2], x_dims[x_ndim - 1],
        platform::errors::InvalidArgument(
            "The inner-most 2 dimensions of Input(X) should be equal."
            "X's shape[-2] = %d and shape[-1] = %d.",
            x_dims[x_ndim - 2], x_dims[x_ndim - 1]));

    MatrixPowerFunction<DeviceContext, T>(X, n, Out, ctx);
  }
};

template <typename DeviceContext, typename T>
void MatrixPowerGradFunction(const Tensor* X, const Tensor* Out,
                             const Tensor* dOut, const int n, Tensor* dX,
                             const paddle::framework::ExecutionContext& ctx) {
  dX->mutable_data<T>(ctx.GetPlace());
  const auto& x_dims = X->dims();

  auto& dev_ctx = ctx.template device_context<DeviceContext>();
  auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);

  if (n == 0) {
    // \nabla X = O
    pten::funcs::SetConstant<DeviceContext, T> zero;
    zero(dev_ctx, dX, static_cast<T>(0));
    return;
  } else if (n == 1) {
    // \nabla X = \nabla Out
    framework::TensorCopy(*dOut, ctx.GetPlace(), dev_ctx, dX);
    return;
  }

  auto trans_desc = math::CreateMatrixDescriptor(x_dims, 0, true);
  auto no_trans_desc = math::CreateMatrixDescriptor(x_dims, 0, false);

  if (n == -1) {
    // \nabla X = Out^{T} * \nabla Out * Out^{T}
    Tensor temp_dx =
        ctx.AllocateTmpTensor<T, DeviceContext>(X->dims(), dev_ctx);
    blas.MatMul(*Out, trans_desc, *dOut, no_trans_desc, static_cast<T>(-1),
                &temp_dx, static_cast<T>(0));
    blas.MatMul(temp_dx, no_trans_desc, *Out, trans_desc, static_cast<T>(1), dX,
                static_cast<T>(0));
    return;
  }

  Tensor new_x = ctx.AllocateTmpTensor<T, DeviceContext>(X->dims(), dev_ctx);
  int new_n = n;
  if (n > 0) {
    // newX = X
    framework::TensorCopy(*X, ctx.GetPlace(), dev_ctx, &new_x);
  } else {
    // newX = X^{-1}, n = -n
    math::MatrixInverseFunctor<DeviceContext, T> mat_inv;
    mat_inv(dev_ctx, *X, &new_x);
    new_n = -n;
  }

  // Use chain rule blow to compute \nabla newX^{n}
  // First, Get newX^{0}, newX^{1}, ..., newX^{n - 1},
  // Note that newX^{0} can be omitted
  std::vector<std::shared_ptr<Tensor>> tensor_list(new_n - 1);
  tensor_list[0] = std::make_shared<Tensor>(new_x);
  int index = 1;
  while (index < new_n - 1) {
    tensor_list[index] = std::make_shared<Tensor>(
        ctx.AllocateTmpTensor<T, DeviceContext>(X->dims(), dev_ctx));
    blas.MatMul(*tensor_list[index - 1], no_trans_desc, new_x, no_trans_desc,
                static_cast<T>(1), tensor_list[index].get(), static_cast<T>(0));
    index++;
  }

  // Second, \nabla newX = \sum_{i = 0}^{n - 1} (newX^{T}^{i}
  //                      * \nabla Out
  //                      * (newX^{T}^{n - i - 1})
  Tensor dx_new = ctx.AllocateTmpTensor<T, DeviceContext>(X->dims(), dev_ctx);
  blas.MatMul(*tensor_list[new_n - 2], trans_desc, *dOut, no_trans_desc,
              static_cast<T>(1), &dx_new, static_cast<T>(0));
  Tensor da_an_minus1 =
      ctx.AllocateTmpTensor<T, DeviceContext>(X->dims(), dev_ctx);
  blas.MatMul(*dOut, no_trans_desc, *tensor_list[new_n - 2], trans_desc,
              static_cast<T>(1), &da_an_minus1, static_cast<T>(0));
  blas.AXPY(X->numel(), static_cast<T>(1), da_an_minus1.data<T>(),
            dx_new.data<T>());
  int start = 0;
  while (start < new_n - 2) {
    Tensor a_da = ctx.AllocateTmpTensor<T, DeviceContext>(X->dims(), dev_ctx);
    Tensor a_da_a = ctx.AllocateTmpTensor<T, DeviceContext>(X->dims(), dev_ctx);
    blas.MatMul(*tensor_list[start], trans_desc, *dOut, no_trans_desc,
                static_cast<T>(1), &a_da, static_cast<T>(0));
    blas.MatMul(a_da, no_trans_desc, *tensor_list[new_n - 3 - start],
                trans_desc, static_cast<T>(1), &a_da_a, static_cast<T>(0));
    blas.AXPY(X->numel(), static_cast<T>(1), a_da_a.data<T>(),
              dx_new.data<T>());
    start++;
  }

  if (n > 0) {
    // \nabla X = \nabla newX
    framework::TensorCopy(dx_new, ctx.GetPlace(), dev_ctx, dX);
  } else {
    // \nabla X = newX^{T} * \nabla newX * newX^{T}
    Tensor temp_dx =
        ctx.AllocateTmpTensor<T, DeviceContext>(X->dims(), dev_ctx);
    blas.MatMul(new_x, trans_desc, dx_new, no_trans_desc, static_cast<T>(-1),
                &temp_dx, static_cast<T>(0));
    blas.MatMul(temp_dx, no_trans_desc, new_x, trans_desc, static_cast<T>(1),
                dX, static_cast<T>(0));
  }
  return;
}

template <typename DeviceContext, typename T>
class MatrixPowerGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* X = ctx.Input<Tensor>("X");
    const Tensor* Out = ctx.Input<Tensor>("Out");
    const Tensor* dOut = ctx.Input<Tensor>(framework::GradVarName("Out"));
    const int n = ctx.Attr<int>("n");
    Tensor* dX = ctx.Output<Tensor>(framework::GradVarName("X"));

    MatrixPowerGradFunction<DeviceContext, T>(X, Out, dOut, n, dX, ctx);
  }
};

}  // namespace operators
}  // namespace paddle
