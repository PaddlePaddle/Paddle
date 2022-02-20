/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/matmul_v2_op.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
static void Mul(const framework::ExecutionContext& ctx, const Tensor& X,
                const Tensor& Y, Tensor* Out) {
  Out->mutable_data<T>(ctx.GetPlace());

  MLUCnnlTensorDesc x_desc(X, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());
  MLUCnnlTensorDesc y_desc(Y, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());
  MLUCnnlTensorDesc out_desc(*Out, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());

  MLUCnnlOpTensorDesc mul_op_desc(CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(),
                                  CNNL_NOT_PROPAGATE_NAN);
  MLUCnnl::OpTensor(ctx, mul_op_desc.get(), x_desc.get(), GetBasePtr(&X),
                    y_desc.get(), GetBasePtr(&Y), out_desc.get(),
                    GetBasePtr(Out), ToCnnlDataType<T>());
}

template <typename T>
static void MatMul2D(const framework::ExecutionContext& ctx, const Tensor& X,
                     const Tensor& Y, Tensor* Out, const bool trans_x,
                     const bool trans_y) {
  Out->mutable_data<T>(ctx.GetPlace());

  MLUCnnlTensorDesc x_desc(X, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());
  MLUCnnlTensorDesc y_desc(Y, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());
  MLUCnnlTensorDesc out_desc(*Out, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());

  MLUCnnl::Matmul(ctx, trans_x, trans_y, x_desc.get(), GetBasePtr(&X),
                  y_desc.get(), GetBasePtr(&Y), out_desc.get(),
                  GetBasePtr(Out));
}

template <typename T>
static void MatMulND(const framework::ExecutionContext& ctx, const Tensor& X,
                     const Tensor& Y, Tensor* Out, const bool trans_x,
                     const bool trans_y) {
  if (!Out->initialized()) {
    Out->mutable_data<T>(ctx.GetPlace());
  }

  MLUCnnlTensorDesc x_desc(X, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());
  MLUCnnlTensorDesc y_desc(Y, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());
  MLUCnnlTensorDesc out_desc(*Out, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());

  MLUCnnl::BatchMatmul(ctx, trans_x, trans_y, x_desc.get(), GetBasePtr(&X),
                       y_desc.get(), GetBasePtr(&Y), out_desc.get(),
                       GetBasePtr(Out));
}

template <typename T>
static void ReduceDims(const framework::ExecutionContext& ctx,
                       const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& bcast_dims, const Tensor& in,
                       Tensor* out) {
  std::vector<int64_t> axes;
  int64_t size = bcast_dims.size();
  int64_t diff = bcast_dims.size() - dims.size();
  for (int64_t i = 0; i < size; ++i) {
    if (i < diff) {
      axes.push_back(i);
      continue;
    }
    if (bcast_dims[i] > dims[i - diff]) {
      axes.push_back(i);
    }
  }
  out->mutable_data<T>(ctx.GetPlace());

  MLUCnnlTensorDesc in_desc(in, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());
  MLUCnnlTensorDesc out_desc(*out, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());

  std::vector<int> reduce_dims(axes.begin(), axes.end());
  MLUCnnlReduceDesc reduce_desc(reduce_dims, CNNL_REDUCE_ADD,
                                ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN,
                                CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES);

  MLUCnnl::Reduce(ctx, true /*need_workspace*/, reduce_desc.get(), nullptr,
                  in_desc.get(), GetBasePtr(&in), 0 /*indices_size*/, nullptr,
                  nullptr, out_desc.get(), GetBasePtr(out));
}

template <typename T>
class MatMulV2MLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<framework::Tensor>("X");
    auto* Y = ctx.Input<framework::Tensor>("Y");
    auto* Out = ctx.Output<framework::Tensor>("Out");
    const bool trans_x = ctx.Attr<bool>("trans_x");
    const bool trans_y = ctx.Attr<bool>("trans_y");

    std::vector<int64_t> x_dims = phi::vectorize(X->dims());
    std::vector<int64_t> y_dims = phi::vectorize(Y->dims());
    std::vector<int64_t> out_dims = phi::vectorize(Out->dims());
    int x_ndim = x_dims.size();
    int y_ndim = y_dims.size();

    // Case 1: [K] x [K] = [1]
    // Equal: [1, K] x [K, 1] = [1, 1] => [1]
    const bool all_one_dim = (x_ndim == 1 && y_ndim == 1);
    if (all_one_dim) {
      Out->Resize({1, 1});
    }

    // Resize dim 1 to 2
    Tensor x_temp, y_temp;
    x_temp.ShareDataWith(*X);
    y_temp.ShareDataWith(*Y);
    if (x_ndim == 1) {
      x_dims.insert(x_dims.begin(), 1);
      x_temp.Resize(phi::make_ddim(x_dims));
      x_ndim = 2;
      // matmul op of mlu needs `std::max(x->dim, y->dim) == out->dim`
      if (out_dims.size() < y_dims.size()) {
        std::vector<int64_t> temp_out_dims(out_dims.begin(), out_dims.end());
        temp_out_dims.insert(temp_out_dims.end() - 1, 1);
        Out->Resize(phi::make_ddim(temp_out_dims));
      }
    }
    if (y_ndim == 1) {
      y_dims.push_back(1);
      y_temp.Resize(phi::make_ddim(y_dims));
      y_ndim = 2;
      // matmul op of mlu needs `std::max(x->dim, y->dim) == out->dim`
      if (out_dims.size() < x_dims.size()) {
        std::vector<int64_t> temp_out_dims(out_dims.begin(), out_dims.end());
        temp_out_dims.push_back(1);
        Out->Resize(phi::make_ddim(temp_out_dims));
      }
    }

    const int K = trans_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
    if (trans_y) {
      PADDLE_ENFORCE_EQ(y_dims[y_ndim - 1], K,
                        platform::errors::InvalidArgument(
                            "Input(Y) has error dim."
                            "Y'dims[%d] must be equal to %d"
                            "But received Y'dims[%d] is %d",
                            y_ndim - 1, K, y_ndim - 1, y_dims[y_ndim - 1]));
    } else {
      PADDLE_ENFORCE_EQ(y_dims[y_ndim - 2], K,
                        platform::errors::InvalidArgument(
                            "Input(Y) has error dim."
                            "Y'dims[%d] must be equal to %d"
                            "But received Y'dims[%d] is %d",
                            y_ndim - 2, K, y_ndim - 2, y_dims[y_ndim - 2]));
    }

    if (x_ndim == 2 && y_ndim == 2) {
      // Case 2: [M, K] x [K, N] = [M, N]
      MatMul2D<T>(ctx, x_temp, y_temp, Out, trans_x, trans_y);
    } else {
      // Case 3: [B, M, K] x [K, N] =  [B, M, N]
      // Case 4: [B, M, K] x  [B, K, N] = [B, M, N]
      MatMulND<T>(ctx, x_temp, y_temp, Out, trans_x, trans_y);
    }

    if (phi::vectorize(Out->dims()) != out_dims) {
      Out->Resize(phi::make_ddim(out_dims));
    }
  }
};

template <typename T>
class MatMulGradV2MLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<framework::Tensor>("X");
    auto* Y = ctx.Input<framework::Tensor>("Y");
    auto* dOut = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dX = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dY = ctx.Output<framework::Tensor>(framework::GradVarName("Y"));
    const bool trans_x = ctx.Attr<bool>("trans_x");
    const bool trans_y = ctx.Attr<bool>("trans_y");

    std::vector<int64_t> x_dims = phi::vectorize(X->dims());
    std::vector<int64_t> y_dims = phi::vectorize(Y->dims());
    std::vector<int64_t> out_dims = phi::vectorize(dOut->dims());
    int x_ndim = x_dims.size();
    int y_ndim = y_dims.size();
    int out_ndim = out_dims.size();

    // Case 1: [K] x [K] = [1]
    if (x_ndim == 1 && y_ndim == 1) {
      if (dX) {
        Mul<T>(ctx, *dOut, *Y, dX);
      }
      if (dY) {
        Mul<T>(ctx, *dOut, *X, dY);
      }
      return;
    }

    // Resize dim 1 to 2
    Tensor x_temp, y_temp, dout_temp;
    x_temp.ShareDataWith(*X);
    y_temp.ShareDataWith(*Y);
    dout_temp.ShareDataWith(*dOut);
    if (x_ndim == 1) {
      x_dims.insert(x_dims.begin(), 1);
      out_dims.insert(out_dims.end() - 1, 1);
      x_temp.Resize(phi::make_ddim(x_dims));
      dout_temp.Resize(phi::make_ddim(out_dims));
      x_ndim = 2;
      out_ndim += 1;
    }
    if (y_ndim == 1) {
      y_dims.push_back(1);
      out_dims.push_back(1);
      y_temp.Resize(phi::make_ddim(y_dims));
      dout_temp.Resize(phi::make_ddim(out_dims));
      y_ndim = 2;
      out_ndim += 1;
    }

    // Case 2: [M, K] x [K, N] = [M, N]
    if (out_ndim == 2) {
      if (dX) {
        dX->Resize(phi::make_ddim(x_dims));
        if (trans_x) {
          MatMul2D<T>(ctx, y_temp, dout_temp, dX, trans_y, true);
        } else {
          MatMul2D<T>(ctx, dout_temp, y_temp, dX, false, !trans_y);
        }
        dX->Resize(X->dims());
      }
      if (dY) {
        dY->Resize(phi::make_ddim(y_dims));
        if (trans_y) {
          MatMul2D<T>(ctx, dout_temp, x_temp, dY, true, trans_x);
        } else {
          MatMul2D<T>(ctx, x_temp, dout_temp, dY, !trans_x, false);
        }
        dY->Resize(Y->dims());
      }
      return;
    }

    // Case 3: [B, M, K] x [K, N] =  [B, M, N]
    // Case 4: [B, M, K] x  [B, K, N] = [B, M, N]
    std::vector<int64_t> x_bcast_dims(out_ndim, 1);
    std::vector<int64_t> y_bcast_dims(out_ndim, 1);
    std::copy(out_dims.begin(), out_dims.end() - 2, x_bcast_dims.begin());
    std::copy(out_dims.begin(), out_dims.end() - 2, y_bcast_dims.begin());
    std::copy(x_dims.end() - 2, x_dims.end(), x_bcast_dims.end() - 2);
    std::copy(y_dims.end() - 2, y_dims.end(), y_bcast_dims.end() - 2);

    if (dX) {
      Tensor dx_temp(X->type());
      if (x_dims != x_bcast_dims) {
        dx_temp.Resize(phi::make_ddim(x_bcast_dims));
      } else {
        dX->mutable_data<T>(ctx.GetPlace());
        dx_temp.ShareDataWith(*dX);
      }

      if (trans_x) {
        MatMulND<T>(ctx, y_temp, dout_temp, &dx_temp, trans_y, true);
      } else {
        MatMulND<T>(ctx, dout_temp, y_temp, &dx_temp, false, !trans_y);
      }

      if (x_dims != x_bcast_dims) {
        ReduceDims<T>(ctx, x_dims, x_bcast_dims, dx_temp, dX);
      }
    }

    if (dY) {
      Tensor dy_temp(Y->type());
      if (y_dims != y_bcast_dims) {
        dy_temp.Resize(phi::make_ddim(y_bcast_dims));
      } else {
        dY->mutable_data<T>(ctx.GetPlace());
        dy_temp.ShareDataWith(*dY);
      }

      if (trans_y) {
        MatMulND<T>(ctx, dout_temp, x_temp, &dy_temp, true, trans_x);
      } else {
        MatMulND<T>(ctx, x_temp, dout_temp, &dy_temp, !trans_x, false);
      }

      if (y_dims != y_bcast_dims) {
        ReduceDims<T>(ctx, y_dims, y_bcast_dims, dy_temp, dY);
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(matmul_v2, ops::MatMulV2MLUKernel<float>,
                       ops::MatMulV2MLUKernel<plat::float16>);
REGISTER_OP_MLU_KERNEL(matmul_v2_grad, ops::MatMulGradV2MLUKernel<float>,
                       ops::MatMulGradV2MLUKernel<plat::float16>);
