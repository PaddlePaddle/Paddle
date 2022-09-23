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

#include <memory>
#include <string>

#include "paddle/fluid/operators/matmul_v2_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using NPUDeviceContext = platform::NPUDeviceContext;

template <typename T>
static void MatMul2D(const framework::ExecutionContext& ctx,
                     const aclrtStream& stream,
                     const Tensor& X,
                     const Tensor& Y,
                     Tensor* Out,
                     const bool trans_x,
                     const bool trans_y) {
  Out->mutable_data<T>(ctx.GetPlace());
  const auto& runner =
      NpuOpRunner("MatMul",
                  {X, Y},
                  {*Out},
                  {{"transpose_x1", trans_x}, {"transpose_x2", trans_y}});
  runner.Run(stream);
}

template <typename T>
static void MatMulND(const framework::ExecutionContext& ctx,
                     const aclrtStream& stream,
                     const Tensor& X,
                     const Tensor& Y,
                     Tensor* Out,
                     const bool trans_x,
                     const bool trans_y) {
  Out->mutable_data<T>(ctx.GetPlace());
  const auto& runner = NpuOpRunner("BatchMatMul",
                                   {X, Y},
                                   {*Out},
                                   {{"adj_x1", trans_x}, {"adj_x2", trans_y}});
  runner.Run(stream);
}

#if (CANN_VERSION_CODE < 504000)
template <>
void MatMulND<phi::dtype::float16>(const framework::ExecutionContext& ctx,
                                   const aclrtStream& stream,
                                   const Tensor& X,
                                   const Tensor& Y,
                                   Tensor* Out,
                                   const bool trans_x,
                                   const bool trans_y) {
  Out->mutable_data<phi::dtype::float16>(ctx.GetPlace());
  Tensor x_fp32, y_fp32, out_fp32;
  x_fp32.Resize(X.dims());
  y_fp32.Resize(Y.dims());
  out_fp32.Resize(Out->dims());
  x_fp32.mutable_data<float>(ctx.GetPlace());
  y_fp32.mutable_data<float>(ctx.GetPlace());
  out_fp32.mutable_data<float>(ctx.GetPlace());

  const auto& cast_x =
      NpuOpRunner("Cast",
                  {X},
                  {x_fp32},
                  {{"dst_type",
                    static_cast<int>(ConvertToNpuDtype(
                        framework::TransToProtoVarType(x_fp32.type())))}});
  cast_x.Run(stream);
  const auto& cast_y =
      NpuOpRunner("Cast",
                  {Y},
                  {y_fp32},
                  {{"dst_type",
                    static_cast<int>(ConvertToNpuDtype(
                        framework::TransToProtoVarType(y_fp32.type())))}});
  cast_y.Run(stream);

  const auto& runner = NpuOpRunner("BatchMatMul",
                                   {x_fp32, y_fp32},
                                   {out_fp32},
                                   {{"adj_x1", trans_x}, {"adj_x2", trans_y}});
  runner.Run(stream);

  const auto& cast_out = NpuOpRunner(
      "Cast",
      {out_fp32},
      {*Out},
      {{"dst_type",
        static_cast<int>(
            ConvertToNpuDtype(framework::TransToProtoVarType(Out->type())))}});
  cast_out.Run(stream);
}
#endif

template <typename T>
static void ReduceDims(const framework::ExecutionContext& ctx,
                       const aclrtStream& stream,
                       const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& brd_dims,
                       const Tensor& in,
                       Tensor* out) {
  std::vector<int64_t> axes;
  int64_t size = brd_dims.size();
  int64_t diff = brd_dims.size() - dims.size();
  for (int64_t i = 0; i < size; ++i) {
    if (i < diff) {
      axes.push_back(i);
      continue;
    }
    if (brd_dims[i] > dims[i - diff]) {
      axes.push_back(i);
    }
  }
  out->mutable_data<T>(ctx.GetPlace());
  const auto& runner = NpuOpRunner(
      "ReduceSumD", {in}, {*out}, {{"axes", axes}, {"keep_dims", false}});
  runner.Run(stream);
}

template <typename T>
class MatMulV2NPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<Tensor>("X");
    auto* Y = ctx.Input<Tensor>("Y");
    auto* Out = ctx.Output<Tensor>("Out");
    const bool trans_x = ctx.Attr<bool>("trans_x");
    const bool trans_y = ctx.Attr<bool>("trans_y");

    std::vector<int64_t> x_dims = phi::vectorize(X->dims());
    std::vector<int64_t> y_dims = phi::vectorize(Y->dims());
    std::vector<int64_t> out_dims = phi::vectorize(Out->dims());
    int x_ndim = x_dims.size();
    int y_ndim = y_dims.size();
    int out_ndim = out_dims.size();

    auto stream = ctx.template device_context<NPUDeviceContext>().stream();

    // Case 1: [K] x [K] = [1]
    if (x_ndim == 1 && y_ndim == 1) {
      PADDLE_ENFORCE_EQ(
          X->numel(),
          Y->numel(),
          platform::errors::InvalidArgument(
              "X's numbers must be equal to Y's numbers,"
              "when X/Y's dims =1. But received X has [%d] elements,"
              "received Y has [%d] elements",
              X->numel(),
              Y->numel()));
      Out->Resize({1});
      Out->mutable_data<T>(ctx.GetPlace());

      const auto& runner = NpuOpRunner("Dot", {*X, *Y}, {*Out});
      runner.Run(stream);
      return;
    }

    // Resize dim 1 to 2
    Tensor x_temp, y_temp;
    x_temp.ShareDataWith(*X);
    y_temp.ShareDataWith(*Y);
    if (x_ndim == 1) {
      x_dims.insert(x_dims.begin(), 1);
      out_dims.insert(out_dims.end() - 1, 1);
      x_temp.Resize(phi::make_ddim(x_dims));
      x_ndim = 2;
      out_ndim += 1;
    }
    if (y_ndim == 1) {
      y_dims.push_back(1);
      out_dims.push_back(1);
      y_temp.Resize(phi::make_ddim(y_dims));
      y_ndim = 2;
      out_ndim += 1;
    }

    const int K = trans_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
    if (trans_y) {
      PADDLE_ENFORCE_EQ(
          y_dims[y_ndim - 1],
          K,
          platform::errors::InvalidArgument("Input(Y) has error dim."
                                            "Y'dims[%d] must be equal to %d"
                                            "But received Y'dims[%d] is %d",
                                            y_ndim - 1,
                                            K,
                                            y_ndim - 1,
                                            y_dims[y_ndim - 1]));
    } else {
      PADDLE_ENFORCE_EQ(
          y_dims[y_ndim - 2],
          K,
          platform::errors::InvalidArgument("Input(Y) has error dim."
                                            "Y'dims[%d] must be equal to %d"
                                            "But received Y'dims[%d] is %d",
                                            y_ndim - 2,
                                            K,
                                            y_ndim - 2,
                                            y_dims[y_ndim - 2]));
    }

    // Case 2: [M, K] x [K, N] = [M, N]
    if (x_ndim == 2 && y_ndim == 2) {
      MatMul2D<T>(ctx, stream, x_temp, y_temp, Out, trans_x, trans_y);
      return;
    }

    // Case 3: [B, M, K] x [K, N] =  [B, M, N], when trans_x = false
    // Equal: [B * M, K] x [K, N] = [B * M, N] => [B, M, N]
    if (trans_x == false && y_ndim == 2) {
      std::vector<int64_t> vec_dim = {x_temp.numel() / K, K};
      x_temp.Resize(phi::make_ddim(vec_dim));
      MatMul2D<T>(ctx, stream, x_temp, y_temp, Out, trans_x, trans_y);
      return;
    }

    // Case 4: [B, M, K] x  [B, K, N] = [B, M, N]
    std::vector<int64_t> x_broadcast_dims(out_ndim, 1);
    std::vector<int64_t> y_broadcast_dims(out_ndim, 1);
    std::copy(out_dims.begin(), out_dims.end() - 2, x_broadcast_dims.begin());
    std::copy(out_dims.begin(), out_dims.end() - 2, y_broadcast_dims.begin());
    std::copy(x_dims.end() - 2, x_dims.end(), x_broadcast_dims.end() - 2);
    std::copy(y_dims.end() - 2, y_dims.end(), y_broadcast_dims.end() - 2);

    Tensor x_temp_brd(X->type());
    if (x_dims == x_broadcast_dims) {
      x_temp_brd.ShareDataWith(*X);
      x_temp_brd.Resize(phi::make_ddim(x_broadcast_dims));
    } else {
      x_temp_brd.Resize(phi::make_ddim(x_broadcast_dims));
      x_temp_brd.mutable_data<T>(ctx.GetPlace());
      NpuOpRunner runner_brd;
      runner_brd.SetType("BroadcastTo")
          .AddInput(x_temp)
          .AddInput(std::move(x_broadcast_dims))
          .AddOutput(x_temp_brd)
          .Run(stream);
    }

    Tensor y_temp_brd(Y->type());
    if (y_dims == y_broadcast_dims) {
      y_temp_brd.ShareDataWith(*Y);
      y_temp_brd.Resize(phi::make_ddim(y_broadcast_dims));
    } else {
      y_temp_brd.Resize(phi::make_ddim(y_broadcast_dims));
      y_temp_brd.mutable_data<T>(ctx.GetPlace());
      NpuOpRunner runner_brd;
      runner_brd.SetType("BroadcastTo")
          .AddInput(y_temp)
          .AddInput(std::move(y_broadcast_dims))
          .AddOutput(y_temp_brd)
          .Run(stream);
    }
    MatMulND<T>(ctx, stream, x_temp_brd, y_temp_brd, Out, trans_x, trans_y);
  }
};

template <typename T>
class MatMulV2GradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<Tensor>("X");
    auto* Y = ctx.Input<Tensor>("Y");
    auto* dOut = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dX = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dY = ctx.Output<Tensor>(framework::GradVarName("Y"));
    const bool trans_x = ctx.Attr<bool>("trans_x");
    const bool trans_y = ctx.Attr<bool>("trans_y");

    std::vector<int64_t> x_dims = phi::vectorize(X->dims());
    std::vector<int64_t> y_dims = phi::vectorize(Y->dims());
    std::vector<int64_t> out_dims = phi::vectorize(dOut->dims());
    int x_ndim = x_dims.size();
    int y_ndim = y_dims.size();
    int out_ndim = out_dims.size();

    auto stream = ctx.template device_context<NPUDeviceContext>().stream();

    // Case 1: [K] x [K] = [1]
    if (x_ndim == 1 && y_ndim == 1) {
      Tensor dout_temp(dOut->type());
      dout_temp.Resize(X->dims());
      dout_temp.mutable_data<T>(ctx.GetPlace());
      NpuOpRunner runner;
      runner.SetType("BroadcastTo")
          .AddInput(*dOut)
          .AddInput(std::move(x_dims))
          .AddOutput(dout_temp)
          .Run(stream);

      if (dX) {
        dX->mutable_data<T>(ctx.GetPlace());
        const auto& runner_dx = NpuOpRunner("Mul", {dout_temp, *Y}, {*dX}, {});
        runner_dx.Run(stream);
      }
      if (dY) {
        dY->mutable_data<T>(ctx.GetPlace());
        const auto& runner_dy = NpuOpRunner("Mul", {dout_temp, *X}, {*dY}, {});
        runner_dy.Run(stream);
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
          MatMul2D<T>(ctx, stream, y_temp, dout_temp, dX, trans_y, true);
        } else {
          MatMul2D<T>(ctx, stream, dout_temp, y_temp, dX, false, !trans_y);
        }
        dX->Resize(X->dims());
      }
      if (dY) {
        dY->Resize(phi::make_ddim(y_dims));
        if (trans_y) {
          MatMul2D<T>(ctx, stream, dout_temp, x_temp, dY, true, trans_x);
        } else {
          MatMul2D<T>(ctx, stream, x_temp, dout_temp, dY, !trans_x, false);
        }
        dY->Resize(Y->dims());
      }
      return;
    }

    const int K = trans_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
    const int N = trans_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];

    // Case 3: [B, M, K] x [K, N] =  [B, M, N], when trans_x = false
    // Equal: [B * M, K] x [K, N] = [B * M, N] => [B, M, N]
    if (trans_x == false && y_ndim == 2) {
      std::vector<int64_t> x_vec_dim = {x_temp.numel() / K, K};
      dout_temp.Resize(
          phi::make_ddim(std::vector<int64_t>{dout_temp.numel() / N, N}));
      if (dX) {
        dX->Resize(phi::make_ddim(x_vec_dim));
        MatMul2D<T>(ctx, stream, dout_temp, y_temp, dX, false, !trans_y);
        dX->Resize(X->dims());
      }
      if (dY) {
        x_temp.Resize(phi::make_ddim(x_vec_dim));
        if (trans_y) {
          MatMul2D<T>(ctx, stream, dout_temp, x_temp, dY, true, false);
        } else {
          MatMul2D<T>(ctx, stream, x_temp, dout_temp, dY, true, false);
        }
      }
      return;
    }

    // Case 4: [B, M, K] x  [B, K, N] = [B, M, N]
    std::vector<int64_t> x_broadcast_dims(out_ndim, 1);
    std::vector<int64_t> y_broadcast_dims(out_ndim, 1);
    std::copy(out_dims.begin(), out_dims.end() - 2, x_broadcast_dims.begin());
    std::copy(out_dims.begin(), out_dims.end() - 2, y_broadcast_dims.begin());
    std::copy(x_dims.end() - 2, x_dims.end(), x_broadcast_dims.end() - 2);
    std::copy(y_dims.end() - 2, y_dims.end(), y_broadcast_dims.end() - 2);

    Tensor x_temp_brd(X->type());
    if (x_dims == x_broadcast_dims) {
      x_temp_brd.ShareDataWith(*X);
      x_temp_brd.Resize(phi::make_ddim(x_broadcast_dims));
    } else {
      x_temp_brd.Resize(phi::make_ddim(x_broadcast_dims));
      x_temp_brd.mutable_data<T>(ctx.GetPlace());
      NpuOpRunner runner_brd;
      runner_brd.SetType("BroadcastTo")
          .AddInput(x_temp)
          .AddInput(std::move(x_broadcast_dims))
          .AddOutput(x_temp_brd)
          .Run(stream);
    }

    Tensor y_temp_brd(Y->type());
    if (y_dims == y_broadcast_dims) {
      y_temp_brd.ShareDataWith(*Y);
      y_temp_brd.Resize(phi::make_ddim(y_broadcast_dims));
    } else {
      y_temp_brd.Resize(phi::make_ddim(y_broadcast_dims));
      y_temp_brd.mutable_data<T>(ctx.GetPlace());
      NpuOpRunner runner_brd;
      runner_brd.SetType("BroadcastTo")
          .AddInput(y_temp)
          .AddInput(std::move(y_broadcast_dims))
          .AddOutput(y_temp_brd)
          .Run(stream);
    }

    if (dX) {
      if (x_dims == x_broadcast_dims) {
        if (trans_x) {
          MatMulND<T>(ctx, stream, y_temp_brd, dout_temp, dX, trans_y, true);
        } else {
          MatMulND<T>(ctx, stream, dout_temp, y_temp_brd, dX, false, !trans_y);
        }
      } else {
        Tensor dx_temp(X->type());
        dx_temp.Resize(phi::make_ddim(x_broadcast_dims));
        if (trans_x) {
          MatMulND<T>(
              ctx, stream, y_temp_brd, dout_temp, &dx_temp, trans_y, true);
        } else {
          MatMulND<T>(
              ctx, stream, dout_temp, y_temp_brd, &dx_temp, false, !trans_y);
        }
        ReduceDims<T>(ctx, stream, x_dims, x_broadcast_dims, dx_temp, dX);
      }
    }
    if (dY) {
      if (y_dims == y_broadcast_dims) {
        if (trans_y) {
          MatMulND<T>(ctx, stream, dout_temp, x_temp_brd, dY, true, trans_x);
        } else {
          MatMulND<T>(ctx, stream, x_temp_brd, dout_temp, dY, !trans_x, false);
        }
      } else {
        Tensor dy_temp(Y->type());
        dy_temp.Resize(phi::make_ddim(y_broadcast_dims));
        if (trans_y) {
          MatMulND<T>(
              ctx, stream, dout_temp, x_temp_brd, &dy_temp, true, trans_x);
        } else {
          MatMulND<T>(
              ctx, stream, x_temp_brd, dout_temp, &dy_temp, !trans_x, false);
        }
        ReduceDims<T>(ctx, stream, y_dims, y_broadcast_dims, dy_temp, dY);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(matmul_v2,
                       ops::MatMulV2NPUKernel<float>,
                       ops::MatMulV2NPUKernel<paddle::platform::float16>);
REGISTER_OP_NPU_KERNEL(matmul_v2_grad,
                       ops::MatMulV2GradNPUKernel<float>,
                       ops::MatMulV2GradNPUKernel<paddle::platform::float16>);
