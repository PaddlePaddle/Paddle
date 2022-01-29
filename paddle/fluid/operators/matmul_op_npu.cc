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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using NPUDeviceContext = platform::NPUDeviceContext;

template <typename T>
static void Mul(const framework::ExecutionContext& ctx,
                const aclrtStream& stream, const Tensor& X, const Tensor& Y,
                Tensor* Out, const float alpha) {
  Out->mutable_data<T>(ctx.GetPlace());

  if (fabs(alpha - 1.0) < std::numeric_limits<float>::epsilon()) {
    const auto& runner_dx = NpuOpRunner("Mul", {X, Y}, {*Out}, {});
    runner_dx.Run(stream);
  } else {
    Tensor Out_temp(Out->type());
    Out_temp.mutable_data<T>(Out->dims(), ctx.GetPlace());
    const auto& runner_dx = NpuOpRunner("Mul", {X, Y}, {Out_temp}, {});
    runner_dx.Run(stream);

    const auto& runner =
        NpuOpRunner("Muls", {Out_temp}, {*Out}, {{"value", alpha}});
    runner.Run(stream);
  }
}

template <typename T>
static void Dot(const framework::ExecutionContext& ctx,
                const aclrtStream& stream, const Tensor& X, const Tensor& Y,
                Tensor* Out, const float alpha) {
  Out->mutable_data<T>(ctx.GetPlace());

  if (fabs(alpha - 1.0) < std::numeric_limits<float>::epsilon()) {
    const auto& runner = NpuOpRunner("Dot", {X, Y}, {*Out});
    runner.Run(stream);
  } else {
    Tensor Out_temp(Out->type());
    Out_temp.mutable_data<T>(Out->dims(), ctx.GetPlace());
    const auto& out_temp_runner = NpuOpRunner("Dot", {X, Y}, {Out_temp});
    out_temp_runner.Run(stream);

    const auto& runner =
        NpuOpRunner("Muls", {Out_temp}, {*Out}, {{"value", alpha}});
    runner.Run(stream);
  }
}

template <typename T>
static void MatMul2D(const framework::ExecutionContext& ctx,
                     const aclrtStream& stream, const Tensor& X,
                     const Tensor& Y, Tensor* Out, const bool trans_x,
                     const bool trans_y, const float alpha) {
  Out->mutable_data<T>(ctx.GetPlace());

  if (fabs(alpha - 1.0) < std::numeric_limits<float>::epsilon()) {
    const auto& runner =
        NpuOpRunner("MatMul", {X, Y}, {*Out},
                    {{"transpose_x1", trans_x}, {"transpose_x2", trans_y}});
    runner.Run(stream);
  } else {
    Tensor Out_temp(Out->type());
    Out_temp.mutable_data<T>(Out->dims(), ctx.GetPlace());
    const auto& out_temp_runner =
        NpuOpRunner("MatMul", {X, Y}, {Out_temp},
                    {{"transpose_x1", trans_x}, {"transpose_x2", trans_y}});
    out_temp_runner.Run(stream);

    const auto& runner =
        NpuOpRunner("Muls", {Out_temp}, {*Out}, {{"value", alpha}});
    runner.Run(stream);
  }
}

template <typename T>
static void MatMulND(const framework::ExecutionContext& ctx,
                     const aclrtStream& stream, const Tensor& X,
                     const Tensor& Y, Tensor* Out, const bool trans_x,
                     const bool trans_y, const float alpha) {
  Out->mutable_data<T>(ctx.GetPlace());

  if (fabs(alpha - 1.0) < std::numeric_limits<float>::epsilon()) {
    const auto& runner =
        NpuOpRunner("BatchMatMul", {X, Y}, {*Out},
                    {{"adj_x1", trans_x}, {"adj_x2", trans_y}});
    runner.Run(stream);
  } else {
    Tensor Out_temp(Out->type());
    Out_temp.mutable_data<T>(Out->dims(), ctx.GetPlace());
    const auto& out_temp_runner =
        NpuOpRunner("BatchMatMul", {X, Y}, {Out_temp},
                    {{"adj_x1", trans_x}, {"adj_x2", trans_y}});
    out_temp_runner.Run(stream);

    const auto& runner =
        NpuOpRunner("Muls", {Out_temp}, {*Out}, {{"value", alpha}});
    runner.Run(stream);
  }
}

template <typename T>
static void ReduceDims(const framework::ExecutionContext& ctx,
                       const aclrtStream& stream,
                       const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& brd_dims, const Tensor& in,
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
  const auto& runner = NpuOpRunner("ReduceSumD", {in}, {*out},
                                   {{"axes", axes}, {"keep_dims", false}});
  runner.Run(stream);
}

template <typename DeviceContext, typename T>
class MatMulNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<framework::Tensor>("X");
    auto* Y = ctx.Input<framework::Tensor>("Y");
    auto* Out = ctx.Output<framework::Tensor>("Out");
    bool transpose_x = ctx.Attr<bool>("transpose_X");
    bool transpose_y = ctx.Attr<bool>("transpose_Y");
    float alpha = static_cast<T>(ctx.Attr<float>("alpha"));

    std::vector<int64_t> x_dims = framework::vectorize(X->dims());
    std::vector<int64_t> y_dims = framework::vectorize(Y->dims());
    std::vector<int64_t> out_dims = framework::vectorize(Out->dims());
    int x_ndim = x_dims.size();
    int y_ndim = y_dims.size();
    int out_ndim = out_dims.size();

    auto stream = ctx.template device_context<NPUDeviceContext>().stream();

    // Case 1: [K] x [K] = [1]
    if (x_ndim == 1 && y_ndim == 1) {
      PADDLE_ENFORCE_EQ(
          X->numel(), Y->numel(),
          platform::errors::InvalidArgument(
              "X's numbers must be equal to Y's numbers,"
              "when X/Y's dims =1. But received X has [%d] elements,"
              "received Y has [%d] elements",
              X->numel(), Y->numel()));
      Out->Resize({1});
      Dot<T>(ctx, stream, *X, *Y, Out, alpha);
      return;
    }

    // Resize dim 1 to 2
    Tensor x_temp, y_temp;
    x_temp.ShareDataWith(*X);
    y_temp.ShareDataWith(*Y);
    if (x_ndim == 1) {
      x_dims.insert(x_dims.begin(), 1);
      out_dims.insert(out_dims.end() - 1, 1);
      x_temp.Resize(framework::make_ddim(x_dims));
      x_ndim = 2;
      out_ndim += 1;
    }
    if (y_ndim == 1) {
      y_dims.push_back(1);
      out_dims.push_back(1);
      y_temp.Resize(framework::make_ddim(y_dims));
      y_ndim = 2;
      out_ndim += 1;
    }

    const int K = transpose_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
    if (transpose_y) {
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

    // Case 2: [M, K] x [K, N] = [M, N]
    if (x_ndim == 2 && y_ndim == 2) {
      MatMul2D<T>(ctx, stream, x_temp, y_temp, Out, transpose_x, transpose_y,
                  alpha);
      return;
    }

    // Case 3: [B, M, K] x [K, N] =  [B, M, N], when transpose_x = false
    // Equal: [B * M, K] x [K, N] = [B * M, N] => [B, M, N]
    if (transpose_x == false && y_ndim == 2) {
      std::vector<int64_t> vec_dim = {x_temp.numel() / K, K};
      x_temp.Resize(framework::make_ddim(vec_dim));
      MatMul2D<T>(ctx, stream, x_temp, y_temp, Out, transpose_x, transpose_y,
                  alpha);
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
      x_temp_brd.Resize(framework::make_ddim(x_broadcast_dims));
    } else {
      x_temp_brd.Resize(framework::make_ddim(x_broadcast_dims));
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
      y_temp_brd.Resize(framework::make_ddim(y_broadcast_dims));
    } else {
      y_temp_brd.Resize(framework::make_ddim(y_broadcast_dims));
      y_temp_brd.mutable_data<T>(ctx.GetPlace());
      NpuOpRunner runner_brd;
      runner_brd.SetType("BroadcastTo")
          .AddInput(y_temp)
          .AddInput(std::move(y_broadcast_dims))
          .AddOutput(y_temp_brd)
          .Run(stream);
    }
    MatMulND<T>(ctx, stream, x_temp_brd, y_temp_brd, Out, transpose_x,
                transpose_y, alpha);
  }
};

template <typename DeviceContext, typename T>
class MatMulGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<framework::Tensor>("X");
    auto* Y = ctx.Input<framework::Tensor>("Y");
    auto* dOut = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dX = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dY = ctx.Output<framework::Tensor>(framework::GradVarName("Y"));
    bool transpose_x = ctx.Attr<bool>("transpose_X");
    bool transpose_y = ctx.Attr<bool>("transpose_Y");
    float alpha = static_cast<T>(ctx.Attr<float>("alpha"));

    std::vector<int64_t> x_dims = framework::vectorize(X->dims());
    std::vector<int64_t> y_dims = framework::vectorize(Y->dims());
    std::vector<int64_t> out_dims = framework::vectorize(dOut->dims());
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
        Mul<T>(ctx, stream, dout_temp, *Y, dX, alpha);
      }
      if (dY) {
        Mul<T>(ctx, stream, dout_temp, *X, dY, alpha);
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
      x_temp.Resize(framework::make_ddim(x_dims));
      dout_temp.Resize(framework::make_ddim(out_dims));
      x_ndim = 2;
      out_ndim += 1;
    }
    if (y_ndim == 1) {
      y_dims.push_back(1);
      out_dims.push_back(1);
      y_temp.Resize(framework::make_ddim(y_dims));
      dout_temp.Resize(framework::make_ddim(out_dims));
      y_ndim = 2;
      out_ndim += 1;
    }

    // Case 2: [M, K] x [K, N] = [M, N]
    if (out_ndim == 2) {
      if (dX) {
        dX->Resize(framework::make_ddim(x_dims));
        if (transpose_x) {
          MatMul2D<T>(ctx, stream, y_temp, dout_temp, dX, transpose_y, true,
                      alpha);
        } else {
          MatMul2D<T>(ctx, stream, dout_temp, y_temp, dX, false, !transpose_y,
                      alpha);
        }
        dX->Resize(X->dims());
      }
      if (dY) {
        dY->Resize(framework::make_ddim(y_dims));
        if (transpose_y) {
          MatMul2D<T>(ctx, stream, dout_temp, x_temp, dY, true, transpose_x,
                      alpha);
        } else {
          MatMul2D<T>(ctx, stream, x_temp, dout_temp, dY, !transpose_x, false,
                      alpha);
        }
        dY->Resize(Y->dims());
      }
      return;
    }

    const int K = transpose_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
    const int N = transpose_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];

    // Case 3: [B, M, K] x [K, N] =  [B, M, N], when transpose_x = false
    // Equal: [B * M, K] x [K, N] = [B * M, N] => [B, M, N]
    if (transpose_x == false && y_ndim == 2) {
      std::vector<int64_t> x_vec_dim = {x_temp.numel() / K, K};
      dout_temp.Resize(
          framework::make_ddim(std::vector<int64_t>{dout_temp.numel() / N, N}));
      if (dX) {
        dX->Resize(framework::make_ddim(x_vec_dim));
        MatMul2D<T>(ctx, stream, dout_temp, y_temp, dX, false, !transpose_y,
                    alpha);
        dX->Resize(X->dims());
      }
      if (dY) {
        x_temp.Resize(framework::make_ddim(x_vec_dim));
        if (transpose_y) {
          MatMul2D<T>(ctx, stream, dout_temp, x_temp, dY, true, false, alpha);
        } else {
          MatMul2D<T>(ctx, stream, x_temp, dout_temp, dY, true, false, alpha);
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
      x_temp_brd.Resize(framework::make_ddim(x_broadcast_dims));
    } else {
      x_temp_brd.Resize(framework::make_ddim(x_broadcast_dims));
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
      y_temp_brd.Resize(framework::make_ddim(y_broadcast_dims));
    } else {
      y_temp_brd.Resize(framework::make_ddim(y_broadcast_dims));
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
        if (transpose_x) {
          MatMulND<T>(ctx, stream, y_temp_brd, dout_temp, dX, transpose_y, true,
                      alpha);
        } else {
          MatMulND<T>(ctx, stream, dout_temp, y_temp_brd, dX, false,
                      !transpose_y, alpha);
        }
      } else {
        Tensor dx_temp(X->type());
        dx_temp.Resize(framework::make_ddim(x_broadcast_dims));
        if (transpose_x) {
          MatMulND<T>(ctx, stream, y_temp_brd, dout_temp, &dx_temp, transpose_y,
                      true, alpha);
        } else {
          MatMulND<T>(ctx, stream, dout_temp, y_temp_brd, &dx_temp, false,
                      !transpose_y, alpha);
        }
        ReduceDims<T>(ctx, stream, x_dims, x_broadcast_dims, dx_temp, dX);
      }
    }
    if (dY) {
      if (y_dims == y_broadcast_dims) {
        if (transpose_y) {
          MatMulND<T>(ctx, stream, dout_temp, x_temp_brd, dY, true, transpose_x,
                      alpha);
        } else {
          MatMulND<T>(ctx, stream, x_temp_brd, dout_temp, dY, !transpose_x,
                      false, alpha);
        }
      } else {
        Tensor dy_temp(Y->type());
        dy_temp.Resize(framework::make_ddim(y_broadcast_dims));
        if (transpose_y) {
          MatMulND<T>(ctx, stream, dout_temp, x_temp_brd, &dy_temp, true,
                      transpose_x, alpha);
        } else {
          MatMulND<T>(ctx, stream, x_temp_brd, dout_temp, &dy_temp,
                      !transpose_x, false, alpha);
        }
        ReduceDims<T>(ctx, stream, y_dims, y_broadcast_dims, dy_temp, dY);
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    matmul, ops::MatMulNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::MatMulNPUKernel<paddle::platform::NPUDeviceContext,
                         paddle::platform::float16>);
REGISTER_OP_NPU_KERNEL(
    matmul_grad,
    ops::MatMulGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::MatMulGradNPUKernel<paddle::platform::NPUDeviceContext,
                             paddle::platform::float16>);
