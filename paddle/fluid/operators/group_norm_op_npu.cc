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

#include <vector>

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename T>
struct GroupNormFunction {
 public:
  explicit GroupNormFunction(const framework::ExecutionContext& ctx)
      : ctx(ctx) {
    place = ctx.GetPlace();
    stream = ctx.template device_context<paddle::platform::NPUDeviceContext>()
                 .stream();
  }
  void ReduceMean(const phi::DenseTensor* x,
                  phi::DenseTensor* y,
                  const std::vector<int>& dim,
                  bool keep_dims = true) {
    //  y should be init first
    const auto& runner = NpuOpRunner(
        "ReduceMeanD", {*x}, {*y}, {{"axes", dim}, {"keep_dims", keep_dims}});
    runner.Run(stream);
  }
  void ReduceSum(const phi::DenseTensor* x,
                 phi::DenseTensor* y,
                 const std::vector<int>& dim,
                 bool keep_dims = true) {
    //  y should be init first
    const auto& runner = NpuOpRunner(
        "ReduceSumD", {*x}, {*y}, {{"axes", dim}, {"keep_dims", keep_dims}});
    runner.Run(stream);
  }
  void Add(const phi::DenseTensor* x,
           const phi::DenseTensor* y,
           phi::DenseTensor* z) {
    //  y should be init first
    const auto& runner = NpuOpRunner("AddV2", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void Sub(const phi::DenseTensor* x,
           const phi::DenseTensor* y,
           phi::DenseTensor* z) {
    //  y should be init first
    const auto& runner = NpuOpRunner("Sub", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void Mul(const phi::DenseTensor* x,
           const phi::DenseTensor* y,
           phi::DenseTensor* z) {
    //  y should be init first
    const auto& runner = NpuOpRunner("Mul", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void Div(const phi::DenseTensor* x,
           const phi::DenseTensor* y,
           phi::DenseTensor* z) {
    //  y should be init first
    const auto& runner = NpuOpRunner("Div", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void DivNoNan(const phi::DenseTensor* x,
                const phi::DenseTensor* y,
                phi::DenseTensor* z) {
    //  y should be init first
    const auto& runner = NpuOpRunner("DivNoNan", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void Transpose(const phi::DenseTensor* x,
                 phi::DenseTensor* y,
                 const std::vector<int>& axis) {
    //  y should be init first
    const auto& runner =
        NpuOpRunner("TransposeD", {*x}, {*y}, {{"perm", axis}});
    runner.Run(stream);
  }
  void Sqrt(const phi::DenseTensor* x, phi::DenseTensor* y) {
    //  y should be init first
    const auto& runner = NpuOpRunner("Sqrt", {*x}, {*y}, {});
    runner.Run(stream);
  }
  void Adds(const phi::DenseTensor* x, float scalar, phi::DenseTensor* y) {
    //  y should be init first
    const auto& runner = NpuOpRunner("Adds", {*x}, {*y}, {{"value", scalar}});
    runner.Run(stream);
  }
  phi::DenseTensor ReduceMeanToNG(const phi::DenseTensor* x,
                                  const DataLayout& data_layout,
                                  const int64_t N,
                                  const int64_t C,
                                  const int64_t H,
                                  const int64_t W,
                                  const int G) {
    phi::DenseTensor y(x->type());
    // y.mutable_data<T>( {N,G,1}, place );
    if (data_layout == DataLayout::kNCHW) {
      y.mutable_data<T>({N, G, 1}, place);
      //  shape of x is [N, G, C*H*W/G]
      this->ReduceMean(x, &y, std::vector<int>{2});
    } else {
      y.mutable_data<T>({N, 1, G}, place);
      //  shape of x is [N, C*H*W/G, G]
      phi::DenseTensor x_trans(x->type());
      x_trans.mutable_data<T>({N, G, C * H * W / G}, place);
      this->Transpose(x, &x_trans, std::vector<int>{0, 2, 1});
      this->ReduceMean(&x_trans, &y, std::vector<int>{2});
    }
    return y;
  }

 private:
  platform::Place place;
  aclrtStream stream;
  const framework::ExecutionContext& ctx;
};

template <typename T>
class GroupNormNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout = phi::StringToDataLayout(data_layout_str);
    const float epsilon = ctx.Attr<float>("epsilon");
    auto* scale = ctx.Input<phi::DenseTensor>("Scale");
    auto* bias = ctx.Input<phi::DenseTensor>("Bias");
    auto* x = ctx.Input<phi::DenseTensor>("X");

    auto* y = ctx.Output<phi::DenseTensor>("Y");
    auto* mean = ctx.Output<phi::DenseTensor>("Mean");
    auto* var = ctx.Output<phi::DenseTensor>("Variance");
    const auto groups = ctx.Attr<int>("groups");

    auto place = ctx.GetPlace();
    phi::DenseTensor xnorm(x->type());
    xnorm.mutable_data<T>(x->dims(), place);
    GroupNormFunction<T> F(ctx);
    if (data_layout != DataLayout::kNCHW) {
      xnorm.Resize({x->dims()[0], x->dims()[3], x->dims()[1], x->dims()[2]});
      F.Transpose(x, &xnorm, std::vector<int>{0, 3, 1, 2});
    } else {
      paddle::framework::TensorCopy(*x, platform::NPUPlace(), &xnorm);
    }
    auto N = xnorm.dims()[0];
    auto C = xnorm.dims()[1];
    auto H = xnorm.dims()[2];
    auto W = xnorm.dims()[3];
    xnorm.Resize({N * groups, C * H * W / groups});
    std::vector<int> axis = {1};
    auto reduce_dim = mean->dims();

    mean->mutable_data<T>({N * groups, 1}, place);
    var->mutable_data<T>({N * groups, 1}, place);
    y->mutable_data<T>(place);
    F.ReduceMean(&xnorm, mean, axis);

    F.Sub(&xnorm, mean, &xnorm);
    phi::DenseTensor sqr(x->type());
    sqr.mutable_data<T>(xnorm.dims(), place);

    F.Mul(&xnorm, &xnorm, &sqr);
    F.ReduceMean(&sqr, var, axis);
    phi::DenseTensor std(x->type());
    std.mutable_data<T>(var->dims(), place);
    F.Adds(var, epsilon, &std);
    F.Sqrt(&std, &std);
    y->Resize(xnorm.dims());
    F.Div(&xnorm, &std, y);
    y->Resize({N, C, H, W});
    if (scale) {
      phi::DenseTensor scale_t(scale->type());
      scale_t.ShareDataWith(*scale);
      scale_t.Resize({C, 1, 1});
      F.Mul(y, &scale_t, y);
    }
    if (bias) {
      phi::DenseTensor bias_t(bias->type());
      bias_t.ShareDataWith(*bias);
      bias_t.Resize({C, 1, 1});
      F.Add(y, &bias_t, y);
    }
    if (data_layout != DataLayout::kNCHW) {
      F.Transpose(y, y, std::vector<int>{0, 2, 3, 1});
      y->Resize({x->dims()});
    }
    mean->Resize(reduce_dim);
    var->Resize(reduce_dim);
  }
};

template <typename T>
class GroupNormGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout = phi::StringToDataLayout(data_layout_str);
    const float epsilon = ctx.Attr<float>("epsilon");
    auto* y = ctx.Input<phi::DenseTensor>("Y");
    auto* var = ctx.Input<phi::DenseTensor>("Variance");

    auto* scale = ctx.Input<phi::DenseTensor>("Scale");
    auto* bias = ctx.Input<phi::DenseTensor>("Bias");
    auto* d_y = ctx.Input<phi::DenseTensor>(framework::GradVarName("Y"));
    const auto G = ctx.Attr<int>("groups");

    // init output
    auto* d_x = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* d_scale =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Scale"));
    auto* d_bias = ctx.Output<phi::DenseTensor>(framework::GradVarName("Bias"));

    GroupNormFunction<T> F(ctx);
    auto place = ctx.GetPlace();
    auto _type = y->type();

    phi::DenseTensor xnorm(_type);
    xnorm.mutable_data<T>(y->dims(), place);
    phi::DenseTensor scale_share(_type);
    scale_share.ShareDataWith(*scale);
    phi::DenseTensor bias_share(_type);
    bias_share.ShareDataWith(*bias);

    int64_t N = y->dims()[0];
    int64_t C, H, W;
    framework::DDim scale_bias_dim;
    if (data_layout == DataLayout::kNCHW) {
      C = y->dims()[1];
      H = y->dims()[2];
      W = y->dims()[3];
      scale_bias_dim = phi::make_ddim({C, 1, 1});
    } else {
      C = y->dims()[3];
      H = y->dims()[1];
      W = y->dims()[2];
      scale_bias_dim = phi::make_ddim({1, 1, C});
    }
    scale_share.Resize(scale_bias_dim);
    bias_share.Resize(scale_bias_dim);
    F.Sub(y, &bias_share, &xnorm);
    F.DivNoNan(&xnorm, &scale_share, &xnorm);

    if (d_bias) {
      d_bias->mutable_data<T>(place);
      if (data_layout == DataLayout::kNCHW) {
        F.ReduceSum(d_y, d_bias, std::vector<int>{0, 2, 3}, false);
      } else {
        F.ReduceSum(d_y, d_bias, std::vector<int>{0, 1, 2}, false);
      }
    }
    if (d_scale) {
      d_scale->mutable_data<T>(place);
      phi::DenseTensor dy_xnorm(_type);
      dy_xnorm.mutable_data<T>(d_y->dims(), place);
      F.Mul(d_y, &xnorm, &dy_xnorm);
      if (data_layout == DataLayout::kNCHW) {
        F.ReduceSum(&dy_xnorm, d_scale, std::vector<int>{0, 2, 3});
      } else {
        F.ReduceSum(&dy_xnorm, d_scale, std::vector<int>{0, 1, 2});
      }
    }

    //  std = Sqrt(var+epsilon), init shape = [ N, G ]
    phi::DenseTensor std(_type);
    std.mutable_data<T>(var->dims(), place);
    F.Adds(var, epsilon, &std);
    F.Sqrt(&std, &std);
    //  d_xnorm_std = dy_proc * scale / std
    phi::DenseTensor d_xnorm_std(_type);
    d_xnorm_std.mutable_data<T>(y->dims(), place);
    F.Mul(d_y, &scale_share, &d_xnorm_std);
    if (data_layout == DataLayout::kNCHW) {
      xnorm.Resize({N, G, C * H * W / G});
      d_xnorm_std.Resize({N, G, C * H * W / G});
      std.Resize({N, G, 1});
    } else {
      xnorm.Resize({N, C * H * W / G, G});
      d_xnorm_std.Resize({N, C * H * W / G, G});
      std.Resize({N, 1, G});
    }
    F.Div(&d_xnorm_std, &std, &d_xnorm_std);

    //  d_x = d_xnorm_std
    //       - Mean ( d_xnorm_std * x_norm, axis=1, keepdim=True ) * x_norm
    //       - Mean ( d_xnorm_std, axis=1, keepdim=True )
    d_x->mutable_data<T>(place);
    d_x->Resize(xnorm.dims());
    F.Mul(&d_xnorm_std, &xnorm, d_x);
    phi::DenseTensor dx1 = F.ReduceMeanToNG(d_x, data_layout, N, C, H, W, G);
    F.Mul(&dx1, &xnorm, d_x);

    phi::DenseTensor dx2 =
        F.ReduceMeanToNG(&d_xnorm_std, data_layout, N, C, H, W, G);

    F.Sub(&d_xnorm_std, d_x, d_x);
    F.Sub(d_x, &dx2, d_x);

    d_x->Resize(y->dims());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(group_norm,
                       ops::GroupNormNPUKernel<float>,
                       ops::GroupNormNPUKernel<plat::float16>);
REGISTER_OP_NPU_KERNEL(group_norm_grad,
                       ops::GroupNormGradNPUKernel<float>,
                       ops::GroupNormGradNPUKernel<plat::float16>);
