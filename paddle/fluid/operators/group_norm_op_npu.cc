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

#include "paddle/fluid/operators/group_norm_op.h"
#include <vector>
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
struct GroupNormFunction {
 public:
  explicit GroupNormFunction(const framework::ExecutionContext& ctx)
      : ctx(ctx) {
    place = ctx.GetPlace();
    stream = ctx.template device_context<paddle::platform::NPUDeviceContext>()
                 .stream();
  }
  void ReduceMean(const Tensor* x, Tensor* y, const std::vector<int>& dim,
                  bool keep_dims = true) {
    //  y should be init first
    const auto& runner = NpuOpRunner("ReduceMeanD", {*x}, {*y},
                                     {{"axes", dim}, {"keep_dims", keep_dims}});
    runner.Run(stream);
  }
  void ReduceStdWithMean(const Tensor* x, const Tensor* mean, Tensor* y,
                         const std::vector<int>& dim, bool keep_dims = true) {
    //  y should be init first
    const auto& runner = NpuOpRunner(
        "ReduceStdWithMean", {*x, *mean}, {*y},
        {{"dim", dim}, {"unbiased", false}, {"keep_dims", keep_dims}});
    runner.Run(stream);
  }
  void ReduceSum(const Tensor* x, Tensor* y, const std::vector<int>& dim,
                 bool keep_dims = true) {
    //  y should be init first
    const auto& runner = NpuOpRunner("ReduceSumD", {*x}, {*y},
                                     {{"axes", dim}, {"keep_dims", keep_dims}});
    runner.Run(stream);
  }
  void Add(const Tensor* x, const Tensor* y, Tensor* z) {
    //  y should be init first
    const auto& runner = NpuOpRunner("AddV2", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void Sub(const Tensor* x, const Tensor* y, Tensor* z) {
    //  y should be init first
    const auto& runner = NpuOpRunner("Sub", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void Mul(const Tensor* x, const Tensor* y, Tensor* z) {
    //  y should be init first
    const auto& runner = NpuOpRunner("Mul", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void Div(const Tensor* x, const Tensor* y, Tensor* z) {
    //  y should be init first
    const auto& runner = NpuOpRunner("Div", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void Transpose(const Tensor* x, Tensor* y, const std::vector<int>& axis) {
    //  y should be init first
    const auto& runner =
        NpuOpRunner("TransposeD", {*x}, {*y}, {{"perm", axis}});
    runner.Run(stream);
  }
  void SquaredDifference(const Tensor* x, const Tensor* y, Tensor* z) {
    //  y should be init first
    const auto& runner = NpuOpRunner("SquaredDifference", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void Sqrt(const Tensor* x, Tensor* y) {
    //  y should be init first
    const auto& runner = NpuOpRunner("Sqrt", {*x}, {*y}, {});
    runner.Run(stream);
  }
  void Adds(const Tensor* x, float scalar, Tensor* y) {
    //  y should be init first
    const auto& runner = NpuOpRunner("Adds", {*x}, {*y}, {{"value", scalar}});
    runner.Run(stream);
  }
  void Muls(const Tensor* x, float scalar, Tensor* y) {
    //  y should be init first
    const auto& runner = NpuOpRunner("Muls", {*x}, {*y}, {{"value", scalar}});
    runner.Run(stream);
  }
  void Pow(const Tensor* x, float factor, Tensor* y) {
    //  y should be init first
    const auto& runner =
        NpuOpRunner("Power", {*x}, {*y}, {{"power", factor},
                                          {"scale", static_cast<float>(1.0)},
                                          {"shift", static_cast<float>(0.0)}});
    runner.Run(stream);
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
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);
    const float epsilon = ctx.Attr<float>("epsilon");
    auto* scale = ctx.Input<Tensor>("Scale");
    auto* bias = ctx.Input<Tensor>("Bias");
    auto* x = ctx.Input<Tensor>("X");

    auto* y = ctx.Output<Tensor>("Y");
    auto* mean = ctx.Output<Tensor>("Mean");
    auto* var = ctx.Output<Tensor>("Variance");
    const auto groups = ctx.Attr<int>("groups");

    auto place = ctx.GetPlace();
    Tensor xnorm(x->type());
    xnorm.mutable_data<T>(x->dims(), place);
    GroupNormFunction<T> F(ctx);
    if (data_layout != DataLayout::kNCHW) {
      xnorm.Resize({x->dims()[0], x->dims()[3], x->dims()[1], x->dims()[2]});
      F.Transpose(x, &xnorm, std::vector<int>{0, 3, 1, 2});
    } else {
      TensorCopy(*x, platform::NPUPlace(), &xnorm);
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
    Tensor sqr(x->type());
    sqr.mutable_data<T>(xnorm.dims(), place);

    F.Mul(&xnorm, &xnorm, &sqr);
    F.ReduceSum(&sqr, var, axis);
    F.Muls(var, static_cast<float>(1.0 / xnorm.dims()[1]), var);
    Tensor std(x->type());
    std.mutable_data<T>(var->dims(), place);
    F.Adds(var, epsilon, &std);
    F.Sqrt(&std, &std);
    y->Resize(xnorm.dims());
    F.Div(&xnorm, &std, y);
    y->Resize({N, C, H, W});
    if (scale) {
      Tensor scale_t(scale->type());
      scale_t.ShareDataWith(*scale);
      scale_t.Resize({C, 1, 1});
      F.Mul(y, &scale_t, y);
    }
    if (bias) {
      Tensor bias_t(bias->type());
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
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);
    const float epsilon = ctx.Attr<float>("epsilon");
    // auto* x = ctx.Input<Tensor>("Y");
    auto* x = ctx.Input<Tensor>("X");
    auto* var = ctx.Input<Tensor>("Variance");

    auto* meanx = ctx.Input<Tensor>("Mean");

    auto* scale = ctx.Input<Tensor>("Scale");
    // auto* bias = ctx.Input<Tensor>("Bias");
    auto* d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto groups = ctx.Attr<int>("groups");

    // init output
    auto* d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto* d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    // LOG(INFO) << bias;

    GroupNormFunction<T> F(ctx);
    auto place = ctx.GetPlace();
    Tensor dy_proc(d_y->type());
    Tensor x_proc(x->type());

    dy_proc.mutable_data<T>(d_y->dims(), place);
    x_proc.mutable_data<T>(x->dims(), place);
    if (data_layout != DataLayout::kNCHW) {
      x_proc.Resize({x->dims()[0], x->dims()[3], x->dims()[1], x->dims()[2]});
      dy_proc.Resize(x_proc.dims());
      F.Transpose(x, &x_proc, std::vector<int>{0, 3, 1, 2});
      F.Transpose(d_y, &dy_proc, std::vector<int>{0, 3, 1, 2});
    } else {
      TensorCopy(*x, platform::NPUPlace(), &x_proc);
      TensorCopy(*d_y, platform::NPUPlace(), &dy_proc);
    }
    auto N = x_proc.dims()[0];
    auto C = x_proc.dims()[1];
    auto H = x_proc.dims()[2];
    auto W = x_proc.dims()[3];

    std::vector<int> axis = {1};

    x_proc.Resize({N * groups, C * H * W / groups});

    //  mean = Mean(X)
    Tensor mean(x->type());

    mean.ShareDataWith(*meanx);
    mean.Resize({N * groups, 1});

    //  x_mean = x_proc - mean
    Tensor x_mean(x->type());
    x_mean.mutable_data<T>(x_proc.dims(), place);
    F.Sub(&x_proc, &mean, &x_mean);

    //  std = Sqrt(var+epsilon)
    Tensor std(x->type());
    std.mutable_data<T>(mean.dims(), place);
    Tensor var_share(x->type());
    var_share.ShareDataWith(*var);
    var_share.Resize(std.dims());
    F.Adds(&var_share, epsilon, &std);
    F.Sqrt(&std, &std);

    //  xnorm = x_mean / std
    Tensor xnorm(x->type());
    xnorm.mutable_data<T>(x_proc.dims(), place);
    F.Div(&x_mean, &std, &xnorm);

    //  d_xnorm = dy_proc * scale
    Tensor d_xnorm(x->type());
    d_xnorm.mutable_data<T>(dy_proc.dims(), place);
    Tensor scale_share(x->type());
    scale_share.ShareDataWith(*scale);
    scale_share.Resize({C, 1, 1});
    F.Mul(&dy_proc, &scale_share, &d_xnorm);
    d_xnorm.Resize(x_proc.dims());
    dy_proc.Resize(x_proc.dims());

    //  d_var = -0.5 * paddle.sum ( d_xnorm * x_mean, axis=1, keepdim=True ) / (
    //  std ** 3 )
    Tensor d_var(x->type());
    d_var.mutable_data<T>(mean.dims(), place);
    Tensor t_gxnorm_xmean(x->type());
    t_gxnorm_xmean.mutable_data<T>(x_proc.dims(), place);
    F.Mul(&d_xnorm, &x_mean, &t_gxnorm_xmean);
    F.ReduceSum(&t_gxnorm_xmean, &d_var, axis);
    Tensor std3(x->type());
    std3.mutable_data<T>(mean.dims(), place);
    F.Pow(&std, static_cast<float>(3), &std3);
    F.Div(&d_var, &std3, &d_var);
    F.Muls(&d_var, static_cast<float>(-0.5), &d_var);

    //  d_mean = -paddle.sum ( d_xnorm, axis=1, keepdim=True ) / std - 2 * dvar
    //  / x_norm.shape[1] * paddle.sum( x_mean, axis=1, keepdim=True )
    Tensor d_mean(x->type());
    d_mean.mutable_data<T>(mean.dims(), place);
    F.ReduceSum(&d_xnorm, &d_mean, axis);
    F.Div(&d_mean, &std, &d_mean);
    Tensor t_dvar_xmean(x->type());
    t_dvar_xmean.mutable_data<T>(mean.dims(), place);
    F.ReduceSum(&x_mean, &t_dvar_xmean, axis);
    F.Mul(&t_dvar_xmean, &d_var, &t_dvar_xmean);
    F.Muls(&t_dvar_xmean, static_cast<float>(2.0 / xnorm.dims()[1]),
           &t_dvar_xmean);
    F.Sub(&t_dvar_xmean, &d_mean, &d_mean);

    //  d_x = d_xnorm / std + 2 * d_var * x_mean / xnorm.shape[1] + d_mean /
    //  xnorm.shape[1]
    d_x->mutable_data<T>(place);
    d_x->Resize(x_proc.dims());
    F.Div(&d_xnorm, &std, d_x);
    F.Mul(&d_var, &x_mean, &x_mean);
    F.Muls(&x_mean, static_cast<float>(2.0 / xnorm.dims()[1]), &x_mean);
    F.Muls(&d_mean, static_cast<float>(1.0 / xnorm.dims()[1]), &d_mean);
    F.Add(d_x, &x_mean, d_x);
    F.Add(d_x, &d_mean, d_x);

    // d_x->mutable_data<T>(place);
    d_scale->mutable_data<T>(place);
    d_bias->mutable_data<T>(place);

    dy_proc.Resize({N, C, W * H});
    if (d_scale) {
      xnorm.Resize({N, C, W * H});
      Tensor d_scale_pre(x->type());
      d_scale_pre.mutable_data<T>({N, C, W * H}, place);
      F.Mul(&dy_proc, &xnorm, &d_scale_pre);
      F.ReduceSum(&d_scale_pre, d_scale, std::vector<int>{0, 2}, false);
    }

    if (d_bias) {
      F.ReduceSum(&dy_proc, d_bias, std::vector<int>{0, 2}, false);
    }
    if (data_layout != DataLayout::kNCHW) {
      d_x->Resize({N, C, H, W});
      F.Transpose(d_x, d_x, std::vector<int>{0, 2, 3, 1});
    }
    d_x->Resize({x->dims()});
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(group_norm, ops::GroupNormNPUKernel<float>,
                       ops::GroupNormNPUKernel<plat::float16>);
REGISTER_OP_NPU_KERNEL(group_norm_grad, ops::GroupNormGradNPUKernel<float>,
                       ops::GroupNormGradNPUKernel<plat::float16>);
