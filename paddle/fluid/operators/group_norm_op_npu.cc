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
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

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
  void DivNoNan(const Tensor* x, const Tensor* y, Tensor* z) {
    //  y should be init first
    const auto& runner = NpuOpRunner("DivNoNan", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void Transpose(const Tensor* x, Tensor* y, const std::vector<int>& axis) {
    //  y should be init first
    const auto& runner =
        NpuOpRunner("TransposeD", {*x}, {*y}, {{"perm", axis}});
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
  Tensor ReduceMeanToNG(const Tensor* x, const DataLayout& data_layout,
                        const int64_t N, const int64_t C, const int64_t H,
                        const int64_t W, const int G) {
    Tensor y(x->type());
    // y.mutable_data<T>( {N,G,1}, place );
    if (data_layout == DataLayout::kNCHW) {
      y.mutable_data<T>({N, G, 1}, place);
      //  shape of x is [N, G, C*H*W/G]
      this->ReduceMean(x, &y, std::vector<int>{2});
    } else {
      y.mutable_data<T>({N, 1, G}, place);
      //  shape of x is [N, C*H*W/G, G]
      Tensor x_trans(x->type());
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
    Tensor sqr(x->type());
    sqr.mutable_data<T>(xnorm.dims(), place);

    F.Mul(&xnorm, &xnorm, &sqr);
    F.ReduceMean(&sqr, var, axis);
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
    auto* y = ctx.Input<Tensor>("Y");
    auto* var = ctx.Input<Tensor>("Variance");

    auto* scale = ctx.Input<Tensor>("Scale");
    auto* bias = ctx.Input<Tensor>("Bias");
    auto* d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto G = ctx.Attr<int>("groups");

    // init output
    auto* d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto* d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    GroupNormFunction<T> F(ctx);
    auto place = ctx.GetPlace();
    auto _type = y->type();

    Tensor xnorm(_type);
    xnorm.mutable_data<T>(y->dims(), place);
    Tensor scale_share(_type);
    scale_share.ShareDataWith(*scale);
    Tensor bias_share(_type);
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
      Tensor dy_xnorm(_type);
      dy_xnorm.mutable_data<T>(d_y->dims(), place);
      F.Mul(d_y, &xnorm, &dy_xnorm);
      if (data_layout == DataLayout::kNCHW) {
        F.ReduceSum(&dy_xnorm, d_scale, std::vector<int>{0, 2, 3});
      } else {
        F.ReduceSum(&dy_xnorm, d_scale, std::vector<int>{0, 1, 2});
      }
    }

    //  std = Sqrt(var+epsilon), init shape = [ N, G ]
    Tensor std(_type);
    std.mutable_data<T>(var->dims(), place);
    F.Adds(var, epsilon, &std);
    F.Sqrt(&std, &std);
    //  d_xnorm_std = dy_proc * scale / std
    Tensor d_xnorm_std(_type);
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
    Tensor dx1 = F.ReduceMeanToNG(d_x, data_layout, N, C, H, W, G);
    F.Mul(&dx1, &xnorm, d_x);

    Tensor dx2 = F.ReduceMeanToNG(&d_xnorm_std, data_layout, N, C, H, W, G);

    F.Sub(&d_xnorm_std, d_x, d_x);
    F.Sub(d_x, &dx2, d_x);

    d_x->Resize(y->dims());
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
