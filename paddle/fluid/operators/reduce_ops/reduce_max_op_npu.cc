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

#include "paddle/fluid/operators/reduce_ops/reduce_min_max_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class ReduceMaxNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* out = ctx.Output<phi::DenseTensor>("Out");
    auto dims = ctx.Attr<std::vector<int>>("dim");
    bool keep_dim = ctx.Attr<bool>("keep_dim");
    bool reduce_all = ctx.Attr<bool>("reduce_all");
    int out_dtype = ctx.Attr<int>("out_dtype");

    auto place = ctx.GetPlace();

    phi::DenseTensor cast_out(x->type());
    cast_out.Resize(out->dims());
    cast_out.mutable_data<T>(place);

    auto cast_out_dtype = framework::TransToProtoVarType(x->dtype());
    if (out_dtype != -1) {
      cast_out_dtype = static_cast<framework::proto::VarType::Type>(out_dtype);
    }

    if (framework::TransToProtoVarType(x->dtype()) != cast_out_dtype) {
      if (cast_out_dtype == framework::proto::VarType::FP32) {
        out->mutable_data<float>(place);
      } else if (cast_out_dtype == framework::proto::VarType::FP16) {
        out->mutable_data<paddle::platform::float16>(place);
      } else if (cast_out_dtype == framework::proto::VarType::INT16) {
        out->mutable_data<int16_t>(place);
      } else if (cast_out_dtype == framework::proto::VarType::INT32) {
        out->mutable_data<int32_t>(place);
      } else if (cast_out_dtype == framework::proto::VarType::INT64) {
        out->mutable_data<int64_t>(place);
      } else if (cast_out_dtype == framework::proto::VarType::FP64) {
        out->mutable_data<double>(place);
      } else if (cast_out_dtype == framework::proto::VarType::BOOL) {
        out->mutable_data<bool>(place);
      }
    } else {
      out->ShareDataWith(cast_out);
    }

    framework::NPUAttributeMap attr_input = {{"axes", dims},
                                             {"keep_dims", keep_dim}};

    if (reduce_all) {
      std::vector<int> dim_vec;
      for (int i = 0; i < x->dims().size(); i++) {
        dim_vec.push_back(i);
      }

      attr_input = {{"axes", dim_vec}, {"keep_dims", keep_dim}};
    }

    const auto& dev_ctx =
        ctx.template device_context<paddle::platform::NPUDeviceContext>();
    if (framework::TransToProtoVarType(x->dtype()) ==
        framework::proto::VarType::INT64) {
      auto op_func = [](const std::vector<phi::DenseTensor>& inputs,
                        const std::vector<phi::DenseTensor>& outputs,
                        const NPUAttributeMap& attrs,
                        const platform::NPUDeviceContext& dev_ctx) {
        const auto& runner =
            NpuOpRunner("ReduceMaxD", {inputs[0]}, {outputs[0]}, attrs);
        runner.Run(dev_ctx.stream());
      };

      NpuOpRunner::TypeAdapter({*x},
                               {cast_out},
                               attr_input,
                               dev_ctx,
                               op_func,
                               {framework::proto::VarType::INT32},
                               {framework::proto::VarType::INT32});
    } else {
      const auto& runner =
          NpuOpRunner("ReduceMaxD", {*x}, {cast_out}, attr_input);
      runner.Run(dev_ctx.stream());
    }

    if (framework::TransToProtoVarType(x->dtype()) != cast_out_dtype) {
      auto dst_dtype = ConvertToNpuDtype(cast_out_dtype);
      const auto& runner_cast =
          NpuOpRunner("Cast",
                      {cast_out},
                      {*out},
                      {{"dst_type", static_cast<int>(dst_dtype)}});
      runner_cast.Run(dev_ctx.stream());
    }
  }
};

template <typename DeviceContext, typename T>
class ReduceMaxGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<phi::DenseTensor>("X");
    auto* out = context.Input<phi::DenseTensor>("Out");
    auto* out_grad =
        context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto reduce_dims = context.Attr<std::vector<int>>("dim");
    bool reduce_all = context.Attr<bool>("reduce_all");
    int in_dtype = context.Attr<int>("in_dtype");

    PADDLE_ENFORCE_EQ(
        in_dtype == -1,
        true,
        platform::errors::InvalidArgument(
            "NPU only support in_dtype == -1 in reduce_max_grad op."));

    auto* x_grad =
        context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    x_grad->mutable_data<T>(context.GetPlace());

    auto& dev_ctx =
        context.template device_context<paddle::platform::NPUDeviceContext>();
    auto place = context.GetPlace();
    auto stream = dev_ctx.stream();

    // broadcast
    auto x_dims_vec = phi::vectorize(x->dims());
    if (reduce_all) {
      reduce_dims.clear();
      for (size_t d = 0; d < x_dims_vec.size(); ++d) {
        reduce_dims.push_back(static_cast<int>(d));
      }
    }

    phi::DenseTensor tmp_out, tmp_out_grad;
    auto tmp_out_dims_vec = x_dims_vec;
    for (auto d : reduce_dims) {
      if (d < 0) {
        d += x_dims_vec.size();
      }
      tmp_out_dims_vec[d] = 1;
    }

    tmp_out.ShareDataWith(*out);
    tmp_out.Resize(phi::make_ddim(tmp_out_dims_vec));
    tmp_out_grad.ShareDataWith(*out_grad);
    tmp_out_grad.Resize(phi::make_ddim(tmp_out_dims_vec));

    phi::DenseTensor transformed_out(x->type());
    transformed_out.Resize(phi::make_ddim(x_dims_vec));
    transformed_out.mutable_data<T>(place);
    NpuOpRunner r_brd_out;
    r_brd_out.SetType("BroadcastTo")
        .AddInput(tmp_out)
        .AddInput(std::move(x_dims_vec))
        .AddOutput(transformed_out)
        .Run(stream);
    phi::DenseTensor transformed_out_grad(x->type());
    transformed_out_grad.Resize(phi::make_ddim(x_dims_vec));
    transformed_out_grad.mutable_data<T>(place);
    NpuOpRunner r_brd_out_grad;
    r_brd_out_grad.SetType("BroadcastTo")
        .AddInput(tmp_out_grad)
        .AddInput(std::move(x_dims_vec))
        .AddOutput(transformed_out_grad)
        .Run(stream);

    // compare
    phi::DenseTensor equal_cond;
    equal_cond.mutable_data<bool>(x_grad->dims(), place);
    const auto& r_equal =
        NpuOpRunner("Equal", {*x, transformed_out}, {equal_cond}, {});
    r_equal.Run(stream);

    // select
    phi::DenseTensor t_zero;
    t_zero.mutable_data<T>(x_grad->dims(), place);
    FillNpuTensorWithConstant(&t_zero, static_cast<T>(0));
    t_zero.Resize(x_grad->dims());

    const auto& r_sel = NpuOpRunner(
        "SelectV2", {equal_cond, transformed_out_grad, t_zero}, {*x_grad}, {});
    r_sel.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(
    reduce_max,
    ops::ReduceMaxNPUKernel<plat::NPUDeviceContext, float>,
    ops::ReduceMaxNPUKernel<plat::NPUDeviceContext, plat::float16>,
    ops::ReduceMaxNPUKernel<plat::NPUDeviceContext, int64_t>,
    ops::ReduceMaxNPUKernel<plat::NPUDeviceContext, int>);
REGISTER_OP_NPU_KERNEL(
    reduce_max_grad,
    ops::ReduceMaxGradNPUKernel<plat::NPUDeviceContext, float>,
    ops::ReduceMaxGradNPUKernel<plat::NPUDeviceContext, plat::float16>,
    ops::ReduceMaxGradNPUKernel<plat::NPUDeviceContext, int64_t>,
    ops::ReduceMaxGradNPUKernel<plat::NPUDeviceContext, int>);
