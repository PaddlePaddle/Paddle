/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licnse. */

#include <memory>
#include <string>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/dropout_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class DropoutNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    // auto* seed = ctx.Input<Tensor>("Seed");
    auto* out = ctx.Output<Tensor>("Out");
    auto* mask = ctx.Output<Tensor>("Mask");

    auto dropout_prob = ctx.Attr<float>("dropout_prob");
    auto is_test = ctx.Attr<bool>("is_test");
    // auto fix_seed = ctx.Attr<bool>("fix_seed");
    // auto seed = ctx.Attr<int>("seed");

    float keep_prob = 1. - dropout_prob;

    Tensor keep_prob_tensor(framework::proto::VarType::FP32);
    keep_prob_tensor.mutable_data<float>({1}, ctx.GetPlace());
    FillNpuTensorWithConstant<float>(&keep_prob_tensor, static_cast<float>(keep_prob));

    mask->mutable_data<T>(ctx.GetPlace());


    const auto& in_dims = x->dims();
    std::vector<int64_t> out_shape = framework::vectorize<int64_t>(in_dims);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // mask->mutable_data<uint8_t>(ctx.GetPlace());

    if (!is_test) {
        Tensor tmp_mask(framework::proto::VarType::UINT8);
        uint32_t length = (x->numel() + 128 - 1) / 128 * 128;
        tmp_mask.Resize(framework::make_ddim({length / 8}));
        tmp_mask.mutable_data<uint8_t>(ctx.GetPlace());

        int seed = 0;
        int seed2 = 0;

        NpuOpRunner runner_gen_mask;
        runner_gen_mask.SetType("DropOutGenMask")
                    .AddInput(framework::vectorize(mask->dims()))
                    .AddInput(keep_prob_tensor)
                    .AddOutput(tmp_mask)
                    .AddAttr("seed", seed)
                    .AddAttr("seed2", seed2);
        runner_gen_mask.Run(stream);

        out->mutable_data<T>(ctx.GetPlace());
        NpuOpRunner runner_dropout;
        runner_dropout.SetType("DropOutDoMask")
                    .AddInput(*x)
                    .AddInput(tmp_mask)
                    .AddInput(keep_prob_tensor)
                    .AddOutput(*out);
        runner_dropout.Run(stream);

        // cast seed from int to float32
        Tensor cast_mask(framework::proto::VarType::BOOL);
        cast_mask.Resize(mask->dims());
        cast_mask.mutable_data<bool>(ctx.GetPlace());
        auto dst_dtype = ConvertToNpuDtype(cast_mask.type());
        const auto& runner_cast_mask =
            NpuOpRunner("Cast", {*out}, {cast_mask},
                        {{"dst_type", static_cast<int>(dst_dtype)}});
        runner_cast_mask.Run(stream);

        // cast mask from int32 to float
        auto dst_dtype2 = ConvertToNpuDtype(mask->type());
        const auto& runner_cast_mask2 =
            NpuOpRunner("Cast", {cast_mask}, {*mask},
                        {{"dst_type", static_cast<int>(dst_dtype2)}});
        runner_cast_mask2.Run(stream);
    }

    

    /*auto* x = ctx.Input<Tensor>("X");
    auto* seed = ctx.Input<Tensor>("Seed");
    auto* out = ctx.Output<Tensor>("Out");
    auto* mask = ctx.Output<Tensor>("Mask");

    auto dropout_prob = ctx.Attr<float>("dropout_prob");
    // auto is_test = ctx.Attr<bool>("is_test");
    // auto fix_seed = ctx.Attr<bool>("fix_seed");
    // auto seed = ctx.Attr<int>("seed");

    // float keep_prob = 1. - dropout_prob;


    // const auto& in_dims = input->dims();

    mask->mutable_data<T>(ctx.GetPlace());
    out->mutable_data<T>(ctx.GetPlace());

    // std::vector<int> offsets(in_dims.size());
    // std::vector<int> size(in_dims.size());

    // UpdateAttr(in_dims, axes, starts, ends, &offsets, &size);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // cast seed from int to float32
    Tensor cast_seed(framework::proto::VarType::FP32);
    cast_seed.Resize(seed->dims());
    cast_seed.mutable_data<float>(ctx.GetPlace());
    auto dst_dtype = ConvertToNpuDtype(cast_seed.type());
    const auto& runner_cast_seed =
        NpuOpRunner("Cast", {*seed}, {cast_seed},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_seed.Run(stream);
    
    const auto& runner_dropout = NpuOpRunner("DropoutV2", {*x, cast_seed}, {*out, *mask, cast_seed},
                                     {{"p", dropout_prob}});
    runner_dropout.Run(stream);*/
  }
};

template <typename DeviceContext, typename T>
class DropoutGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* mask = ctx.Input<Tensor>("Mask");

    auto dropout_prob = ctx.Attr<float>("dropout_prob");

    dx->mutable_data<T>(ctx.GetPlace());

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner = NpuOpRunner("MaskedScale", {*dout, *mask}, {*dx},
                                     {{"value", static_cast<float>(1./(1-dropout_prob))}});
    runner.Run(stream);


    /*
    auto* input = ctx.Input<Tensor>("Input");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dinput = ctx.Output<Tensor>(framework::GradVarName("Input"));

    auto axes = ctx.Attr<std::vector<int>>("axes");
    auto starts = ctx.Attr<std::vector<int>>("starts");
    auto ends = ctx.Attr<std::vector<int>>("ends");
    const auto& in_dims = input->dims();
    int rank = in_dims.size();

    std::vector<int> offsets(rank);
    std::vector<int> size(rank);
    UpdateAttr(in_dims, axes, starts, ends, &offsets, &size);

    std::vector<std::vector<int64_t>> paddings(rank, std::vector<int64_t>(2));
    for (int i = 0; i < rank; ++i) {
      paddings[i][0] = static_cast<int64_t>(offsets[i]);
      paddings[i][1] = static_cast<int64_t>(in_dims[i] - size[i] - offsets[i]);
    }

    dinput->mutable_data<T>(ctx.GetPlace());
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    const auto& runner =
        NpuOpRunner("PadD", {*dout}, {*dinput}, {{"paddings", paddings}});
    runner.Run(stream);*/
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    dropout, ops::DropoutNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::DropoutNPUKernel<paddle::platform::NPUDeviceContext,
                        paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    dropout_grad,
    ops::DropoutGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::DropoutGradNPUKernel<paddle::platform::NPUDeviceContext,
                            paddle::platform::float16>);

