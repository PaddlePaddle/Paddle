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
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename T>
class MaskedSelectedNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto input = ctx.Input<phi::DenseTensor>("X");
    auto mask = ctx.Input<phi::DenseTensor>("Mask");
    auto out = ctx.Output<phi::DenseTensor>("Y");

    auto input_dim = input->dims();
    auto mask_dim = mask->dims();
    PADDLE_ENFORCE_EQ(
        input_dim,
        mask_dim,
        platform::errors::InvalidArgument(
            "The dim size of input and mask in OP(masked_selected) "
            "must be equal, but got input dim:(%ld), mask dim: "
            "(%ld). Please check input "
            "value.",
            input_dim,
            mask_dim));

    auto& dev_ctx =
        ctx.template device_context<paddle::platform::NPUDeviceContext>();
    auto stream = dev_ctx.stream();

    Tensor mask_int32, out_size;
    std::vector<int32_t> out_size_vec;
    mask_int32.mutable_data<int32_t>(mask->dims(), ctx.GetPlace());
    out_size.mutable_data<int32_t>({1}, ctx.GetPlace());
    {
      const auto& cast_runner = NpuOpRunner(
          "Cast",
          {*mask},
          {mask_int32},
          {{"dst_type",
            static_cast<int32_t>(
                ConvertToNpuDtype(framework::proto::VarType::INT32))}});
      cast_runner.Run(stream);

      mask_int32.Resize({mask_int32.numel()});
      NpuOpRunner sum_runner;
      sum_runner.SetType("ReduceSum");
      sum_runner.AddInput(mask_int32);
      sum_runner.AddInput(std::vector<int32_t>({0}));
      sum_runner.AddOutput(out_size);
      sum_runner.AddAttr("keep_dims", false);
      sum_runner.Run(stream);
      paddle::framework::TensorToVector(out_size, dev_ctx, &out_size_vec);
    }

    out->Resize({out_size_vec[0]});
    out->mutable_data<T>(ctx.GetPlace());

    Tensor topkv2_out, indices;
    topkv2_out.mutable_data<int32_t>({out_size_vec[0]}, ctx.GetPlace());
    indices.mutable_data<int32_t>({out_size_vec[0]}, ctx.GetPlace());
    {
      NpuOpRunner topkv2_runner;
      topkv2_runner.SetType("TopKV2")
          .AddInput(mask_int32)
          .AddInput(out_size)
          .AddOutput(topkv2_out)
          .AddOutput(indices)
          .AddAttr("sorted", false)
          .AddAttr("dim", 0)
          .AddAttr("largest", true)
          .Run(stream);
      // TopKV2 may be unstable
      NpuOpRunner topkv2_runner2;
      topkv2_runner2.SetType("TopKV2")
          .AddInput(indices)
          .AddInput(out_size)
          .AddOutput(topkv2_out)
          .AddOutput(indices)
          .AddAttr("sorted", true)
          .AddAttr("dim", 0)
          .AddAttr("largest", false)
          .Run(stream);

      Tensor input_tmp;
      input_tmp.ShareDataWith(*input);
      input_tmp.Resize({input->numel()});
      const auto& gather_runner = NpuOpRunner(
          "GatherV2D", {input_tmp, topkv2_out}, {*out}, {{"axis", 0}});
      gather_runner.Run(stream);
    }
  }
};

template <typename T>
class MaskedSelectedGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto mask = ctx.Input<phi::DenseTensor>("Mask");
    auto y_grad = ctx.Input<phi::DenseTensor>(framework::GradVarName("Y"));
    auto x_grad = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));

    x_grad->mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx =
        ctx.template device_context<paddle::platform::NPUDeviceContext>();
    auto stream = dev_ctx.stream();

    Tensor mask_int32, out_size;
    std::vector<int32_t> out_size_vec;
    mask_int32.mutable_data<int32_t>(mask->dims(), ctx.GetPlace());
    out_size.mutable_data<int32_t>({1}, ctx.GetPlace());
    {
      const auto& cast_runner = NpuOpRunner(
          "Cast",
          {*mask},
          {mask_int32},
          {{"dst_type",
            static_cast<int32_t>(
                ConvertToNpuDtype(framework::proto::VarType::INT32))}});
      cast_runner.Run(stream);

      mask_int32.Resize({mask_int32.numel()});
      NpuOpRunner sum_runner;
      sum_runner.SetType("ReduceSum");
      sum_runner.AddInput(mask_int32);
      sum_runner.AddInput(std::vector<int32_t>({0}));
      sum_runner.AddOutput(out_size);
      sum_runner.AddAttr("keep_dims", false);
      sum_runner.Run(stream);
      paddle::framework::TensorToVector(out_size, dev_ctx, &out_size_vec);
    }

    Tensor topkv2_out, indices;
    topkv2_out.mutable_data<int32_t>({out_size_vec[0]}, ctx.GetPlace());
    indices.mutable_data<int32_t>({out_size_vec[0]}, ctx.GetPlace());
    {
      NpuOpRunner topkv2_runner;
      topkv2_runner.SetType("TopKV2")
          .AddInput(mask_int32)
          .AddInput(out_size)
          .AddOutput(topkv2_out)
          .AddOutput(indices)
          .AddAttr("sorted", false)
          .AddAttr("dim", 0)
          .AddAttr("largest", true)
          .Run(stream);

      NpuOpRunner topkv2_runner2;
      topkv2_runner2.SetType("TopKV2")
          .AddInput(indices)
          .AddInput(out_size)
          .AddOutput(topkv2_out)
          .AddOutput(indices)
          .AddAttr("sorted", true)
          .AddAttr("dim", 0)
          .AddAttr("largest", false)
          .Run(stream);

      topkv2_out.Resize({out_size_vec[0], 1});
      x_grad->Resize({x_grad->numel()});
      NpuOpRunner scatter_runner;
      scatter_runner.SetType("ScatterNd");
      scatter_runner.AddInput(topkv2_out);
      scatter_runner.AddInput(*y_grad);
      scatter_runner.AddInput(
          std::vector<int32_t>({static_cast<int32_t>(x_grad->numel())}));
      scatter_runner.AddOutput(*x_grad);
      scatter_runner.Run(stream);
      x_grad->Resize(mask->dims());
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(masked_select,
                       ops::MaskedSelectedNPUKernel<plat::float16>,
                       ops::MaskedSelectedNPUKernel<float>,
                       ops::MaskedSelectedNPUKernel<int>,
                       ops::MaskedSelectedNPUKernel<int64_t>);
REGISTER_OP_NPU_KERNEL(masked_select_grad,
                       ops::MaskedSelectedGradNPUKernel<plat::float16>,
                       ops::MaskedSelectedGradNPUKernel<float>,
                       ops::MaskedSelectedGradNPUKernel<int>,
                       ops::MaskedSelectedGradNPUKernel<int64_t>);
