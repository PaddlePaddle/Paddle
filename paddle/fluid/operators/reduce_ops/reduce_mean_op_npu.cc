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
#include "paddle/fluid/operators/elementwise/elementwise_npu.h"
#include "paddle/fluid/operators/reduce_ops/reduce_mean_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename T>
class NPUReduceMeanOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("X");
    auto* output = ctx.Output<phi::DenseTensor>("Out");
    output->mutable_data<T>(ctx.GetPlace());

    bool reduce_all = ctx.Attr<bool>("reduce_all");
    auto dims = ctx.Attr<std::vector<int>>("dim");
    bool keep_dim = ctx.Attr<bool>("keep_dim");

    auto input_dims = input->dims();
    if (reduce_all) {
      dims.clear();
      for (int i = 0; i < input_dims.size(); i++) {
        dims.push_back(static_cast<int>(i));
      }
    }

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    NpuOpRunner runner;
    runner.SetType("ReduceMean")
        .AddInput(*input)
        .AddInput(std::move(dims))
        .AddOutput(*output)
        .AddAttrs({{"keep_dims", keep_dim}})
        .Run(stream);
  }
};

template <typename T>
class NPUReduceMeanGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("X");
    auto* output_grad =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* input_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    input_grad->mutable_data<T>(ctx.GetPlace());

    bool reduce_all = ctx.Attr<bool>("reduce_all");
    auto reduce_dims = ctx.Attr<std::vector<int>>("dim");
    auto input_dims = input->dims();

    int reduce_numel = 1;
    if (reduce_all) {
      reduce_dims.clear();
      for (int d = 0; d < input_dims.size(); ++d) {
        reduce_dims.push_back(static_cast<int>(d));
      }
    }
    for (auto& d : reduce_dims) {
      if (d < 0) {
        d = d + input_dims.size();
      }
      reduce_numel *= input_dims[d];
    }

    phi::DenseTensor tensor_value(input_grad->dtype());
    tensor_value.mutable_data<T>({1}, ctx.GetPlace());
    FillNpuTensorWithConstant<T>(
        &tensor_value, static_cast<T>(1.0f / static_cast<T>(reduce_numel)));

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    NpuOpRunner runner;
    runner.SetType("Fill")
        .AddInput(phi::vectorize(input_dims))
        .AddInput(tensor_value)
        .AddOutput(*input_grad)
        .Run(stream);

    phi::DenseTensor transformed_input_grad, transformed_out_grad;
    phi::DenseTensor tmp_output_grad;
    auto tmp_output_dims = input_dims;
    for (auto d : reduce_dims) {
      tmp_output_dims[d] = 1;
    }
    tmp_output_grad.ShareDataWith(*output_grad);
    tmp_output_grad.Resize(tmp_output_dims);
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::NPUDeviceContext>();
    NpuElementWiseOpBroadcast<T>(dev_ctx,
                                 input_grad,
                                 &tmp_output_grad,
                                 0,
                                 &transformed_input_grad,
                                 &transformed_out_grad);
    const auto& runner2 =
        NpuOpRunner("Mul",
                    {transformed_input_grad, transformed_out_grad},
                    {*input_grad},
                    {});
    runner2.Run(stream);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(reduce_mean, ops::NPUReduceMeanOpKernel<float>);
REGISTER_OP_NPU_KERNEL(reduce_mean_grad, ops::NPUReduceMeanGradOpKernel<float>);
