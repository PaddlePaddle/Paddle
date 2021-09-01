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

#include <iostream>
#include <memory>
#include <string>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class LookupTableV2NPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *ids_t = ctx.Input<framework::LoDTensor>("Ids");      // int tensor
    auto *output_t = ctx.Output<framework::LoDTensor>("Out");  // float tensor
    auto *table_t = ctx.Input<framework::LoDTensor>("W");

    // It seems cann 20.1 accepts int64, but cann 20.2+ not.
    PADDLE_ENFORCE_EQ(ids_t->type(), framework::proto::VarType::INT32,
                      platform::errors::Unimplemented(
                          "The index of LookupTableV2 should be int32."));

    auto *table_var = ctx.InputVar("W");
    PADDLE_ENFORCE_EQ(
        table_var->IsType<framework::LoDTensor>(), true,
        platform::errors::InvalidArgument("npu only accept LoDTensor"));
    output_t->mutable_data<T>(ctx.GetPlace());

    // FIXME(baiyangfan) Fix padding_idx grad bugs.
    if (ctx.Attr<int64_t>("padding_idx") > 0) {
      PADDLE_ENFORCE_EQ(table_t->dims().size(), 2,
                        platform::errors::InvalidArgument(
                            "npu only accept the dims of table_t == 2"));
      framework::LoDTensor tensor_tmp;
      auto tmp_shape = framework::make_ddim({1, table_t->dims()[1]});
      tensor_tmp.mutable_data<T>(tmp_shape, ctx.GetPlace());
      FillNpuTensorWithConstant<T>(&tensor_tmp, static_cast<T>(0.0));

      auto stream =
          ctx.template device_context<paddle::platform::NPUDeviceContext>()
              .stream();
      std::vector<framework::Tensor> inputs = {*table_t, tensor_tmp};
      std::vector<std::string> names = {"x_w", "tmp_w"};
      auto out_shape =
          framework::make_ddim({table_t->dims()[0] + 1, table_t->dims()[1]});
      framework::Tensor table_t_tmp;
      table_t_tmp.mutable_data<T>(out_shape, ctx.GetPlace());
      NpuOpRunner runner_concat{
          "ConcatD",
          {inputs},
          {table_t_tmp},
          {{"concat_dim", 0}, {"N", static_cast<int>(inputs.size())}}};
      runner_concat.AddInputNames(names);
      runner_concat.Run(stream);

      NpuOpRunner runner;
      runner.SetType("GatherV2")
          .AddInput(table_t_tmp)
          .AddInput(*ids_t)
          .AddInput(std::vector<int32_t>{0})
          .AddOutput(*output_t);
      runner.Run();
    } else {
      NpuOpRunner runner;
      runner.SetType("GatherV2")
          .AddInput(*table_t)
          .AddInput(*ids_t)
          .AddInput(std::vector<int32_t>{0})
          .AddOutput(*output_t);
      runner.Run();
    }
  }
};

template <typename T>
class LookupTableV2GradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *ids_t = ctx.Input<framework::LoDTensor>("Ids");
    auto *output_grad_t =
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto *table_grad_t =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("W"));
    table_grad_t->mutable_data<T>(ctx.GetPlace());

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    /* EmbeddingDenseGrad has bug on large shape, temporarily disable it.

    int embedding_dim = table_grad_t->dims()[1];
    if (embedding_dim % 32 == 0) {
      // NOTE(pangyoki): The embedding_dim of Tensor used in
      // EmbeddingDenseGrad must be an integer multiple of 32.
      int num_weights = table_grad_t->dims()[0];
      const auto &runner =
          NpuOpRunner("EmbeddingDenseGrad", {*output_grad_t, *ids_t},
                      {*table_grad_t}, {{"num_weights", num_weights},
                                        {"padding_idx", -1},
                                        {"scale_grad_by_freq", false}});
      runner.Run(stream);
      return;
    }
    */

    if (ctx.Attr<int64_t>("padding_idx") > 0) {
      framework::LoDTensor tensor_tmp;
      auto tmp_shape = framework::make_ddim({1, table_grad_t->dims()[1]});
      tensor_tmp.mutable_data<T>(tmp_shape, ctx.GetPlace());
      FillNpuTensorWithConstant<T>(&tensor_tmp, static_cast<T>(0.0));

      std::vector<framework::Tensor> inputs = {*table_grad_t, tensor_tmp};
      std::vector<std::string> names = {"x_w_grad", "tmp_w_grad"};
      auto out_shape = framework::make_ddim(
          {table_grad_t->dims()[0] + 1, table_grad_t->dims()[1]});
      framework::Tensor table_t_tmp;
      table_t_tmp.mutable_data<T>(out_shape, ctx.GetPlace());
      NpuOpRunner runner_concat{
          "ConcatD",
          {inputs},
          {table_t_tmp},
          {{"concat_dim", 0}, {"N", static_cast<int>(inputs.size())}}};
      runner_concat.AddInputNames(names);
      runner_concat.Run(stream);

      const auto &runner_zeros =
          NpuOpRunner("ZerosLike", {table_t_tmp}, {table_t_tmp});
      runner_zeros.Run(stream);

      // NOTE(zhiqiu): It seems in cann 20.1, the first input and output
      // can be different tensor, but in cann 20.2+, it does inplace operation.
      // Thus, the first input and output should be same tensor.
      const auto &runner_scatter =
          NpuOpRunner("ScatterAdd", {table_t_tmp, *ids_t, *output_grad_t},
                      {table_t_tmp}, {{"use_locking", true}});
      runner_scatter.Run(stream);

      framework::Tensor out1;
      framework::Tensor out2;
      auto out_shape1 = framework::make_ddim(
          {table_t_tmp.dims()[0] - 1, table_t_tmp.dims()[1]});
      auto out_shape2 = framework::make_ddim({1, table_t_tmp.dims()[1]});
      out1.mutable_data<T>(out_shape1, ctx.GetPlace());
      out2.mutable_data<T>(out_shape2, ctx.GetPlace());
      std::vector<framework::Tensor> outputs = {out1, out2};
      std::vector<int> sections = {static_cast<int>(table_t_tmp.dims()[0] - 1),
                                   1};
      framework::NPUAttributeMap attr_input = {
          {"size_splits", sections},
          {"split_dim", 0},
          {"num_split", static_cast<int32_t>(sections.size())}};
      NpuOpRunner runner_split;
      runner_split.SetType("SplitVD")
          .AddInput(table_t_tmp)
          .AddOutputs(outputs)
          .AddAttrs(attr_input);
      runner_split.Run(stream);
      table_grad_t->ShareDataWith(out1);
    } else {
      const auto &runner_zeros =
          NpuOpRunner("ZerosLike", {*table_grad_t}, {*table_grad_t});
      runner_zeros.Run(stream);
      // NOTE(zhiqiu): It seems in cann 20.1, the first input and output
      // can be different tensor, but in cann 20.2+, it does inplace operation.
      // Thus, the first input and output should be same tensor.
      const auto &runner_scatter =
          NpuOpRunner("ScatterAdd", {*table_grad_t, *ids_t, *output_grad_t},
                      {*table_grad_t}, {{"use_locking", true}});
      runner_scatter.Run(stream);
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    lookup_table_v2,
    ops::LookupTableV2NPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::LookupTableV2NPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::LookupTableV2NPUKernel<paddle::platform::NPUDeviceContext,
                                paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    lookup_table_v2_grad, ops::LookupTableV2GradNPUKernel<float>,
    ops::LookupTableV2GradNPUKernel<int>,
    ops::LookupTableV2GradNPUKernel<paddle::platform::float16>);
