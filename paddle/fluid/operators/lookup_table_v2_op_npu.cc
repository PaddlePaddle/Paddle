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
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
constexpr int64_t kNoPadding = -1;

template <typename DeviceContext, typename T>
class LookupTableV2NPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *ids_t = ctx.Input<framework::LoDTensor>("Ids");      // int tensor
    auto *output_t = ctx.Output<framework::LoDTensor>("Out");  // float tensor
    auto *table_t = ctx.Input<framework::LoDTensor>("W");

    auto *table_var = ctx.InputVar("W");
    PADDLE_ENFORCE_EQ(
        table_var->IsType<framework::LoDTensor>(), true,
        platform::errors::InvalidArgument("npu only accept LoDTensor"));
    output_t->mutable_data<T>(ctx.GetPlace());

    int64_t padding_idx = ctx.Attr<int64_t>("padding_idx");
    if (padding_idx == kNoPadding) {
      NpuOpRunner runner;
      runner.SetType("GatherV2")
          .AddInput(*table_t)
          .AddInput(*ids_t)
          .AddInput(std::vector<int32_t>{0})
#if (CANN_VERSION_CODE >= 503003)
          .AddAttrs({{"batch_dims", 0}})
#endif
          .AddOutput(*output_t);
      runner.Run();
    } else {
      Tensor tmp_table_t(table_t->type());
      tmp_table_t.mutable_data<T>(table_t->dims(), ctx.GetPlace());

      Tensor index;
      index.mutable_data<int32_t>({1, 1}, ctx.GetPlace());
      FillNpuTensorWithConstant<int32_t>(&index,
                                         static_cast<int32_t>(padding_idx));

      auto updata_dim = phi::make_ddim({1, table_t->dims()[1]});
      Tensor update;
      update.mutable_data<T>(updata_dim, ctx.GetPlace());
      FillNpuTensorWithConstant<T>(&update, static_cast<T>(0));
      update.Resize(updata_dim);

      NpuOpRunner update_runner;
      update_runner.SetType("TensorScatterUpdate")
          .AddInput(*table_t)
          .AddInput(index)
          .AddInput(update)
          .AddOutput(tmp_table_t);
      update_runner.Run();

      NpuOpRunner runner;
      runner.SetType("GatherV2")
          .AddInput(tmp_table_t)
          .AddInput(*ids_t)
          .AddInput(std::vector<int32_t>{0})
#if (CANN_VERSION_CODE >= 503003)
          .AddAttrs({{"batch_dims", 0}})
#endif
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
    int64_t padding_idx = ctx.Attr<int64_t>("padding_idx");

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

    const auto &runner_zeros =
        NpuOpRunner("ZerosLike", {*table_grad_t}, {*table_grad_t});
    runner_zeros.Run(stream);

    if (padding_idx == kNoPadding) {
      // NOTE(zhiqiu): It seems in cann 20.1, the first input and output
      // can be different tensor, but in cann 20.2+, it does inplace operation.
      // Thus, the first input and output should be same tensor.
      const auto &runner_scatter =
          NpuOpRunner("ScatterAdd", {*table_grad_t, *ids_t, *output_grad_t},
                      {*table_grad_t}, {{"use_locking", true}});
      runner_scatter.Run(stream);
    } else {
      Tensor casted_ids_t;
      if (framework::TransToProtoVarType(ids_t->dtype()) !=
          framework::proto::VarType::INT32) {
        casted_ids_t.mutable_data<int32_t>(ids_t->dims(), ctx.GetPlace());
        const auto &cast_runner = NpuOpRunner("Cast", {*ids_t}, {casted_ids_t},
                                              {{"dst_type", ACL_INT32}});
        cast_runner.Run(stream);
      } else {
        casted_ids_t.ShareDataWith(*ids_t);
      }
      auto table_grad_dims = table_grad_t->dims();

      NpuOpRunner runner;
      runner.SetType("UnsortedSegmentSum")
          .AddInput(*output_grad_t)
          .AddInput(casted_ids_t)
          .AddInput(std::vector<int64_t>{table_grad_dims[0]})
          .AddOutput(*table_grad_t);
      runner.Run(stream);
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
