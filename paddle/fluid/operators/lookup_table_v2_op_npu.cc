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

    NpuOpRunner runner;
    runner.SetType("GatherV2")
        .AddInput(*table_t)
        .AddInput(*ids_t)
        .AddInput(std::vector<int32_t>{0})
        .AddOutput(*output_t);
    runner.Run();

    auto dev_ctx = static_cast<platform::NPUDeviceContext *>(
        platform::DeviceContextPool::Instance().Get(ctx.GetPlace()));
    auto stream = dev_ctx->stream();
    auto w_name = ctx.InputName("W");
    VLOG(10) << "input W name:" << w_name;
    if (w_name == "word_embedding_0" || w_name == "word_embedding_1" ||
        w_name == "word_embedding_2" || w_name == "word_embedding_3" ||
        w_name == "word_embedding_4" || w_name == "word_embedding_5" ||
        w_name == "word_embedding_6" || w_name == "word_embedding_7") {
      PADDLE_ENFORCE_NPU_SUCCESS(aclrtSynchronizeStream(stream));

      // get weight last two lines
      {
        std::vector<T> w_vec;
        framework::TensorToVector(*table_t, *dev_ctx, &w_vec);
        int64_t width = table_t->dims()[1];
        int64_t height = table_t->dims()[0];
        printf("%s:", w_name.c_str());
        for (int64_t i = 0; i < height; i++) {
          printf("embedding row %ld:", i);
          for (int64_t w = 0; w < 10; w++) {
            printf("%f,", static_cast<float>(w_vec[i * width + w]));
          }
          printf("\n");
        }
        printf("\n");

        printf("%s last line:", w_name.c_str());
        for (int64_t i = (height - 1) * width; i < 10; i++) {
          printf("%f,", static_cast<float>(w_vec[i]));
        }
        printf("\n");
      }

      // get ids_t
      std::vector<int32_t> ids_vec;
      std::vector<T> out_vec;
      framework::TensorToVector(*ids_t, *dev_ctx, &ids_vec);
      framework::TensorToVector(*output_t, *dev_ctx, &out_vec);

      const auto &out = output_t;
      const int64_t batch_size = out->dims()[0];
      const int64_t height = out->dims()[1];
      const int64_t width = out->dims()[2];
      VLOG(10) << "batchsize:" << batch_size << ", height:" << height
               << ", width:" << width << ", out_dims:" << out->dims();

      printf("%s lines:", w_name.c_str());
      for (int64_t i = 0; i < batch_size; i++) {
        printf("batch %ld:\n", i);
        for (int64_t h = 0; h < height; h++) {
          printf("\tidt row %ld idx:%d:", h, ids_vec[i * height + h]);
          for (int64_t w = 0; w < 10; w++) {
            printf("%f,", static_cast<float>(
                              out_vec[i * height * width + h * width + w]));
          }
          printf("\n");
        }
        printf("\n");
      }
      printf("\n");
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
