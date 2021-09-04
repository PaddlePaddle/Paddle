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
#include <memory>
#include <string>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/collective/c_embedding_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/platform/npu_info.h"

namespace paddle {
namespace operators {

template <typename T>
void shard_index(const Tensor &table_t, const Tensor &ids_t, int64_t start_idx,
                 Tensor id_t, const framework::ExecutionContext &context) {
  int table_w = table_t.dims()[0];

  auto stream =
      context.template device_context<paddle::platform::NPUDeviceContext>()
          .stream();
  framework::Tensor id_t_d;
  id_t_d.mutable_data<T>(ids_t.dims(), context.GetPlace());
  FillNpuTensorWithConstant(&id_t_d, static_cast<T>(0.0));
  id_t_d.Resize(ids_t.dims());
  framework::Tensor id_t_u;
  id_t_u.mutable_data<T>(ids_t.dims(), context.GetPlace());
  FillNpuTensorWithConstant(&id_t_u, static_cast<T>(table_w - 1));
  id_t_u.Resize(ids_t.dims());

  framework::Tensor id_matched_d;
  id_matched_d.mutable_data<bool>(ids_t.dims(), context.GetPlace());
  framework::Tensor id_matched_u;
  id_matched_u.mutable_data<bool>(ids_t.dims(), context.GetPlace());
  framework::Tensor ignore_tensor;
  ignore_tensor.mutable_data<T>(ids_t.dims(), context.GetPlace());
  FillNpuTensorWithConstant(&ignore_tensor, static_cast<T>(table_w));
  ignore_tensor.Resize(ids_t.dims());

  NpuOpRunner sub_runner;
  sub_runner.SetType("Sub")
      .AddInput(ids_t)
      .AddInput(std::vector<int>{static_cast<int>(start_idx)})
      .AddOutput(id_t);
  sub_runner.Run();

  NpuOpRunner lessequal1_runner;
  lessequal1_runner.SetType("LessEqual")
      .AddInput(id_t)
      .AddInput(id_t_u)
      .AddOutput(id_matched_u);
  lessequal1_runner.Run();

  NpuOpRunner lessequal2_runner;
  lessequal2_runner.SetType("LessEqual")
      .AddInput(id_t_d)
      .AddInput(id_t)
      .AddOutput(id_matched_d);
  lessequal2_runner.Run();

  NpuOpRunner("Equal", {id_matched_u, id_matched_d}, {id_matched_d}, {})
      .Run(stream);
  NpuOpRunner("Select", {id_matched_d, id_t, ignore_tensor}, {id_t}, {})
      .Run(stream);
}

template <typename T>
class CEmbeddingNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *table_t = context.Input<LoDTensor>("W");
    auto *ids_t = context.Input<LoDTensor>("Ids");
    auto *output_t = context.Output<LoDTensor>("Out");
    VLOG(10) << "debug check 1";

    auto stream =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    const int64_t start_idx = context.Attr<int64_t>("start_index");
    framework::Tensor id_t;
    id_t.mutable_data<int>(ids_t->dims(), context.GetPlace());
    shard_index<int>(*table_t, *ids_t, start_idx, id_t, context);

    PADDLE_ENFORCE_EQ(table_t->dims().size(), 2,
                      platform::errors::InvalidArgument(
                          "npu only accept the dims of table_t == 2"));
    PADDLE_ENFORCE_EQ(table_t->dims()[1] % 64, 0, "must align by 64");
    auto pad_shape =
        framework::make_ddim({table_t->dims()[0] + 1, table_t->dims()[1]});
    framework::LoDTensor table_t_pad;

    size_t mem_size = table_t->memory_size();
    size_t line_mem_size =
        table_t->dims()[1] * framework::SizeOfType(table_t->type());

    VLOG(10) << "mem_size:" << mem_size << ",line_mem_size:" << line_mem_size
             << ", pad_shape:" << pad_shape
             << ", table_dims:" << table_t->dims();

    uint8_t *pad_data = reinterpret_cast<uint8_t *>(
        table_t_pad.mutable_data<T>(pad_shape, context.GetPlace()));
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclrtMemcpyAsync(pad_data, mem_size, table_t->data<T>(), mem_size,
                         ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtMemsetAsync(
        pad_data + mem_size, line_mem_size, 0, line_mem_size, stream));

    output_t->mutable_data<T>(context.GetPlace());
    NpuOpRunner runner;
    runner.SetType("GatherV2")
        .AddInput(table_t_pad)
        .AddInput(id_t)
        .AddInput(std::vector<int32_t>{0})
        .AddOutput(*output_t);
    runner.Run();
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtSynchronizeStream(stream));
    VLOG(10) << "debug check 9";
  }
};

template <typename T>
class CEmbeddingGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    VLOG(10) << "grad debug check 0";
    const int64_t start_idx = context.Attr<int64_t>("start_index");
    auto ids_t = context.Input<LoDTensor>("Ids");
    auto d_output_t = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto table_grad_t = context.Output<LoDTensor>(framework::GradVarName("W"));
    VLOG(10) << "grad debug check 1";

    auto stream =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    framework::Tensor id_t;
    id_t.mutable_data<int>(ids_t->dims(), context.GetPlace());
    shard_index<int>(*table_grad_t, *ids_t, start_idx, id_t, context);
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtSynchronizeStream(stream));
    VLOG(10) << "grad debug check 2";

    auto pad_shape = framework::make_ddim(
        {table_grad_t->dims()[0] + 1, table_grad_t->dims()[1]});
    framework::LoDTensor table_t_pad;

    size_t mem_size = table_grad_t->memory_size();
    size_t line_mem_size =
        table_grad_t->dims()[1] * framework::SizeOfType(table_grad_t->type());

    uint8_t *pad_data = reinterpret_cast<uint8_t *>(
        table_t_pad.mutable_data<T>(pad_shape, context.GetPlace()));
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclrtMemcpyAsync(pad_data, mem_size, table_grad_t->data<T>(), mem_size,
                         ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtMemsetAsync(
        pad_data + mem_size, line_mem_size, 0, line_mem_size, stream));
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtSynchronizeStream(stream));
    VLOG(10) << "grad debug check 3";

    // NOTE(zhiqiu): It seems in cann 20.1, the first input and output
    // can be different tensor, but in cann 20.2+, it does inplace operation.
    // Thus, the first input and output should be same tensor.
    const auto &runner_scatter =
        NpuOpRunner("ScatterAdd", {table_t_pad, id_t, *d_output_t},
                    {table_t_pad}, {{"use_locking", true}});
    runner_scatter.Run(stream);

    // copy to src
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclrtMemcpyAsync(table_grad_t->data<T>(), mem_size, pad_data, mem_size,
                         ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtSynchronizeStream(stream));
    VLOG(10) << "grad debug check 4";
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(c_embedding, ops::CEmbeddingNPUKernel<float>,
                       ops::CEmbeddingNPUKernel<double>,
                       ops::CEmbeddingNPUKernel<plat::float16>);
REGISTER_OP_NPU_KERNEL(c_embedding_grad, ops::CEmbeddingGradNPUKernel<float>,
                       ops::CEmbeddingGradNPUKernel<double>,
                       ops::CEmbeddingGradNPUKernel<plat::float16>);
