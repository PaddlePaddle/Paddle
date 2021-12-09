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
#include "paddle/fluid/platform/device/npu/npu_info.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename T>
inline void FillNPU(Tensor *dst, T val,
                    const framework::ExecutionContext &context) {
  Tensor value(dst->type());
  value.mutable_data<T>({1}, context.GetPlace());
  FillNpuTensorWithConstant<T>(&value, static_cast<T>(val));

  auto stream =
      context.template device_context<paddle::platform::NPUDeviceContext>()
          .stream();

  const auto &runner = NpuOpRunner(
      "FillD", {value}, {*dst}, {{"dims", framework::vectorize(dst->dims())}});
  runner.Run(stream);
}

template <typename T>
void shard_index(const Tensor &table_t, const Tensor &ids_t, int64_t start_idx,
                 const Tensor &id_t,
                 const framework::ExecutionContext &context) {
  const int height = table_t.dims()[0];

  auto stream =
      context.template device_context<paddle::platform::NPUDeviceContext>()
          .stream();
  framework::Tensor id_t_d;
  id_t_d.mutable_data<T>(ids_t.dims(), context.GetPlace());
  FillNPU(&id_t_d, static_cast<T>(0.0), context);
  id_t_d.Resize(ids_t.dims());

  framework::Tensor id_t_u;
  id_t_u.mutable_data<T>(ids_t.dims(), context.GetPlace());
  FillNPU(&id_t_u, static_cast<T>(height - 1), context);
  id_t_u.Resize(ids_t.dims());

  framework::Tensor id_matched_d;
  id_matched_d.mutable_data<bool>(ids_t.dims(), context.GetPlace());
  framework::Tensor id_matched_u;
  id_matched_u.mutable_data<bool>(ids_t.dims(), context.GetPlace());
  framework::Tensor ignore_tensor;
  ignore_tensor.mutable_data<T>(ids_t.dims(), context.GetPlace());
  FillNPU(&ignore_tensor, static_cast<T>(height), context);
  ignore_tensor.Resize(ids_t.dims());

  NpuOpRunner sub_runner;
#if (CANN_VERSION_CODE >= 503003)
  Tensor factor_tensor(ids_t.type());
  factor_tensor.mutable_data<T>({1}, context.GetPlace());
  TensorFromVector(std::vector<T>{static_cast<T>(start_idx)},
                   context.device_context(), &factor_tensor);
  sub_runner.SetType("Sub")
      .AddInput(ids_t)
      .AddInput(factor_tensor)
      .AddOutput(id_t);
#else
  sub_runner.SetType("Sub")
      .AddInput(ids_t)
      .AddInput(std::vector<T>{static_cast<T>(start_idx)})
      .AddOutput(id_t);
#endif
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

template <typename TIds, typename T>
void NPUGetIdsEmbedding(const framework::ExecutionContext &context) {
  auto *table_t = context.Input<LoDTensor>("W");
  auto *ids_t = context.Input<LoDTensor>("Ids");
  auto *output_t = context.Output<LoDTensor>("Out");
  const int64_t start_idx = context.Attr<int64_t>("start_index");

  auto stream =
      context.template device_context<paddle::platform::NPUDeviceContext>()
          .stream();

  framework::Tensor ids_t_local;
  ids_t_local.mutable_data<TIds>(ids_t->dims(), context.GetPlace());
  shard_index<TIds>(*table_t, *ids_t, start_idx, ids_t_local, context);

  auto pad_shape =
      framework::make_ddim({table_t->dims()[0] + 1, table_t->dims()[1]});
  framework::LoDTensor table_t_pad;

  size_t mem_size = table_t->numel() * framework::SizeOfType(table_t->type());
  size_t line_mem_size =
      table_t->dims()[1] * framework::SizeOfType(table_t->type());
  PADDLE_ENFORCE_EQ(line_mem_size % 64, 0,
                    platform::errors::InvalidArgument(
                        "NPU only accept the second dim must align by 64"));

  VLOG(10) << "mem_size:" << mem_size << ",line_mem_size:" << line_mem_size
           << ", pad_shape:" << pad_shape << ", table_dims:" << table_t->dims();

  uint8_t *pad_data = reinterpret_cast<uint8_t *>(
      table_t_pad.mutable_data<T>(pad_shape, context.GetPlace()));
  platform::NPUMemcpyAsync(pad_data, table_t->data<T>(), mem_size,
                           ACL_MEMCPY_DEVICE_TO_DEVICE, stream, mem_size);
  platform::NPUMemsetAsync(pad_data + mem_size, 0, line_mem_size, stream,
                           line_mem_size);

  output_t->mutable_data<T>(context.GetPlace());
  NpuOpRunner runner;
  runner.SetType("GatherV2")
      .AddInput(table_t_pad)
      .AddInput(ids_t_local)
      .AddInput(std::vector<int32_t>{0})
#if (CANN_VERSION_CODE >= 503003)
      .AddAttrs({{"batch_dims", 0}})
#endif
      .AddOutput(*output_t);
  runner.Run();
}

template <typename T>
class CEmbeddingNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *ids_t = context.Input<LoDTensor>("Ids");

    const auto &index_type = ids_t->type();
    if (index_type == framework::proto::VarType::INT32) {
      NPUGetIdsEmbedding<int32_t, T>(context);
    } else {
      PADDLE_THROW(platform::errors::Unavailable(
          "NPU c_embedding ids only support int32."));
    }
  }
};

template <typename TIds, typename T>
void NPUUpdateEmbedding(const framework::ExecutionContext &context) {
  // get inputs
  const int64_t start_idx = context.Attr<int64_t>("start_index");
  auto ids_t = context.Input<LoDTensor>("Ids");
  auto d_output_t = context.Input<LoDTensor>(framework::GradVarName("Out"));
  auto table_t = context.Input<Tensor>("W");
  auto table_grad_t = context.Output<LoDTensor>(framework::GradVarName("W"));

  VLOG(10) << "ids_t:" << ids_t << ", d_output_t:" << d_output_t
           << ", table_t:" << table_t << ", table_grad_t" << table_grad_t;

  auto stream =
      context.template device_context<paddle::platform::NPUDeviceContext>()
          .stream();

  // convert ids_t to local valid
  framework::Tensor ids_t_local;
  ids_t_local.mutable_data<TIds>(ids_t->dims(), context.GetPlace());
  shard_index<TIds>(*table_t, *ids_t, start_idx, ids_t_local, context);

  // padding table_t -> table_t_pad
  auto pad_shape =
      framework::make_ddim({table_t->dims()[0] + 1, table_t->dims()[1]});
  framework::LoDTensor table_t_pad;

  // set table_t_pad to zero
  uint8_t *pad_data = reinterpret_cast<uint8_t *>(
      table_t_pad.mutable_data<T>(pad_shape, context.GetPlace()));
  size_t table_t_pad_mem_size =
      table_t_pad.numel() * framework::SizeOfType(table_t_pad.type());
  platform::NPUMemsetAsync(pad_data, 0, table_t_pad_mem_size, stream,
                           table_t_pad_mem_size);

  // NOTE(zhiqiu): It seems in cann 20.1, the first input and output
  // can be different tensor, but in cann 20.2+, it does inplace operation.
  // Thus, the first input and output should be same tensor.
  const auto &runner_scatter =
      NpuOpRunner("ScatterAdd", {table_t_pad, ids_t_local, *d_output_t},
                  {table_t_pad}, {{"use_locking", true}});
  runner_scatter.Run(stream);

  // copy table_t_pad to table_t
  T *dst = table_grad_t->mutable_data<T>(table_t->dims(), context.GetPlace());
  const size_t mem_size =
      table_grad_t->numel() * framework::SizeOfType(table_grad_t->type());

  // check align
  size_t line_mem_size =
      table_grad_t->dims()[1] * framework::SizeOfType(table_grad_t->type());
  PADDLE_ENFORCE_EQ(line_mem_size % 64, 0,
                    platform::errors::InvalidArgument(
                        "NPU only accept the second dim must align by 64"));

  platform::NPUMemcpyAsync(dst, pad_data, mem_size, ACL_MEMCPY_DEVICE_TO_DEVICE,
                           stream, mem_size);
}

template <typename T>
class CEmbeddingGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *ids_t = context.Input<LoDTensor>("Ids");

    const auto &index_type = ids_t->type();
    if (index_type == framework::proto::VarType::INT32) {
      NPUUpdateEmbedding<int32_t, T>(context);
    } else {
      PADDLE_THROW(
          platform::errors::Unavailable("c_embedding ids only support int32."));
    }
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
