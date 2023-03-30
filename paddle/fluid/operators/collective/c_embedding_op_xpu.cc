/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/c_embedding_op.h"
#include "paddle/fluid/platform/device/device_wrapper.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class CEmbeddingOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* table_t = ctx.Input<phi::DenseTensor>("W");
    auto* ids_t = ctx.Input<phi::DenseTensor>("Ids");
    auto* output_t = ctx.Output<phi::DenseTensor>("Out");
    const int64_t start_index = ctx.Attr<int64_t>("start_index");
    const T* table_data = table_t->data<T>();
    T* output_data = output_t->mutable_data<T>(ctx.GetPlace());

    const int64_t height = table_t->dims()[0];
    const int64_t width = table_t->dims()[1];

    // int embedding(Context* ctx, const T* x, const TID* indices, T* y, int xm,
    // int n, int ym, int padding_idx, TID start_index = 0);

    // xm: table height: number of entries of table.
    // n: embedding dim: number of float value within single entry.
    // ym: number of elements of input ids.

    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    const auto& index_type = framework::TransToProtoVarType(ids_t->dtype());
    if (index_type == framework::proto::VarType::INT32) {
      int r = xpu::embedding(dev_ctx.x_context(),
                             table_data,
                             ids_t->data<int32_t>(),
                             output_data,
                             height,
                             width,
                             ids_t->numel(),
                             -1,
                             static_cast<int32_t>(start_index));
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "embedding");
    } else if (index_type == framework::proto::VarType::INT64) {
      int r = xpu::embedding(dev_ctx.x_context(),
                             table_data,
                             ids_t->data<int64_t>(),
                             output_data,
                             height,
                             width,
                             ids_t->numel(),
                             -1,
                             static_cast<int64_t>(start_index));
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "embedding");
    } else {
      PADDLE_THROW(platform::errors::Unavailable(
          "XPU c_embedding ids only support int32 or int64."));
    }
  }
};

template <typename DeviceContext, typename T>
class CEmbeddingGradOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const int64_t start_idx = context.Attr<int64_t>("start_index");
    auto ids_t = context.Input<phi::DenseTensor>("Ids");
    auto d_output_t =
        context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto table_t = context.Input<phi::DenseTensor>("W");
    auto table_grad_t =
        context.Output<phi::DenseTensor>(framework::GradVarName("W"));

    auto& dev_ctx = context.template device_context<phi::XPUContext>();
    table_grad_t->Resize(table_t->dims());
    dev_ctx.template Alloc(table_grad_t, table_t->dtype());
    T* table_grad_data = static_cast<T*>(table_grad_t->data());

    size_t table_t_mem_size =
        table_t->numel() * phi::SizeOf(table_grad_t->dtype());
    size_t table_grad_t_mem_size =
        table_grad_t->numel() *
        framework::SizeOfType(
            framework::TransToProtoVarType(table_grad_t->dtype()));

    VLOG(10) << "table_dims:" << table_t->dims()
             << ", table_t memory_size:" << table_t_mem_size
             << ", table_grad_t memory_size:" << table_grad_t_mem_size
             << ", start_index:" << start_idx;

    int r = xpu::constant(
        dev_ctx.x_context(), table_grad_data, table_grad_t->numel(), (T)0);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
    const T* d_output_data = d_output_t->data<T>();

    const int64_t height = table_t->dims()[0];
    const int64_t width = table_t->dims()[1];

    const auto& index_type = framework::TransToProtoVarType(ids_t->dtype());
    if (index_type == framework::proto::VarType::INT32) {
      r = xpu::embedding_grad(dev_ctx.x_context(),
                              d_output_data,
                              ids_t->data<int32_t>(),
                              table_grad_data,
                              height,
                              width,
                              ids_t->numel(),
                              -1,
                              static_cast<int32_t>(start_idx));
    } else if (index_type == framework::proto::VarType::INT64) {
      r = xpu::embedding_grad(dev_ctx.x_context(),
                              d_output_data,
                              ids_t->data<int64_t>(),
                              table_grad_data,
                              height,
                              width,
                              ids_t->numel(),
                              -1,
                              static_cast<int64_t>(start_idx));
    } else {
      PADDLE_THROW(platform::errors::Unavailable(
          "XPU c_embedding ids only support int32 or int64."));
    }
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "embedding_grad");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_XPU_KERNEL(
    c_embedding,
    ops::CEmbeddingOpXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    c_embedding_grad,
    ops::CEmbeddingGradOpXPUKernel<paddle::platform::XPUDeviceContext, float>);
