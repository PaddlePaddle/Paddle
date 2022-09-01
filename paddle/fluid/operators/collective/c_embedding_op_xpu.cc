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

using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class CEmbeddingOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* table_t = ctx.Input<LoDTensor>("W");
    auto* ids_t = ctx.Input<LoDTensor>("Ids");
    auto* output_t = ctx.Output<LoDTensor>("Out");
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

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_XPU_KERNEL(
    c_embedding,
    ops::CEmbeddingOpXPUKernel<paddle::platform::XPUDeviceContext, float>);
