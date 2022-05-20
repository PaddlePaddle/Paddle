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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
constexpr int64_t kNoPadding = -1;

template <typename T>
class LookupTableV2MLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *ids_t = ctx.Input<framework::LoDTensor>("Ids");      // int tensor
    auto *output_t = ctx.Output<framework::LoDTensor>("Out");  // float tensor
    auto *table_t = ctx.Input<framework::LoDTensor>("W");

    auto *table_var = ctx.InputVar("W");
    PADDLE_ENFORCE_EQ(
        table_var->IsType<framework::LoDTensor>(), true,
        platform::errors::InvalidArgument("mlu only accept LoDTensor"));
    output_t->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc ids_desc(*ids_t);
    MLUCnnlTensorDesc table_desc(*table_t);
    MLUCnnlTensorDesc output_desc(*output_t);

    int64_t padding_idx = ctx.Attr<int64_t>("padding_idx");
    if (padding_idx == kNoPadding) {
      MLUCnnl::GatherFunctor(ctx, /*axis=*/0, /*batch_dims=*/0,
                             table_desc.get(), GetBasePtr(table_t),
                             ids_desc.get(), GetBasePtr(ids_t),
                             output_desc.get(), GetBasePtr(output_t));
    } else {
      Tensor tmp_table_t(table_t->type());
      tmp_table_t.mutable_data<T>(table_t->dims(), ctx.GetPlace());

      Tensor index;
      index.mutable_data<int32_t>({1, 1}, ctx.GetPlace());
      auto idx_value = static_cast<int32_t>(padding_idx);
      MLUCnnlTensorDesc index_desc(index);
      MLUCnnl::Fill(ctx, CNNL_POINTER_MODE_HOST, &idx_value, index_desc.get(),
                    GetBasePtr(&index));

      auto update_dim = phi::make_ddim({1, table_t->dims()[1]});
      Tensor update;
      update.mutable_data<T>(update_dim, ctx.GetPlace());

      auto update_value = static_cast<T>(0);
      MLUCnnlTensorDesc update_desc(update);
      MLUCnnl::Fill(ctx, CNNL_POINTER_MODE_HOST, &update_value,
                    update_desc.get(), GetBasePtr(&update));

      MLUCnnlTensorDesc tmp_table_desc(tmp_table_t);
      MLUCnnl::ScatterNd(
          ctx, CNNL_SCATTERND_UPDATE, index_desc.get(), GetBasePtr(&index),
          update_desc.get(), GetBasePtr(&update), table_desc.get(),
          GetBasePtr(table_t), tmp_table_desc.get(), GetBasePtr(&tmp_table_t));

      MLUCnnl::GatherFunctor(ctx, /*axis=*/0, /*batch_dims=*/0,
                             tmp_table_desc.get(), GetBasePtr(&tmp_table_t),
                             ids_desc.get(), GetBasePtr(ids_t),
                             output_desc.get(), GetBasePtr(output_t));
    }
  }
};

template <typename T>
class LookupTableV2GradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *ids_t = ctx.Input<framework::LoDTensor>("Ids");
    auto *output_grad_t =
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto *table_grad_t =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("W"));
    table_grad_t->mutable_data<T>(ctx.GetPlace());

    int padding_idx = static_cast<int>(ctx.Attr<int64_t>("padding_idx"));

    Tensor ids_int32(ids_t->dtype());
    if (ids_t->dtype() != DataType::INT32) {
      ids_int32.mutable_data<int>(ids_t->dims(), ctx.GetPlace());
      MLUCnnlTensorDesc ids_desc(*ids_t);
      MLUCnnlTensorDesc ids_int32_desc(ids_int32);
      auto cast_type = GetCastDataType(ids_t->dtype(), DataType::INT32);
      MLUCnnl::Cast(ctx, cast_type, ids_desc.get(), GetBasePtr(ids_t),
                    ids_int32_desc.get(), GetBasePtr(&ids_int32));
    } else {
      ids_int32 = *ids_t;
    }

    MLUCnnlTensorDesc ids_int32_desc(ids_int32);
    MLUCnnlTensorDesc output_grad_desc(*output_grad_t);
    MLUCnnlTensorDesc table_grad_desc(*table_grad_t);

    MLUCnnl::EmbeddingBackward(ctx, padding_idx, false, ids_int32_desc.get(),
                               GetBasePtr(&ids_int32), output_grad_desc.get(),
                               GetBasePtr(output_grad_t), table_grad_desc.get(),
                               GetBasePtr(table_grad_t));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(lookup_table_v2, ops::LookupTableV2MLUKernel<float>,
                       ops::LookupTableV2MLUKernel<int>,
                       ops::LookupTableV2MLUKernel<plat::float16>);

REGISTER_OP_MLU_KERNEL(lookup_table_v2_grad,
                       ops::LookupTableV2GradMLUKernel<float>,
                       ops::LookupTableV2GradMLUKernel<int>,
                       ops::LookupTableV2GradMLUKernel<plat::float16>);
