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

template <typename T>
class LookupTableV2MLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *ids_t = ctx.Input<phi::DenseTensor>("Ids");      // int tensor
    auto *output_t = ctx.Output<phi::DenseTensor>("Out");  // float tensor
    auto *table_t = ctx.Input<phi::DenseTensor>("W");
    int padding_idx = static_cast<int>(ctx.Attr<int64_t>("padding_idx"));

    auto *table_var = ctx.InputVar("W");
    PADDLE_ENFORCE_EQ(
        table_var->IsType<phi::DenseTensor>(),
        true,
        platform::errors::InvalidArgument("mlu only accept phi::DenseTensor"));
    output_t->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc ids_desc(*ids_t);
    MLUCnnlTensorDesc table_desc(*table_t);
    MLUCnnlTensorDesc output_desc(*output_t);

    MLUCnnl::EmbeddingForward(ctx,
                              padding_idx,
                              table_desc.get(),
                              GetBasePtr(table_t),
                              ids_desc.get(),
                              static_cast<const int *>(GetBasePtr(ids_t)),
                              output_desc.get(),
                              GetBasePtr(output_t));
  }
};

template <typename T>
class LookupTableV2GradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *table_var = ctx.InputVar("W");
    PADDLE_ENFORCE_EQ(
        table_var->IsType<phi::DenseTensor>(),
        true,
        platform::errors::PermissionDenied(
            "Unsupported Variable Type , idx in "
            "LookupTableV2GradMLUKernel should be phi::DenseTensor."));
    bool is_sparse = ctx.Attr<bool>("is_sparse");
    PADDLE_ENFORCE_EQ(
        is_sparse,
        false,
        platform::errors::InvalidArgument(
            "LookupTableV2GradMLUKernel dose NOT support is_sparse = True."));
    auto *ids_t = ctx.Input<phi::DenseTensor>("Ids");
    auto *output_grad_t =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto *table_grad_t =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("W"));
    table_grad_t->mutable_data<T>(ctx.GetPlace());

    int padding_idx = static_cast<int>(ctx.Attr<int64_t>("padding_idx"));

    int64_t ids_numel = ids_t->numel();
    PADDLE_ENFORCE_EQ(
        ids_numel <= std::numeric_limits<int32_t>::max(),
        true,
        platform::errors::OutOfRange(
            "Number of ids greater than int32_t::max , please check "
            "number of ids in LookupTableV2GradMLUKernel."));

    phi::DenseTensor ids_int32(ids_t->dtype());
    if (ids_t->dtype() != DataType::INT32) {
      ids_int32.mutable_data<int>(ids_t->dims(), ctx.GetPlace());
      MLUCnnlTensorDesc ids_desc(*ids_t);
      MLUCnnlTensorDesc ids_int32_desc(ids_int32);
      auto cast_type = GetCastDataType(ids_t->dtype(), DataType::INT32);
      MLUCnnl::Cast(ctx,
                    cast_type,
                    ids_desc.get(),
                    GetBasePtr(ids_t),
                    ids_int32_desc.get(),
                    GetBasePtr(&ids_int32));
    } else {
      ids_int32 = *ids_t;
    }

    MLUCnnlTensorDesc ids_int32_desc(ids_int32);
    MLUCnnlTensorDesc output_grad_desc(*output_grad_t);
    MLUCnnlTensorDesc table_grad_desc(*table_grad_t);

    MLUCnnl::EmbeddingBackward(ctx,
                               padding_idx,
                               false,
                               ids_int32_desc.get(),
                               GetBasePtr(&ids_int32),
                               output_grad_desc.get(),
                               GetBasePtr(output_grad_t),
                               table_grad_desc.get(),
                               GetBasePtr(table_grad_t));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(lookup_table_v2,
                       ops::LookupTableV2MLUKernel<float>,
                       ops::LookupTableV2MLUKernel<int>,
                       ops::LookupTableV2MLUKernel<plat::float16>);

REGISTER_OP_MLU_KERNEL(lookup_table_v2_grad,
                       ops::LookupTableV2GradMLUKernel<float>,
                       ops::LookupTableV2GradMLUKernel<plat::float16>);
