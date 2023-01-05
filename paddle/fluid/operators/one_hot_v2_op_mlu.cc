
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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"
#include "paddle/fluid/operators/utils.h"

namespace paddle {
namespace operators {

template <typename T>
class OneHotV2MLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MLUDeviceContext>();
    auto* in = ctx.Input<phi::DenseTensor>("X");
    auto* out = ctx.Output<phi::DenseTensor>("Out");
    int depth = ctx.Attr<int>("depth");
    if (ctx.HasInput("depth_tensor")) {
      std::vector<int32_t> depth_data;
      depth_data =
          GetDataFromTensor<int>(ctx.Input<phi::DenseTensor>("depth_tensor"));
      depth = depth_data[0];

      auto out_dims = out->dims();
      out_dims[out_dims.size() - 1] = depth;
      out->Resize(out_dims);
    }
    out->mutable_data<float>(ctx.GetPlace());

    float on_value = 1.0f, off_value = 0.0f;
    const int in_off_dim[1] = {1};
    phi::DenseTensor on_value_tensor =
        ctx.AllocateTmpTensor<float, MLUDeviceContext>(
            framework::DDim(in_off_dim, 1), dev_ctx);
    phi::DenseTensor off_value_tensor =
        ctx.AllocateTmpTensor<float, MLUDeviceContext>(
            framework::DDim(in_off_dim, 1), dev_ctx);
    FillMLUTensorWithHostValue(ctx, on_value, &on_value_tensor);
    FillMLUTensorWithHostValue(ctx, off_value, &off_value_tensor);

    if (framework::TransToProtoVarType(in->dtype()) ==
        framework::proto::VarType::INT32) {
      MLUCnnlTensorDesc desc_indices(*in);
      MLUCnnl::OneHot(ctx,
                      desc_indices.get(),
                      GetBasePtr(in),
                      depth,
                      GetBasePtr(&on_value_tensor),
                      GetBasePtr(&off_value_tensor),
                      -1,
                      ToCnnlDataType(out->dtype()),
                      GetBasePtr(out));
    } else {
      phi::DenseTensor transformed_in;
      transformed_in.mutable_data<int32_t>(in->dims(), dev_ctx.GetPlace());
      // use cnnlCast to cast int64_t to int32_t then do one_hot
      MLUCnnlTensorDesc in_desc(*in);
      MLUCnnlTensorDesc transformed_in_desc(transformed_in);
      cnnlCastDataType_t cast_type = GetCastDataType(
          framework::TransToProtoVarType(in->dtype()),
          framework::TransToProtoVarType(transformed_in.dtype()));
      MLUCnnl::Cast(ctx,
                    cast_type,
                    in_desc.get(),
                    GetBasePtr(in),
                    transformed_in_desc.get(),
                    GetBasePtr(&transformed_in));
      MLUCnnl::OneHot(ctx,
                      transformed_in_desc.get(),
                      GetBasePtr(&transformed_in),
                      depth,
                      GetBasePtr(&on_value_tensor),
                      GetBasePtr(&off_value_tensor),
                      -1,
                      ToCnnlDataType(out->dtype()),
                      GetBasePtr(out));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(one_hot_v2,
                       ops::OneHotV2MLUKernel<int32_t>,
                       ops::OneHotV2MLUKernel<int64_t>);
