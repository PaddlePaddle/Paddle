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
class ScaleMLUKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& ctx) const {
    auto& dev_ctx = GetDevCtxFromCTX(ctx);
    auto* in_var = ctx.InputVar("X");
    auto* in = framework::GetLoDTensorOrSelectedRowsValueFromVar(*in_var);

    // cnnl require input, scale, bias with same type. And all in device side.
    auto& scale = ctx.Attr<float>("scale");
    framework::Tensor scale_tensor;
    if (ctx.HasInput("ScaleTensor")) {
      framework::Tensor float_scale_tensor =
          *ctx.Input<framework::Tensor>("ScaleTensor");
      if (framework::TransToProtoVarType(float_scale_tensor.dtype()) !=
          framework::TransToProtoVarType(in->dtype())) {
        scale_tensor = ctx.AllocateTmpTensor<T, MLUDeviceContext>({1}, dev_ctx);
        MLUCnnlTensorDesc float_scale_desc(float_scale_tensor);
        MLUCnnlTensorDesc final_scale_desc(scale_tensor);
        cnnlCastDataType_t cast_type = GetCastDataType(
            framework::TransToProtoVarType(float_scale_tensor.dtype()),
            framework::TransToProtoVarType(scale_tensor.dtype()));
        MLUCnnl::Cast(ctx, cast_type, float_scale_desc.get(),
                      GetBasePtr(&float_scale_tensor), final_scale_desc.get(),
                      GetBasePtr(&scale_tensor));
      } else {
        scale_tensor = float_scale_tensor;
      }
    } else {
      scale_tensor = ctx.AllocateTmpTensor<T, MLUDeviceContext>({1}, dev_ctx);
      MLUCnnlTensorDesc scale_desc(scale_tensor);
      MLUCnnl::Fill(ctx, scale, scale_desc.get(), GetBasePtr(&scale_tensor));
    }

    auto& bias = ctx.Attr<float>("bias");
    framework::Tensor bias_tensor =
        ctx.AllocateTmpTensor<T, MLUDeviceContext>({1}, dev_ctx);
    MLUCnnlTensorDesc bias_desc(bias_tensor);
    MLUCnnl::Fill(ctx, bias, bias_desc.get(), GetBasePtr(&bias_tensor));

    auto* out_var = ctx.OutputVar("Out");
    if (in_var->IsType<pten::SelectedRows>() && in_var != out_var) {
      auto& in_slr = in_var->Get<pten::SelectedRows>();
      auto* out_slr = out_var->GetMutable<pten::SelectedRows>();
      out_slr->set_rows(in_slr.rows());
      out_slr->set_height(in_slr.height());
    }
    auto* out =
        framework::GetMutableLoDTensorOrSelectedRowsValueFromVar(out_var);
    out->mutable_data<T>(in->place());

    MLUCnnlTensorDesc input_desc(*in);
    MLUCnnlTensorDesc scale_desc(scale_tensor);
    MLUCnnlTensorDesc output_desc(*out);

    const int axis = std::max(in->dims().size() - 1, 0);
    auto bias_after_scale = ctx.Attr<bool>("bias_after_scale");
    if (bias_after_scale) {
      MLUCnnl::Scale(ctx, axis, input_desc.get(), GetBasePtr(in),
                     scale_desc.get(), GetBasePtr(&scale_tensor),
                     bias_desc.get(), GetBasePtr(&bias_tensor),
                     output_desc.get(), GetBasePtr(out));
    } else {
      framework::Tensor new_bias_tensor =
          ctx.AllocateTmpTensor<T, MLUDeviceContext>({1}, dev_ctx);
      MLUCnnlTensorDesc new_bias_desc(new_bias_tensor);

      MLUCnnlOpTensorDesc mul_op_desc(
          CNNL_OP_TENSOR_MUL,
          ToCnnlDataType(framework::TransToProtoVarType(in->dtype())),
          CNNL_NOT_PROPAGATE_NAN);
      MLUCnnl::OpTensor(
          ctx, mul_op_desc.get(), scale_desc.get(), GetBasePtr(&scale_tensor),
          bias_desc.get(), GetBasePtr(&bias_tensor), new_bias_desc.get(),
          GetBasePtr(&new_bias_tensor),
          ToCnnlDataType(framework::TransToProtoVarType(in->dtype())));
      MLUCnnl::Scale(ctx, axis, input_desc.get(), GetBasePtr(in),
                     scale_desc.get(), GetBasePtr(&scale_tensor),
                     new_bias_desc.get(), GetBasePtr(&new_bias_tensor),
                     output_desc.get(), GetBasePtr(out));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_MLU_KERNEL(scale, ops::ScaleMLUKernel<float>,
                       ops::ScaleMLUKernel<paddle::platform::float16>);
