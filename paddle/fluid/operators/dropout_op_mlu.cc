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

using Tensor = phi::DenseTensor;

template <typename T>
class DropoutMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* out = ctx.Output<phi::DenseTensor>("Out");
    auto dropout_prob = ctx.Attr<float>("dropout_prob");
    auto is_test = ctx.Attr<bool>("is_test");
    auto* seed_tensor =
        ctx.HasInput("Seed") ? ctx.Input<phi::DenseTensor>("Seed") : nullptr;
    auto dropout_implementation =
        ctx.Attr<std::string>("dropout_implementation");

    const bool is_upscale = (dropout_implementation == "upscale_in_train");

    out->mutable_data<T>(ctx.GetPlace());
    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc out_desc(*out);

    if (is_test && is_upscale) {
      // dropout op for inference: out = input.
      framework::TensorCopy(
          *x,
          ctx.GetPlace(),
          ctx.template device_context<platform::MLUDeviceContext>(),
          out);
      return;
    } else if (!is_test) {
      // dropout op for training: out = input * mask / ( 1.0 - dropout_prob ) or
      // out = input * mask.
      int seed_data = 0;
      if (seed_tensor) {
        if (platform::is_mlu_place(seed_tensor->place())) {
          memory::Copy(platform::CPUPlace(),
                       &seed_data,
                       seed_tensor->place(),
                       seed_tensor->data<int>(),
                       sizeof(int));
        } else {
          seed_data = *(seed_tensor->data<int>());
        }
      } else {
        seed_data = ctx.Attr<bool>("fix_seed") ? ctx.Attr<int>("seed") : 0;
      }

      auto* mask = ctx.Output<phi::DenseTensor>("Mask");
      mask->mutable_data<uint8_t>(ctx.GetPlace());
      MLUCnnlTensorDesc mask_desc(*mask);
      // Special case when dropout_prob is 1.0
      if (dropout_prob == 1.0f) {
        auto value_t = static_cast<T>(0.0f);
        MLUCnnl::Fill(ctx,
                      CNNL_POINTER_MODE_HOST,
                      &value_t,
                      out_desc.get(),
                      GetBasePtr(out));
        MLUCnnl::Fill(ctx,
                      CNNL_POINTER_MODE_HOST,
                      &value_t,
                      mask_desc.get(),
                      GetBasePtr(mask));
        return;
      }

      // create mlu random generator
      const int device_id = ctx.GetPlace().GetDeviceId();
      auto mlu_gen_random = GetMLURandomGenerator(ctx, device_id, seed_data);

      // compute out = input * mask / ( 1.0 - dropout_prob )
      MLUCnnl::FusedDropout(ctx,
                            mlu_gen_random->get(),
                            x_desc.get(),
                            GetBasePtr(x),
                            dropout_prob,
                            GetBasePtr(&(mlu_gen_random->get_state())),
                            mask_desc.get(),
                            GetBasePtr(mask),
                            out_desc.get(),
                            GetBasePtr(out));

      if (is_upscale) {
        return;
      }
    }

    // In downgrade_in_infer mode, need to multiply (1.0f - dropout_prob).
    Tensor scale_tensor(x->dtype());
    Tensor bias_tensor(x->dtype());
    scale_tensor.mutable_data<T>({1}, ctx.GetPlace());
    bias_tensor.mutable_data<T>({1}, ctx.GetPlace());
    MLUCnnlTensorDesc scale_desc(scale_tensor);
    MLUCnnlTensorDesc bias_desc(bias_tensor);
    FillMLUTensorWithHostValue(
        ctx, static_cast<T>(1.0f - dropout_prob), &scale_tensor);
    FillMLUTensorWithHostValue(ctx, static_cast<T>(0.0f), &bias_tensor);

    MLUCnnl::Scale(ctx,
                   0,
                   is_test ? x_desc.get() : out_desc.get(),
                   is_test ? GetBasePtr(x) : GetBasePtr(out),
                   scale_desc.get(),
                   GetBasePtr(&scale_tensor),
                   bias_desc.get(),
                   GetBasePtr(&bias_tensor),
                   out_desc.get(),
                   GetBasePtr(out));
  }
};

template <typename T>
class DropoutGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(!ctx.Attr<bool>("is_test"),
                      true,
                      platform::errors::InvalidArgument(
                          "GradOp is only callable when is_test is false"));
    auto* grad_x = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* grad_out = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* mask = ctx.Input<phi::DenseTensor>("Mask");
    auto dropout_prob = ctx.Attr<float>("dropout_prob");
    auto dropout_impl = ctx.Attr<std::string>("dropout_implementation");

    grad_x->mutable_data<T>(ctx.GetPlace());
    MLUCnnlTensorDesc grad_x_desc(*grad_x);

    if (dropout_prob == 1.) {
      auto value_t = static_cast<T>(0.0f);
      MLUCnnl::Fill(ctx,
                    CNNL_POINTER_MODE_HOST,
                    &value_t,
                    grad_x_desc.get(),
                    GetBasePtr(grad_x));
      return;
    }

    // cast mask from uint8 to float32/float16
    Tensor cast_mask(grad_x->dtype());
    cast_mask.Resize(mask->dims());
    cast_mask.mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc mask_desc(*mask);
    MLUCnnlTensorDesc cast_mask_desc(cast_mask);
    cnnlCastDataType_t cast_type =
        GetCastDataType(framework::TransToProtoVarType(mask->dtype()),
                        framework::TransToProtoVarType(cast_mask.dtype()));

    MLUCnnl::Cast(ctx,
                  cast_type,
                  mask_desc.get(),
                  GetBasePtr(mask),
                  cast_mask_desc.get(),
                  GetBasePtr(&cast_mask));

    const bool is_upscale = (dropout_impl == "upscale_in_train");
    const float scale = is_upscale ? (1.0f / (1.0f - dropout_prob)) : (1.0f);

    auto data_type = ToCnnlDataType<T>();
    MLUCnnlTensorDesc grad_out_desc(*grad_out);
    MLUCnnlOpTensorDesc op_tensor_desc(
        CNNL_OP_TENSOR_MUL, data_type, CNNL_NOT_PROPAGATE_NAN);
    MLUCnnl::OpTensor(ctx,
                      op_tensor_desc.get(),
                      cast_mask_desc.get(),
                      GetBasePtr(&cast_mask),
                      grad_out_desc.get(),
                      GetBasePtr(grad_out),
                      grad_x_desc.get(),
                      GetBasePtr(grad_x),
                      data_type,
                      scale);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(dropout,
                       ops::DropoutMLUKernel<float>,
                       ops::DropoutMLUKernel<plat::float16>);

REGISTER_OP_MLU_KERNEL(dropout_grad,
                       ops::DropoutGradMLUKernel<float>,
                       ops::DropoutGradMLUKernel<plat::float16>);
