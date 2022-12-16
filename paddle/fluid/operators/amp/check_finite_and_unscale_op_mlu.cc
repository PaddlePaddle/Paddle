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
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <typename T>
class CheckFiniteAndUnscaleMLUKernel : public framework::OpKernel<T> {
  using MPDType = typename details::MPTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto& dev_ctx = ctx.template device_context<platform::MLUDeviceContext>();
    const auto xs = ctx.MultiInput<phi::DenseTensor>("X");
    const auto* scale = ctx.Input<phi::DenseTensor>("Scale");
    auto outs = ctx.MultiOutput<phi::DenseTensor>("Out");
    auto* found_inf = ctx.Output<phi::DenseTensor>("FoundInfinite");

    found_inf->mutable_data<bool>(dev_ctx.GetPlace());

    MLUCnnlTensorDesc scale_desc(*scale);
    MLUCnnlTensorDesc found_inf_desc(
        *found_inf, CNNL_LAYOUT_ARRAY, ToCnnlDataType<bool>());

    for (size_t i = 0; i < xs.size(); ++i) {
      const auto* x = xs[i];
      auto* out = outs[i];
      out->mutable_data<T>(ctx.GetPlace());

      // check is_finite or is_nan
      phi::DenseTensor is_finite(found_inf->type());
      if (i != 0) {
        is_finite.Resize(phi::make_ddim({1}));
        is_finite.mutable_data<bool>(ctx.GetPlace());
      } else {
        is_finite.ShareDataWith(*found_inf);
      }

      MLUCnnlTensorDesc x_desc(*x);
      MLUCnnlTensorDesc out_desc(*out);

      MLUCnnl::IsNanInf(
          ctx, x_desc.get(), GetBasePtr(x), GetBasePtr(&is_finite));

      // save is_finite by logical_and op after checking every input
      if (i != 0) {
        MLUCnnlTensorDesc is_finite_desc(
            is_finite, CNNL_LAYOUT_ARRAY, ToCnnlDataType<bool>());
        MLUCnnl::Logic(ctx,
                       CNNL_LOGIC_OP_OR,
                       found_inf_desc.get(),
                       GetBasePtr(found_inf),
                       is_finite_desc.get(),
                       GetBasePtr(&is_finite),
                       found_inf_desc.get(),
                       GetBasePtr(found_inf));
      }

      // The normal logic is :
      // out = in, if found_inf = true
      // out = in/scale, if found_inf = false
      // But when found_inf is true, the data of Out should not be used.
      // So, on MLU, we always compute out with in/scale.
      phi::DenseTensor float_x;
      phi::DenseTensor float_out;
      if (std::is_same<T, paddle::platform::float16>::value) {
        float_x.Resize(x->dims());
        float_out.Resize(out->dims());
        float_x.mutable_data<MPDType>(ctx.GetPlace());
        float_out.mutable_data<MPDType>(ctx.GetPlace());

        MLUCnnlTensorDesc float_x_desc(float_x);
        MLUCnnlTensorDesc float_out_desc(float_out);
        auto cast_fp16_type =
            GetCastDataType(DataType::FLOAT16, DataType::FLOAT32);
        MLUCnnl::Cast(ctx,
                      cast_fp16_type,
                      x_desc.get(),
                      GetBasePtr(x),
                      float_x_desc.get(),
                      GetBasePtr(&float_x));

        MLUCnnl::Div(ctx,
                     CNNL_COMPUTATION_HIGH_PRECISION,
                     float_x_desc.get(),
                     GetBasePtr(&float_x),
                     scale_desc.get(),
                     GetBasePtr(scale),
                     float_out_desc.get(),
                     GetBasePtr(&float_out));

        auto cast_fp32_type =
            GetCastDataType(DataType::FLOAT32, DataType::FLOAT16);
        MLUCnnl::Cast(ctx,
                      cast_fp32_type,
                      float_out_desc.get(),
                      GetBasePtr(&float_out),
                      out_desc.get(),
                      GetBasePtr(out));
      } else {
        MLUCnnl::Div(ctx,
                     CNNL_COMPUTATION_HIGH_PRECISION,
                     x_desc.get(),
                     GetBasePtr(x),
                     scale_desc.get(),
                     GetBasePtr(scale),
                     out_desc.get(),
                     GetBasePtr(out));
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_MLU_KERNEL(check_finite_and_unscale,
                       ops::CheckFiniteAndUnscaleMLUKernel<float>,
                       ops::CheckFiniteAndUnscaleMLUKernel<plat::float16>);
