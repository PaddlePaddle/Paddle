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

using Tensor = phi::DenseTensor;
using DDim = framework::DDim;

template <typename T>
class LayerNormMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto begin_norm_axis = ctx.Attr<int>("begin_norm_axis");
    const auto epsilon = ctx.Attr<float>("epsilon");
    const auto* x = ctx.Input<phi::DenseTensor>("X");
    const auto* scale = ctx.Input<phi::DenseTensor>("Scale");
    const auto* bias = ctx.Input<phi::DenseTensor>("Bias");
    auto* y = ctx.Output<phi::DenseTensor>("Y");
    auto* mean = ctx.Output<phi::DenseTensor>("Mean");
    auto* variance = ctx.Output<phi::DenseTensor>("Variance");

    auto place = ctx.GetPlace();

    y->mutable_data<T>(place);
    mean->mutable_data<T>(place);
    variance->mutable_data<T>(place);

    const auto& x_dims = x->dims();
    std::vector<int> scale_bias_axes;
    std::vector<int> mean_var_axes;
    for (auto i = 0; i < x_dims.size(); ++i) {
      if (i >= begin_norm_axis) {
        scale_bias_axes.push_back(x_dims[i]);
      } else {
        mean_var_axes.push_back(x_dims[i]);
      }
    }

    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc y_desc(*y);
    MLUCnnlTensorDesc mean_var_desc(
        mean_var_axes.size(), mean_var_axes.data(), ToCnnlDataType<T>());
    // cnnl only support both of scale and bias is NULL or not.
    if (!scale && !bias) {
      MLUCnnl::LayerNormForward(ctx,
                                begin_norm_axis,
                                x_desc.get(),
                                GetBasePtr(x),
                                nullptr /*scale_bias_desc*/,
                                nullptr /*scale*/,
                                nullptr /*bias*/,
                                epsilon,
                                y_desc.get(),
                                GetBasePtr(y),
                                mean_var_desc.get(),
                                GetBasePtr(mean),
                                GetBasePtr(variance));
    } else {
      Tensor tmp_scale(x->dtype());
      if (!scale) {
        tmp_scale.mutable_data<T>(phi::make_ddim(scale_bias_axes), place);
        FillMLUTensorWithHostValue(ctx, static_cast<T>(1), &tmp_scale);
      } else {
        tmp_scale = *scale;
      }

      Tensor tmp_bias(x->dtype());
      if (!bias) {
        tmp_bias.mutable_data<T>(phi::make_ddim(scale_bias_axes), place);
        FillMLUTensorWithHostValue(ctx, static_cast<T>(0), &tmp_bias);
      } else {
        tmp_bias = *bias;
      }

      // scale and bias should have same type with x/y
      MLUCnnlTensorDesc float32_desc(
          scale_bias_axes.size(), scale_bias_axes.data(), CNNL_DTYPE_FLOAT);
      MLUCnnlTensorDesc float16_desc(
          scale_bias_axes.size(), scale_bias_axes.data(), CNNL_DTYPE_HALF);
      cnnlCastDataType_t cast_type = GetCastDataType(VT::FP32, VT::FP16);

      Tensor final_scale(x->dtype());
      if (final_scale.dtype() == DataType::FLOAT16 &&
          tmp_scale.dtype() == DataType::FLOAT32) {
        final_scale.mutable_data<T>(phi::make_ddim(scale_bias_axes), place);
        // cast scale to fp16
        MLUCnnl::Cast(ctx,
                      cast_type,
                      float32_desc.get(),
                      GetBasePtr(&tmp_scale),
                      float16_desc.get(),
                      GetBasePtr(&final_scale));
      } else {
        final_scale = tmp_scale;
      }

      Tensor final_bias(x->dtype());
      if (final_bias.dtype() == DataType::FLOAT16 &&
          tmp_bias.dtype() == DataType::FLOAT32) {
        final_bias.mutable_data<T>(phi::make_ddim(scale_bias_axes), place);
        // cast bias to fp16
        MLUCnnl::Cast(ctx,
                      cast_type,
                      float32_desc.get(),
                      GetBasePtr(&tmp_bias),
                      float16_desc.get(),
                      GetBasePtr(&final_bias));
      } else {
        final_bias = tmp_bias;
      }

      MLUCnnlTensorDesc scale_bias_desc(
          scale_bias_axes.size(), scale_bias_axes.data(), ToCnnlDataType<T>());
      MLUCnnl::LayerNormForward(ctx,
                                begin_norm_axis,
                                x_desc.get(),
                                GetBasePtr(x),
                                scale_bias_desc.get(),
                                GetBasePtr(&final_scale),
                                GetBasePtr(&final_bias),
                                epsilon,
                                y_desc.get(),
                                GetBasePtr(y),
                                mean_var_desc.get(),
                                GetBasePtr(mean),
                                GetBasePtr(variance));
    }
  }
};

template <typename T>
class LayerNormGradMLUKernel : public framework::OpKernel<T> {
  using MPDType = typename details::MPTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto begin_norm_axis = ctx.Attr<int>("begin_norm_axis");
    const auto* x = ctx.Input<phi::DenseTensor>("X");
    const auto* mean = ctx.Input<phi::DenseTensor>("Mean");
    const auto* variance = ctx.Input<phi::DenseTensor>("Variance");
    const auto* scale = ctx.Input<phi::DenseTensor>("Scale");
    const auto* dy = ctx.Input<phi::DenseTensor>(framework::GradVarName("Y"));
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* dscale =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Scale"));
    auto* dbias = ctx.Output<phi::DenseTensor>(framework::GradVarName("Bias"));

    auto place = ctx.GetPlace();
    dx->mutable_data<T>(place);

    const auto& x_dims = x->dims();
    std::vector<int> scale_bias_axes;
    std::vector<int> mean_var_axes;
    for (auto i = 0; i < x_dims.size(); ++i) {
      if (i >= begin_norm_axis) {
        scale_bias_axes.push_back(x_dims[i]);
      } else {
        mean_var_axes.push_back(x_dims[i]);
      }
    }

    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc dy_desc(*dy);
    MLUCnnlTensorDesc mean_var_desc(
        mean_var_axes.size(), mean_var_axes.data(), ToCnnlDataType<T>());
    MLUCnnlTensorDesc dx_desc(*dx);

    Tensor tmp_scale(x->dtype());
    if (!scale) {
      tmp_scale.mutable_data<T>(phi::make_ddim(scale_bias_axes), place);
      FillMLUTensorWithHostValue(ctx, static_cast<T>(1), &tmp_scale);
    } else {
      tmp_scale = *scale;
    }

    MLUCnnlTensorDesc float32_desc(
        scale_bias_axes.size(), scale_bias_axes.data(), CNNL_DTYPE_FLOAT);
    MLUCnnlTensorDesc float16_desc(
        scale_bias_axes.size(), scale_bias_axes.data(), CNNL_DTYPE_HALF);
    cnnlCastDataType_t cast_fp32_to_fp16 = GetCastDataType(VT::FP32, VT::FP16);
    cnnlCastDataType_t cast_fp16_to_fp32 = GetCastDataType(VT::FP16, VT::FP32);

    Tensor final_scale(x->dtype());
    if (final_scale.dtype() == DataType::FLOAT16 &&
        tmp_scale.dtype() == DataType::FLOAT32) {
      final_scale.mutable_data<T>(phi::make_ddim(scale_bias_axes), place);
      // cast scale to fp16
      MLUCnnl::Cast(ctx,
                    cast_fp32_to_fp16,
                    float32_desc.get(),
                    GetBasePtr(&tmp_scale),
                    float16_desc.get(),
                    GetBasePtr(&final_scale));
    } else {
      final_scale = tmp_scale;
    }

    Tensor tmp_dscale(x->dtype());
    if (dscale && (tmp_dscale.dtype() == dscale->dtype())) {
      dscale->mutable_data<T>(place);
      tmp_dscale = *dscale;
    } else {
      tmp_dscale.mutable_data<T>(phi::make_ddim(scale_bias_axes), place);
    }
    Tensor tmp_dbias(x->dtype());
    if (dbias && (tmp_dbias.dtype() == dbias->dtype())) {
      dbias->mutable_data<T>(place);
      tmp_dbias = *dbias;
    } else {
      tmp_dbias.mutable_data<T>(phi::make_ddim(scale_bias_axes), place);
    }

    MLUCnnlTensorDesc scale_desc(
        scale_bias_axes.size(), scale_bias_axes.data(), ToCnnlDataType<T>());
    MLUCnnl::LayerNormBackward(ctx,
                               begin_norm_axis,
                               x_desc.get(),
                               GetBasePtr(x),
                               dy_desc.get(),
                               GetBasePtr(dy),
                               scale_desc.get(),
                               GetBasePtr(&final_scale),
                               mean_var_desc.get(),
                               GetBasePtr(mean),
                               GetBasePtr(variance),
                               dx_desc.get(),
                               GetBasePtr(dx),
                               GetBasePtr(&tmp_dscale),
                               GetBasePtr(&tmp_dbias));

    if (dscale && (tmp_dscale.dtype() == DataType::FLOAT16 &&
                   dscale->dtype() == DataType::FLOAT32)) {
      dscale->mutable_data<MPDType>(place);
      MLUCnnl::Cast(ctx,
                    cast_fp16_to_fp32,
                    float16_desc.get(),
                    GetBasePtr(&tmp_dscale),
                    float32_desc.get(),
                    GetBasePtr(dscale));
    }
    if (dbias && (tmp_dbias.dtype() == DataType::FLOAT16 &&
                  dbias->dtype() == DataType::FLOAT32)) {
      dbias->mutable_data<MPDType>(place);
      MLUCnnl::Cast(ctx,
                    cast_fp16_to_fp32,
                    float16_desc.get(),
                    GetBasePtr(&tmp_dbias),
                    float32_desc.get(),
                    GetBasePtr(dbias));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(layer_norm,
                       ops::LayerNormMLUKernel<float>,
                       ops::LayerNormMLUKernel<plat::float16>);
REGISTER_OP_MLU_KERNEL(layer_norm_grad,
                       ops::LayerNormGradMLUKernel<float>,
                       ops::LayerNormGradMLUKernel<plat::float16>);
