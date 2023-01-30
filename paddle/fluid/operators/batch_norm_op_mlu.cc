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

#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/batch_norm_op.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <typename T>
class MLUBatchNormOpKernel : public framework::OpKernel<T> {
  using MPDType = typename details::MPTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto &place = ctx.GetPlace();
    const float epsilon = ctx.Attr<float>("epsilon");
    float momentum = ctx.Attr<float>("momentum");
    const bool is_test = ctx.Attr<bool>("is_test");
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    const bool trainable_stats = ctx.Attr<bool>("trainable_statistics");
    bool test_mode = is_test && (!trainable_stats);

    bool global_stats = test_mode || use_global_stats;

    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
<<<<<<< HEAD
    DataLayout data_layout = phi::StringToDataLayout(data_layout_str);

    const auto *x = ctx.Input<phi::DenseTensor>("X");
=======
    DataLayout data_layout = framework::StringToDataLayout(data_layout_str);

    const auto *x = ctx.Input<Tensor>("X");
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    const auto &x_dims = x->dims();
    PADDLE_ENFORCE_GE(
        x_dims.size(),
        2,
        platform::errors::InvalidArgument(
            "The size of input X's dimensions should be larger than 1."
            "But received: the size of input X's dimensions is [%d]",
            x_dims.size()));
    PADDLE_ENFORCE_LE(
        x_dims.size(),
        5,
        platform::errors::InvalidArgument(
            "The size of input X's dimensions should be less than 6."
            "But received: the size of input X's dimensions is [%d]",
            x_dims.size()));
    const int N = x_dims[0];
    const int C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);
    const int sample_size = x->numel() / N / C;

<<<<<<< HEAD
    const auto *running_mean = ctx.Input<phi::DenseTensor>("Mean");
    const auto *running_var = ctx.Input<phi::DenseTensor>("Variance");
    const auto *scale = ctx.Input<phi::DenseTensor>("Scale");
    const auto *bias = ctx.Input<phi::DenseTensor>("Bias");

    auto *y = ctx.Output<phi::DenseTensor>("Y");
    auto *mean_out = ctx.Output<phi::DenseTensor>("MeanOut");
    auto *variance_out = ctx.Output<phi::DenseTensor>("VarianceOut");
    auto *saved_mean = ctx.Output<phi::DenseTensor>("SavedMean");
    auto *saved_variance = ctx.Output<phi::DenseTensor>("SavedVariance");
=======
    const auto *running_mean = ctx.Input<Tensor>("Mean");
    const auto *running_var = ctx.Input<Tensor>("Variance");
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");

    auto *y = ctx.Output<Tensor>("Y");
    auto *mean_out = ctx.Output<Tensor>("MeanOut");
    auto *variance_out = ctx.Output<Tensor>("VarianceOut");
    auto *saved_mean = ctx.Output<Tensor>("SavedMean");
    auto *saved_variance = ctx.Output<Tensor>("SavedVariance");
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    // alloc memory
    y->mutable_data<T>(place);
    mean_out->mutable_data<MPDType>(place);
    variance_out->mutable_data<MPDType>(place);
    saved_mean->mutable_data<MPDType>(place);
    saved_variance->mutable_data<MPDType>(place);

<<<<<<< HEAD
    phi::DenseTensor transformed_x;
    phi::DenseTensor transformed_y;
=======
    Tensor transformed_x;
    Tensor transformed_y;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    const int transformed_dim_size = 4;
    const int transformed_shape[transformed_dim_size] = {N, sample_size, 1, C};
    MLUCnnlTensorDesc transformed_desc(transformed_dim_size,
                                       transformed_shape,
                                       ToCnnlDataType<T>(),
                                       CNNL_LAYOUT_NHWC);
    MLUCnnlTensorDesc others_input_desc(*scale);
    // input dimension is 2 and the format is NCHW. The input can be regarded as
    // NHWC format. Don't need to transpose.
    bool need_transpose =
        (data_layout == DataLayout::kNCHW && x_dims.size() != 2);
    if (need_transpose) {
      auto &dev_ctx = ctx.template device_context<MLUDeviceContext>();
      transformed_x = ctx.AllocateTmpTensor<T, MLUDeviceContext>(
          framework::DDim(transformed_shape, transformed_dim_size), dev_ctx);
      transformed_y = ctx.AllocateTmpTensor<T, MLUDeviceContext>(
          framework::DDim(transformed_shape, transformed_dim_size), dev_ctx);

      const int x_reshaped[] = {N, C, sample_size, 1};
      MLUCnnlTensorDesc x_reshaped_desc(
          transformed_dim_size, x_reshaped, ToCnnlDataType<T>());
      const std::vector<int> perm = {0, 2, 3, 1};
      MLUCnnl::Transpose(ctx,
                         perm,
                         transformed_dim_size,
                         x_reshaped_desc.get(),
                         GetBasePtr(x),
                         transformed_desc.get(),
                         GetBasePtr(&transformed_x));
    } else {
      transformed_x = *x;
      transformed_y = *y;
    }

    if (ctx.HasInput("MomentumTensor")) {
<<<<<<< HEAD
      const auto *mom_tensor = ctx.Input<phi::DenseTensor>("MomentumTensor");
      phi::DenseTensor mom_cpu;
=======
      const auto *mom_tensor = ctx.Input<Tensor>("MomentumTensor");
      Tensor mom_cpu;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
      framework::TensorCopySync(*mom_tensor, platform::CPUPlace(), &mom_cpu);
      momentum = mom_cpu.data<float>()[0];
    }

    MLUCnnl::FusedBatchNorm(ctx,
                            !global_stats,
                            transformed_desc.get(),
                            GetBasePtr(&transformed_x),
                            others_input_desc.get(),
                            GetBasePtr(scale),
                            GetBasePtr(bias),
                            GetBasePtr(running_mean),
                            GetBasePtr(running_var),
                            epsilon,
                            momentum,
                            transformed_desc.get(),
                            GetBasePtr(&transformed_y),
                            GetBasePtr(mean_out),
                            GetBasePtr(variance_out),
                            GetBasePtr(saved_mean),
                            GetBasePtr(saved_variance));

    if (need_transpose) {
      const int y_reshaped[] = {N, C, sample_size, 1};
      MLUCnnlTensorDesc y_reshaped_desc(
          transformed_dim_size, y_reshaped, ToCnnlDataType<T>());
      const std::vector<int> perm = {0, 3, 1, 2};
      MLUCnnl::Transpose(ctx,
                         perm,
                         transformed_y.dims().size(),
                         transformed_desc.get(),
                         GetBasePtr(&transformed_y),
                         y_reshaped_desc.get(),
                         GetBasePtr(y));
    }
  }
};

template <typename T>
class MLUBatchNormGradOpKernel : public framework::OpKernel<T> {
  using MPDType = typename details::MPTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
<<<<<<< HEAD
    const auto *x = ctx.Input<phi::DenseTensor>("X");
    const auto *d_y = ctx.Input<phi::DenseTensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<phi::DenseTensor>("Scale");
    const auto *bias = ctx.Input<phi::DenseTensor>("Bias");
    const auto *saved_mean = ctx.Input<phi::DenseTensor>("SavedMean");
    // SavedVariance have been reverted in forward operator
    const auto *saved_inv_variance =
        ctx.Input<phi::DenseTensor>("SavedVariance");
=======
    const auto *x = ctx.Input<Tensor>("X");
    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");
    const auto *saved_mean = ctx.Input<Tensor>("SavedMean");
    // SavedVariance have been reverted in forward operator
    const auto *saved_inv_variance = ctx.Input<Tensor>("SavedVariance");
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    const bool is_test = ctx.Attr<bool>("is_test");
    const float epsilon = ctx.Attr<float>("epsilon");
<<<<<<< HEAD
    DataLayout data_layout = phi::StringToDataLayout(data_layout_str);

    auto *d_x = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto *d_scale =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<phi::DenseTensor>(framework::GradVarName("Bias"));
=======
    DataLayout data_layout = framework::StringToDataLayout(data_layout_str);

    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    auto &dev_ctx = ctx.template device_context<MLUDeviceContext>();
    auto d_x_tmp =
        ctx.AllocateTmpTensor<T, MLUDeviceContext>(x->dims(), dev_ctx);
    auto scale_grad_tmp = ctx.AllocateTmpTensor<MPDType, MLUDeviceContext>(
        scale->dims(), dev_ctx);
    auto bias_grad_tmp =
        ctx.AllocateTmpTensor<MPDType, MLUDeviceContext>(bias->dims(), dev_ctx);

    if (d_x == nullptr) {
      d_x = &d_x_tmp;
    }
    if (d_scale == nullptr) {
      d_scale = &scale_grad_tmp;
    }
    if (d_bias == nullptr) {
      d_bias = &bias_grad_tmp;
    }

    const auto &place = ctx.GetPlace();
    d_x->mutable_data<T>(place);
    d_scale->mutable_data<MPDType>(place);
    d_bias->mutable_data<MPDType>(place);

    use_global_stats = is_test || use_global_stats;

    const auto &x_dims = x->dims();
    PADDLE_ENFORCE_GE(
        x_dims.size(),
        2,
        platform::errors::InvalidArgument(
            "The size of input X's dimensions should be larger than 1."
            "But received: the size of input X's dimensions is [%d]",
            x_dims.size()));
    PADDLE_ENFORCE_LE(
        x_dims.size(),
        5,
        platform::errors::InvalidArgument(
            "The size of input X's dimensions should be less than 6."
            "But received: the size of input X's dimensions is [%d]",
            x_dims.size()));
    const int N = x_dims[0];
    const int C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);
    const int sample_size = x->numel() / N / C;

<<<<<<< HEAD
    phi::DenseTensor transformed_d_y;
    phi::DenseTensor transformed_x;
    phi::DenseTensor transformed_d_x;
=======
    Tensor transformed_d_y;
    Tensor transformed_x;
    Tensor transformed_d_x;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    const int transformed_dim_size = 4;
    const int transformed_shape[transformed_dim_size] = {N, sample_size, 1, C};

    MLUCnnlTensorDesc transformed_desc(transformed_dim_size,
                                       transformed_shape,
                                       ToCnnlDataType<T>(),
                                       CNNL_LAYOUT_NHWC);
    MLUCnnlTensorDesc others_input_desc(*scale);

    bool need_transpose =
        (data_layout == DataLayout::kNCHW && x_dims.size() != 2);
    if (need_transpose) {
      transformed_d_y = ctx.AllocateTmpTensor<T, MLUDeviceContext>(
          framework::DDim(transformed_shape, transformed_dim_size), dev_ctx);
      transformed_x = ctx.AllocateTmpTensor<T, MLUDeviceContext>(
          framework::DDim(transformed_shape, transformed_dim_size), dev_ctx);
      transformed_d_x = ctx.AllocateTmpTensor<T, MLUDeviceContext>(
          framework::DDim(transformed_shape, transformed_dim_size), dev_ctx);
      const int org_reshaped[] = {N, C, sample_size, 1};
      MLUCnnlTensorDesc org_reshaped_desc(
          transformed_dim_size, org_reshaped, ToCnnlDataType<T>());
      const std::vector<int> perm = {0, 2, 3, 1};
      MLUCnnl::Transpose(ctx,
                         perm,
                         transformed_dim_size,
                         org_reshaped_desc.get(),
                         GetBasePtr(d_y),
                         transformed_desc.get(),
                         GetBasePtr(&transformed_d_y));
      MLUCnnl::Transpose(ctx,
                         perm,
                         transformed_dim_size,
                         org_reshaped_desc.get(),
                         GetBasePtr(x),
                         transformed_desc.get(),
                         GetBasePtr(&transformed_x));
    } else {
      transformed_d_y = *d_y;
      transformed_x = *x;
      transformed_d_x = *d_x;
    }

    if (use_global_stats) {
<<<<<<< HEAD
      const auto *running_mean = ctx.Input<phi::DenseTensor>("Mean");
      const auto *running_variance = ctx.Input<phi::DenseTensor>("Variance");
=======
      const auto *running_mean = ctx.Input<Tensor>("Mean");
      const auto *running_variance = ctx.Input<Tensor>("Variance");
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
      MLUCnnl::FusedBatchNormGrad(ctx,
                                  false /*is_training*/,
                                  transformed_desc.get(),
                                  GetBasePtr(&transformed_d_y),
                                  transformed_desc.get(),
                                  GetBasePtr(&transformed_x),
                                  others_input_desc.get(),
                                  GetBasePtr(scale),
                                  GetBasePtr(running_mean),
                                  GetBasePtr(running_variance),
                                  epsilon,
                                  transformed_desc.get(),
                                  GetBasePtr(&transformed_d_x),
                                  GetBasePtr(d_scale),
                                  GetBasePtr(d_bias));
    } else {
      MLUCnnl::FusedBatchNormGrad(ctx,
                                  true /*is_training*/,
                                  transformed_desc.get(),
                                  GetBasePtr(&transformed_d_y),
                                  transformed_desc.get(),
                                  GetBasePtr(&transformed_x),
                                  others_input_desc.get(),
                                  GetBasePtr(scale),
                                  GetBasePtr(saved_mean),
                                  GetBasePtr(saved_inv_variance),
                                  epsilon,
                                  transformed_desc.get(),
                                  GetBasePtr(&transformed_d_x),
                                  GetBasePtr(d_scale),
                                  GetBasePtr(d_bias));
    }

    if (need_transpose) {
      const int d_x_reshaped[] = {N, C, sample_size, 1};
      MLUCnnlTensorDesc d_x_reshaped_desc(
          transformed_dim_size, d_x_reshaped, ToCnnlDataType<T>());
      const std::vector<int> perm = {0, 3, 1, 2};
      MLUCnnl::Transpose(ctx,
                         perm,
                         transformed_dim_size,
                         transformed_desc.get(),
                         GetBasePtr(&transformed_d_x),
                         d_x_reshaped_desc.get(),
                         GetBasePtr(d_x));
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(batch_norm,
                       ops::MLUBatchNormOpKernel<float>,
                       ops::MLUBatchNormOpKernel<plat::float16>);
REGISTER_OP_MLU_KERNEL(batch_norm_grad,
                       ops::MLUBatchNormGradOpKernel<float>,
                       ops::MLUBatchNormGradOpKernel<plat::float16>);
