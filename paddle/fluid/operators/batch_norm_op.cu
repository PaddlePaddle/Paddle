/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <cfloat>
#include <string>
#include <vector>
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/operators/batch_norm_op.h"
#include "paddle/fluid/operators/norm_utils.cu.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/pten/kernels/funcs/math_function.h"

DECLARE_bool(cudnn_batchnorm_spatial_persistent);

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;
template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;
template <typename T>
using BatchNormParamType = typename CudnnDataType<T>::BatchNormParamType;



template <typename T>
class BatchNormKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    
  }
};



template <typename T>
class BatchNormGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {

    PADDLE_ENFORCE_EQ(
        x_dims.size() >= 2 && x_dims.size() <= 5, true,
        platform::errors::InvalidArgument(
            "The size of input's dimensions should be between 2 and 5."
            "But received: the size of input's dimensions is [%d],"
            "the dimensions of input is [%s]",
            x_dims.size(), x_dims));
    int N, C, H, W, D;
    ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);

    // init output
    if (d_x) {
      d_x->mutable_data<T>(ctx.GetPlace());
    }

    if (d_scale && d_bias) {
      d_scale->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
      d_bias->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
    }
    PADDLE_ENFORCE_EQ(
        scale->dims().size(), 1UL,
        platform::errors::InvalidArgument(
            "The size of scale's dimensions must equal to 1. But received: "
            "the size of scale's dimensions is [%d], the dimensions of scale "
            "is [%s].",
            scale->dims().size(), scale->dims()));
    PADDLE_ENFORCE_EQ(
        scale->dims()[0], C,
        platform::errors::InvalidArgument(
            "The first dimension of scale must equal to Channels[%d]. But "
            "received: the first dimension of scale is [%d]",
            C, scale->dims()[0]));

    auto dtype = platform::CudnnDataType<T>::type;
    const auto *reserve_space = ctx.Input<Tensor>("ReserveSpace");
#ifdef PADDLE_WITH_HIP
    auto compute_format = data_layout == DataLayout::kNHWC ? DataLayout::kNHWC
                                                           : DataLayout::kNCHW;

// TODO(wangran16): wait for MIOpen to improve the performance of BN
// HIP do not support compute format of NHWC
// auto compute_format = DataLayout::kNCHW;
#else
    const bool fast_nhwc_batch_norm =
        dtype == CUDNN_DATA_HALF && FLAGS_cudnn_batchnorm_spatial_persistent &&
        reserve_space != nullptr;
    auto compute_format =
        fast_nhwc_batch_norm && data_layout == DataLayout::kNHWC
            ? DataLayout::kNHWC
            : DataLayout::kNCHW;
#endif

    Tensor transformed_x(x->type());
    Tensor transformed_d_y(d_y->type());
    Tensor transformed_d_x;
    if (data_layout == DataLayout::kNHWC &&
        compute_format == DataLayout::kNCHW && x_dims.size() > 2) {
      VLOG(3) << "Transform input tensor from NHWC to NCHW.";
      ResizeToChannelFirst<platform::CUDADeviceContext, T>(ctx, x,
                                                           &transformed_x);
      TransToChannelFirst<platform::CUDADeviceContext, T>(ctx, x,
                                                          &transformed_x);
      ResizeToChannelFirst<platform::CUDADeviceContext, T>(ctx, d_y,
                                                           &transformed_d_y);
      TransToChannelFirst<platform::CUDADeviceContext, T>(ctx, d_y,
                                                          &transformed_d_y);
      if (d_x) {
        ResizeToChannelFirst<platform::CUDADeviceContext, T>(ctx, d_x,
                                                             &transformed_d_x);
      }
    } else {
      transformed_x.ShareDataWith(*x);
      transformed_d_y.ShareDataWith(*d_y);
      if (d_x) {
        transformed_d_x.ShareDataWith(*d_x);
      }
    }

    std::vector<int> dims;
    std::vector<int> strides;
    if (compute_format == DataLayout::kNCHW) {
      dims = {N, C, H, W, D};
      strides = {C * H * W * D, H * W * D, W * D, D, 1};
    } else {
      dims = {N, C, H, W, D};
      strides = {H * W * C * D, 1, W * D * C, D * C, C};
    }

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    const int num = transformed_x.numel();
#ifdef HIPCC
    const int block = 256;
#else
    const int block = 512;
#endif
    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    const int max_blocks = std::max(max_threads / block, 1);
    int grid1 = (num + block - 1) / block;
    int grid2 = std::min(C, max_blocks);
    auto stream = dev_ctx.stream();
    InplaceHelper<T> inplace_functor;

    if (!use_global_stats) {
      if ((N * H * W * D) == 1) {
        if (d_x) {
          framework::TensorCopy(*d_y, ctx.GetPlace(), d_x);
        }
        pten::funcs::SetConstant<platform::CUDADeviceContext,
                                 BatchNormParamType<T>>
            functor;
        functor(dev_ctx, d_scale, static_cast<BatchNormParamType<T>>(0));
        functor(dev_ctx, d_bias, static_cast<BatchNormParamType<T>>(0));
        return;
      }

// ------------------- cudnn descriptors ---------------------
#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// miopenTensorDescriptor_t data_desc_;
// miopenTensorDescriptor_t bn_param_desc_;
// miopenBatchNormMode_t mode_;

// PADDLE_ENFORCE_GPU_SUCCESS(
//     platform::dynload::miopenCreateTensorDescriptor(&data_desc_));
// PADDLE_ENFORCE_GPU_SUCCESS(
//     platform::dynload::miopenCreateTensorDescriptor(&bn_param_desc_));
#else
      cudnnTensorDescriptor_t data_desc_;
      cudnnTensorDescriptor_t bn_param_desc_;
      cudnnBatchNormMode_t mode_;

      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnCreateTensorDescriptor(&data_desc_));
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnCreateTensorDescriptor(&bn_param_desc_));
#endif
      if (epsilon <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
        LOG(ERROR) << "Provided epsilon is smaller than "
                   << "CUDNN_BN_MIN_EPSILON. Setting it to "
                   << "CUDNN_BN_MIN_EPSILON instead.";
      }
      epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);
#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// mode_ = miopenBNSpatial;
#elif CUDNN_VERSION_MIN(7, 0, 1)
      if (FLAGS_cudnn_batchnorm_spatial_persistent) {
        mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
      } else if (H == 1 && W == 1) {
        mode_ = CUDNN_BATCHNORM_PER_ACTIVATION;
      } else {
        mode_ = CUDNN_BATCHNORM_SPATIAL;
      }
#else
      if (H == 1 && W == 1) {
        mode_ = CUDNN_BATCHNORM_PER_ACTIVATION;
      } else {
        mode_ = CUDNN_BATCHNORM_SPATIAL;
      }
#endif  // CUDNN_VERSION_MIN(7, 0, 1)

#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::miopenSetTensorDescriptor(
//     data_desc_, CudnnDataType<T>::type,
//     x_dims.size() > 3 ? x_dims.size() : 4, const_cast<int *>(dims.data()),
//     const_cast<int *>(strides.data())));
// PADDLE_ENFORCE_GPU_SUCCESS(
//     platform::dynload::miopenDeriveBNTensorDescriptor(bn_param_desc_,
//                                                       data_desc_, mode_));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
          data_desc_, CudnnDataType<T>::type,
          x_dims.size() > 3 ? x_dims.size() : 4, dims.data(), strides.data()));
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnDeriveBNTensorDescriptor(bn_param_desc_,
                                                           data_desc_, mode_));
#endif

      const auto *saved_mean = ctx.Input<Tensor>("SavedMean");
      const auto *saved_var = ctx.Input<Tensor>("SavedVariance");
      const auto *saved_mean_data =
          saved_mean->template data<BatchNormParamType<T>>();
      const auto *saved_var_data =
          saved_var->template data<BatchNormParamType<T>>();

      if (is_inplace) {
        inplace_functor(compute_format, transformed_x.data<T>(),
                        scale->template data<BatchNormParamType<T>>(),
                        bias->template data<BatchNormParamType<T>>(),
                        saved_mean_data, saved_var_data, epsilon, C, H * W * D,
                        num, transformed_x.data<T>(), grid2, block, stream);
      }

      // This branch calls CUDNN APIs
      if (d_x && d_scale && d_bias) {
        bool called = false;
#if CUDNN_VERSION_MIN(7, 4, 1)
        called = true;
        size_t workspace_size = 0;
        void *workspace_ptr = nullptr;
        Tensor workspace_tensor;
        auto reserve_space_size = reserve_space->memory_size();
        // --------------- cudnn batchnorm workspace ---------------
        PADDLE_ENFORCE_GPU_SUCCESS(
            platform::dynload::
                cudnnGetBatchNormalizationBackwardExWorkspaceSize(
                    /*handle=*/dev_ctx.cudnn_handle(),
                    /*mode=*/mode_,
                    /*bnIps=*/CUDNN_BATCHNORM_OPS_BN,
                    /*xDesc=*/data_desc_,
                    /*yDesc=*/data_desc_,
                    /*dyDesc=*/data_desc_,
                    /*dzDesc=*/nullptr,
                    /*dxDesc=*/data_desc_,
                    /*bnScaleBiasMeanVarDesc=*/bn_param_desc_,
                    /*activationDesc=*/nullptr,
                    /*sizeInBytes=*/&workspace_size));

        workspace_ptr = workspace_tensor.mutable_data(
            ctx.GetPlace(), transformed_x.type(), workspace_size);

        PADDLE_ENFORCE_GPU_SUCCESS(
            platform::dynload::cudnnBatchNormalizationBackwardEx(
                /*handle=*/dev_ctx.cudnn_handle(),
                /*mode=*/mode_,
                /*bnOps=*/CUDNN_BATCHNORM_OPS_BN,
                /*alphaDataDiff=*/CudnnDataType<T>::kOne(),
                /*betaDataDiff=*/CudnnDataType<T>::kZero(),
                /*alphaParamDiff=*/CudnnDataType<T>::kOne(),
                /*betaParamDiff=*/CudnnDataType<T>::kZero(),
                /*xDesc=*/data_desc_,
                /*xData=*/transformed_x.template data<T>(),
                /*yDesc=*/nullptr,
                /*yData=*/nullptr,
                /*dyDesc=*/data_desc_,
                /*dyData=*/transformed_d_y.template data<T>(),
                /*dzDesc=*/nullptr,
                /*dzData=*/nullptr,
                /*dxDesc=*/data_desc_,
                /*dxData=*/transformed_d_x.template mutable_data<T>(
                    ctx.GetPlace()),
                /*dBnScaleBiasDesc=*/bn_param_desc_,
                /*bnScaleData=*/scale->template data<BatchNormParamType<T>>(),
                /*bnBiasData=*/nullptr,
                /*dBnScaleData=*/d_scale
                    ->template mutable_data<BatchNormParamType<T>>(
                        ctx.GetPlace()),
                /*dBnBiasData=*/d_bias
                    ->template mutable_data<BatchNormParamType<T>>(
                        ctx.GetPlace()),
                /*epsilon=*/epsilon,
                /*savedMean=*/saved_mean_data,
                /*savedInvVariance=*/saved_var_data,
                /*activationDesc=*/nullptr,
                /*workspace=*/workspace_ptr,
                /*workSpaceSizeInBytes=*/workspace_size,
                /*reserveSpace=*/const_cast<T *>(
                    reserve_space->template data<T>()),
                /*reserveSpaceSizeInBytes=*/reserve_space_size));
#endif  // CUDNN_VERSION_MIN(7, 4, 1)
        if (!called) {
#ifdef PADDLE_WITH_HIP
          if (compute_format == DataLayout::kNCHW) {
            BNBackward<
                T, block,
                DataLayout::kNCHW><<<grid2, block, 0, dev_ctx.stream()>>>(
                transformed_d_y.template data<T>(),
                transformed_x.template data<T>(),
                scale->template data<BatchNormParamType<T>>(), saved_mean_data,
                saved_var_data, C, N, H * W * D, epsilon,
                transformed_d_x.template data<T>(),
                d_scale->template mutable_data<BatchNormParamType<T>>(
                    ctx.GetPlace()),
                d_bias->template mutable_data<BatchNormParamType<T>>(
                    ctx.GetPlace()));
          } else {
            BNBackward<
                T, block,
                DataLayout::kNHWC><<<grid2, block, 0, dev_ctx.stream()>>>(
                transformed_d_y.template data<T>(),
                transformed_x.template data<T>(),
                scale->template data<BatchNormParamType<T>>(), saved_mean_data,
                saved_var_data, C, N, H * W * D, epsilon,
                transformed_d_x.template data<T>(),
                d_scale->template mutable_data<BatchNormParamType<T>>(
                    ctx.GetPlace()),
                d_bias->template mutable_data<BatchNormParamType<T>>(
                    ctx.GetPlace()));
          }

// TODO(wangran16): wait for MIOpen to improve the performance of BN
// PADDLE_ENFORCE_GPU_SUCCESS(
//     platform::dynload::miopenBatchNormalizationBackward(
//         dev_ctx.cudnn_handle(), mode_, CudnnDataType<T>::kOne(),
//         CudnnDataType<T>::kZero(), CudnnDataType<T>::kOne(),
//         CudnnDataType<T>::kZero(), data_desc_,
//         transformed_x.template data<T>(), data_desc_,
//         transformed_d_y.template data<T>(), data_desc_,
//         transformed_d_x.template mutable_data<T>(ctx.GetPlace()),
//         bn_param_desc_, scale->template data<BatchNormParamType<T>>(),
//         d_scale->template mutable_data<BatchNormParamType<T>>(
//             ctx.GetPlace()),
//         d_bias->template mutable_data<BatchNormParamType<T>>(
//             ctx.GetPlace()),
//         epsilon, saved_mean_data, saved_var_data));
#else
          PADDLE_ENFORCE_GPU_SUCCESS(
              platform::dynload::cudnnBatchNormalizationBackward(
                  dev_ctx.cudnn_handle(), mode_, CudnnDataType<T>::kOne(),
                  CudnnDataType<T>::kZero(), CudnnDataType<T>::kOne(),
                  CudnnDataType<T>::kZero(), data_desc_,
                  transformed_x.template data<T>(), data_desc_,
                  transformed_d_y.template data<T>(), data_desc_,
                  transformed_d_x.template mutable_data<T>(ctx.GetPlace()),
                  bn_param_desc_, scale->template data<BatchNormParamType<T>>(),
                  d_scale->template mutable_data<BatchNormParamType<T>>(
                      ctx.GetPlace()),
                  d_bias->template mutable_data<BatchNormParamType<T>>(
                      ctx.GetPlace()),
                  epsilon, saved_mean_data, saved_var_data));
#endif
        }

        if (data_layout == DataLayout::kNHWC &&
            compute_format == DataLayout::kNCHW) {
          VLOG(3) << "Transform batchnorm output from NCHW to NHWC";
          TransToChannelLast<paddle::platform::CUDADeviceContext, T>(
              ctx, &transformed_d_x, d_x);
        }
      } else {
        // This branch call CUDA kernels
        if (compute_format == DataLayout::kNCHW) {
          if (d_x) {
            BNBackwardData<T, block, framework::DataLayout::kNCHW><<<
                grid2, block, 0, dev_ctx.stream()>>>(
                d_y->data<T>(), scale->data<BatchNormParamType<T>>(),
                saved_mean_data, x->data<T>(), saved_var_data, C, N, H * W * D,
                d_x->data<T>());
          }
          if (d_scale && d_bias) {
            KeBNBackwardScaleBias<
                T, block,
                framework::DataLayout::kNCHW><<<grid2, block, 0, stream>>>(
                d_y->data<T>(), x->data<T>(), saved_mean_data, saved_var_data,
                epsilon, N, C, H * W * D,
                d_scale->data<BatchNormParamType<T>>(),
                d_bias->data<BatchNormParamType<T>>());
          }
        } else {
          if (d_x) {
            BNBackwardData<T, block, framework::DataLayout::kNHWC><<<
                grid2, block, 0, dev_ctx.stream()>>>(
                d_y->data<T>(), scale->data<BatchNormParamType<T>>(),
                saved_mean_data, x->data<T>(), saved_var_data, C, N, H * W * D,
                d_x->data<T>());
          }
          if (d_scale && d_bias) {
            KeBNBackwardScaleBias<
                T, block,
                framework::DataLayout::kNHWC><<<grid2, block, 0, stream>>>(
                d_y->data<T>(), x->data<T>(), saved_mean_data, saved_var_data,
                epsilon, N, C, H * W * D,
                d_scale->data<BatchNormParamType<T>>(),
                d_bias->data<BatchNormParamType<T>>());
          }
        }
      }

#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// clean when exit.
// PADDLE_ENFORCE_GPU_SUCCESS(
//     platform::dynload::miopenDestroyTensorDescriptor(data_desc_));
// PADDLE_ENFORCE_GPU_SUCCESS(
//     platform::dynload::miopenDestroyTensorDescriptor(bn_param_desc_));
#else
      // clean when exit.
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnDestroyTensorDescriptor(data_desc_));
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnDestroyTensorDescriptor(bn_param_desc_));
#endif
    } else {
      const auto *running_mean = ctx.Input<Tensor>("Mean");
      const auto *running_var = ctx.Input<Tensor>("Variance");

      const auto *running_mean_data =
          running_mean->template data<BatchNormParamType<T>>();
      const auto *running_var_data =
          running_var->template data<BatchNormParamType<T>>();

      if (is_inplace) {
        auto px = *x;
        inplace_functor(data_layout, px.mutable_data<T>(ctx.GetPlace()),
                        scale->template data<BatchNormParamType<T>>(),
                        bias->template data<BatchNormParamType<T>>(),
                        running_mean_data, running_var_data, epsilon, C,
                        H * W * D, num, x->data<T>(), grid2, block, stream);
      }

      if (compute_format == DataLayout::kNCHW) {
        if (d_x) {
          KeBNBackwardData<
              T, framework::DataLayout::kNCHW><<<grid1, block, 0, stream>>>(
              d_y->data<T>(), scale->data<BatchNormParamType<T>>(),
              running_var_data, epsilon, C, H * W, num, d_x->data<T>());
        }
        if (d_scale && d_bias) {
          KeBNBackwardScaleBias<
              T, block,
              framework::DataLayout::kNCHW><<<grid2, block, 0, stream>>>(
              d_y->data<T>(), x->data<T>(), running_mean_data, running_var_data,
              epsilon, N, C, H * W * D, d_scale->data<BatchNormParamType<T>>(),
              d_bias->data<BatchNormParamType<T>>());
        }
      } else {
        if (d_x) {
          KeBNBackwardData<
              T, framework::DataLayout::kNHWC><<<grid1, block, 0, stream>>>(
              d_y->data<T>(), scale->data<BatchNormParamType<T>>(),
              running_var_data, epsilon, C, H * W, num, d_x->data<T>());
        }
        if (d_scale && d_bias) {
          KeBNBackwardScaleBias<
              T, block,
              framework::DataLayout::kNHWC><<<grid2, block, 0, stream>>>(
              d_y->data<T>(), x->data<T>(), running_mean_data, running_var_data,
              epsilon, N, C, H * W * D, d_scale->data<BatchNormParamType<T>>(),
              d_bias->data<BatchNormParamType<T>>());
        }
      }
    }
  }
};

template <typename T>
class BatchNormDoubleGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *X = ctx.Input<Tensor>("X");
    const auto *Scale = ctx.Input<Tensor>("Scale");
    const auto *dY = ctx.Input<Tensor>("DY");
    const auto *Saved_mean = ctx.Input<Tensor>("SavedMean");
    const auto *Saved_variance = ctx.Input<Tensor>("SavedVariance");
    const double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    const bool is_test = ctx.Attr<bool>("is_test");

    PADDLE_ENFORCE_EQ(
        is_test, false,
        platform::errors::InvalidArgument(
            "`is_test = True` CANNOT be used in train program. If "
            "you want to use global status in pre_train model, "
            "please set `use_global_stats = True`"));

    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);

    const auto *ddX = ctx.Input<Tensor>("DDX");
    const auto *ddScale = ctx.Input<Tensor>("DDScale");
    const auto *ddBias = ctx.Input<Tensor>("DDBias");

    auto *dX = ctx.Output<Tensor>("DX");
    auto *dScale = ctx.Output<Tensor>("DScale");
    auto *ddY = ctx.Output<Tensor>("DDY");

    NormDoubleGradFunctor<platform::CUDADeviceContext, T>(
        ctx, data_layout, X, Scale, dY, Saved_mean, Saved_variance, epsilon,
        use_global_stats, ddX, ddScale, ddBias, dX, dScale, ddY);
  }
};

}  // namespace operators
}  // namespace paddle

// namespace ops = paddle::operators;
// namespace plat = paddle::platform;
// #ifdef PADDLE_WITH_HIP
// // MIOPEN do not support double
// REGISTER_OP_CUDA_KERNEL(
//     batch_norm, ops::BatchNormKernel<plat::CUDADeviceContext, float>,
//     ops::BatchNormKernel<plat::CUDADeviceContext, plat::float16>);
// REGISTER_OP_CUDA_KERNEL(
//     batch_norm_grad, ops::BatchNormGradKernel<plat::CUDADeviceContext, float>,
//     ops::BatchNormGradKernel<plat::CUDADeviceContext, plat::float16>);
// REGISTER_OP_CUDA_KERNEL(
//     batch_norm_grad_grad,
//     ops::BatchNormDoubleGradKernel<plat::CUDADeviceContext, float>);
// #else
// REGISTER_OP_CUDA_KERNEL(
//     batch_norm, ops::BatchNormKernel<plat::CUDADeviceContext, float>,
//     ops::BatchNormKernel<plat::CUDADeviceContext, double>,
//     ops::BatchNormKernel<plat::CUDADeviceContext, plat::float16>);
// REGISTER_OP_CUDA_KERNEL(
//     batch_norm_grad, ops::BatchNormGradKernel<plat::CUDADeviceContext, float>,
//     ops::BatchNormGradKernel<plat::CUDADeviceContext, double>,
//     ops::BatchNormGradKernel<plat::CUDADeviceContext, plat::float16>);
// REGISTER_OP_CUDA_KERNEL(
//     batch_norm_grad_grad,
//     ops::BatchNormDoubleGradKernel<plat::CUDADeviceContext, float>,
//     ops::BatchNormDoubleGradKernel<plat::CUDADeviceContext, double>);
// #endif
