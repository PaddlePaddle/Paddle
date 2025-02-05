// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/kernels/fusion/gpu/cudnn_fusion_helper.h"

namespace phi {
namespace fusion {
namespace dynload = phi::dynload;

template <typename T>
using ScalingParamType =
    typename phi::backends::gpu::CudnnDataType<T>::ScalingParamType;

#if CUDNN_VERSION >= 8000

static size_t RoundUp(int64_t a, int64_t b) { return (a + b - 1) / b * b; }

template <typename T>
struct NormConvolutionArgs {
  NormConvolutionArgs() {
    dtype = phi::backends::gpu::CudnnDataType<T>::type;
    format = CUDNN_TENSOR_NHWC;
    compute_type = phi::backends::gpu::CudnnDataType<float>::type;
  }

  void Set(const phi::GPUContext &ctx,
           const std::vector<int> &input_shape,
           const std::vector<int> &filter_shape,
           const std::vector<int> &output_shape,
           int padding,
           int stride,
           int dilation,
           int group) {
    PADDLE_ENFORCE_LT(
        ctx.GetComputeCapability(),
        90,
        common::errors::PreconditionNotMet(
            "Expect compute compatibility to be less than 90, but got %d. "
            "CUDNN FusedOps is no longer available on H100 and later "
            "devices.",
            ctx.GetComputeCapability()));
    PADDLE_ENFORCE_EQ(
        input_shape.size(),
        4U,
        common::errors::InvalidArgument(
            "The size of input_shape is expected to 4. But received "
            "input_shape's size is %d, input_shape is [%s].",
            input_shape.size(),
            common::make_ddim(input_shape)));
    PADDLE_ENFORCE_EQ(
        filter_shape.size(),
        4U,
        common::errors::InvalidArgument(
            "The size of filter_shape is expected to 4. But received "
            "filter_shape's size is %d, filter_shape is [%s].",
            filter_shape.size(),
            common::make_ddim(filter_shape)));
    PADDLE_ENFORCE_EQ(filter_shape[1] == filter_shape[2] &&
                          (filter_shape[1] == 1 || filter_shape[1] == 3),
                      true,
                      common::errors::InvalidArgument(
                          "The filter_shape is expected to store as nhwc, and "
                          "h = w = 1 or 3. But received filter_shape is [%s].",
                          common::make_ddim(filter_shape)));
    PADDLE_ENFORCE_EQ((filter_shape[0] % 32 == 0 && filter_shape[3] % 8 == 0),
                      true,
                      common::errors::InvalidArgument(
                          "The input channel is expected to be multiple of 8, "
                          "and the output channel is expected to be multiple "
                          "of 32. But received input channel is %d, output "
                          "channel is %d.",
                          filter_shape[3],
                          filter_shape[0]));
    PADDLE_ENFORCE_EQ(
        output_shape.size(),
        4U,
        common::errors::InvalidArgument(
            "The size of output_shape is expected to 4. But received "
            "filter_shape's size is %d, filter_shape is [%s].",
            output_shape.size(),
            common::make_ddim(output_shape)));
    is_support = IsSupport(ctx, filter_shape, stride, dilation, group);
    PADDLE_ENFORCE_EQ(
        is_support,
        true,
        common::errors::InvalidArgument(
            "Current test is only supported in the platforms with "
            "compatibility greater than or equal to 70 and the kernel size "
            "must be equal to 1 or 3. When the kernel size is 1, "
            "the stride must be 1 if the compatibility is equal to 70. "
            "Besides, the dilation and group must be equal to 1. But received "
            "compatibility is %d, kernel size is %d, stride is %d, "
            "dilation is %d, group is %d",
            ctx.GetComputeCapability(),
            filter_shape[1],
            stride,
            dilation,
            group));

    for (size_t i = 0; i < input_shape.size(); ++i) {
      in_dims.push_back(input_shape[i]);
    }
    for (size_t i = 0; i < filter_shape.size(); ++i) {
      filter_dims.push_back(filter_shape[i]);
    }
    paddings = {padding, padding};
    strides = {stride, stride};
    dilations = {dilation, dilation};

    in_desc.set(input_shape, format, dtype);
    filter_desc.set(filter_shape, format, dtype, group);
    out_desc.set(output_shape, format, dtype);

    int output_channel = filter_shape[0];
    std::vector<int> stats_shape = {1, 1, 1, output_channel};
    out_stats_desc.set(stats_shape, format, compute_type);

    conv_desc.set(dtype, paddings, strides, dilations, false, group);
  }

  bool IsSupport(const phi::GPUContext &ctx,
                 const std::vector<int> &filter_shape,
                 int stride,
                 int dilation,
                 int group) {
    int kernel_size = filter_shape[1];
    if (dilation != 1 || group != 1) {
      return false;
    }
    if (ctx.GetComputeCapability() == 70) {
      if ((kernel_size == 3) || ((kernel_size == 1) && (stride == 1))) {
        return true;
      }
    } else if (ctx.GetComputeCapability() > 70) {
      if ((kernel_size == 3) || (kernel_size == 1)) {
        return true;
      }
    }
    return false;
  }

  cudnnDataType_t dtype;
  cudnnTensorFormat_t format;
  cudnnDataType_t compute_type;

  std::vector<int64_t> in_dims;
  std::vector<int64_t> filter_dims;
  std::vector<int> strides;
  std::vector<int> paddings;
  std::vector<int> dilations;

  phi::backends::gpu::TensorDescriptor in_desc;
  phi::backends::gpu::FilterDescriptor filter_desc;
  phi::backends::gpu::TensorDescriptor out_desc;
  phi::backends::gpu::TensorDescriptor out_stats_desc;
  phi::backends::gpu::ConvolutionDescriptor conv_desc;

  bool is_support;
};

template <typename T>
class CudnnNormConvolution {
 public:
  CudnnNormConvolution(const phi::GPUContext &ctx,
                       const std::vector<int> &input_shape,
                       const std::vector<int> &filter_shape,
                       const std::vector<int> &output_shape,
                       const int &padding,
                       const int &stride,
                       const int &dilation,
                       const int &group) {
    args_.Set(ctx,
              input_shape,
              filter_shape,
              output_shape,
              padding,
              stride,
              dilation,
              group);
  }
  ~CudnnNormConvolution() {}

  void Forward(const phi::GPUContext &ctx,
               const phi::DenseTensor &input,
               const phi::DenseTensor &filter,
               phi::DenseTensor *output,
               phi::DenseTensor *sum,
               phi::DenseTensor *sum_of_squares) {
    auto cudnn_handle = ctx.cudnn_handle();

    CudnnFusionOp *fwd_op = GetForwardOp(ctx);
    size_t workspace_size = RoundUp(
        static_cast<int64_t>(fwd_op->GetWorkspaceSizeInBytes(cudnn_handle)),
        512);

    // Set variant_param
    // input ptr
    T *input_ptr = const_cast<T *>(input.data<T>());
    T *filter_ptr = const_cast<T *>(filter.data<T>());
    fwd_op->SetOpVariantParamAttrPtr(CUDNN_PTR_XDATA, input_ptr);
    fwd_op->SetOpVariantParamAttrPtr(CUDNN_PTR_WDATA, filter_ptr);
    fwd_op->SetOpVariantParamAttrPtr(
        CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES, &workspace_size);

    // output ptr
    T *output_ptr = ctx.template Alloc<T>(output, output->numel() * sizeof(T));
    float *sum_ptr =
        ctx.template Alloc<float>(sum, sum->numel() * sizeof(float));
    float *sum_of_squares_ptr = ctx.template Alloc<float>(
        sum_of_squares, sum_of_squares->numel() * sizeof(float));
    fwd_op->SetOpVariantParamAttrPtr(CUDNN_PTR_YDATA, output_ptr);
    fwd_op->SetOpVariantParamAttrPtr(CUDNN_PTR_YSUM, sum_ptr);
    fwd_op->SetOpVariantParamAttrPtr(CUDNN_PTR_YSQSUM, sum_of_squares_ptr);

    ctx.cudnn_workspace_handle().RunFunc(
        [&](void *workspace_ptr) {
          // workspace ptr
          fwd_op->SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE, workspace_ptr);
          // fused op execute
          fwd_op->Execute(cudnn_handle);
        },
        workspace_size);
  }

 private:
  CudnnFusionOp *GetForwardOp(const phi::GPUContext &ctx) {
    phi::funcs::AlgorithmsCache<CudnnFusionOp *> &cache =
        *(CudnnFusionOpCache::Instance().GetForward());

    CudnnFusionOp *fwd_op = cache.GetAlgorithm(
        args_.in_dims,
        args_.filter_dims,
        args_.strides,
        args_.paddings,
        args_.dilations,
        0,
        static_cast<int64_t>(args_.dtype),
        [&]() {
          CudnnFusionOp *fwd_op =
              new CudnnFusionOp(CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS);

          // Set constant_param
          fwd_op->SetOpConstParamAttr({CUDNN_PARAM_XDATA_PLACEHOLDER,
                                       CUDNN_PARAM_WDATA_PLACEHOLDER,
                                       CUDNN_PARAM_YDATA_PLACEHOLDER},
                                      CUDNN_PTR_16B_ALIGNED);
          fwd_op->SetOpConstParamAttr(
              {CUDNN_PARAM_YSUM_PLACEHOLDER, CUDNN_PARAM_YSQSUM_PLACEHOLDER},
              CUDNN_PTR_16B_ALIGNED);

          // conv desc
          fwd_op->SetOpConstParamDesc(CUDNN_PARAM_CONV_DESC,
                                      args_.conv_desc.desc());
          // input desc
          fwd_op->SetOpConstParamDesc(CUDNN_PARAM_XDESC, args_.in_desc.desc());
          // filter desc
          fwd_op->SetOpConstParamDesc(CUDNN_PARAM_WDESC,
                                      args_.filter_desc.desc());
          // output desc
          fwd_op->SetOpConstParamDesc(CUDNN_PARAM_YDESC, args_.out_desc.desc());
          // output_stats desc
          fwd_op->SetOpConstParamDesc(CUDNN_PARAM_YSTATS_DESC,
                                      args_.out_stats_desc.desc());
          // batch_norm mode
          fwd_op->SetOpConstParamAttr(CUDNN_PARAM_BN_MODE,
                                      CUDNN_BATCHNORM_SPATIAL_PERSISTENT);

          // Make cudnn fused ops plan
          fwd_op->GetWorkspaceSizeInBytes(ctx.cudnn_handle());
          return fwd_op;
        });
    return fwd_op;
  }

 private:
  NormConvolutionArgs<T> args_;
};

template <typename T>
class CudnnNormConvolutionGrad {
 public:
  CudnnNormConvolutionGrad(const phi::GPUContext &ctx,
                           const std::vector<int> &input_shape,
                           const std::vector<int> &filter_shape,
                           const std::vector<int> &output_shape,
                           const int &padding,
                           const int &stride,
                           const int &dilation,
                           const int &group) {
    args_.Set(ctx,
              input_shape,
              filter_shape,
              output_shape,
              padding,
              stride,
              dilation,
              group);
    dgrad_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  }
  ~CudnnNormConvolutionGrad() {}

  void Backward(const phi::GPUContext &ctx,
                const phi::DenseTensor &input,
                const phi::DenseTensor &filter,
                const phi::DenseTensor &output_grad,
                phi::DenseTensor *input_grad,
                phi::DenseTensor *filter_grad,
                bool use_addto = false) {
    T *input_ptr = const_cast<T *>(input.data<T>());
    T *filter_ptr = const_cast<T *>(filter.data<T>());
    T *output_grad_ptr = const_cast<T *>(output_grad.data<T>());

    if (filter_grad) {
      T *filter_grad_ptr =
          ctx.template Alloc<T>(filter_grad, filter_grad->numel() * sizeof(T));
      BackwardFilter(ctx, output_grad_ptr, input_ptr, filter_grad_ptr);
    }
    if (input_grad) {
      T *input_grad_ptr =
          ctx.template Alloc<T>(input_grad, input_grad->numel() * sizeof(T));
      BackwardData(ctx, output_grad_ptr, filter_ptr, input_grad_ptr, use_addto);
    }
  }

 private:
  void BackwardFilter(const phi::GPUContext &ctx,
                      T *output_grad_ptr,
                      T *input_ptr,
                      T *filter_grad_ptr) {
    auto cudnn_handle = ctx.cudnn_handle();

    CudnnFusionOp *wgrad_op = GetBackwardFilterOp(ctx);
    size_t workspace_size = RoundUp(
        static_cast<int64_t>(wgrad_op->GetWorkspaceSizeInBytes(cudnn_handle)),
        512);

    wgrad_op->SetOpVariantParamAttrPtr(CUDNN_PTR_XDATA, input_ptr);
    wgrad_op->SetOpVariantParamAttrPtr(CUDNN_PTR_DYDATA, output_grad_ptr);
    wgrad_op->SetOpVariantParamAttrPtr(CUDNN_PTR_DWDATA, filter_grad_ptr);
    wgrad_op->SetOpVariantParamAttrPtr(
        CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES, &workspace_size);

    ctx.cudnn_workspace_handle().RunFunc(
        [&](void *workspace_ptr) {
          // workspace ptr
          wgrad_op->SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE,
                                             workspace_ptr);
          // fused op execute
          wgrad_op->Execute(cudnn_handle);
        },
        workspace_size);
  }

  void BackwardData(const phi::GPUContext &ctx,
                    T *output_grad_ptr,
                    T *filter_ptr,
                    T *input_grad_ptr,
                    bool use_addto = false) {
    auto cudnn_handle = ctx.cudnn_handle();
    size_t workspace_size = GetWorkspaceSizeBwdData(ctx);

    // Convolution dgrad followed optionally by batchnorm dgrad
    ScalingParamType<T> alpha = 1.0f;
    ScalingParamType<T> beta = use_addto ? 1.0f : 0.0f;
    ctx.cudnn_workspace_handle().RunFunc(
        [&](void *cudnn_workspace_ptr) {
          PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnConvolutionBackwardData(
              cudnn_handle,
              &alpha,
              args_.filter_desc.desc(),
              filter_ptr,
              args_.out_desc.desc(),
              output_grad_ptr,
              args_.conv_desc.desc(),
              dgrad_algo_,
              cudnn_workspace_ptr,
              workspace_size,
              &beta,
              args_.in_desc.desc(),
              input_grad_ptr));
        },
        workspace_size);
  }

  CudnnFusionOp *GetBackwardFilterOp(const phi::GPUContext &ctx) {
    phi::funcs::AlgorithmsCache<CudnnFusionOp *> &cache =
        *(CudnnFusionOpCache::Instance().GetBackward());

    CudnnFusionOp *wgrad_op = cache.GetAlgorithm(
        args_.in_dims,
        args_.filter_dims,
        args_.strides,
        args_.paddings,
        args_.dilations,
        0,
        static_cast<int64_t>(args_.dtype),
        [&]() {
          CudnnFusionOp *wgrad_op =
              new CudnnFusionOp(CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD);

          wgrad_op->SetOpConstParamAttr({CUDNN_PARAM_DYDATA_PLACEHOLDER,
                                         CUDNN_PARAM_XDATA_PLACEHOLDER,
                                         CUDNN_PARAM_DWDATA_PLACEHOLDER},
                                        CUDNN_PTR_16B_ALIGNED);

          // conv desc
          wgrad_op->SetOpConstParamDesc(CUDNN_PARAM_CONV_DESC,
                                        args_.conv_desc.desc());
          // input desc
          wgrad_op->SetOpConstParamDesc(CUDNN_PARAM_XDESC,
                                        args_.in_desc.desc());
          // filter desc
          wgrad_op->SetOpConstParamDesc(CUDNN_PARAM_DWDESC,
                                        args_.filter_desc.desc());
          // output desc
          wgrad_op->SetOpConstParamDesc(CUDNN_PARAM_DYDESC,
                                        args_.out_desc.desc());
          wgrad_op->SetOpConstParamAttr(CUDNN_PARAM_BN_MODE,
                                        CUDNN_BATCHNORM_SPATIAL_PERSISTENT);

          // Make cudnn fused ops plan
          wgrad_op->GetWorkspaceSizeInBytes(ctx.cudnn_handle());
          return wgrad_op;
        });
    return wgrad_op;
  }

  size_t GetWorkspaceSizeBwdData(const phi::GPUContext &ctx) {
    size_t workspace_size = 0U;
    auto handle = ctx.cudnn_handle();
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle,
            args_.filter_desc.desc(),
            args_.out_desc.desc(),
            args_.conv_desc.desc(),
            args_.in_desc.desc(),
            dgrad_algo_,
            &workspace_size));
    return RoundUp(workspace_size, 512);
  }

 private:
  NormConvolutionArgs<T> args_;
  cudnnConvolutionBwdDataAlgo_t dgrad_algo_;
};

#endif
}  // namespace fusion
}  // namespace phi
