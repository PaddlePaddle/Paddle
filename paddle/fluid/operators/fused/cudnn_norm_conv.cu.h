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

#pragma once

#include "paddle/fluid/operators/fused/cudnn_fusion_helper.h"
#include "paddle/fluid/platform/cudnn_desc.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
namespace dynload = platform::dynload;

template <typename T>
using ScalingParamType = typename platform::CudnnDataType<T>::ScalingParamType;

#if CUDNN_VERSION >= 8000

static size_t RoundUp(int64_t a, int64_t b) { return (a + b - 1) / b * b; }

template <typename T>
struct NormConvolutionArgs {
  NormConvolutionArgs() {
    dtype = platform::CudnnDataType<T>::type;
    format = CUDNN_TENSOR_NHWC;
    compute_type = platform::CudnnDataType<float>::type;
  }

  void Set(const std::vector<int> &input_shape,
           const std::vector<int> &filter_shape,
           const std::vector<int> &output_shape, int padding, int stride,
           int dilation, int group) {
    for (size_t i = 0; i < input_shape.size(); ++i) {
      in_dims[i] = input_shape[i];
    }
    for (size_t i = 0; i < filter_shape.size(); ++i) {
      filter_dims[i] = filter_shape[i];
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

  cudnnDataType_t dtype;
  cudnnTensorFormat_t format;
  cudnnDataType_t compute_type;

  std::vector<int64_t> in_dims;
  std::vector<int64_t> filter_dims;
  std::vector<int> strides;
  std::vector<int> paddings;
  std::vector<int> dilations;

  platform::TensorDescriptor in_desc;
  platform::FilterDescriptor filter_desc;
  platform::TensorDescriptor out_desc;
  platform::TensorDescriptor out_stats_desc;
  platform::ConvolutionDescriptor conv_desc;
};

template <typename T>
class CudnnNormConvolution {
 public:
  CudnnNormConvolution(const platform::CUDADeviceContext &ctx,
                       const std::vector<int> &input_shape,
                       const std::vector<int> &filter_shape,
                       const std::vector<int> &output_shape, const int &padding,
                       const int &stride, const int &dilation,
                       const int &group) {
    args_.Set(input_shape, filter_shape, output_shape, padding, stride,
              dilation, group);
  }
  ~CudnnNormConvolution() {}

  void Forward(const platform::CUDADeviceContext &ctx, T *input_ptr,
               T *filter_ptr, T *output_ptr, float *sum_ptr,
               float *sum_of_squares_ptr) {
    auto cudnn_handle = ctx.cudnn_handle();

    CudnnFusionOp *fwd_op = GetForwardOp(ctx);
    size_t workspace_size = RoundUp(
        static_cast<int64_t>(fwd_op->GetWorkspaceSizeInBytes(cudnn_handle)),
        512);

    // Set variant_param
    // input ptr
    fwd_op->SetOpVariantParamAttrPtr(CUDNN_PTR_XDATA, input_ptr);
    fwd_op->SetOpVariantParamAttrPtr(CUDNN_PTR_WDATA, filter_ptr);
    fwd_op->SetOpVariantParamAttrPtr(
        CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES, &workspace_size);

    // output ptr
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
  CudnnFusionOp *GetForwardOp(const platform::CUDADeviceContext &ctx) {
    framework::AlgorithmsCache<CudnnFusionOp *> &cache =
        *(CudnnFusionOpCache::Instance().Get());

    CudnnFusionOp *fwd_op = cache.GetAlgorithm(
        args_.in_dims, args_.filter_dims, args_.strides, args_.paddings,
        args_.dilations, 0, static_cast<int64_t>(args_.dtype), [&]() {
          CudnnFusionOp *fwd_op =
              new CudnnFusionOp(CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS);

          // Set constant_param
          fwd_op->SetOpConstParamAttr(
              {CUDNN_PARAM_XDATA_PLACEHOLDER, CUDNN_PARAM_WDATA_PLACEHOLDER,
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
  CudnnNormConvolutionGrad(const platform::CUDADeviceContext &ctx,
                           const std::vector<int> &input_shape,
                           const std::vector<int> &filter_shape,
                           const std::vector<int> &output_shape,
                           const int &padding, const int &stride,
                           const int &dilation, const int &group) {
    args_.Set(input_shape, filter_shape, output_shape, padding, stride,
              dilation, group);
    dgrad_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  }
  ~CudnnNormConvolutionGrad() {}

  void Backward(const platform::CUDADeviceContext &ctx, T *input_ptr,
                T *output_ptr, T *filter_ptr, T *input_grad_ptr,
                T *filter_grad_ptr) {
    if (filter_grad_ptr) {
      BackwardFilter(ctx, input_ptr, output_ptr, filter_ptr, filter_grad_ptr);
    }
    if (input_grad_ptr) {
      BackwardData(ctx, input_ptr, output_ptr, filter_ptr, input_grad_ptr);
    }
  }

 private:
  void BackwardFilter(const platform::CUDADeviceContext &ctx, T *input_ptr,
                      T *output_ptr, T *filter_ptr, T *filter_grad_ptr) {
    auto cudnn_handle = ctx.cudnn_handle();

    CudnnFusionOp *wgrad_op = GetBackwardFilterOp(ctx);
    size_t workspace_size = RoundUp(
        static_cast<int64_t>(wgrad_op->GetWorkspaceSizeInBytes(cudnn_handle)),
        512);

    wgrad_op->SetOpVariantParamAttrPtr(CUDNN_PTR_XDATA, input_ptr);
    wgrad_op->SetOpVariantParamAttrPtr(CUDNN_PTR_DYDATA, output_ptr);
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

  void BackwardData(const platform::CUDADeviceContext &ctx, T *input_ptr,
                    T *output_ptr, T *filter_ptr, T *input_grad_ptr,
                    bool use_addto = false) {
    auto cudnn_handle = ctx.cudnn_handle();
    size_t workspace_size = GetWorkspaceSizeBwdData(ctx);

    // Convolution dgrad followed optionally by batchnorm dgrad
    ScalingParamType<T> alpha = 1.0f;
    ScalingParamType<T> beta = use_addto ? 1.0f : 0.0f;
    ctx.cudnn_workspace_handle().RunFunc(
        [&](void *cudnn_workspace_ptr) {
          PADDLE_ENFORCE_CUDA_SUCCESS(
              platform::dynload::cudnnConvolutionBackwardData(
                  cudnn_handle, &alpha, args_.filter_desc.desc(), filter_ptr,
                  args_.out_desc.desc(), output_ptr, args_.conv_desc.desc(),
                  dgrad_algo_, cudnn_workspace_ptr, workspace_size, &beta,
                  args_.in_desc.desc(), input_grad_ptr));
        },
        workspace_size);
  }

  CudnnFusionOp *GetBackwardFilterOp(const platform::CUDADeviceContext &ctx) {
    framework::AlgorithmsCache<CudnnFusionOp *> &cache =
        *(CudnnFusionOpCache::Instance().Get());

    CudnnFusionOp *wgrad_op = cache.GetAlgorithm(
        args_.in_dims, args_.filter_dims, args_.strides, args_.paddings,
        args_.dilations, 0, static_cast<int64_t>(args_.dtype), [&]() {
          CudnnFusionOp *wgrad_op =
              new CudnnFusionOp(CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD);

          wgrad_op->SetOpConstParamAttr(
              {CUDNN_PARAM_DYDATA_PLACEHOLDER, CUDNN_PARAM_XDATA_PLACEHOLDER,
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

  size_t GetWorkspaceSizeBwdData(const platform::CUDADeviceContext &ctx) {
    size_t workspace_size = 0U;
    auto handle = ctx.cudnn_handle();
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle, args_.filter_desc.desc(), args_.out_desc.desc(),
            args_.conv_desc.desc(), args_.in_desc.desc(), dgrad_algo_,
            &workspace_size));
    return RoundUp(workspace_size, 512);
  }

 private:
  NormConvolutionArgs<T> args_;
  cudnnConvolutionBwdDataAlgo_t dgrad_algo_;
};

#endif
}  // namespace operators
}  // namespace paddle
