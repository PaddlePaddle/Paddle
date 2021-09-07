// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/fused/cudnn_fusion_helper.h"
#include "paddle/fluid/operators/fused/resnet_unit_op.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
namespace dynload = platform::dynload;
template <typename T>
class CuDNNNormConvolutionOp {
 public:
  CuDNNNormConvolutionOp()
#if CUDNN_VERSION >= 8000
      : fwd_op_(CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS)
#endif
  {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateTensorDescriptor(&in_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateTensorDescriptor(&out_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateTensorDescriptor(&out_stats_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateFilterDescriptor(&filter_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateConvolutionDescriptor(&conv_desc_));
  }

  ~CuDNNNormConvolutionOp() {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(in_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(out_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(out_stats_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyFilterDescriptor(filter_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyConvolutionDescriptor(conv_desc_));
  }

  void Init(const framework::ExecutionContext &ctx,
            const std::vector<int> &input_shape,
            const std::vector<int> &filter_shape,
            const std::vector<int> &output_shape) {
#if CUDNN_VERSION < 8000
    LOG(FATAL) << "cuDNN version 8.0 or later is required.";
#else
    cudnn_fwd_compute_type_ = platform::CudnnDataType<float>::type;
    dtype_ = platform::CudnnDataType<T>::type;
    format_ = CUDNN_TENSOR_NHWC;

    InitDescriptors(ctx, input_shape, filter_shape, output_shape);
    GetTempSize(ctx);
#endif  // CUDNN_VERSION >= 8000
  }

  void Forward(const framework::ExecutionContext &ctx, const T *input_ptr,
               const T *filter_ptr, T *output_ptr, float *sum_ptr,
               float *sum_of_squares_ptr) {
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();
#if CUDNN_VERSION < 8000
    LOG(FATAL) << "cuDNN version 8.0 or later is required.";
#else
    // Set variant_param
    // input ptr
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_XDATA,
                                     const_cast<T *>(input_ptr));
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_WDATA,
                                     const_cast<T *>(filter_ptr));
    fwd_op_.SetOpVariantParamAttrPtr(
        CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES, &fwd_workspace_byte_);
    // output ptr
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_YDATA, output_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_YSUM, sum_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_YSQSUM, sum_of_squares_ptr);
    workspace_handle.RunFunc(
        [&](void *workspace_ptr) {
          // workspace ptr
          fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE, workspace_ptr);
          // fused op execute
          fwd_op_.Execute(handle);
        },
        fwd_workspace_byte_);
#endif  // CUDNN_VERSION < 8000
  }

  // TBD
  void Backward(const framework::ExecutionContext &ctx) {}

 private:
  void InitDescriptors(const framework::ExecutionContext &ctx,
                       const std::vector<int> &input_shape,
                       const std::vector<int> &filter_shape,
                       const std::vector<int> &output_shape) {
#if CUDNN_VERSION < 8000
    LOG(FATAL) << "cuDNN version 8.0 or later is required.";
#else
    // Set constant_param
    fwd_op_.SetOpConstParamAttr(
        {CUDNN_PARAM_XDATA_PLACEHOLDER, CUDNN_PARAM_WDATA_PLACEHOLDER,
         CUDNN_PARAM_YDATA_PLACEHOLDER},
        CUDNN_PTR_16B_ALIGNED);
    fwd_op_.SetOpConstParamAttr(
        {CUDNN_PARAM_YSUM_PLACEHOLDER, CUDNN_PARAM_YSQSUM_PLACEHOLDER},
        CUDNN_PTR_16B_ALIGNED);

    auto pad = ctx.Attr<int>("pad");
    auto stride = ctx.Attr<int>("stride");
    auto dilate = ctx.Attr<int>("dilate");
    auto group = ctx.Attr<int>("group");
    int output_channel = filter_shape[0];
    std::vector<int> stats_shape = {1, output_channel, 1, 1};

    // set conv desc
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetConvolution2dDescriptor(
        conv_desc_, pad, pad, stride, stride, dilate, dilate,
        CUDNN_CROSS_CORRELATION, cudnn_fwd_compute_type_));
    cudnnMathType_t math_type = CUDNN_TENSOR_OP_MATH;
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnSetConvolutionMathType(conv_desc_, math_type));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnSetConvolutionGroupCount(conv_desc_, group));
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_CONV_DESC, conv_desc_);

    // set input desc
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptorEx(
        in_desc_, format_, dtype_, input_shape.size(), input_shape.data()));
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_XDESC, in_desc_);

    // set filter desc
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetFilter4dDescriptor(
        filter_desc_, dtype_, format_, filter_shape[0], filter_shape[1],
        filter_shape[2], filter_shape[3]));
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_WDESC, filter_desc_);

    // set output desc
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptorEx(
        out_desc_, format_, dtype_, output_shape.size(), output_shape.data()));
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_YDESC, out_desc_);

    // set output_stats desc
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptorEx(
        out_stats_desc_, format_, cudnn_fwd_compute_type_, stats_shape.size(),
        stats_shape.data()));
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_YSTATS_DESC, out_stats_desc_);

    fwd_op_.SetOpConstParamAttr(CUDNN_PARAM_BN_MODE, CUDNN_BATCHNORM_SPATIAL);

#endif  // CUDNN_VERSION < 8000
  }

  void GetTempSize(const framework::ExecutionContext &ctx) {
#if CUDNN_VERSION < 8000
    LOG(FATAL) << "cuDNN version 8.0 or later is required.";
#else
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    fwd_workspace_byte_ = fwd_op_.GetWorkspaceSizeInBytes(handle);
#endif  // CUDNN_VERSION < 8000
  }

  size_t fwd_workspace_byte_ = 0;

  cudnnDataType_t dtype_;
  cudnnDataType_t cudnn_fwd_compute_type_;
  cudnnTensorDescriptor_t in_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnTensorDescriptor_t out_stats_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnTensorFormat_t format_;

#if CUDNN_VERSION >= 8000
  CuDNNFusionOp fwd_op_;
#endif
};
}  // namespace operators
}  // namespace paddle
