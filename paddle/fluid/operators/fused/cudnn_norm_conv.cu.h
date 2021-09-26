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

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
namespace dynload = platform::dynload;

template <typename T>
using ScalingParamType = typename platform::CudnnDataType<T>::ScalingParamType;

#if CUDNN_VERSION >= 8000
template <typename T>
class CudnnNormConvolutionOp {
 public:
  CudnnNormConvolutionOp()
      : fwd_op_(CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS),
        bwd_wgrad_op_(CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD),
        back_conv_dgrad_algo_(CUDNN_CONVOLUTION_BWD_DATA_ALGO_1) {}
  ~CudnnNormConvolutionOp() {}

  void Init(const platform::CUDADeviceContext &ctx,
            const std::vector<int> &input_shape,
            const std::vector<int> &filter_shape,
            const std::vector<int> &output_shape, const int &pad,
            const int &stride, const int &dilate, const int &group) {
    cudnn_fwd_compute_type_ = platform::CudnnDataType<float>::type;
    dtype_ = platform::CudnnDataType<T>::type;
    format_ = CUDNN_TENSOR_NHWC;

    InitDescriptors(ctx, input_shape, filter_shape, output_shape, pad, stride,
                    dilate, group);
    GetWorkspaceSize(ctx);
  }

  void Forward(const platform::CUDADeviceContext &ctx, T *input_ptr,
               T *filter_ptr, T *output_ptr, float *sum_ptr,
               float *sum_of_squares_ptr) {
    auto handle = ctx.cudnn_handle();
    auto workspace_handle = ctx.cudnn_workspace_handle();
    // Set variant_param
    // input ptr
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_XDATA, input_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_WDATA, filter_ptr);
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
  }

  // TBD
  void Backward(const platform::CUDADeviceContext &ctx, T *input_ptr,
                T *output_ptr, T *filter_ptr, T *input_grad_ptr,
                T *filter_grad_ptr) {
    auto handle = ctx.cudnn_handle();
    auto workspace_handle = ctx.cudnn_workspace_handle();
    auto bwd_workspace_byte_ =
        std::max(bwd_wgrad_workspace_byte_, bwd_dgrad_conv_workspace_byte_);

    bwd_wgrad_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_XDATA, input_ptr);
    bwd_wgrad_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_DYDATA, output_ptr);
    bwd_wgrad_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_DWDATA, filter_grad_ptr);
    bwd_wgrad_op_.SetOpVariantParamAttrPtr(
        CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES,
        &bwd_wgrad_workspace_byte_);
    // bwd_wgrad_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQSCALE,
    // equiv_scale_ptr);
    // bwd_wgrad_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQBIAS,
    // equiv_bias_ptr);
    workspace_handle.RunFunc(
        [&](void *workspace_ptr) {
          // workspace ptr
          bwd_wgrad_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE,
                                                 workspace_ptr);
          // fused op execute
          bwd_wgrad_op_.Execute(handle);
        },
        bwd_wgrad_workspace_byte_);

    // DGRAD - Convolution dgrad followed optionally by batchnorm dgrad
    ScalingParamType<T> alpha = 1.0f;
    ScalingParamType<T> beta = 0.0f;
    workspace_handle.RunFunc(
        [&](void *cudnn_workspace_ptr) {
          PADDLE_ENFORCE_CUDA_SUCCESS(
              platform::dynload::cudnnConvolutionBackwardData(
                  handle, &alpha, filter_desc_.desc(), filter_ptr,
                  out_desc_.desc(), output_ptr, conv_desc_.desc(),
                  back_conv_dgrad_algo_, cudnn_workspace_ptr,
                  bwd_workspace_byte_, &beta, in_desc_.desc(), input_grad_ptr));
        },
        bwd_dgrad_conv_workspace_byte_);
  }

 private:
  size_t RoundUp(int64_t a, int64_t b) { return (a + b - 1) / b * b; }

  void InitDescriptors(const platform::CUDADeviceContext &ctx,
                       const std::vector<int> &input_shape,
                       const std::vector<int> &filter_shape,
                       const std::vector<int> &output_shape, const int &pad,
                       const int &stride, const int &dilate, const int &group) {
    // Set constant_param
    fwd_op_.SetOpConstParamAttr(
        {CUDNN_PARAM_XDATA_PLACEHOLDER, CUDNN_PARAM_WDATA_PLACEHOLDER,
         CUDNN_PARAM_YDATA_PLACEHOLDER},
        CUDNN_PTR_16B_ALIGNED);
    fwd_op_.SetOpConstParamAttr(
        {CUDNN_PARAM_YSUM_PLACEHOLDER, CUDNN_PARAM_YSQSUM_PLACEHOLDER},
        CUDNN_PTR_16B_ALIGNED);

    bwd_wgrad_op_.SetOpConstParamAttr(
        {CUDNN_PARAM_DYDATA_PLACEHOLDER, CUDNN_PARAM_XDATA_PLACEHOLDER,
         CUDNN_PARAM_DWDATA_PLACEHOLDER},
        CUDNN_PTR_16B_ALIGNED);

    std::vector<int> pad_vec = {pad, pad};
    std::vector<int> stride_vec = {stride, stride};
    std::vector<int> dilate_vec = {dilate, dilate};
    int output_channel = filter_shape[0];
    std::vector<int> stats_shape = {1, 1, 1, output_channel};

    // set conv desc
    conv_desc_.set(dtype_, pad_vec, stride_vec, dilate_vec, false, group);
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_CONV_DESC, conv_desc_.desc());
    bwd_wgrad_op_.SetOpConstParamDesc(CUDNN_PARAM_CONV_DESC, conv_desc_.desc());

    // set input desc
    in_desc_.set(input_shape, format_, dtype_);
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_XDESC, in_desc_.desc());
    bwd_wgrad_op_.SetOpConstParamDesc(CUDNN_PARAM_XDESC, in_desc_.desc());

    // set filter desc
    filter_desc_.set(filter_shape, format_, dtype_, group);
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_WDESC, filter_desc_.desc());
    bwd_wgrad_op_.SetOpConstParamDesc(CUDNN_PARAM_DWDESC, filter_desc_.desc());

    // set output desc
    out_desc_.set(output_shape, format_, dtype_);
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_YDESC, out_desc_.desc());
    bwd_wgrad_op_.SetOpConstParamDesc(CUDNN_PARAM_DYDESC, out_desc_.desc());

    // set output_stats desc
    out_stats_desc_.set(stats_shape, format_, cudnn_fwd_compute_type_);
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_YSTATS_DESC,
                                out_stats_desc_.desc());

    fwd_op_.SetOpConstParamAttr(CUDNN_PARAM_BN_MODE,
                                CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
    bwd_wgrad_op_.SetOpConstParamAttr(CUDNN_PARAM_BN_MODE,
                                      CUDNN_BATCHNORM_SPATIAL_PERSISTENT);

    auto handle = ctx.cudnn_handle();
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle, filter_desc_.desc(), out_desc_.desc(), conv_desc_.desc(),
            in_desc_.desc(), back_conv_dgrad_algo_,
            &bwd_dgrad_conv_workspace_byte_));
  }

  void GetWorkspaceSize(const platform::CUDADeviceContext &ctx) {
    auto handle = ctx.cudnn_handle();
    fwd_workspace_byte_ = fwd_op_.GetWorkspaceSizeInBytes(handle);
    bwd_wgrad_workspace_byte_ = bwd_wgrad_op_.GetWorkspaceSizeInBytes(handle);
    const size_t dptr_alignment = 512;
    bwd_wgrad_workspace_byte_ =
        RoundUp(bwd_wgrad_workspace_byte_, dptr_alignment);
  }

  size_t fwd_workspace_byte_ = 0;
  size_t bwd_wgrad_workspace_byte_ = 0;
  size_t bwd_dgrad_conv_workspace_byte_ = 0;

  cudnnDataType_t dtype_;
  cudnnDataType_t cudnn_fwd_compute_type_;
  platform::TensorDescriptor in_desc_;
  platform::FilterDescriptor filter_desc_;
  platform::TensorDescriptor out_desc_;
  platform::TensorDescriptor out_stats_desc_;
  platform::ConvolutionDescriptor conv_desc_;
  cudnnTensorFormat_t format_;

  CudnnFusionOp fwd_op_;
  CudnnFusionOp bwd_wgrad_op_;
  cudnnConvolutionBwdDataAlgo_t back_conv_dgrad_algo_;
};
#endif
}  // namespace operators
}  // namespace paddle
