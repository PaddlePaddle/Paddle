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

#if CUDNN_VERSION >= 8000
template <typename T>
class CudnnNormConvolutionOp {
 public:
  CudnnNormConvolutionOp()
      : fwd_op_(CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS) {}
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
  void Backward(const platform::CUDADeviceContext &ctx) {}

 private:
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

    std::vector<int> pad_vec = {pad, pad};
    std::vector<int> stride_vec = {stride, stride};
    std::vector<int> dilate_vec = {dilate, dilate};
    int output_channel = filter_shape[0];
    std::vector<int> stats_shape = {1, 1, 1, output_channel};

    // set conv desc
    conv_desc_.set(dtype_, pad_vec, stride_vec, dilate_vec, false, group);
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_CONV_DESC, conv_desc_.desc());

    // set input desc
    in_desc_.set(input_shape, format_, dtype_);
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_XDESC, in_desc_.desc());

    // set filter desc
    filter_desc_.set(filter_shape, format_, dtype_, group);
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_WDESC, filter_desc_.desc());

    // set output desc
    out_desc_.set(output_shape, format_, dtype_);
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_YDESC, out_desc_.desc());

    // set output_stats desc
    out_stats_desc_.set(stats_shape, format_, cudnn_fwd_compute_type_);
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_YSTATS_DESC,
                                out_stats_desc_.desc());

    fwd_op_.SetOpConstParamAttr(CUDNN_PARAM_BN_MODE, CUDNN_BATCHNORM_SPATIAL);
  }

  void GetWorkspaceSize(const platform::CUDADeviceContext &ctx) {
    auto handle = ctx.cudnn_handle();
    fwd_workspace_byte_ = fwd_op_.GetWorkspaceSizeInBytes(handle);
  }

  size_t fwd_workspace_byte_ = 0;

  cudnnDataType_t dtype_;
  cudnnDataType_t cudnn_fwd_compute_type_;
  platform::TensorDescriptor in_desc_;
  platform::FilterDescriptor filter_desc_;
  platform::TensorDescriptor out_desc_;
  platform::TensorDescriptor out_stats_desc_;
  platform::ConvolutionDescriptor conv_desc_;
  cudnnTensorFormat_t format_;

  CudnnFusionOp fwd_op_;
};
#endif
}  // namespace operators
}  // namespace paddle
