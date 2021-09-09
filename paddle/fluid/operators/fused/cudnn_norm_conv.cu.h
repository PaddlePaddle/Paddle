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
#if CUDNN_VERSION < 8000
  LOG(ERROR) << "cuDNN version 8.0 or later is required.";
#else

 public:
  CuDNNNormConvolutionOp()
      : fwd_op_(CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS) {}
  ~CuDNNNormConvolutionOp() {}

  void Init(const framework::ExecutionContext &ctx,
            const std::vector<int> &input_shape,
            const std::vector<int> &filter_shape,
            const std::vector<int> &output_shape) {
    cudnn_fwd_compute_type_ = platform::CudnnDataType<float>::type;
    dtype_ = platform::CudnnDataType<T>::type;
    format_ = CUDNN_TENSOR_NHWC;

    InitDescriptors(ctx, input_shape, filter_shape, output_shape);
    GetWorkspaceSize(ctx);
  }

  void Forward(const framework::ExecutionContext &ctx, const T *input_ptr,
               const T *filter_ptr, T *output_ptr, float *sum_ptr,
               float *sum_of_squares_ptr) {
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();
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
  }

  // TBD
  void Backward(const framework::ExecutionContext &ctx) {}

 private:
  void InitDescriptors(const framework::ExecutionContext &ctx,
                       const std::vector<int> &input_shape,
                       const std::vector<int> &filter_shape,
                       const std::vector<int> &output_shape) {
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
    std::vector<int> p_vec = {pad, pad};
    std::vector<int> s_vec = {stride, stride};
    std::vector<int> d_vec = {dilate, dilate};
    auto group = ctx.Attr<int>("group");
    int output_channel = filter_shape[0];
    std::vector<int> stats_shape = {1, 1, 1, output_channel};

    // set conv desc
    conv_desc_.set(dtype_, p_vec, s_vec, d_vec, false, group);
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

  void GetWorkspaceSize(const framework::ExecutionContext &ctx) {
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
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

  CuDNNFusionOp fwd_op_;
#endif
};
}  // namespace operators
}  // namespace paddle
