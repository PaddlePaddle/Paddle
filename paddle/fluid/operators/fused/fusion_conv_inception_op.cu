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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/conv_cudnn_op_cache.h"
#include "paddle/fluid/platform/cudnn_helper.h"

DECLARE_uint64(conv_workspace_size_limit);

namespace paddle {
namespace operators {

#if CUDNN_VERSION >= 7100
using Tensor = framework::Tensor;
using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using ScopedFilterDescriptor = platform::ScopedFilterDescriptor;
using ScopedConvolutionDescriptor = platform::ScopedConvolutionDescriptor;
using ScopedActivationDescriptor = platform::ScopedActivationDescriptor;
using DataLayout = platform::DataLayout;

using ScopedPoolingDescriptor = platform::ScopedPoolingDescriptor;
using PoolingMode = platform::PoolingMode;
template <typename T>
using ScalingParamType = typename platform::CudnnDataType<T>::ScalingParamType;

template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;

template <typename T>
class CUDNNConvInceptionFusionOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto* input = ctx.Input<Tensor>("Input");
    auto filters = ctx.MultiInput<framework::Tensor>("Filter");
    auto bias = ctx.MultiInput<framework::Tensor>("Bias");

    auto* output = ctx.Output<Tensor>("Output");
    auto temp_outs = ctx.MultiOutput<framework::Tensor>("TempOutput");

    const std::string pool_type = ctx.Attr<std::string>("pooling_type");
    const std::string activation = ctx.Attr<std::string>("activation");
    const bool exclusive = ctx.Attr<bool>("exclusive");

    int64_t user_workspace_size =
        static_cast<size_t>(ctx.Attr<int>("workspace_size_MB"));

    const T* input_data = input->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    T* temp_data = temp_outs[0]->mutable_data<T>(input->dims(), ctx.GetPlace());

    DataLayout layout = DataLayout::kNCHW;
    std::vector<int> in_dim = framework::vectorize2int(input->dims());

    // ------------------- cudnn descriptors ---------------------
    PoolingMode pooling_mode;
    if (pool_type == "max") {
      pooling_mode = PoolingMode::kMaximum;
    } else {
      pooling_mode = exclusive ? PoolingMode::kAverageExclusive
                               : (PoolingMode::kAverageInclusive);
    }
    std::vector<int> k0x0 = {0, 0};
    std::vector<int> k1x1 = {1, 1};
    std::vector<int> k1x1_2 = {1, 1};
    std::vector<int> k3x3 = {3, 3};
    ScopedPoolingDescriptor pool_desc;
    ScopedActivationDescriptor act_desc;
    ScopedTensorDescriptor out_pool_desc;
    ScopedTensorDescriptor input_desc;
    cudnnPoolingDescriptor_t cudnn_pool_desc =
        pool_desc.descriptor(pooling_mode, k3x3, k1x1, k1x1);

    cudnnTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
        layout, framework::vectorize2int(input->dims()));
    cudnnTensorDescriptor_t pool_out_desc = out_pool_desc.descriptor<T>(
        layout, framework::vectorize2int(input->dims()));

    cudnnDataType_t cudnn_dtype = CudnnDataType<T>::type;
    cudnnTensorDescriptor_t* out_desc = new cudnnTensorDescriptor_t[4];
    cudnnFilterDescriptor_t* filter_desc = new cudnnFilterDescriptor_t[4];
    cudnnTensorDescriptor_t* bias_desc = new cudnnTensorDescriptor_t[4];
    cudnnTensorDescriptor_t* in_desc = new cudnnTensorDescriptor_t[4];
    cudnnConvolutionDescriptor_t* conv_desc =
        new cudnnConvolutionDescriptor_t[4];
    for (int i = 0; i < 4; ++i) {
      CUDNN_ENFORCE(
          platform::dynload::cudnnCreateFilterDescriptor(&filter_desc[i]));
      CUDNN_ENFORCE(
          platform::dynload::cudnnCreateTensorDescriptor(&bias_desc[i]));
      CUDNN_ENFORCE(
          platform::dynload::cudnnCreateTensorDescriptor(&in_desc[i]));
      CUDNN_ENFORCE(
          platform::dynload::cudnnCreateTensorDescriptor(&out_desc[i]));
      CUDNN_ENFORCE(
          platform::dynload::cudnnCreateConvolutionDescriptor(&conv_desc[i]));
    }

    std::vector<std::vector<int>> filter_dims;
    std::vector<std::vector<int>> bias_dims;
    std::vector<std::vector<int>> in_dims;
    std::vector<std::vector<int>> out_dims;
    std::vector<std::vector<int>> in_strides;
    std::vector<std::vector<int>> out_strides;
    std::vector<std::vector<int>> bias_strides;

    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
    int n = in_dim[0];
    int h = in_dim[2];
    int w = in_dim[3];
    int oc = output->dims()[1];

    cudnnDataType_t compute_type = (cudnn_dtype == CUDNN_DATA_DOUBLE)
                                       ? CUDNN_DATA_DOUBLE
                                       : CUDNN_DATA_FLOAT;

    for (int i = 0; i < 4; ++i) {
      filter_dims.push_back(framework::vectorize2int(filters[i]->dims()));
      CUDNN_ENFORCE(platform::dynload::cudnnSetFilterNdDescriptor(
          filter_desc[i], cudnn_dtype, format, 4, filter_dims[i].data()));
      bias_dims.push_back({1, filter_dims[i][0], 1, 1});
      bias_strides.push_back({filter_dims[i][0], 1, 1, 1});
      CUDNN_ENFORCE(platform::dynload::cudnnSetTensorNdDescriptor(
          bias_desc[i], cudnn_dtype, 4, bias_dims[i].data(),
          bias_strides[i].data()));
      in_dims.push_back({n, filter_dims[i][1], h, w});
      out_dims.push_back({n, filter_dims[i][0], h, w});
      in_strides.push_back({filter_dims[i][1] * h * w, h * w, w, 1});
      out_strides.push_back({oc * h * w, h * w, w, 1});

      if (i < 2) {
        CUDNN_ENFORCE(platform::dynload::cudnnSetConvolutionNdDescriptor(
            conv_desc[i], 2, k0x0.data(), k1x1.data(), k1x1.data(),
            CUDNN_CROSS_CORRELATION, compute_type));
      } else {
        CUDNN_ENFORCE(platform::dynload::cudnnSetConvolutionNdDescriptor(
            conv_desc[i], 2, k1x1.data(), k1x1.data(), k1x1.data(),
            CUDNN_CROSS_CORRELATION, compute_type));
      }
      CUDNN_ENFORCE(platform::dynload::cudnnSetConvolutionMathType(
          conv_desc[i], CUDNN_DEFAULT_MATH));
    }
    in_dims[2][1] *= 2;
    in_strides[2][0] = oc * h * w;
    out_strides[2][0] = filter_dims[2][0] * h * w;  // this out is continuous.
    in_strides[3][0] = filter_dims[2][0] * h * w;
    CUDNN_ENFORCE(
        platform::dynload::cudnnSetConvolutionGroupCount(conv_desc[2], 2));

    cudnnConvolutionFwdAlgo_t algo[4];
    auto handle = dev_ctx.cudnn_handle();
    size_t workspace_size_in_bytes = 0;  // final workspace to allocate.

    size_t workspace_size_limit = 0;
    if (FLAGS_conv_workspace_size_limit > 0 || user_workspace_size > 0) {
      int64_t max_user_size =
          std::max(static_cast<int64_t>(FLAGS_conv_workspace_size_limit),
                   user_workspace_size);
      workspace_size_limit = max_user_size * 1024 * 1024;
    }

    for (int i = 0; i < 4; ++i) {
      CUDNN_ENFORCE(platform::dynload::cudnnSetTensorNdDescriptor(
          in_desc[i], cudnn_dtype, 4, in_dims[i].data(), in_strides[i].data()));
      CUDNN_ENFORCE(platform::dynload::cudnnSetTensorNdDescriptor(
          out_desc[i], cudnn_dtype, 4, out_dims[i].data(),
          out_strides[i].data()));
      CUDNN_ENFORCE(platform::dynload::cudnnGetConvolutionForwardAlgorithm(
          handle, in_desc[i], filter_desc[i], conv_desc[i], out_desc[i],
          CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, workspace_size_limit,
          &algo[i]));
      size_t tmp_size = 0;
      CUDNN_ENFORCE(platform::dynload::cudnnGetConvolutionForwardWorkspaceSize(
          handle, in_desc[i], filter_desc[i], conv_desc[i], out_desc[i],
          algo[i], &tmp_size));
      workspace_size_in_bytes = std::max(workspace_size_in_bytes, tmp_size);
    }
    cudnnActivationDescriptor_t cudnn_act_desc =
        act_desc.descriptor<T>(activation);

    int oc0 = filter_dims[0][0];
    int oc1 = filter_dims[1][0] - filter_dims[2][1] * 2;
    int oc3 = filter_dims[3][0];
    int oc2 = oc - oc0 - oc1 - oc3;

    // branch1: pool + 1x1 conv
    ScalingParamType<T> alpha = 1.0f, beta = 0.0f;
    CUDNN_ENFORCE(platform::dynload::cudnnPoolingForward(
        handle, cudnn_pool_desc, &alpha, cudnn_input_desc, input_data, &beta,
        pool_out_desc, temp_data));

    std::vector<const void*> in_datas;
    in_datas.push_back(static_cast<const void*>(temp_data));
    in_datas.push_back(static_cast<const void*>(input_data));
    in_datas.push_back(
        static_cast<const void*>(output_data + (oc0 + oc1) * h * w));
    T* temp2_data = temp_outs[1]->mutable_data<T>(
        framework::make_ddim(out_dims[2]), ctx.GetPlace());
    in_datas.push_back(static_cast<const void*>(temp2_data + oc2 * h * w));

    std::vector<void*> out_datas;
    out_datas.push_back(static_cast<void*>(output_data));
    out_datas.push_back(static_cast<void*>(output_data + oc0 * h * w));
    out_datas.push_back(static_cast<void*>(temp2_data));
    out_datas.push_back(
        static_cast<void*>(output_data + (oc0 + oc1 + oc2) * h * w));

    for (int i = 0; i < 4; ++i) {
      auto func = [&](void* cudnn_workspace) {
        CUDNN_ENFORCE(platform::dynload::cudnnConvolutionBiasActivationForward(
            handle, &alpha, in_desc[i], in_datas[i], filter_desc[i],
            static_cast<const void*>(filters[i]->data<T>()), conv_desc[i],
            algo[i], cudnn_workspace, workspace_size_in_bytes, &beta,
            out_desc[i], out_datas[i], bias_desc[i],
            static_cast<const void*>(bias[i]->data<T>()), cudnn_act_desc,
            out_desc[i], out_datas[i]));
      };
      auto workspace_handle = dev_ctx.cudnn_workspace_handle();
      workspace_handle.RunFunc(func, workspace_size_in_bytes);
    }

    cudnnTensorDescriptor_t x_desc;
    cudnnTensorDescriptor_t y_desc;
    CUDNN_ENFORCE(platform::dynload::cudnnCreateTensorDescriptor(&x_desc));
    CUDNN_ENFORCE(platform::dynload::cudnnCreateTensorDescriptor(&y_desc));
    CUDNN_ENFORCE(platform::dynload::cudnnSetTensorNdDescriptor(
        x_desc, cudnn_dtype, 4, out_dims[3].data(), out_strides[2].data()));
    CUDNN_ENFORCE(platform::dynload::cudnnSetTensorNdDescriptor(
        y_desc, cudnn_dtype, 4, out_dims[3].data(), out_strides[3].data()));
    CUDNN_ENFORCE(platform::dynload::cudnnTransformTensor(
        handle, CudnnDataType<T>::kOne(), x_desc,
        static_cast<const void*>(out_datas[2]), CudnnDataType<T>::kZero(),
        y_desc, static_cast<void*>(output_data + (oc0 + oc1) * h * w)));

    for (int i = 0; i < 4; ++i) {
      CUDNN_ENFORCE(
          platform::dynload::cudnnDestroyTensorDescriptor(in_desc[i]));
      CUDNN_ENFORCE(
          platform::dynload::cudnnDestroyTensorDescriptor(out_desc[i]));
      CUDNN_ENFORCE(
          platform::dynload::cudnnDestroyFilterDescriptor(filter_desc[i]));
      CUDNN_ENFORCE(
          platform::dynload::cudnnDestroyTensorDescriptor(bias_desc[i]));
      CUDNN_ENFORCE(
          platform::dynload::cudnnDestroyConvolutionDescriptor(conv_desc[i]));
    }
    CUDNN_ENFORCE(platform::dynload::cudnnDestroyTensorDescriptor(x_desc));
    CUDNN_ENFORCE(platform::dynload::cudnnDestroyTensorDescriptor(y_desc));
  }
};
#endif

}  // namespace operators
}  // namespace paddle

#if CUDNN_VERSION >= 7100
namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(conv2d_inception_fusion,
                        ops::CUDNNConvInceptionFusionOpKernel<float>,
                        ops::CUDNNConvInceptionFusionOpKernel<double>);
#endif
