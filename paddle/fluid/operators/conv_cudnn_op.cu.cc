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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/conv_cudnn_op_cache.h"
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/platform/assert.h"
#include "paddle/fluid/platform/cudnn_helper.h"
#include "paddle/fluid/platform/cudnn_workspace_helper.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/profiler.h"

DEFINE_bool(cudnn_deterministic, false,
            "Whether allow using an autotuning algorithm for convolution "
            "operator. The autotuning algorithm may be non-deterministic. If "
            "true, the algorithm is deterministic.");
DEFINE_uint64(conv_workspace_size_limit,
              paddle::platform::kDefaultConvWorkspaceSizeLimitMB,
              "cuDNN convolution workspace limit in MB unit.");
DEFINE_bool(cudnn_exhaustive_search, false,
            "Whether enable exhaustive search for cuDNN convolution or "
            "not, defalut is False.");

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using ScopedFilterDescriptor = platform::ScopedFilterDescriptor;
using ScopedConvolutionDescriptor = platform::ScopedConvolutionDescriptor;
using DataLayout = platform::DataLayout;
template <typename T>
using ScalingParamType = typename platform::CudnnDataType<T>::ScalingParamType;
using framework::AlgorithmsCache;

template <typename T>
class CUDNNConvOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");
    auto* input = ctx.Input<Tensor>("Input");
    auto* filter = ctx.Input<Tensor>("Filter");
    auto* output = ctx.Output<Tensor>("Output");

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    int64_t user_workspace_size =
        static_cast<size_t>(ctx.Attr<int>("workspace_size_MB"));
    bool exhaustive_search =
        FLAGS_cudnn_exhaustive_search || ctx.Attr<bool>("exhaustive_search");

    const T* input_data = input->data<T>();
    const T* filter_data = filter->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());

    // ------------------- cudnn descriptors ---------------------
    ScopedTensorDescriptor input_desc;
    ScopedTensorDescriptor output_desc;
    ScopedFilterDescriptor filter_desc;
    ScopedConvolutionDescriptor conv_desc;
    DataLayout layout = DataLayout::kNCHW;
    if (input->dims().size() == 5) {
      layout = DataLayout::kNCDHW;
    }

    cudnnConvolutionDescriptor_t cudnn_conv_desc =
        conv_desc.descriptor<T>(paddings, strides, dilations);

#if CUDNN_VERSION_MIN(7, 0, 1)
    // cudnn 7 can support groups, no need to do it mannually
    // FIXME(typhoonzero): find a better way to disable groups
    // rather than setting it to 1.
    CUDNN_ENFORCE(platform::dynload::cudnnSetConvolutionGroupCount(
        cudnn_conv_desc, groups));
    groups = 1;
#endif

    cudnnTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
        layout, framework::vectorize2int(input->dims()), groups);
    cudnnTensorDescriptor_t cudnn_output_desc = output_desc.descriptor<T>(
        layout, framework::vectorize2int(output->dims()), groups);
    cudnnFilterDescriptor_t cudnn_filter_desc = filter_desc.descriptor<T>(
        layout, framework::vectorize2int(filter->dims()), groups);

    int input_channels = input->dims()[1];
    int input_height, input_width, input_depth;
    if (input->dims().size() == 5) {
      input_depth = input->dims()[2];
      input_height = input->dims()[3];
      input_width = input->dims()[4];
    } else {  // dim size is enforced in InferShape
      input_depth = 1;
      input_height = input->dims()[2];
      input_width = input->dims()[3];
    }
    int output_channels = filter->dims()[0];
    int output_height, output_width, output_depth;
    if (output->dims().size() == 5) {
      output_depth = output->dims()[2];
      output_height = output->dims()[3];
      output_width = output->dims()[4];
    } else {
      output_depth = 1;
      output_height = output->dims()[2];
      output_width = output->dims()[3];
    }

    int group_offset_in =
        input_channels / groups * input_height * input_width * input_depth;
    int group_offset_out =
        output_channels / groups * output_height * output_width * output_depth;
    int group_offset_filter = filter->numel() / groups;
    // ------------------- cudnn conv workspace ---------------------
    size_t workspace_size_in_bytes;  // final workspace to allocate.
    size_t workspace_size_limit = 0;
    if (FLAGS_conv_workspace_size_limit > 0 || user_workspace_size > 0) {
      int64_t max_user_size =
          std::min(static_cast<int64_t>(FLAGS_conv_workspace_size_limit),
                   user_workspace_size);
      workspace_size_limit = max_user_size * 1024 * 1024;
    }

    // ------------------- cudnn conv algorithm ---------------------
    cudnnConvolutionFwdAlgo_t algo;
    auto handle = dev_ctx.cudnn_handle();

    bool half_float = false;
#if CUDA_VERSION >= 9000 && CUDNN_VERSION_MIN(7, 0, 1)
    // Tensor core is supported since the volta GPU and
    // is only enabled when input and filter data are float16
    if (dev_ctx.GetComputeCapability() >= 70 &&
        std::is_same<T, platform::float16>::value) {
      CUDNN_ENFORCE(platform::dynload::cudnnSetConvolutionMathType(
          cudnn_conv_desc, CUDNN_TENSOR_OP_MATH));
      // Currently tensor core is only enabled using this algo
      algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
      half_float = true;
      VLOG(5) << "use cudnn_tensor_op_math";
    } else {
      CUDNN_ENFORCE(platform::dynload::cudnnSetConvolutionMathType(
          cudnn_conv_desc, CUDNN_DEFAULT_MATH));
      VLOG(5) << "NOT use cudnn_tensor_op_math";
    }
#endif

    auto x_dims = framework::vectorize(input->dims());
    auto f_dims = framework::vectorize(filter->dims());
    if ((!exhaustive_search) && (!half_float)) {
      CUDNN_ENFORCE(platform::dynload::cudnnGetConvolutionForwardAlgorithm(
          handle, cudnn_input_desc, cudnn_filter_desc, cudnn_conv_desc,
          cudnn_output_desc, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
          workspace_size_limit, &algo));
      VLOG(3) << "cuDNN forward algo " << algo;
    } else if (exhaustive_search && (!half_float)) {
      AlgorithmsCache<cudnnConvolutionFwdAlgo_t>& algo_cache =
          ctx.GetKernelConfig<AlgorithmsCache<cudnnConvolutionFwdAlgo_t>>(0);

      algo = algo_cache.GetAlgorithm(
          x_dims, f_dims, strides, paddings, dilations, 0, [&]() {
            int returned_algo_count;
            std::array<cudnnConvolutionFwdAlgoPerf_t, kNUM_CUDNN_FWD_ALGS>
                fwd_perf_stat;

            auto cudnn_workspace_handle = dev_ctx.cudnn_workspace_handle();
            cudnn_workspace_handle.RunFunc(
                [&](void* cudnn_workspace_ptr) {
                  CUDNN_ENFORCE(
                      platform::dynload::cudnnFindConvolutionForwardAlgorithmEx(
                          handle, cudnn_input_desc, input_data,
                          cudnn_filter_desc, filter_data, cudnn_conv_desc,
                          cudnn_output_desc, output_data, kNUM_CUDNN_FWD_ALGS,
                          &returned_algo_count, fwd_perf_stat.data(),
                          cudnn_workspace_ptr, workspace_size_limit));
                },
                workspace_size_limit);

            VLOG(3) << "Perf result: (algo: stat, time, memory)";
            for (int i = 0; i < returned_algo_count; ++i) {
              const auto& stat = fwd_perf_stat[i];
              VLOG(3) << stat.algo << ": " << stat.status << " " << stat.time
                      << " " << stat.memory;
            }
            return fwd_perf_stat[0].algo;
          });
      VLOG(3) << "choose algo " << algo;
    } else {
      PADDLE_ENFORCE(half_float,
                     "cuDNN exhaustive search doesn't support half float.");
    }

    // get workspace size able to allocate
    CUDNN_ENFORCE(platform::dynload::cudnnGetConvolutionForwardWorkspaceSize(
        handle, cudnn_input_desc, cudnn_filter_desc, cudnn_conv_desc,
        cudnn_output_desc, algo, &workspace_size_in_bytes));
    // It is possible for float16 on Volta GPU to allocate more memory than
    // the limit because the algo is overrided to use tensor core.
    PADDLE_ENFORCE_LE(workspace_size_in_bytes, workspace_size_limit,
                      "workspace_size to be allocated exceeds the limit");

    // ------------------- cudnn conv forward ---------------------
    ScalingParamType<T> alpha = 1.0f, beta = 0.0f;
    auto cudnn_workspace_handle = dev_ctx.cudnn_workspace_handle();
    for (int i = 0; i < groups; i++) {
      cudnn_workspace_handle.RunFunc(
          [&](void* cudnn_workspace_ptr) {
            CUDNN_ENFORCE(platform::dynload::cudnnConvolutionForward(
                handle, &alpha, cudnn_input_desc,
                input_data + i * group_offset_in, cudnn_filter_desc,
                filter_data + i * group_offset_filter, cudnn_conv_desc, algo,
                cudnn_workspace_ptr, workspace_size_in_bytes, &beta,
                cudnn_output_desc, output_data + i * group_offset_out));
          },
          workspace_size_in_bytes);
    }
  }
};

template <typename T>
class CUDNNConvGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");
    auto input = ctx.Input<Tensor>("Input");
    auto filter = ctx.Input<Tensor>("Filter");
    auto output_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));
    auto input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto filter_grad = ctx.Output<Tensor>(framework::GradVarName("Filter"));

    const T* input_data = input->data<T>();
    const T* output_grad_data = output_grad->data<T>();
    const T* filter_data = filter->data<T>();

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    int64_t user_workspace_size =
        static_cast<size_t>(ctx.Attr<int>("workspace_size_MB"));
    bool exhaustive_search =
        FLAGS_cudnn_exhaustive_search || ctx.Attr<bool>("exhaustive_search");
    if (exhaustive_search && FLAGS_cudnn_deterministic) {
      PADDLE_THROW(
          "Cann't set exhaustive_search True and "
          "FLAGS_cudnn_deterministic True at same time.");
    }

    // ------------------- cudnn descriptors ---------------------
    ScopedTensorDescriptor input_desc;
    ScopedTensorDescriptor output_grad_desc;

    ScopedFilterDescriptor filter_desc;
    ScopedFilterDescriptor filter_grad_desc;
    ScopedConvolutionDescriptor conv_desc;
    DataLayout layout = DataLayout::kNCHW;
    if (input->dims().size() == 5) {
      layout = DataLayout::kNCDHW;
    }

    cudnnConvolutionDescriptor_t cudnn_conv_desc =
        conv_desc.descriptor<T>(paddings, strides, dilations);

#if CUDNN_VERSION_MIN(7, 0, 1)
    // cudnn 7 can support groups, no need to do it mannually
    // FIXME(typhoonzero): find a better way to disable groups
    // rather than setting it to 1.
    CUDNN_ENFORCE(platform::dynload::cudnnSetConvolutionGroupCount(
        cudnn_conv_desc, groups));
    groups = 1;
#endif

    cudnnTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
        layout, framework::vectorize2int(input->dims()), groups);
    cudnnTensorDescriptor_t cudnn_output_grad_desc =
        output_grad_desc.descriptor<T>(
            layout, framework::vectorize2int(output_grad->dims()), groups);
    cudnnFilterDescriptor_t cudnn_filter_desc = filter_desc.descriptor<T>(
        layout, framework::vectorize2int(filter->dims()), groups);

#if CUDA_VERSION >= 9000 && CUDNN_VERSION_MIN(7, 0, 1)
    // Enable Tensor Core for cudnn backward
    if (dev_ctx.GetComputeCapability() >= 70 &&
        std::type_index(typeid(T)) ==
            std::type_index(typeid(platform::float16))) {
      CUDNN_ENFORCE(platform::dynload::cudnnSetConvolutionMathType(
          cudnn_conv_desc, CUDNN_TENSOR_OP_MATH));
      VLOG(5) << "use cudnn_tensor_op_math for backward";
    } else {
      CUDNN_ENFORCE(platform::dynload::cudnnSetConvolutionMathType(
          cudnn_conv_desc, CUDNN_DEFAULT_MATH));
      VLOG(5) << "NOT use cudnn_tensor_op_math for backward";
    }
#endif

    int input_channels = input->dims()[1];
    int input_height, input_width, input_depth;
    if (input->dims().size() == 5) {
      input_depth = input->dims()[2];
      input_height = input->dims()[3];
      input_width = input->dims()[4];
    } else {  // dim size is enforced in InferShape
      input_depth = 1;
      input_height = input->dims()[2];
      input_width = input->dims()[3];
    }

    int output_grad_channels = filter->dims()[0];
    int output_grad_height, output_grad_width, output_grad_depth;
    if (input->dims().size() == 5) {
      output_grad_depth = output_grad->dims()[2];
      output_grad_height = output_grad->dims()[3];
      output_grad_width = output_grad->dims()[4];
    } else {
      output_grad_depth = 1;
      output_grad_height = output_grad->dims()[2];
      output_grad_width = output_grad->dims()[3];
    }

    int group_offset_in =
        input_channels / groups * input_height * input_width * input_depth;
    int group_offset_out = output_grad_channels / groups * output_grad_height *
                           output_grad_width * output_grad_depth;
    int group_offset_filter = filter->numel() / groups;
    // ------------------- cudnn backward algorithm ---------------------
    cudnnConvolutionBwdDataAlgo_t data_algo;
    cudnnConvolutionBwdFilterAlgo_t filter_algo;
    size_t workspace_size_in_bytes = 0, tmp_size = 0;
    size_t workspace_size_limit = 0;
    if (FLAGS_conv_workspace_size_limit > 0 || user_workspace_size > 0) {
      int64_t max_user_size =
          std::min(static_cast<int64_t>(FLAGS_conv_workspace_size_limit),
                   user_workspace_size);
      workspace_size_limit = max_user_size * 1024 * 1024;
    }

    auto x_dims = framework::vectorize(input->dims());
    auto f_dims = framework::vectorize(filter->dims());
    auto handle = dev_ctx.cudnn_handle();
    if (input_grad) {
      T* input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());
      if (exhaustive_search) {
        AlgorithmsCache<cudnnConvolutionBwdDataAlgo_t>& data_algo_cache =
            ctx.GetKernelConfig<AlgorithmsCache<cudnnConvolutionBwdDataAlgo_t>>(
                0);

        data_algo = data_algo_cache.GetAlgorithm(
            x_dims, f_dims, strides, paddings, dilations, 0, [&]() {
              int returned_algo_count;
              std::array<cudnnConvolutionBwdDataAlgoPerf_t,
                         kNUM_CUDNN_BWD_DATA_ALGS>
                  data_perf_stat;

              auto cudnn_workspace_handle = dev_ctx.cudnn_workspace_handle();
              cudnn_workspace_handle.RunFunc(
                  [&](void* cudnn_workspace_ptr) {
                    CUDNN_ENFORCE(
                        platform::dynload::
                            cudnnFindConvolutionBackwardDataAlgorithmEx(
                                handle, cudnn_filter_desc, filter_data,
                                cudnn_output_grad_desc, output_grad_data,
                                cudnn_conv_desc, cudnn_input_desc,
                                input_grad_data, kNUM_CUDNN_BWD_DATA_ALGS,
                                &returned_algo_count, data_perf_stat.data(),
                                cudnn_workspace_ptr, workspace_size_limit));
                  },
                  workspace_size_limit);

              VLOG(3) << "Perf result: (algo: stat, time, memory)";
              for (int i = 0; i < returned_algo_count; ++i) {
                const auto& stat = data_perf_stat[i];
                VLOG(3) << stat.algo << ": " << stat.status << " " << stat.time
                        << " " << stat.memory;
              }
              return data_perf_stat[0].algo;
            });
        VLOG(3) << "cuDNN backward data algo " << data_algo;
      } else if (FLAGS_cudnn_deterministic) {
        data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
      } else {
        CUDNN_ENFORCE(
            platform::dynload::cudnnGetConvolutionBackwardDataAlgorithm(
                handle, cudnn_filter_desc,
                // dyDesc: Handle to the previously initialized input
                // differential
                // tensor descriptor.
                cudnn_output_grad_desc, cudnn_conv_desc,
                // dxDesc: Handle to the previously initialized output tensor
                // descriptor.
                cudnn_input_desc,
                CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
                workspace_size_limit, &data_algo));
      }
      CUDNN_ENFORCE(
          platform::dynload::cudnnGetConvolutionBackwardDataWorkspaceSize(
              handle, cudnn_filter_desc, cudnn_output_grad_desc,
              cudnn_conv_desc, cudnn_input_desc, data_algo, &tmp_size));
      workspace_size_in_bytes = std::max(workspace_size_in_bytes, tmp_size);
    }

    if (filter_grad) {
      T* filter_grad_data = filter_grad->mutable_data<T>(ctx.GetPlace());
      if (exhaustive_search) {
        AlgorithmsCache<cudnnConvolutionBwdFilterAlgo_t>& f_algo_cache =
            ctx.GetKernelConfig<
                AlgorithmsCache<cudnnConvolutionBwdFilterAlgo_t>>(1);

        filter_algo = f_algo_cache.GetAlgorithm(
            x_dims, f_dims, strides, paddings, dilations, 0, [&]() {
              int returned_algo_count;
              std::array<cudnnConvolutionBwdFilterAlgoPerf_t,
                         kNUM_CUDNN_BWD_FILTER_ALGS>
                  filter_perf_stat;

              auto cudnn_workspace_handle = dev_ctx.cudnn_workspace_handle();
              cudnn_workspace_handle.RunFunc(
                  [&](void* cudnn_workspace_ptr) {
                    CUDNN_ENFORCE(
                        platform::dynload::
                            cudnnFindConvolutionBackwardFilterAlgorithmEx(
                                handle, cudnn_input_desc, input_data,
                                cudnn_output_grad_desc, output_grad_data,
                                cudnn_conv_desc, cudnn_filter_desc,
                                filter_grad_data, kNUM_CUDNN_BWD_FILTER_ALGS,
                                &returned_algo_count, filter_perf_stat.data(),
                                cudnn_workspace_ptr, workspace_size_limit));
                  },
                  workspace_size_limit);
              return filter_perf_stat[0].algo;
            });
        VLOG(3) << "cuDNN backward filter algo " << filter_algo;
      } else if (FLAGS_cudnn_deterministic) {
        filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
      } else {
        CUDNN_ENFORCE(
            platform::dynload::cudnnGetConvolutionBackwardFilterAlgorithm(
                handle, cudnn_input_desc, cudnn_output_grad_desc,
                cudnn_conv_desc, cudnn_filter_desc,
                CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                workspace_size_limit, &filter_algo));
      }
      CUDNN_ENFORCE(
          platform::dynload::cudnnGetConvolutionBackwardFilterWorkspaceSize(
              handle, cudnn_input_desc, cudnn_output_grad_desc, cudnn_conv_desc,
              cudnn_filter_desc, filter_algo, &tmp_size));
      workspace_size_in_bytes = std::max(workspace_size_in_bytes, tmp_size);
    }

    auto cudnn_workspace_handle = dev_ctx.cudnn_workspace_handle();
    // ------------------- cudnn conv backward data ---------------------
    ScalingParamType<T> alpha = 1.0f, beta = 0.0f;
    if (input_grad) {
      T* input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());
      // Because beta is zero, it is unnecessary to reset input_grad.

      for (int i = 0; i < groups; i++) {
        cudnn_workspace_handle.RunFunc(
            [&](void* cudnn_workspace_ptr) {
              CUDNN_ENFORCE(platform::dynload::cudnnConvolutionBackwardData(
                  handle, &alpha, cudnn_filter_desc,
                  filter_data + i * group_offset_filter, cudnn_output_grad_desc,
                  output_grad_data + i * group_offset_out, cudnn_conv_desc,
                  data_algo, cudnn_workspace_ptr, workspace_size_in_bytes,
                  &beta, cudnn_input_desc,
                  input_grad_data + i * group_offset_in));
            },
            workspace_size_in_bytes);
      }
    }
    // ------------------- cudnn conv backward filter ---------------------
    if (filter_grad) {
      T* filter_grad_data = filter_grad->mutable_data<T>(ctx.GetPlace());
      // Because beta is zero, it is unnecessary to reset filter_grad.
      for (int i = 0; i < groups; i++) {
        cudnn_workspace_handle.RunFunc(
            [&](void* cudnn_workspace_ptr) {
              CUDNN_ENFORCE(platform::dynload::cudnnConvolutionBackwardFilter(
                  handle, &alpha, cudnn_input_desc,
                  input_data + i * group_offset_in, cudnn_output_grad_desc,
                  output_grad_data + i * group_offset_out, cudnn_conv_desc,
                  filter_algo, cudnn_workspace_ptr, workspace_size_in_bytes,
                  &beta, cudnn_filter_desc,
                  filter_grad_data + i * group_offset_filter));
            },
            workspace_size_in_bytes);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace plat = paddle::platform;
REGISTER_OP_KERNEL(conv2d, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvOpKernel<float>,
                   paddle::operators::CUDNNConvOpKernel<double>,
                   paddle::operators::CUDNNConvOpKernel<plat::float16>);
REGISTER_OP_KERNEL(conv2d_grad, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvGradOpKernel<float>,
                   paddle::operators::CUDNNConvGradOpKernel<double>,
                   paddle::operators::CUDNNConvGradOpKernel<plat::float16>);

REGISTER_OP_KERNEL(conv3d, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvOpKernel<float>,
                   paddle::operators::CUDNNConvOpKernel<double>,
                   paddle::operators::CUDNNConvOpKernel<plat::float16>);
REGISTER_OP_KERNEL(conv3d_grad, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvGradOpKernel<float>,
                   paddle::operators::CUDNNConvGradOpKernel<double>);
