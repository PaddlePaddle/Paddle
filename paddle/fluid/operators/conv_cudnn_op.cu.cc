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
#include "paddle/fluid/operators/conv_cudnn_helper.h"
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

static inline void GetNCDHW(const framework::DDim& dims,
                            const DataLayout& layout, int* N, int* C, int* D,
                            int* H, int* W) {
  *N = dims[0];
  *C = layout == DataLayout::kNCHW ? dims[1] : dims[dims.size() - 1];
  int i = layout == DataLayout::kNCHW ? 0 : 1;
  if (dims.size() == 5) {
    *D = dims[2 - i];
    *H = dims[3 - i];
    *W = dims[4 - i];
  } else {
    *D = 1;
    *H = dims[2 - i];
    *W = dims[3 - i];
  }
}

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

    int i_n, i_c, i_d, i_h, i_w;
    GetNCDHW(input->dims(), DataLayout::kNCHW, &i_n, &i_c, &i_d, &i_h, &i_w);
    int o_n, o_c, o_d, o_h, o_w;
    GetNCDHW(output->dims(), DataLayout::kNCHW, &o_n, &o_c, &o_d, &o_h, &o_w);

    int group_offset_in = i_c / groups * i_h * i_w * i_d;
    int group_offset_out = o_c / groups * o_h * o_w * o_d;
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
    bool half_float = false;

#if CUDA_VERSION >= 9000 && CUDNN_VERSION_MIN(7, 0, 1)
    // Tensor core is supported since the volta GPU and
    // is only enabled when input and filter data are float16
    if (dev_ctx.GetComputeCapability() >= 70 &&
        std::type_index(typeid(T)) ==
            std::type_index(typeid(platform::float16))) {
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

    auto handle = dev_ctx.cudnn_handle();
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();
    auto x_dims = framework::vectorize(input->dims());
    auto f_dims = framework::vectorize(filter->dims());

    // TODO(dangqingqing) simplify the following code by SearchAlgorithm in
    // conv_cudnn_helper.h
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

            auto cudnn_find_func = [&](void* cudnn_workspace) {
              CUDNN_ENFORCE(
                  platform::dynload::cudnnFindConvolutionForwardAlgorithmEx(
                      handle, cudnn_input_desc, input_data, cudnn_filter_desc,
                      filter_data, cudnn_conv_desc, cudnn_output_desc,
                      output_data, kNUM_CUDNN_FWD_ALGS, &returned_algo_count,
                      fwd_perf_stat.data(), cudnn_workspace,
                      workspace_size_limit));
            };
            workspace_handle.RunFuncSync(cudnn_find_func, workspace_size_limit);

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

    // Allocate on GPU memory
    Tensor cudnn_workspace =
        ctx.AllocateTmpTensor<int8_t, platform::CUDADeviceContext>(
            framework::make_ddim(
                {static_cast<int64_t>(workspace_size_in_bytes)}),
            dev_ctx);
    void* cudnn_workspace_ptr =
        static_cast<void*>(cudnn_workspace.data<int8_t>());
    VLOG(2) << "Cudnn workspace size fwd: "
            << static_cast<double>(workspace_size_in_bytes) / (1 << 20)
            << " MB";
    // ------------------- cudnn conv forward ---------------------
    ScalingParamType<T> alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < groups; i++) {
      CUDNN_ENFORCE(platform::dynload::cudnnConvolutionForward(
          handle, &alpha, cudnn_input_desc, input_data + i * group_offset_in,
          cudnn_filter_desc, filter_data + i * group_offset_filter,
          cudnn_conv_desc, algo, cudnn_workspace_ptr, workspace_size_in_bytes,
          &beta, cudnn_output_desc, output_data + i * group_offset_out));
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

    int i_n, i_c, i_d, i_h, i_w;
    GetNCDHW(input->dims(), DataLayout::kNCHW, &i_n, &i_c, &i_d, &i_h, &i_w);
    int o_n, o_c, o_d, o_h, o_w;
    GetNCDHW(output_grad->dims(), DataLayout::kNCHW, &o_n, &o_c, &o_d, &o_h,
             &o_w);

    int group_offset_in = i_c / groups * i_h * i_w * i_d;
    int group_offset_out = o_c / groups * o_h * o_w * o_d;
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

    Tensor cudnn_workspace;
    void* cudnn_workspace_ptr = nullptr;
    if ((input_data || filter_data) && exhaustive_search) {
      cudnn_workspace =
          ctx.AllocateTmpTensor<int8_t, platform::CUDADeviceContext>(
              framework::make_ddim(
                  {static_cast<int64_t>(workspace_size_limit)}),
              dev_ctx);
      cudnn_workspace_ptr = static_cast<void*>(cudnn_workspace.data<int8_t>());
    }

    // TODO(dangqingqing) simplify the following code by SearchAlgorithm in
    // conv_cudnn_helper.h
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

              CUDNN_ENFORCE(platform::dynload::
                                cudnnFindConvolutionBackwardDataAlgorithmEx(
                                    handle, cudnn_filter_desc, filter_data,
                                    cudnn_output_grad_desc, output_grad_data,
                                    cudnn_conv_desc, cudnn_input_desc,
                                    input_grad_data, kNUM_CUDNN_BWD_DATA_ALGS,
                                    &returned_algo_count, data_perf_stat.data(),
                                    cudnn_workspace_ptr, workspace_size_limit));

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

              CUDNN_ENFORCE(
                  platform::dynload::
                      cudnnFindConvolutionBackwardFilterAlgorithmEx(
                          handle, cudnn_input_desc, input_data,
                          cudnn_output_grad_desc, output_grad_data,
                          cudnn_conv_desc, cudnn_filter_desc, filter_grad_data,
                          kNUM_CUDNN_BWD_FILTER_ALGS, &returned_algo_count,
                          filter_perf_stat.data(), cudnn_workspace_ptr,
                          workspace_size_limit));
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

    // ------------------- cudnn conv workspace ---------------------
    if (!cudnn_workspace_ptr) {
      cudnn_workspace =
          ctx.AllocateTmpTensor<int8_t, platform::CUDADeviceContext>(
              framework::make_ddim(
                  {static_cast<int64_t>(workspace_size_in_bytes)}),
              dev_ctx);
      cudnn_workspace_ptr = static_cast<void*>(cudnn_workspace.data<int8_t>());
      VLOG(2) << "Cudnn workspace size bwd: "
              << static_cast<double>(workspace_size_in_bytes) / (1 << 20)
              << " MB";
    }

    // ------------------- cudnn conv backward data ---------------------
    ScalingParamType<T> alpha = 1.0f, beta = 0.0f;
    if (input_grad) {
      T* input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());
      // Because beta is zero, it is unnecessary to reset input_grad.

      for (int i = 0; i < groups; i++) {
        CUDNN_ENFORCE(platform::dynload::cudnnConvolutionBackwardData(
            handle, &alpha, cudnn_filter_desc,
            filter_data + i * group_offset_filter, cudnn_output_grad_desc,
            output_grad_data + i * group_offset_out, cudnn_conv_desc, data_algo,
            cudnn_workspace_ptr, workspace_size_in_bytes, &beta,
            cudnn_input_desc, input_grad_data + i * group_offset_in));
      }
    }
    // ------------------- cudnn conv backward filter ---------------------
    if (filter_grad) {
      T* filter_grad_data = filter_grad->mutable_data<T>(ctx.GetPlace());
      // Because beta is zero, it is unnecessary to reset filter_grad.
      for (int i = 0; i < groups; i++) {
        CUDNN_ENFORCE(platform::dynload::cudnnConvolutionBackwardFilter(
            handle, &alpha, cudnn_input_desc, input_data + i * group_offset_in,
            cudnn_output_grad_desc, output_grad_data + i * group_offset_out,
            cudnn_conv_desc, filter_algo, cudnn_workspace_ptr,
            workspace_size_in_bytes, &beta, cudnn_filter_desc,
            filter_grad_data + i * group_offset_filter));
      }
    }
  }
};

/*
 * Inputs:  I, W, dO, ddI, ddW
 * Outputs: ddO, dW, dI
 * ddo = conv(ddI, W) + conv(I, ddW)
 * dW = conv_bp_filter(ddI, dO)
 * dI = conv_bp_data(ddW, dO)
 */
template <typename T>
class CUDNNConvDoubleGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");
    auto X = ctx.Input<Tensor>("Input");
    auto W = ctx.Input<Tensor>("Filter");
    auto dO = ctx.Input<Tensor>("DOutput");
    auto ddX = ctx.Input<Tensor>("DDInput");
    auto ddW = ctx.Input<Tensor>("DDFilter");

    auto ddO = ctx.Output<Tensor>("DDOutput");
    auto dW = ctx.Output<Tensor>("DFilter");
    auto dX = ctx.Output<Tensor>("DInput");

    const T* x = X->data<T>();
    const T* dy = dO->data<T>();
    const T* w = W->data<T>();

    const T* ddx = nullptr;
    const T* ddw = nullptr;
    T *dw, *dx, *ddy;
    dw = dx = ddy = nullptr;

    const std::vector<int>& strides = ctx.Attr<std::vector<int>>("strides");
    const std::vector<int>& paddings = ctx.Attr<std::vector<int>>("paddings");
    const std::vector<int>& dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    bool exhaustive_search =
        FLAGS_cudnn_exhaustive_search || ctx.Attr<bool>("exhaustive_search");
    bool deterministic = FLAGS_cudnn_deterministic;
    if (exhaustive_search && deterministic) {
      PADDLE_THROW(
          "Cann't set exhaustive_search True and "
          "FLAGS_cudnn_deterministic True at same time.");
    }

    int iwo_group = groups;
    int c_group = 1;
#if CUDNN_VERSION_MIN(7, 0, 1)
    iwo_group = 1;
    c_group = groups;
#endif
    auto dtype = platform::CudnnDataType<T>::type;

    auto handle = dev_ctx.cudnn_handle();

    ConvArgs args1{ddX, W, ddO, strides, paddings, dilations};
    ConvArgs args2{X, ddW, ddO, strides, paddings, dilations};
    ConvArgs args3{ddX, dW, dO, strides, paddings, dilations};
    ConvArgs args4{dX, ddW, dO, strides, paddings, dilations};

    cudnnConvolutionFwdAlgo_t fwd_algo1 =
        static_cast<cudnnConvolutionFwdAlgo_t>(0);
    cudnnConvolutionFwdAlgo_t fwd_algo2 =
        static_cast<cudnnConvolutionFwdAlgo_t>(0);
    cudnnConvolutionBwdDataAlgo_t data_algo =
        static_cast<cudnnConvolutionBwdDataAlgo_t>(0);
    cudnnConvolutionBwdFilterAlgo_t filter_algo =
        static_cast<cudnnConvolutionBwdFilterAlgo_t>(0);

    auto layout = GetCudnnTensorFormat(DataLayout::kNCHW);

    // ddo = conv(ddI, W) + conv(I, ddW)
    size_t workspace_size = 0;
    if (ddO) {
      ddy = ddO->mutable_data<T>(ctx.GetPlace());
      args1.handle = handle;
      args1.idesc.set(*ddX, iwo_group);
      args1.wdesc.set(*W, layout, iwo_group);
      args1.odesc.set(*ddO, iwo_group);
      args1.cdesc.set(dtype, paddings, strides, dilations, c_group);

      using search1 = SearchAlgorithm<cudnnConvolutionFwdAlgoPerf_t>;
      fwd_algo1 = search1::Find<T>(args1, exhaustive_search, false, 0, ctx);
      workspace_size = search1::GetWorkspaceSize(args1, fwd_algo1);

      if (ddW) {
        ddw = ddW->data<T>();
        args2.handle = handle;
        args2.idesc.set(*X, iwo_group);
        args2.wdesc.set(*ddW, layout, iwo_group);
        args2.odesc.set(*ddO, iwo_group);
        args2.cdesc.set(dtype, paddings, strides, dilations, c_group);

        using search2 = SearchAlgorithm<cudnnConvolutionFwdAlgoPerf_t>;
        fwd_algo2 = search2::Find<T>(args2, exhaustive_search, false, 0, ctx);
        workspace_size = std::max(workspace_size,
                                  search2::GetWorkspaceSize(args2, fwd_algo2));
      }
    }

    if (dW) {
      dw = dW->mutable_data<T>(ctx.GetPlace());
      args3.handle = handle;
      args3.idesc.set(*ddX, iwo_group);
      args3.wdesc.set(*dW, layout, iwo_group);
      args3.odesc.set(*dO, iwo_group);
      args3.cdesc.set(dtype, paddings, strides, dilations, c_group);

      using search3 = SearchAlgorithm<cudnnConvolutionBwdFilterAlgoPerf_t>;
      filter_algo =
          search3::Find<T>(args3, exhaustive_search, deterministic, 1, ctx);
      workspace_size = std::max(workspace_size,
                                search3::GetWorkspaceSize(args3, filter_algo));
    }

    if (ddW && dX) {
      dx = dX->mutable_data<T>(ctx.GetPlace());
      args4.handle = handle;
      args4.idesc.set(*dX, iwo_group);
      args4.wdesc.set(*ddW, layout, iwo_group);
      args4.odesc.set(*dO, iwo_group);
      args4.cdesc.set(dtype, paddings, strides, dilations, c_group);

      using search4 = SearchAlgorithm<cudnnConvolutionBwdDataAlgoPerf_t>;
      data_algo =
          search4::Find<T>(args4, exhaustive_search, deterministic, 2, ctx);
      workspace_size =
          std::max(workspace_size, search4::GetWorkspaceSize(args4, data_algo));
    }

    int i_n, i_c, i_d, i_h, i_w;
    GetNCDHW(X->dims(), DataLayout::kNCHW, &i_n, &i_c, &i_d, &i_h, &i_w);
    int o_n, o_c, o_d, o_h, o_w;
    GetNCDHW(dO->dims(), DataLayout::kNCHW, &o_n, &o_c, &o_d, &o_h, &o_w);

    int group_offset_in = i_c / groups * i_h * i_w * i_d;
    int group_offset_out = o_c / groups * o_h * o_w * o_d;
    int group_offset_filter = W->numel() / groups;

    ScalingParamType<T> alpha = 1.0f, beta = 0.0f;
    auto wkspace_handle = dev_ctx.cudnn_workspace_handle();

    if (ddO) {
      ddx = ddX->data<T>();
      for (int i = 0; i < groups; i++) {
        wkspace_handle.RunFunc(
            [&](void* workspace_ptr) {
              CUDNN_ENFORCE(platform::dynload::cudnnConvolutionForward(
                  handle, &alpha, args1.idesc.desc(), ddx + i * group_offset_in,
                  args1.wdesc.desc(), w + i * group_offset_filter,
                  args1.cdesc.desc(), fwd_algo1, workspace_ptr, workspace_size,
                  &beta, args1.odesc.desc(), ddy + i * group_offset_out));
            },
            workspace_size);
      }
      if (ddW) {
        for (int i = 0; i < groups; i++) {
          wkspace_handle.RunFunc(
              [&](void* workspace_ptr) {
                CUDNN_ENFORCE(platform::dynload::cudnnConvolutionForward(
                    handle, &alpha, args2.idesc.desc(), x + i * group_offset_in,
                    args2.wdesc.desc(), ddw + i * group_offset_filter,
                    args2.cdesc.desc(), fwd_algo2, workspace_ptr,
                    workspace_size, &alpha, args2.odesc.desc(),
                    ddy + i * group_offset_out));
              },
              workspace_size);
        }
      }
    }

    if (dW) {
      ddx = ddX->data<T>();
      for (int i = 0; i < groups; i++) {
        wkspace_handle.RunFunc(
            [&](void* workspace_ptr) {
              CUDNN_ENFORCE(platform::dynload::cudnnConvolutionBackwardFilter(
                  handle, &alpha, args3.idesc.desc(), ddx + i * group_offset_in,
                  args3.odesc.desc(), dy + i * group_offset_out,
                  args3.cdesc.desc(), filter_algo, workspace_ptr,
                  workspace_size, &beta, args3.wdesc.desc(),
                  dw + i * group_offset_filter));
            },
            workspace_size);
      }
    }

    if (dX && ddW) {
      ddw = ddW->data<T>();
      for (int i = 0; i < groups; i++) {
        wkspace_handle.RunFunc(
            [&](void* workspace_ptr) {
              CUDNN_ENFORCE(platform::dynload::cudnnConvolutionBackwardData(
                  handle, &alpha, args4.wdesc.desc(),
                  ddw + i * group_offset_filter, args4.odesc.desc(),
                  dy + i * group_offset_out, args4.cdesc.desc(), data_algo,
                  workspace_ptr, workspace_size, &beta, args4.idesc.desc(),
                  dx + i * group_offset_in));
            },
            workspace_size);
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
REGISTER_OP_KERNEL(
    conv2d_grad_grad, CUDNN, plat::CUDAPlace,
    paddle::operators::CUDNNConvDoubleGradOpKernel<float>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<double>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<plat::float16>);

REGISTER_OP_KERNEL(conv3d, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvOpKernel<float>,
                   paddle::operators::CUDNNConvOpKernel<double>,
                   paddle::operators::CUDNNConvOpKernel<plat::float16>);
REGISTER_OP_KERNEL(conv3d_grad, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvGradOpKernel<float>,
                   paddle::operators::CUDNNConvGradOpKernel<double>);
