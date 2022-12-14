// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/padding.h"
#include "paddle/phi/kernels/gpudnn/conv_cudnn_fuse.h"
#include "paddle/phi/kernels/gpudnn/conv_gpudnn_info.h"

namespace phi {
using ScopedTensorDescriptor = phi::backends::gpu::ScopedTensorDescriptor;
using ScopedFilterDescriptor = phi::backends::gpu::ScopedFilterDescriptor;
using ScopedConvolutionDescriptor =
    phi::backends::gpu::ScopedConvolutionDescriptor;
using ScopedActivationDescriptor =
    phi::backends::gpu::ScopedActivationDescriptor;
using GPUDNNDataLayout = phi::backends::gpu::DataLayout;

template <typename T>
using ScalingParamType =
    typename phi::backends::gpu::CudnnDataType<T>::ScalingParamType;

template <typename T, typename Context>
void ConvFusionKernel(const Context& dev_ctx,
                      const DenseTensor& input,
                      const DenseTensor& filter,
                      const DenseTensor& bias,
                      const paddle::optional<DenseTensor>& residual,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings_t,
                      const std::string& padding_algorithm,
                      const std::string& activation,
                      const std::vector<int>& dilations_t,
                      int groups,
                      int search_times,
                      const std::vector<int>& channels,
                      int user_workspace_size,
                      bool exhaustive_search,
                      DenseTensor* output,
                      std::vector<DenseTensor*> outs) {
  dev_ctx.template Alloc<T>(output);
  for (size_t i = 0; i < outs.size(); ++i) {
    dev_ctx.template Alloc<T>(outs[i]);
  }
  std::vector<int> paddings = paddings_t;
  std::vector<int> dilations = dilations_t;
  exhaustive_search = FLAGS_cudnn_exhaustive_search || exhaustive_search;

  LOG(INFO) << "JZZ exhaustive_search: " << exhaustive_search;

  const T* filter_data = filter.data<T>();
  const T* bias_data = bias.data<T>();

  DenseTensor transformed_input_channel(input.dtype());
  DenseTensor transformed_output(output->dtype());
  transformed_input_channel = input;
  transformed_output = *output;
  T* output_data = transformed_output.data<T>();

  const T* residual_data = residual ? residual->data<T>() : output_data;

  // update padding and dilation
  auto in_dims = transformed_input_channel.dims();
  auto filter_dims = filter.dims();
  phi::DDim in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());

  phi::DDim filter_data_dims =
      phi::slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  int data_dim = strides.size();  // 2d or 3d
  bool is_sys_pad = phi::funcs::IsSymmetricPadding(paddings, data_dim);

  DenseTensor transformed_input;
  std::vector<int> padding_common(data_dim, 0);
  if (!is_sys_pad) {
    std::vector<int> padding_diff(data_dim);
    std::vector<int> new_input_shape_vec(data_dim + 2);
    new_input_shape_vec[0] = transformed_input_channel.dims()[0];
    new_input_shape_vec[1] = transformed_input_channel.dims()[1];

    std::vector<int> input_pad(transformed_input_channel.dims().size() * 2, 0);
    for (size_t i = 0; i < data_dim; ++i) {
      padding_diff[i] = std::abs(paddings[2 * i] - paddings[2 * i + 1]);
      padding_common[i] = std::min(paddings[2 * i], paddings[2 * i + 1]);
      new_input_shape_vec[i + 2] =
          transformed_input_channel.dims()[i + 2] + padding_diff[i];
      input_pad[2 * i + 4] = paddings[2 * i] - padding_common[i];
      input_pad[2 * i + 4 + 1] = paddings[2 * i + 1] - padding_common[i];
    }
    phi::DDim new_input_shape(phi::make_ddim(new_input_shape_vec));
    transformed_input.Resize(new_input_shape);
    dev_ctx.template Alloc<T>(&transformed_input);
    const int rank = transformed_input_channel.dims().size();
    T pad_value(0.0);
    switch (rank) {
      case 4: {
        phi::funcs::PadFunction<phi::GPUContext, T, 4>(
            dev_ctx,
            input_pad,
            transformed_input_channel,
            pad_value,
            &transformed_input);
      } break;
      case 5: {
        phi::funcs::PadFunction<phi::GPUContext, T, 5>(
            dev_ctx,
            input_pad,
            transformed_input_channel,
            pad_value,
            &transformed_input);
      } break;
      default:
        PADDLE_THROW(phi::errors::PermissionDenied(
            "Operator Conv2DFusion expects Input to be a 4-D or 5-D Tensor. "
            "But received the actual dimension = %d, shape = [%s].",
            rank,
            transformed_input_channel.dims()));
    }
  } else {
    transformed_input = transformed_input_channel;
    if (paddings.size() == data_dim) {
      for (size_t i = 0; i < data_dim; ++i) {
        padding_common[i] = paddings[i];
      }
    } else {
      for (size_t i = 0; i < data_dim; ++i) {
        padding_common[i] = paddings[2 * i];
      }
    }
  }

  const T* input_data = transformed_input.data<T>();
  // ------------------- cudnn descriptors ---------------------
  ScopedTensorDescriptor input_desc;
  ScopedTensorDescriptor output_desc;
  ScopedFilterDescriptor filter_desc;
  ScopedTensorDescriptor bias_desc;
  ScopedConvolutionDescriptor conv_desc;
  ScopedActivationDescriptor act_desc;
  GPUDNNDataLayout layout = GPUDNNDataLayout::kNCHW;
  if (input.dims().size() == 5) {
    layout = GPUDNNDataLayout::kNCDHW;
  }
#ifdef PADDLE_WITH_HIP
  miopenConvolutionDescriptor_t cudnn_conv_desc =
      conv_desc.descriptor<T>(padding_common, strides, dilations);
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::miopenSetConvolutionGroupCount(cudnn_conv_desc, groups));
  // Now only support NCHW
  std::vector<int> bias_dim = {
      1, static_cast<int>(transformed_output.dims()[1]), 1, 1};
  miopenTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
      layout, phi::vectorize<int>(transformed_input.dims()));
  miopenTensorDescriptor_t cudnn_output_desc = output_desc.descriptor<T>(
      layout, phi::vectorize<int>(transformed_output.dims()));
  miopenTensorDescriptor_t cudnn_filter_desc =
      filter_desc.descriptor<T>(layout, phi::vectorize<int>(filter->dims()));
  miopenTensorDescriptor_t cudnn_bias_desc =
      bias_desc.descriptor<T>(layout, bias_dim);
  miopenActivationDescriptor_t cudnn_act_desc =
      act_desc.descriptor<T>(activation);

  miopenConvFwdAlgorithm_t algo;
  auto handle = dev_ctx.cudnn_handle();
  auto workspace_handle = dev_ctx.cudnn_workspace_handle();

  auto x_dims = phi::vectorize(transformed_input.dims());
  auto f_dims = phi::vectorize(filter.dims());

  size_t workspace_size = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::miopenConvolutionForwardGetWorkSpaceSize(handle,
                                                        cudnn_filter_desc,
                                                        cudnn_input_desc,
                                                        cudnn_conv_desc,
                                                        cudnn_output_desc,
                                                        &workspace_size));
  int find_count;
  miopenConvAlgoPerf_t find_result;
  auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::miopenFindConvolutionForwardAlgorithm(handle,
                                                       cudnn_input_desc,
                                                       input_data,
                                                       cudnn_filter_desc,
                                                       filter_data,
                                                       cudnn_conv_desc,
                                                       cudnn_output_desc,
                                                       output_data,
                                                       phi::kNUM_CUDNN_FWD_ALGS,
                                                       &find_count,
                                                       &find_result,
                                                       cudnn_workspace_ptr,
                                                       workspace_size,
                                                       false));
  };
  workspace_handle.RunFuncSync(cudnn_find_func, workspace_size);
  algo = find_result.fwd_algo;
  VLOG(3) << "cuDNN forward algo " << algo;

  {
    ScalingParamType<T> alpha = 1.0f, beta = 0.0f;
    auto cudnn_func = [&](void* cudnn_workspace) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::miopenConvolutionForward(handle,
                                            &alpha,
                                            cudnn_input_desc,
                                            input_data,
                                            cudnn_filter_desc,
                                            filter_data,
                                            cudnn_conv_desc,
                                            algo,
                                            &beta,
                                            cudnn_output_desc,
                                            output_data,
                                            cudnn_workspace,
                                            workspace_size));
    };
    workspace_handle.RunFunc(cudnn_func, workspace_size);
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::miopenConvolutionForwardBias(handle,
                                              &alpha,
                                              cudnn_bias_desc,
                                              bias_data,
                                              &beta,
                                              cudnn_output_desc,
                                              output_data));
    if (activation != "identity") {
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::miopenActivationForward(handle,
                                           cudnn_act_desc,
                                           &alpha,
                                           cudnn_output_desc,
                                           output_data,
                                           &beta,
                                           cudnn_output_desc,
                                           output_data));
    }
    if (residual) {
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::miopenOpTensor(handle,
                                                         miopenTensorOpAdd,
                                                         &alpha,
                                                         cudnn_output_desc,
                                                         output_data,
                                                         &alpha,
                                                         cudnn_output_desc,
                                                         residual_data,
                                                         &beta,
                                                         cudnn_output_desc,
                                                         output_data));
    }
  }
#else  // PADDLE_WITH_HIP
  cudnnConvolutionDescriptor_t cudnn_conv_desc =
      conv_desc.descriptor<T>(padding_common, strides, dilations);
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cudnnSetConvolutionGroupCount(cudnn_conv_desc, groups));

  cudnnTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
      layout, phi::vectorize<int>(transformed_input.dims()));
  cudnnTensorDescriptor_t cudnn_output_desc = output_desc.descriptor<T>(
      layout, phi::vectorize<int>(transformed_output.dims()));
  cudnnFilterDescriptor_t cudnn_filter_desc =
      filter_desc.descriptor<T>(layout, phi::vectorize<int>(filter.dims()));
  // Now only support NCHW
  std::vector<int> bias_dim = {
      1, static_cast<int>(transformed_output.dims()[1]), 1, 1};
  cudnnTensorDescriptor_t cudnn_bias_desc =
      bias_desc.descriptor<T>(layout, bias_dim);
  cudnnActivationDescriptor_t cudnn_act_desc =
      act_desc.descriptor<T>(activation);

  // ------------------- cudnn conv workspace ---------------------
  size_t workspace_size_in_bytes;  // final workspace to allocate.
  size_t workspace_size_limit = 0;
  if (FLAGS_conv_workspace_size_limit > 0 || user_workspace_size > 0) {
    int64_t max_user_size =
        std::min(static_cast<int64_t>(FLAGS_conv_workspace_size_limit),
                 static_cast<int64_t>(user_workspace_size));
    workspace_size_limit = max_user_size * 1024 * 1024;
  }

  // ------------------- cudnn conv algorithm ---------------------
  cudnnConvolutionFwdAlgo_t algo;
  auto handle = dev_ctx.cudnn_handle();
  auto workspace_handle = dev_ctx.cudnn_workspace_handle();
  auto dtype = phi::backends::gpu::CudnnDataType<T>::type;

  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cudnnSetConvolutionMathType(
      cudnn_conv_desc, CUDNN_DEFAULT_MATH));
  if (dtype == CUDNN_DATA_HALF) {
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cudnnSetConvolutionMathType(
        cudnn_conv_desc, CUDNN_TENSOR_OP_MATH));
  }
  // #if CUDA_VERSION >= 11000 && CUDNN_VERSION >= 8000
  //     if (!platform::allow_tf32_cudnn) {
  //     PADDLE_ENFORCE_GPU_SUCCESS(dynload::cudnnSetConvolutionMathType(
  //         cudnn_conv_desc, CUDNN_FMA_MATH));
  //     }
  // #endif  // CUDA_VERSION >= 11000 && CUDNN_VERSION >= 8000

  auto x_dims = phi::vectorize(transformed_input.dims());
  auto f_dims = phi::vectorize(filter.dims());
  if (!exhaustive_search) {
#if CUDNN_VERSION >= 8000
    int perf_count;
    int best_algo_idx = 0;
    size_t tmp_size = 0;
    std::unique_ptr<cudnnConvolutionFwdAlgoPerf_t[]> perf_results(
        new cudnnConvolutionFwdAlgoPerf_t[phi::kNUM_CUDNN_FWD_ALGS]);
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cudnnGetConvolutionForwardAlgorithm_v7(
        handle,
        cudnn_input_desc,
        cudnn_filter_desc,
        cudnn_conv_desc,
        cudnn_output_desc,
        phi::kNUM_CUDNN_FWD_ALGS,
        &perf_count,
        perf_results.get()));
    algo = (perf_results.get())[best_algo_idx].algo;
#else
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cudnnGetConvolutionForwardAlgorithm(
        handle,
        cudnn_input_desc,
        cudnn_filter_desc,
        cudnn_conv_desc,
        cudnn_output_desc,
        CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
        workspace_size_limit,
        &algo));
#endif
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        cudnn_input_desc,
        cudnn_filter_desc,
        cudnn_conv_desc,
        cudnn_output_desc,
        algo,
        &workspace_size_in_bytes));
    if (workspace_size_in_bytes > workspace_size_limit)
      workspace_size_limit = workspace_size_in_bytes;
    VLOG(3) << "cuDNN forward algo " << algo;
  } else {
    std::function<SearchFuseResult<cudnnConvolutionFwdAlgo_t>()> search_func =
        [&]() -> SearchFuseResult<cudnnConvolutionFwdAlgo_t> {
      int returned_algo_count;
      SearchFuseResult<cudnnConvolutionFwdAlgo_t> fwd_result;
      std::array<cudnnConvolutionFwdAlgoPerf_t, phi::kNUM_CUDNN_FWD_ALGS>
          fwd_perf_stat;
      auto cudnn_find_func = [&](void* cudnn_workspace) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            dynload::cudnnFindConvolutionForwardAlgorithmEx(
                handle,
                cudnn_input_desc,
                input_data,
                cudnn_filter_desc,
                filter_data,
                cudnn_conv_desc,
                cudnn_output_desc,
                output_data,
                phi::kNUM_CUDNN_FWD_ALGS,
                &returned_algo_count,
                fwd_perf_stat.data(),
                cudnn_workspace,
                workspace_size_limit));
      };
      workspace_handle.RunFuncSync(cudnn_find_func, workspace_size_limit);
      VLOG(3) << "Perf result: (algo: stat, time, memory)";
      for (int i = 0; i < returned_algo_count; ++i) {
        const auto& stat = fwd_perf_stat[i];
        VLOG(3) << stat.algo << ": " << stat.status << " " << stat.time << " "
                << stat.memory;
      }

      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::cudnnGetConvolutionForwardWorkspaceSize(
              handle,
              cudnn_input_desc,
              cudnn_filter_desc,
              cudnn_conv_desc,
              cudnn_output_desc,
              fwd_perf_stat[0].algo,
              &workspace_size_in_bytes));
      // PADDLE_ENFORCE_LE(
      //     workspace_size_in_bytes,
      //     workspace_size_limit,
      //     platform::errors::InvalidArgument(
      //         "The actual workspace size to be allocated for cuDNN is
      //         expected " "to be less than the limit. But received: the
      //         actual workspace " "size = %d, limit = %d.",
      //         workspace_size_in_bytes,
      //         workspace_size_limit));

      fwd_result.algo = fwd_perf_stat[0].algo;
      fwd_result.workspace_size = workspace_size_in_bytes;
      return fwd_result;
    };
    AlgorithmsCache<SearchFuseResult<cudnnConvolutionFwdAlgo_t>>& algo_cache =
        *(ConvSearchCache::Instance().GetConvFusion());
    // int search_times = dev_ctx.Attr<int>("search_times");
    SearchFuseResult<cudnnConvolutionFwdAlgo_t> algo_result;
    search_times = std::max(
        static_cast<int>(FLAGS_cudnn_exhaustive_search_times), search_times);
    // TODO(dangqingqing): Unify this if-else.
    if (search_times > 0) {
      // The searched algo will be cached by `search_times` times for
      // different input dimension. For other dimensions, select the algo
      // of closest area.
      algo_result = algo_cache.GetAlgorithm(
          x_dims[2] * x_dims[3], search_times, 0, search_func);
      algo = algo_result.algo;
      workspace_size_in_bytes = algo_result.workspace_size;
    } else {
      algo_result = algo_cache.GetAlgorithm(
          x_dims, f_dims, strides, paddings, dilations, 0, dtype, search_func);
      algo = algo_result.algo;
      workspace_size_in_bytes = algo_result.workspace_size;
    }
    VLOG(3) << "choose algo " << algo;
  }
  if ((activation == "identity") && (!residual)) {
    // Only the CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM algo is
    // enabled with CUDNN_ACTIVATION_IDENTITY in cuDNN lib.
    // But test in some case, the speed is slower, change to use
    // cudnnConvolutionForward and cudnnAddTensor
    // ------------- cudnn conv forward and bias add ---------------------
    ScalingParamType<T> alpha = 1.0f, beta = 0.0f;
    auto cudnn_func = [&](void* cudnn_workspace) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::cudnnConvolutionForward(handle,
                                           &alpha,
                                           cudnn_input_desc,
                                           input_data,
                                           cudnn_filter_desc,
                                           filter_data,
                                           cudnn_conv_desc,
                                           algo,
                                           cudnn_workspace,
                                           workspace_size_in_bytes,
                                           &beta,
                                           cudnn_output_desc,
                                           output_data));
    };
    workspace_handle.RunFunc(cudnn_func, workspace_size_in_bytes);
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cudnnAddTensor(handle,
                                                       &alpha,
                                                       cudnn_bias_desc,
                                                       bias_data,
                                                       &alpha,
                                                       cudnn_output_desc,
                                                       output_data));
  } else {
    if (activation == "identity") {
      algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    }
    // ------------------- cudnn conv+bias+act forward --------------------
    ScalingParamType<T> alpha1 = 1.0f;
    ScalingParamType<T> alpha2 = residual ? 1.0f : 0.0f;
    auto cudnn_func = [&](void* cudnn_workspace) {
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cudnnConvolutionBiasActivationForward(
          handle,
          &alpha1,
          cudnn_input_desc,
          input_data,
          cudnn_filter_desc,
          filter_data,
          cudnn_conv_desc,
          algo,
          cudnn_workspace,
          workspace_size_in_bytes,
          &alpha2,
          cudnn_output_desc,
          residual_data,
          cudnn_bias_desc,
          bias_data,
          cudnn_act_desc,
          cudnn_output_desc,
          output_data));
    };
    workspace_handle.RunFunc(cudnn_func, workspace_size_in_bytes);
  }
#endif
  if (channels.size()) {
    if (x_dims[0] == 1) {
      // share data with Output
      phi::DenseTensor t;
      t.ShareDataWith(*output);
      auto y_dims = output->dims();
      t.Resize({y_dims[1], y_dims[2], y_dims[3]});
      int s = 0;
      for (size_t i = 0; i < channels.size(); ++i) {
        int e = s + channels[i];
        outs[i]->ShareDataWith(t.Slice(s, e));
        outs[i]->Resize({x_dims[0], channels[i], y_dims[2], y_dims[3]});
        s = e;
      }
    } else {
      // TODO(qingiqng): do copy when batch size large than 1
      PADDLE_THROW(phi::errors::Unimplemented(
          "Input with batch size greater than 1 is unsupported. The received "
          "batch size is %d, Input's shape is [%s].",
          x_dims[0],
          phi::make_ddim(x_dims)));
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    conv2d_fusion, GPUDNN, ALL_LAYOUT, phi::ConvFusionKernel, float, double) {}
