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

#ifdef PADDLE_WITH_CUDA
#include <xxhash.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>

#include "paddle/phi/backends/dynload/cudnn.h"
#include "paddle/phi/backends/gpu/cuda/cudnn_desc.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/impl/conv_cudnn_impl.h"
#include "paddle/utils/optional.h"

namespace phi {
namespace fusion {

namespace {
class CudnnConvDescManager {
 public:
  static CudnnConvDescManager* Instance() {
    static CudnnConvDescManager global;
    return &global;
  }

  struct CudnnCacheInfo {
    phi::backends::gpu::TensorDescriptor x_desc;
    phi::backends::gpu::FilterDescriptor w_desc;
    phi::backends::gpu::TensorDescriptor b_desc;
    phi::backends::gpu::TensorDescriptor o_desc;
    phi::backends::gpu::ConvolutionDescriptor conv_desc;
    phi::backends::gpu::ActivationDescriptor act_desc;
    size_t workspace_size;
    cudnnConvolutionFwdAlgo_t algo;

    std::vector<int> paddings;
    std::vector<int> dilations;
    std::vector<int> input_pad;
    std::vector<int> new_input_shape_vec;
    bool is_sys_pad;
  };

  CudnnCacheInfo* GetCudnnCacheInfo(
      const std::vector<int>& input_dims,
      const std::vector<int>& filter_dims,
      const std::vector<int>& bias_dims,
      const std::vector<int>& output_dims,
      const std::vector<int>& paddings,
      const std::vector<int>& strides,
      const std::vector<int>& dilations,
      phi::DataType input_dtype,
      int groups,
      cudnnDataType_t dtype,
      cudnnTensorFormat_t format,
      const std::function<void(cudnnConvolutionFwdAlgo_t*,
                               size_t*,
                               cudnnTensorDescriptor_t,
                               cudnnFilterDescriptor_t,
                               cudnnTensorDescriptor_t,
                               cudnnConvolutionDescriptor_t)>& search_func,
      const std::string& act,
      double value_max = std::numeric_limits<double>::max()) {
    // std::hash takes about 5us, xxhash can optimize to 2.5us.
    XXH64_state_t* const state = XXH64_createState();
    if (state == nullptr) {
      CHECK(false);
    }
    XXH64_hash_t const seed = 0;
    if (XXH64_reset(state, seed) == XXH_ERROR) {
      CHECK(false);
    }
    XXH64_update(state, input_dims.data(), input_dims.size() * sizeof(int));
    XXH64_update(state, filter_dims.data(), filter_dims.size() * sizeof(int));
    XXH64_update(state, bias_dims.data(), bias_dims.size() * sizeof(int));
    // XXH64_update(state, output_dims.data(), output_dims.size() *
    // sizeof(int));
    XXH64_update(state, paddings.data(), paddings.size() * sizeof(int));
    XXH64_update(state, strides.data(), strides.size() * sizeof(int));
    XXH64_update(state, dilations.data(), dilations.size() * sizeof(int));
    XXH64_update(state, &input_dtype, sizeof(int));
    XXH64_update(state, &groups, sizeof(int));
    XXH64_update(state, &dtype, sizeof(int));
    XXH64_update(state, &format, sizeof(int));
    XXH64_update(state, &act, act.size() * sizeof(char));
    // XXH64_update(state, &value_max, sizeof(double));
    XXH64_hash_t hash_key = XXH64_digest(state);
    XXH64_freeState(state);

    if (!cudnn_conv_cache_.count(hash_key)) {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      if (!cudnn_conv_cache_.count(hash_key)) {
        CudnnCacheInfo cache_info;

        cache_info.x_desc = GetTensorDescInfo(input_dims, input_dtype, format);
        cache_info.w_desc = GetFilterDescInfo(filter_dims, input_dtype, format);
        cache_info.b_desc = GetTensorDescInfo(bias_dims, input_dtype, format);
        cache_info.o_desc = GetTensorDescInfo(output_dims, input_dtype, format);
        cache_info.conv_desc =
            GetConvDescInfo(paddings, strides, dilations, groups, dtype);
        cache_info.act_desc = GetActivationDescInfo(act, value_max);

        size_t workspace_size;
        cudnnConvolutionFwdAlgo_t algo;
        search_func(&algo,
                    &workspace_size,
                    cache_info.x_desc.desc(),
                    cache_info.w_desc.desc(),
                    cache_info.o_desc.desc(),
                    cache_info.conv_desc.desc());
        cache_info.workspace_size = workspace_size;
        cache_info.algo = algo;

        cudnn_conv_cache_[hash_key] = std::move(cache_info);
      }
    }

    return &cudnn_conv_cache_.at(hash_key);
  }

  struct ConvAttrCacheInfo {
    std::vector<int> paddings;
    std::vector<int> dilations;
    std::vector<int> input_pad;
    std::vector<int> new_input_shape_vec;
    bool is_sys_pad;
  };
  ConvAttrCacheInfo* GetConvAttr(const std::vector<int>& paddings_t,
                                 const std::vector<int>& dilations_t,
                                 const std::string& padding_algorithm,
                                 const std::vector<int>& input_dims,
                                 const std::vector<int>& filter_dims,
                                 const std::vector<int>& strides,
                                 cudnnTensorFormat_t format) {
    // std::hash takes about 5us, xxhash can optimize to 2.5us.
    XXH64_state_t* const state = XXH64_createState();
    if (state == nullptr) {
      CHECK(false);
    }
    XXH64_hash_t const seed = 0;
    if (XXH64_reset(state, seed) == XXH_ERROR) {
      CHECK(false);
    }
    XXH64_update(state, paddings_t.data(), paddings_t.size() * sizeof(int));
    XXH64_update(state, dilations_t.data(), dilations_t.size() * sizeof(int));
    XXH64_update(state, input_dims.data(), input_dims.size() * sizeof(int));
    XXH64_update(state, filter_dims.data(), filter_dims.size() * sizeof(int));
    XXH64_update(state, strides.data(), strides.size() * sizeof(int));
    XXH64_update(state, &format, sizeof(int));
    XXH64_update(
        state, &padding_algorithm, padding_algorithm.size() * sizeof(char));
    XXH64_hash_t hash_key = XXH64_digest(state);
    XXH64_freeState(state);

    if (!conv_attr_cache_.count(hash_key)) {
      std::lock_guard<std::mutex> lock(attr_mutex_);
      if (!conv_attr_cache_.count(hash_key)) {
        auto printVec = [](const std::vector<int>& vec) {
          std::ostringstream os;
          for (size_t i = 0; i < vec.size(); ++i) os << vec[i] << ", ";
          return os.str();
        };
        ConvAttrCacheInfo cache;
        auto paddings = paddings_t;
        auto dilations = dilations_t;
        std::vector<int> in_data_dims(input_dims.size() - 2);
        std::vector<int> ksize(filter_dims.size() - 2);
        if (format == CUDNN_TENSOR_NHWC) {
          for (size_t i = 1; i < input_dims.size() - 1; ++i) {
            in_data_dims[i - 1] = input_dims[i];
          }
          for (size_t i = 1; i < filter_dims.size() - 1; ++i) {
            ksize[i - 1] = filter_dims[i];
          }
        } else {
          for (size_t i = 2; i < input_dims.size(); ++i) {
            in_data_dims[i - 2] = input_dims[i];
          }
          for (size_t i = 2; i < filter_dims.size(); ++i) {
            ksize[i - 2] = filter_dims[i];
          }
        }
        phi::UpdatePaddingAndDilation(&paddings,
                                      &dilations,
                                      padding_algorithm,
                                      make_ddim(in_data_dims),
                                      strides,
                                      ksize);

        int data_dim = strides.size();  // 2d or 3d
        bool is_sys_pad = funcs::IsSymmetricPadding(paddings, data_dim);
        std::vector<int> padding_common(data_dim, 0);
        if (!is_sys_pad) {
          std::vector<int> padding_diff(data_dim);
          std::vector<int> new_input_shape_vec(data_dim + 2);
          new_input_shape_vec[0] = input_dims[0];

          if (format == CUDNN_TENSOR_NCHW) {
            new_input_shape_vec[1] = input_dims[1];
          } else {
            new_input_shape_vec[data_dim + 1] = input_dims[data_dim + 1];
          }

          std::vector<int> input_pad(input_dims.size() * 2, 0);
          for (size_t i = 0; i < data_dim; ++i) {
            padding_diff[i] = std::abs(paddings[2 * i] - paddings[2 * i + 1]);
            padding_common[i] = std::min(paddings[2 * i], paddings[2 * i + 1]);
            if (format == CUDNN_TENSOR_NCHW) {
              new_input_shape_vec[i + 2] = input_dims[i + 2] + padding_diff[i];
            } else {
              new_input_shape_vec[i + 1] = input_dims[i + 1] + padding_diff[i];
            }
            if (format == CUDNN_TENSOR_NCHW) {
              input_pad[2 * i + 4] = paddings[2 * i] - padding_common[i];
              input_pad[2 * i + 4 + 1] =
                  paddings[2 * i + 1] - padding_common[i];
            } else {
              input_pad[2 * i + 2] = paddings[2 * i] - padding_common[i];
              input_pad[2 * i + 2 + 1] =
                  paddings[2 * i + 1] - padding_common[i];
            }
          }

          cache.is_sys_pad = false;
          cache.input_pad = input_pad;
          cache.new_input_shape_vec = new_input_shape_vec;
        } else {
          cache.is_sys_pad = true;
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

        cache.dilations = dilations;
        cache.paddings = padding_common;
        conv_attr_cache_[hash_key] = cache;
      }
    }

    return &conv_attr_cache_.at(hash_key);
  }

 private:
  phi::backends::gpu::TensorDescriptor GetTensorDescInfo(
      const std::vector<int>& input_dims,
      phi::DataType input_dtype,
      cudnnTensorFormat_t input_format) {
    phi::backends::gpu::TensorDescriptor desc;
    desc.set(
        input_dims, input_format, backends::gpu::ToCudnnDataType(input_dtype));
    return desc;
  }

  phi::backends::gpu::FilterDescriptor GetFilterDescInfo(
      const std::vector<int>& input_dims,
      phi::DataType input_dtype,
      cudnnTensorFormat_t input_format) {
    phi::backends::gpu::FilterDescriptor desc;
    desc.set(
        input_dims, input_format, backends::gpu::ToCudnnDataType(input_dtype));
    return desc;
  }

  phi::backends::gpu::ConvolutionDescriptor GetConvDescInfo(
      const std::vector<int>& paddings,
      const std::vector<int>& strides,
      const std::vector<int>& dilations,
      int groups,
      cudnnDataType_t dtype) {
    phi::backends::gpu::ConvolutionDescriptor conv_desc;
    conv_desc.set(dtype,
                  paddings,
                  strides,
                  dilations,
                  paddle::platform::AllowTF32Cudnn(),
                  groups);
    return conv_desc;
  }

  phi::backends::gpu::ActivationDescriptor GetActivationDescInfo(
      const std::string& act,
      double value_max = std::numeric_limits<double>::max()) {
    phi::backends::gpu::ActivationDescriptor desc;
    cudnnActivationMode_t mode;
    double relu_ceiling = 0.0;
    if (act == "identity") {
      mode = CUDNN_ACTIVATION_IDENTITY;
    } else if (act == "relu") {
      mode = CUDNN_ACTIVATION_RELU;
    } else if (act == "relu6") {
      relu_ceiling = 6.0;
      mode = CUDNN_ACTIVATION_CLIPPED_RELU;
    } else if (act == "sigmoid") {
      mode = CUDNN_ACTIVATION_SIGMOID;
    } else if (act == "relux") {
      relu_ceiling = value_max;
      mode = CUDNN_ACTIVATION_CLIPPED_RELU;
    } else if (act == "tanh") {
      mode = CUDNN_ACTIVATION_TANH;
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Unknown CUDNN activation string: %s.", act));
    }
    desc.set(mode, relu_ceiling);
    return desc;
  }

  std::mutex cache_mutex_;
  std::unordered_map<size_t, CudnnCacheInfo> cudnn_conv_cache_;

  std::mutex attr_mutex_;
  std::unordered_map<size_t, ConvAttrCacheInfo> conv_attr_cache_;
};
}  // namespace

template <typename T, typename Context>
void ConvFusionKernel(const Context& ctx,
                      const DenseTensor& input,
                      const DenseTensor& filter,
                      const DenseTensor& bias,
                      const paddle::optional<DenseTensor>& residual,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings_t,
                      const std::string& padding_algorithm,
                      const std::vector<int>& dilations_t,
                      int groups,
                      const std::string& data_format,
                      const std::string& activation,
                      bool exhaustive_search,
                      const std::vector<int>& channels,
                      int user_workspace_size,
                      DenseTensor* output,
                      std::vector<DenseTensor*> outs) {
  auto handle = ctx.cudnn_handle();
  ctx.template Alloc<T>(output);
  auto workspace_handle = ctx.cudnn_workspace_handle();

  exhaustive_search = FLAGS_cudnn_exhaustive_search || exhaustive_search;
  bool deterministic = FLAGS_cudnn_deterministic;
  PADDLE_ENFORCE_EQ(exhaustive_search && deterministic,
                    false,
                    phi::errors::InvalidArgument(
                        "Cann't set exhaustive_search True and "
                        "FLAGS_cudnn_deterministic True at same time."));

  size_t workspace_size_limit = 0;
  if (FLAGS_conv_workspace_size_limit > 0 || user_workspace_size > 0) {
    int64_t max_user_size =
        std::min(static_cast<int64_t>(FLAGS_conv_workspace_size_limit),
                 static_cast<int64_t>(user_workspace_size));
    workspace_size_limit = max_user_size * 1024 * 1024;
  }

  auto dtype = phi::backends::gpu::CudnnDataType<T>::type;
  const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");
  // Choose NHWC or NCHW by data_format attr.
  auto compute_format = channel_last ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;
  VLOG(3) << "Compute ConvFusionOp with cuDNN:"
          << " data_format=" << data_format << " compute_format="
          << (compute_format == CUDNN_TENSOR_NHWC ? "NHWC" : "NCHW");

  auto* conv_attr_cache = CudnnConvDescManager::Instance()->GetConvAttr(
      paddings_t,
      dilations_t,
      padding_algorithm,
      phi::vectorize<int>(input.dims()),
      phi::vectorize<int>(filter.dims()),
      strides,
      compute_format);

  DenseTensor transformed_input;
  auto unsys_pad_process = [&](const std::vector<int>& new_input_shape_vec,
                               const std::vector<int>& input_pad) {
    DDim new_input_shape(make_ddim(new_input_shape_vec));
    transformed_input.Resize(new_input_shape);
    ctx.template Alloc<T>(&transformed_input);

    const int rank = input.dims().size();
    T pad_value(0.0);
    switch (rank) {
      case 4: {
        funcs::PadFunction<Context, T, 4>(
            ctx, input_pad, input, pad_value, &transformed_input);
      } break;
      case 5: {
        funcs::PadFunction<Context, T, 5>(
            ctx, input_pad, input, pad_value, &transformed_input);
      } break;
      default:
        PADDLE_THROW(phi::errors::InvalidArgument(
            "ConvOp only support tensors with 4 or 5 dimensions."));
    }
  };
  if (conv_attr_cache->is_sys_pad) {
    transformed_input.ShareDataWith(input);
  } else {
    unsys_pad_process(conv_attr_cache->new_input_shape_vec,
                      conv_attr_cache->input_pad);
  }

  std::vector<int> b_dims(input.dims().size(), 1);
  if (compute_format == CUDNN_TENSOR_NCHW) {
    b_dims[1] = static_cast<int>(bias.dims()[0]);
  } else {
    b_dims[input.dims().size() - 1] = static_cast<int>(bias.dims()[0]);
  }

  std::function<void(cudnnConvolutionFwdAlgo_t*,
                     size_t*,
                     cudnnTensorDescriptor_t,
                     cudnnFilterDescriptor_t,
                     cudnnTensorDescriptor_t,
                     cudnnConvolutionDescriptor_t)>
      search_func = [&](cudnnConvolutionFwdAlgo_t* cudnn_algo,
                        size_t* wks_bytes,
                        cudnnTensorDescriptor_t x_desc,
                        cudnnFilterDescriptor_t w_desc,
                        cudnnTensorDescriptor_t o_desc,
                        cudnnConvolutionDescriptor_t cudnn_conv_desc) {
        if (!exhaustive_search) {
#if CUDNN_VERSION >= 8000
          int perf_count;
          int best_algo_idx = 0;
          size_t tmp_size = 0;
          std::unique_ptr<cudnnConvolutionFwdAlgoPerf_t[]> perf_results(
              new cudnnConvolutionFwdAlgoPerf_t[phi::kNUM_CUDNN_FWD_ALGS]);
          PADDLE_ENFORCE_GPU_SUCCESS(
              phi::dynload::cudnnGetConvolutionForwardAlgorithm_v7(
                  handle,
                  x_desc,
                  w_desc,
                  cudnn_conv_desc,
                  o_desc,
                  phi::kNUM_CUDNN_FWD_ALGS,
                  &perf_count,
                  perf_results.get()));
          *cudnn_algo = (perf_results.get())[best_algo_idx].algo;
#else
          PADDLE_ENFORCE_GPU_SUCCESS(
              phi::dynload::cudnnGetConvolutionForwardAlgorithm(
                  handle,
                  x_desc,
                  w_desc,
                  cudnn_conv_desc,
                  o_desc,
                  CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                  workspace_size_limit,
                  cudnn_algo));
#endif
          PADDLE_ENFORCE_GPU_SUCCESS(
              phi::dynload::cudnnGetConvolutionForwardWorkspaceSize(
                  handle,
                  x_desc,
                  w_desc,
                  cudnn_conv_desc,
                  o_desc,
                  *cudnn_algo,
                  wks_bytes));
        } else {
          std::array<cudnnConvolutionFwdAlgoPerf_t, phi::kNUM_CUDNN_FWD_ALGS>
              fwd_perf_stat;
          int returned_algo_count;
          auto cudnn_find_func = [&](void* cudnn_workspace) {
            PADDLE_ENFORCE_GPU_SUCCESS(
                phi::dynload::cudnnFindConvolutionForwardAlgorithmEx(
                    handle,
                    x_desc,
                    transformed_input.data(),
                    w_desc,
                    filter.data(),
                    cudnn_conv_desc,
                    o_desc,
                    output->data(),
                    phi::kNUM_CUDNN_FWD_ALGS,
                    &returned_algo_count,
                    fwd_perf_stat.data(),
                    cudnn_workspace,
                    workspace_size_limit));
          };
          workspace_handle.RunFuncSync(cudnn_find_func, workspace_size_limit);
          *cudnn_algo = fwd_perf_stat[0].algo;

          PADDLE_ENFORCE_GPU_SUCCESS(
              phi::dynload::cudnnGetConvolutionForwardWorkspaceSize(
                  handle,
                  x_desc,
                  w_desc,
                  cudnn_conv_desc,
                  o_desc,
                  fwd_perf_stat[0].algo,
                  wks_bytes));
        }
      };

  auto cudnn_cache_info = CudnnConvDescManager::Instance()->GetCudnnCacheInfo(
      phi::vectorize<int>(transformed_input.dims()),
      phi::vectorize<int>(filter.dims()),
      b_dims,
      phi::vectorize<int>(output->dims()),
      conv_attr_cache->paddings,
      strides,
      conv_attr_cache->dilations,
      transformed_input.dtype(),
      groups,
      phi::backends::gpu::CudnnDataType<T>::type,
      compute_format,
      search_func,
      activation);

  auto x_desc = cudnn_cache_info->x_desc.desc();
  auto w_desc = cudnn_cache_info->w_desc.desc();
  auto b_desc = cudnn_cache_info->b_desc.desc();
  auto o_desc = cudnn_cache_info->o_desc.desc();
  auto cudnn_conv_desc = cudnn_cache_info->conv_desc.desc();
  auto act_desc = cudnn_cache_info->act_desc.desc();
  auto algo = cudnn_cache_info->algo;
  auto workspace_size = cudnn_cache_info->workspace_size;

  if ((activation == "identity") && (!residual.get_ptr())) {
    // Only the CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM algo is
    // enabled with CUDNN_ACTIVATION_IDENTITY in cuDNN lib.
    // But test in some case, the speed is slower, change to use
    // cudnnConvolutionForward and cudnnAddTensor
    // ------------- cudnn conv forward and bias add ---------------------
    ScalingParamType<T> alpha = 1.0f, beta = 0.0f;
    auto cudnn_func = [&](void* cudnn_workspace) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cudnnConvolutionForward(handle,
                                                &alpha,
                                                x_desc,
                                                transformed_input.data(),
                                                w_desc,
                                                filter.data(),
                                                cudnn_conv_desc,
                                                algo,
                                                cudnn_workspace,
                                                workspace_size,
                                                &beta,
                                                o_desc,
                                                output->data()));
    };
    workspace_handle.RunFunc(cudnn_func, workspace_size);
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnAddTensor(
        handle, &alpha, b_desc, bias.data(), &alpha, o_desc, output->data()));
  } else {
    // Only the CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_​PRECOMP_GEMM algo is
    // enabled with CUDNN_ACTIVATION_IDENTITY.
    if (activation == "identity") {
      algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    }

    ScalingParamType<T> alpha = 1.0f;
    ScalingParamType<T> beta = residual.get_ptr() ? 1.0f : 0.0f;
    auto cudnn_func = [&](void* cudnn_workspace) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cudnnConvolutionBiasActivationForward(
              handle,
              &alpha,
              x_desc,
              transformed_input.data(),
              w_desc,
              filter.data(),
              cudnn_conv_desc,
              algo,
              cudnn_workspace,
              workspace_size,
              &beta,
              o_desc,
              residual.get_ptr() ? residual->data() : output->data(),
              b_desc,
              bias.data(),
              act_desc,
              o_desc,
              output->data()));
    };
    workspace_handle.RunFunc(cudnn_func, workspace_size);
  }

  if (!channels.empty()) {
    if (transformed_input.dims()[0] == 1 &&
        compute_format == CUDNN_TENSOR_NCHW) {
      // share data with Output
      phi::DenseTensor t;
      t.ShareDataWith(*output);
      auto y_dims = output->dims();
      t.Resize({y_dims[1], y_dims[2], y_dims[3]});
      int s = 0;
      for (size_t i = 0; i < channels.size(); ++i) {
        int e = s + channels[i];
        outs[i]->ShareDataWith(t.Slice(s, e));
        outs[i]->Resize(
            {transformed_input.dims()[0], channels[i], y_dims[2], y_dims[3]});
        s = e;
      }
    } else {
      // TODO(qingiqng): do copy when batch size large than 1
      PADDLE_THROW(phi::errors::Unimplemented(
          "Input with batch size greater than 1 is unsupported. The received "
          "batch size is %d, Input's shape is [%s].",
          transformed_input.dims()[0],
          transformed_input.dims()));
    }
  }
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(conv2d_fusion,  // cuda_only
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::fusion::ConvFusionKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#endif
