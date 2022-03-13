/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/operators/conv_base_helper.h"
#include "paddle/fluid/platform/cuda_graph_with_memory_pool.h"
#include "paddle/fluid/platform/kernel_metric_tools.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace paddle {
namespace operators {

using ConvArgs = ConvArgsBase<cudnnHandle_t, cudnnDataType_t>;

template <typename Algo_t>
struct AlgoResult {
  Algo_t algo = static_cast<Algo_t>(0);
  float time = -1.f;
  size_t workspace_size = -1;
};

template <typename DeviceContext, typename T, size_t D>
static void RemovePaddingSlice(const phi::GPUContext& context,
                               const Tensor* input, Tensor* out,
                               const std::vector<int>& starts,
                               const std::vector<int>& axes) {
  auto& place = *context.eigen_device();
  auto in_dims = input->dims();
  auto new_out_dims = out->dims();
  auto offsets = Eigen::DSizes<Eigen::DenseIndex, D>();
  auto extents = Eigen::DSizes<Eigen::DenseIndex, D>();
  for (size_t i = 0; i < D; ++i) {
    offsets[i] = 0;
    extents[i] = new_out_dims[i];
  }

  for (size_t i = 0; i < axes.size(); ++i) {
    int start = starts[i];
    if (start < 0) {
      start = (start + in_dims[axes[i]]);
    }
    start = std::max(start, 0);
    offsets[axes[i]] = start;
  }
  auto in_t =
      framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
          *input);

  auto out_t =
      framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
          *out, new_out_dims);
  phi::funcs::EigenSlice<std::decay_t<decltype(place)>, T, D>::Eval(
      place, out_t, in_t, offsets, extents);
}

template <typename PerfType, typename AlgoResult_t>
void ChooseAlgoByWorkspace(PerfType* perf_results, size_t perf_num,
                           size_t workspace_byte, AlgoResult_t* algo_result) {
  for (size_t i = 0; i < perf_num; ++i) {
    auto result = perf_results[i];
    if (result.status == CUDNN_STATUS_SUCCESS &&
        result.memory < workspace_byte) {
      algo_result->algo = result.algo;
      algo_result->time = result.time;
      algo_result->time = result.memory;
      VLOG(3) << "    algo: " << result.algo << ", time: " << result.time
              << " ms, wksp = " << result.memory
              << ", status = " << result.status;
      return;
    }
  }
  VLOG(3) << "Can not find alog that requires memory < "
          << static_cast<double>(workspace_byte) / (1 << 20) << " MB";
}

static void SetConvMathType(const phi::GPUContext& ctx, cudnnDataType_t dtype,
                            const platform::ConvolutionDescriptor& cdesc) {
#if CUDA_VERSION >= 9000 && CUDNN_VERSION_MIN(7, 0, 1)
  auto& dev_ctx = ctx;
  if (dev_ctx.GetComputeCapability() >= 70 && dtype == CUDNN_DATA_HALF) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetConvolutionMathType(
        cdesc.desc(), CUDNN_TENSOR_OP_MATH));
    VLOG(5) << "use cudnn_tensor_op_math";
#if CUDA_VERSION >= 11000
#if CUDNN_VERSION_MIN(8, 1, 0)
  } else if (dev_ctx.GetComputeCapability() >= 80 &&
             dtype == CUDNN_DATA_BFLOAT16) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetConvolutionMathType(
        cdesc.desc(), CUDNN_TENSOR_OP_MATH));
#endif  // CUDNN_VERSION_MIN(8, 1, 0)
  } else if (dtype == CUDNN_DATA_FLOAT && !cdesc.allow_tf32_) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetConvolutionMathType(
        cdesc.desc(), CUDNN_FMA_MATH));
#endif  // CUDA_VERSION >= 11000
  } else {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetConvolutionMathType(
        cdesc.desc(), CUDNN_DEFAULT_MATH));
    VLOG(5) << "NOT use cudnn_tensor_op_math";
  }
#endif
}

template <typename perf_t>
struct SearchAlgorithm {};

template <>
struct SearchAlgorithm<cudnnConvolutionFwdAlgoPerf_t> {
  using perf_t = cudnnConvolutionFwdAlgoPerf_t;
  using algo_result_t = AlgoResult<cudnnConvolutionFwdAlgo_t>;

  template <typename T>
  static algo_result_t Find(const ConvArgs& args, bool exhaustive_search,
                            bool deterministic, const phi::GPUContext& ctx) {
    auto dtype = platform::CudnnDataType<T>::type;
    size_t workspace_size_limit =
        platform::KernelMetricsTool().GpuMemoryQuery();
    size_t workspace_size = 0;
    algo_result_t result;
    SetConvMathType(ctx, dtype, args.cdesc);

    if (!exhaustive_search && !deterministic) {
#if CUDNN_VERSION >= 7001
      int perf_count;
      int best_algo_idx = 0;
      std::unique_ptr<perf_t[]> perf_results(new perf_t[kNUM_CUDNN_FWD_ALGS]);
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnGetConvolutionForwardAlgorithm_v7(
              args.handle, args.idesc.desc(), args.wdesc.desc(),
              args.cdesc.desc(), args.odesc.desc(), kNUM_CUDNN_FWD_ALGS,
              &perf_count, perf_results.get()));
      result.algo = (perf_results.get())[best_algo_idx].algo;
      workspace_size = (perf_results.get())[best_algo_idx].memory;

      if (workspace_size > workspace_size_limit) {
#if CUDNN_VERSION >= 8000
        // cudnnGetConvolutionForwardAlgorithm is removed in CUDNN-8
        ChooseAlgoByWorkspace<perf_t, algo_result_t>(
            perf_results.get(), kNUM_CUDNN_FWD_ALGS, workspace_size_limit,
            &result);
#else
        VLOG(1) << "Fallback to non-v7 method to find conv algorithm becasue "
                   "the workspace size request("
                << workspace_size << ") exceeds the limit("
                << workspace_size_limit << ")";
        PADDLE_ENFORCE_GPU_SUCCESS(
            platform::dynload::cudnnGetConvolutionForwardAlgorithm(
                args.handle, args.idesc.desc(), args.wdesc.desc(),
                args.cdesc.desc(), args.odesc.desc(),
                CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                workspace_size_limit, &(result.algo)));
#endif
      }
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnGetConvolutionForwardAlgorithm(
              args.handle, args.idesc.desc(), args.wdesc.desc(),
              args.cdesc.desc(), args.odesc.desc(),
              CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
              workspace_size_limit, &(result.algo)));
#endif
    } else if (deterministic) {
      result.algo = static_cast<cudnnConvolutionFwdAlgo_t>(1);
    } else {
      auto& dev_ctx = ctx;
      auto workspace_handle = dev_ctx.cudnn_workspace_handle();
      auto x_dims = phi::vectorize(args.x->dims());
      auto w_dims = phi::vectorize(args.w->dims());
      VLOG(10) << "cudnnConvolutionFwdAlgoPerf_t:"
               << ", x_dims:" << x_dims << ", w_dims:" << w_dims << ", args.s"
               << args.s << ", args.p" << args.p << ", args.d" << args.d;

      AlgorithmsCache<cudnnConvolutionFwdAlgo_t>& algo_cache =
          *(framework::ConvSearchCache::Instance().GetForward());

      result.algo = algo_cache.GetAlgorithm(
          x_dims, w_dims, args.s, args.p, args.d, 0,
          static_cast<int64_t>(args.cudnn_dtype), [&]() {
            int returned_algo_count;
            std::array<perf_t, kNUM_CUDNN_FWD_ALGS> perf_stat;

            auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
              PADDLE_ENFORCE_GPU_SUCCESS(
                  platform::dynload::cudnnFindConvolutionForwardAlgorithmEx(
                      args.handle, args.idesc.desc(), args.x->data<T>(),
                      args.wdesc.desc(), args.w->data<T>(), args.cdesc.desc(),
                      args.odesc.desc(), const_cast<T*>(args.o->data<T>()),
                      kNUM_CUDNN_FWD_ALGS, &returned_algo_count,
                      perf_stat.data(), cudnn_workspace_ptr,
                      workspace_size_limit));
            };
            workspace_handle.RunFuncSync(cudnn_find_func, workspace_size_limit);

            VLOG(3) << "FwdAlgo Perf result: (algo: stat, time, memory)";
            for (int i = 0; i < returned_algo_count; ++i) {
              const auto& stat = perf_stat[i];
              VLOG(3) << stat.algo << ": " << stat.status << " " << stat.time
                      << " " << stat.memory;
            }
            result.time = perf_stat[0].time;
            return perf_stat[0].algo;
          });
    }
    VLOG(3) << "choose algo " << result.algo;
    return result;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args,
                                 cudnnConvolutionFwdAlgo_t algo) {
    size_t workspace_size = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetConvolutionForwardWorkspaceSize(
            args.handle, args.idesc.desc(), args.wdesc.desc(),
            args.cdesc.desc(), args.odesc.desc(), algo, &workspace_size));
    return workspace_size;
  }
};

template <>
struct SearchAlgorithm<cudnnConvolutionBwdDataAlgoPerf_t> {
  using perf_t = cudnnConvolutionBwdDataAlgoPerf_t;
  using algo_result_t = AlgoResult<cudnnConvolutionBwdDataAlgo_t>;

  template <typename T>
  static algo_result_t Find(const ConvArgs& args, bool exhaustive_search,
                            bool deterministic, const phi::GPUContext& ctx) {
    auto dtype = platform::CudnnDataType<T>::type;
    size_t workspace_size_limit =
        platform::KernelMetricsTool().GpuMemoryQuery();
    size_t workspace_size = 0;
    algo_result_t result;
    SetConvMathType(ctx, dtype, args.cdesc);

    if (!exhaustive_search && !deterministic) {
#if CUDNN_VERSION >= 7001
      int perf_count;
      int best_algo_idx = 0;
      std::unique_ptr<perf_t[]> perf_results(
          new perf_t[kNUM_CUDNN_BWD_DATA_ALGS]);
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnGetConvolutionBackwardDataAlgorithm_v7(
              args.handle, args.wdesc.desc(), args.odesc.desc(),
              args.cdesc.desc(), args.idesc.desc(), kNUM_CUDNN_BWD_DATA_ALGS,
              &perf_count, perf_results.get()));
      result.algo = (perf_results.get())[best_algo_idx].algo;

#if CUDNN_VERSION < 7500
      int stride_dim = args.x->dims().size() - 2;
      bool blacklist = std::any_of(args.s.begin(), args.s.begin() + stride_dim,
                                   [=](int n) { return n != 1; });
      if (blacklist && (static_cast<cudnnConvolutionBwdDataAlgo_t>(
                            perf_results[best_algo_idx].algo) ==
                            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING ||
                        static_cast<cudnnConvolutionBwdDataAlgo_t>(
                            perf_results[best_algo_idx].algo) ==
                            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)) {
        result.algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
      }
#endif
      workspace_size = GetWorkspaceSize(args, result.algo);
      if (workspace_size > workspace_size_limit) {
#if CUDNN_VERSION >= 8000
        // cudnnGetConvolutionBackwardDataAlgorithm is removed in CUDNN-8
        ChooseAlgoByWorkspace<perf_t, algo_result_t>(
            perf_results.get(), kNUM_CUDNN_BWD_DATA_ALGS, workspace_size_limit,
            &result);
#else
        VLOG(1) << "Fallback to non-v7 method to find conv algorithm becasue "
                   "the workspace size request("
                << workspace_size << ") exceeds the limit("
                << workspace_size_limit << ")";
        PADDLE_ENFORCE_GPU_SUCCESS(
            platform::dynload::cudnnGetConvolutionBackwardDataAlgorithm(
                args.handle, args.wdesc.desc(), args.odesc.desc(),
                args.cdesc.desc(), args.idesc.desc(),
                CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
                workspace_size_limit, &(result.algo)));
#endif
      }
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnGetConvolutionBackwardDataAlgorithm(
              args.handle, args.wdesc.desc(), args.odesc.desc(),
              args.cdesc.desc(), args.idesc.desc(),
              CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
              workspace_size_limit, &(result.algo)));
#endif
    } else if (deterministic) {
      result.algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    } else {
      auto& dev_ctx = ctx;
      auto workspace_handle = dev_ctx.cudnn_workspace_handle();
      auto x_dims = phi::vectorize(args.x->dims());
      auto w_dims = phi::vectorize(args.w->dims());
      VLOG(10) << "cudnnConvolutionFwdAlgoPerf_t"
               << ", x_dims:" << x_dims << ", w_dims:" << w_dims << ", args.s"
               << args.s << ", args.p" << args.p << ", args.d" << args.d;

      AlgorithmsCache<cudnnConvolutionBwdDataAlgo_t>& algo_cache =
          *(framework::ConvSearchCache::Instance().GetBackwardData());

      result.algo = algo_cache.GetAlgorithm(
          x_dims, w_dims, args.s, args.p, args.d, 0,
          static_cast<int64_t>(args.cudnn_dtype), [&]() {
            int returned_algo_count;
            std::array<perf_t, kNUM_CUDNN_BWD_DATA_ALGS> perf_stat;

            auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
              PADDLE_ENFORCE_GPU_SUCCESS(
                  platform::dynload::
                      cudnnFindConvolutionBackwardDataAlgorithmEx(
                          args.handle, args.wdesc.desc(), args.w->data<T>(),
                          args.odesc.desc(), args.o->data<T>(),
                          args.cdesc.desc(), args.idesc.desc(),
                          const_cast<T*>(args.x->data<T>()),
                          kNUM_CUDNN_BWD_DATA_ALGS, &returned_algo_count,
                          perf_stat.data(), cudnn_workspace_ptr,
                          workspace_size_limit));
            };
            workspace_handle.RunFuncSync(cudnn_find_func, workspace_size_limit);

            VLOG(3) << "BwdDataAlgo Perf result: (algo: stat, time, memory)";
            for (int i = 0; i < returned_algo_count; ++i) {
              const auto& stat = perf_stat[i];
              VLOG(3) << stat.algo << ": " << stat.status << " " << stat.time
                      << " " << stat.memory;
            }
            result.time = perf_stat[0].time;
            return perf_stat[0].algo;
          });
    }
    VLOG(3) << "choose algo " << result.algo;
    return result;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args,
                                 cudnnConvolutionBwdDataAlgo_t algo) {
    size_t workspace_size = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetConvolutionBackwardDataWorkspaceSize(
            args.handle, args.wdesc.desc(), args.odesc.desc(),
            args.cdesc.desc(), args.idesc.desc(), algo, &workspace_size));
    return workspace_size;
  }
};

template <>
struct SearchAlgorithm<cudnnConvolutionBwdFilterAlgoPerf_t> {
  using perf_t = cudnnConvolutionBwdFilterAlgoPerf_t;
  using algo_result_t = AlgoResult<cudnnConvolutionBwdFilterAlgo_t>;

  template <typename T>
  static algo_result_t Find(const ConvArgs& args, bool exhaustive_search,
                            bool deterministic, const phi::GPUContext& ctx) {
    algo_result_t result;
    platform::CUDAGraphCaptureModeGuard guard;
    auto dtype = platform::CudnnDataType<T>::type;
    size_t workspace_size_limit =
        platform::KernelMetricsTool().GpuMemoryQuery();
    size_t workspace_size = 0;
    SetConvMathType(ctx, dtype, args.cdesc);

    if (!exhaustive_search && !deterministic) {
#if CUDNN_VERSION >= 7001
      using perf_t = cudnnConvolutionBwdFilterAlgoPerf_t;
      int perf_count;
      int best_algo_idx = 0;
      std::unique_ptr<perf_t[]> perf_results(
          new perf_t[kNUM_CUDNN_BWD_FILTER_ALGS]);
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnGetConvolutionBackwardFilterAlgorithm_v7(
              args.handle, args.idesc.desc(), args.odesc.desc(),
              args.cdesc.desc(), args.wdesc.desc(), kNUM_CUDNN_BWD_FILTER_ALGS,
              &perf_count, perf_results.get()));
      result.algo = (perf_results.get())[best_algo_idx].algo;
      workspace_size = (perf_results.get())[best_algo_idx].memory;

      if (workspace_size > workspace_size_limit) {
        workspace_size = workspace_size_limit;
#if CUDNN_VERSION >= 8000
        // cudnnGetConvolutionBackwardFilterAlgorithm is removed in CUDNN-8
        ChooseAlgoByWorkspace<perf_t, algo_result_t>(
            perf_results.get(), kNUM_CUDNN_BWD_FILTER_ALGS,
            workspace_size_limit, &result);
#else
        VLOG(1) << "Fallback to non-v7 method to find conv algorithm becasue "
                   "the workspace size request("
                << workspace_size << ") exceeds the limit("
                << workspace_size_limit << ")";
        PADDLE_ENFORCE_GPU_SUCCESS(
            platform::dynload::cudnnGetConvolutionBackwardFilterAlgorithm(
                args.handle, args.idesc.desc(), args.odesc.desc(),
                args.cdesc.desc(), args.wdesc.desc(),
                CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                workspace_size_limit, &(result.algo)));
#endif
      }
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnGetConvolutionBackwardFilterAlgorithm(
              args.handle, args.idesc.desc(), args.odesc.desc(),
              args.cdesc.desc(), args.wdesc.desc(),
              CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
              workspace_size_limit, &(result.algo)));
#endif
    } else if (deterministic) {
      result.algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    } else {
      auto& dev_ctx = ctx;
      auto workspace_handle = dev_ctx.cudnn_workspace_handle();
      auto x_dims = phi::vectorize(args.x->dims());
      auto w_dims = phi::vectorize(args.w->dims());
      VLOG(10) << "cudnnConvolutionFwdAlgoPerf_t:"
               << ", x_dims:" << x_dims << ", w_dims:" << w_dims << ", args.s"
               << args.s << ", args.p" << args.p << ", args.d" << args.d;

      AlgorithmsCache<cudnnConvolutionBwdFilterAlgo_t>& algo_cache =
          *(framework::ConvSearchCache::Instance().GetBackwardFilter());

      if (dtype != CUDNN_DATA_HALF) {
        result.algo = algo_cache.GetAlgorithm(
            x_dims, w_dims, args.s, args.p, args.d, 0,
            static_cast<int64_t>(args.cudnn_dtype), [&]() {
              int returned_algo_count;
              std::array<perf_t, kNUM_CUDNN_BWD_FILTER_ALGS> perf_stat;
              auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
                PADDLE_ENFORCE_GPU_SUCCESS(
                    platform::dynload::
                        cudnnFindConvolutionBackwardFilterAlgorithmEx(
                            args.handle, args.idesc.desc(), args.x->data<T>(),
                            args.odesc.desc(), args.o->data<T>(),
                            args.cdesc.desc(), args.wdesc.desc(),
                            const_cast<T*>(args.w->data<T>()),
                            kNUM_CUDNN_BWD_FILTER_ALGS, &returned_algo_count,
                            perf_stat.data(), cudnn_workspace_ptr,
                            workspace_size_limit));
              };
              workspace_handle.RunFuncSync(cudnn_find_func,
                                           workspace_size_limit);

              VLOG(3)
                  << "BwdFilterAlgo Perf result: (algo: stat, time, memory)";
              for (int i = 0; i < returned_algo_count; ++i) {
                const auto& stat = perf_stat[i];
                VLOG(3) << stat.algo << ": " << stat.status << " " << stat.time
                        << " " << stat.memory;
              }
              result.time = perf_stat[0].time;
              return perf_stat[0].algo;
            });
      } else {
        int max_algos = 0;
#if CUDNN_VERSION_MIN(7, 0, 1)
        PADDLE_ENFORCE_GPU_SUCCESS(
            platform::dynload::
                cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(args.handle,
                                                                   &max_algos));
#endif
        result.algo = algo_cache.GetAlgorithm(
            x_dims, w_dims, args.s, args.p, args.d, 0,
            static_cast<int64_t>(args.cudnn_dtype), [&]() {
              algo_result_t algo_result;
              std::vector<perf_t> perf_results(max_algos);
              int actual_algos = 0;
              PADDLE_ENFORCE_GPU_SUCCESS(
                  platform::dynload::
                      cudnnFindConvolutionBackwardFilterAlgorithm(
                          args.handle, args.idesc.desc(), args.odesc.desc(),
                          args.cdesc.desc(), args.wdesc.desc(),
                          perf_results.size(), &actual_algos,
                          perf_results.data()));
              perf_results.resize(actual_algos);
              ChooseAlgo<perf_t>(perf_results, workspace_size_limit,
                                 &algo_result);
              result.time = algo_result.time;
              return algo_result.algo;
            });
      }
    }
    VLOG(3) << "choose algo " << result.algo;
    return result;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args,
                                 cudnnConvolutionBwdFilterAlgo_t algo) {
    platform::CUDAGraphCaptureModeGuard guard;
    size_t workspace_size = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetConvolutionBackwardFilterWorkspaceSize(
            args.handle, args.idesc.desc(), args.odesc.desc(),
            args.cdesc.desc(), args.wdesc.desc(), algo, &workspace_size));
    return workspace_size;
  }

 private:
  template <typename Perf_t>
  static void ChooseAlgo(const std::vector<Perf_t>& perf_results,
                         size_t workspace_byte, algo_result_t* algo_result) {
    VLOG(3) << "=========BwdFilterAlgo Perf result=========";
    for (const auto& result : perf_results) {
      auto math_type_str = "False";
      if (result.mathType == CUDNN_TENSOR_OP_MATH) {
        math_type_str = "True";
      }
      VLOG(3) << "    algo: " << result.algo
              << ", TensorCore: " << math_type_str << ", time: " << result.time
              << " ms"
              << ", wksp = " << result.memory << ", status = " << result.status;
    }

    for (size_t i = 0; i != perf_results.size(); ++i) {
      const auto& result = perf_results[i];
      if (result.status == CUDNN_STATUS_SUCCESS &&
          (result.memory <= workspace_byte)) {
        if ((result.mathType == CUDNN_TENSOR_OP_MATH) &&
            (i != perf_results.size() - 1)) {
          const auto& next_result = perf_results[i + 1];
          if (next_result.status == CUDNN_STATUS_SUCCESS &&
              next_result.algo == result.algo &&
              next_result.memory == result.memory &&
              next_result.mathType != CUDNN_TENSOR_OP_MATH &&
              next_result.time < 1.01 * result.time) {
            // Skip over this result- it's not really a Tensor Core algo.
            // Because it is only 1% performance difference.
            // Prefer to choose the next equivalent non-Tensor Core algo.
            continue;
          }
        }
        algo_result->algo = result.algo;
        algo_result->time = result.time;
        auto math_type_str = "0";
        if (result.mathType == CUDNN_TENSOR_OP_MATH) {
          math_type_str = "1";
        }
        VLOG(3) << "    choose algo: " << result.algo
                << ", TC: " << math_type_str << ", time: " << result.time
                << " ms"
                << ", wksp = " << result.memory
                << ", status = " << result.status;
        break;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
