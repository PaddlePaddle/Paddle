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

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/conv_search_cache.h"
#include "paddle/fluid/framework/operator_kernel_configs.h"
#include "paddle/fluid/operators/conv_cudnn_op_cache.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/platform/cuda_graph_with_memory_pool.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/autotune/cache.h"
#include "paddle/phi/kernels/autotune/switch_autotune.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = platform::DataLayout;
using framework::AlgorithmsCache;
using framework::ConvSearchCache;

template <typename T>
using ScalingParamType = typename platform::CudnnDataType<T>::ScalingParamType;

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

  int start;
  for (size_t i = 0; i < axes.size(); ++i) {
    start = starts[i];
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
  EigenSlice<std::decay_t<decltype(place)>, T, D>::Eval(place, out_t, in_t,
                                                        offsets, extents);
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
  out << "[";
  for (auto const& tmp : v) out << tmp << ",";
  out << "]";
  return out;
}

template <typename PerfType, typename AlgoType>
void ChooseAlgo(const std::vector<PerfType>& perf_results,
                size_t workspace_byte, AlgoType* algo) {
  VLOG(3) << "=========BwdFilterAlgo Perf result=========";
  for (const auto& result : perf_results) {
    auto math_type_str = "False";
    if (result.mathType == CUDNN_TENSOR_OP_MATH) {
      math_type_str = "True";
    }
    VLOG(3) << "    algo: " << result.algo << ", TensorCore: " << math_type_str
            << ", time: " << result.time << " ms"
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
      *algo = result.algo;
      auto math_type_str = "0";
      if (result.mathType == CUDNN_TENSOR_OP_MATH) {
        math_type_str = "1";
      }
      VLOG(3) << "    choose algo: " << result.algo << ", TC: " << math_type_str
              << ", time: " << result.time << " ms"
              << ", wksp = " << result.memory << ", status = " << result.status;
      break;
    }
  }
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

/* After adopting cuDNN find api, more than 1 algorithm acquired. Some
  of them may perf as well as the bese algo, but occupy less memory.*/
template <typename PerfT, typename AlgoT>
static AlgoT ChooseAlgoByWorkspace(PerfT* perf_stats, int perf_num,
                                   size_t workspace_byte,
                                   bool is_find = false) {
  PADDLE_ENFORCE_GT(
      perf_num, 0,
      platform::errors::PreconditionNotMet(
          "At least one algorithm shall be acquired, but now is 0."));

  auto best_stat = perf_stats[0];
  auto best_algo = best_stat.algo;
  for (int i = 0; i < perf_num; ++i) {
    auto temp_stat = perf_stats[i];
    VLOG(3) << "  algo: " << temp_stat.algo << ", time: " << temp_stat.time
            << " ms, wksp = " << temp_stat.memory
            << ", status = " << temp_stat.status;

    if (is_find) {
      if (temp_stat.status != CUDNN_STATUS_SUCCESS) {
        continue;
      }

      float perf_diff = (temp_stat.time - best_stat.time) / best_stat.time;
      if ((perf_diff < 0.01) && (i > 0)) {
        size_t mem_diff = temp_stat.memory - best_stat.memory;
        best_stat = mem_diff < 0 ? temp_stat : best_stat;
        best_algo = mem_diff < 0 ? temp_stat.algo : best_algo;
      }
    } else {
      if (temp_stat.status == CUDNN_STATUS_SUCCESS &&
          temp_stat.memory < workspace_byte) {
        best_algo = temp_stat.algo;
      }
    }
  }
  return best_algo;
}

struct ConvArgs {
  cudnnHandle_t handle;
  platform::TensorDescriptor idesc, odesc;
  platform::FilterDescriptor wdesc;
  platform::ConvolutionDescriptor cdesc;
  const framework::Tensor *x, *w, *o;
  cudnnDataType_t cudnn_dtype;

  // strides
  std::vector<int> s;
  // paddings
  std::vector<int> p;
  // dilations
  std::vector<int> d;

  ConvArgs(const framework::Tensor* x, const framework::Tensor* w,
           const framework::Tensor* o, const std::vector<int> s,
           const std::vector<int> p, const std::vector<int> d,
           cudnnDataType_t dtype)
      : x(x), w(w), o(o), s(s), p(p), d(d), cudnn_dtype(dtype) {}

  template <typename T>
  static size_t GetConvKey(const ConvArgs* args) {
    auto x_shape = phi::vectorize(args->x->dims());
    auto w_shape = phi::vectorize(args->w->dims());
    auto key = phi::autotune::ConvKey(
        x_shape, w_shape, args->p, args->s, args->d,
        paddle::experimental::CppTypeToDataType<T>::Type());
    return key;
  }
};

template <typename perf_t>
struct SearchAlgorithm {};

/* cuDNNv7 forward algorithm searcher, consist of three operation
   modes, namely: debug mode, heuristic mode, and exhaustive search
   mode.
   As well as one workspace size acquirsition function with
   respect to the chosen alogrithm. */
template <>
struct SearchAlgorithm<cudnnConvolutionFwdAlgoPerf_t> {
  using perf_t = cudnnConvolutionFwdAlgoPerf_t;
  using algo_t = cudnnConvolutionFwdAlgo_t;

  template <typename T>
  static algo_t Find(const ConvArgs& args, bool exhaustive_search,
                     bool deterministic, const phi::GPUContext& ctx) {
    algo_t algo;
    int perf_count;
    size_t workspace_size_limit = FLAGS_conv_workspace_size_limit << 20;
    auto dtype = platform::CudnnDataType<T>::type;
    SetConvMathType(ctx, dtype, args.cdesc);

    // Debug mode.
    if (deterministic) {
      algo = static_cast<algo_t>(1);
      VLOG(3) << "Debug Fwd choosen algo is : " << algo;
      return algo;
    }

    /* Exhaustive search mode.
      1. Once turning on exhaustive FLAGS, always get exhaustive_search.
      2. Once turning on auto-tune, running heuristic search(default mode)
      before
         auto-tune process, running exhaustive_search during mentioned process.
    */
    bool open_tune = phi::autotune::AutoTuneCache::Instance().GetTuneStatus();
    if (exhaustive_search || open_tune) {
      std::string op_type("conv_fwd");
      auto key = ConvArgs::GetConvKey<T>(&args);
      auto& cache =
          phi::autotune::AutoTuneCache::Instance().RegisterOrGet(op_type);

      if (cache.Find(key) == true) {
        algo = static_cast<algo_t>(cache.Get(key));
        VLOG(3) << "Cached fwd choosen algo is : " << algo;
        return algo;
      }

      bool is_auto_tune =
          phi::autotune::AutoTuneStatus::Instance().UseAutoTune();
      if (exhaustive_search || is_auto_tune) {
        std::array<perf_t, kNUM_CUDNN_FWD_ALGS> perf_results;
        auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              paddle::platform::dynload::cudnnFindConvolutionForwardAlgorithmEx(
                  args.handle, args.idesc.desc(), args.x->data<T>(),
                  args.wdesc.desc(), args.w->data<T>(), args.cdesc.desc(),
                  args.odesc.desc(), const_cast<T*>(args.o->data<T>()),
                  kNUM_CUDNN_FWD_ALGS, &perf_count, perf_results.data(),
                  cudnn_workspace_ptr, workspace_size_limit));
        };
        auto handle = ctx.cudnn_workspace_handle();
        handle.RunFuncSync(cudnn_find_func, workspace_size_limit);
        VLOG(3) << "FwdAlgo Perf result: (algo: stat, time, memory)";
        algo = ChooseAlgoByWorkspace<perf_t, algo_t>(perf_results.data(),
                                                     perf_count, 0, true);
        cache.Set(key, static_cast<int64_t>(algo));
        VLOG(3) << "Exhaustive fwd choosen algo is : " << algo;
        return algo;
      }
    }

// Heuristic search mode, also default mode.
#if CUDNN_VERSION >= 7001
    int best_algo_idx = 0;
    size_t workspace_size = 0;
    std::unique_ptr<perf_t[]> perf_results(new perf_t[kNUM_CUDNN_FWD_ALGS]);
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetConvolutionForwardAlgorithm_v7(
            args.handle, args.idesc.desc(), args.wdesc.desc(),
            args.cdesc.desc(), args.odesc.desc(), kNUM_CUDNN_FWD_ALGS,
            &perf_count, perf_results.get()));
    algo = (perf_results.get())[best_algo_idx].algo;
    workspace_size = (perf_results.get())[best_algo_idx].memory;

    if (workspace_size > workspace_size_limit) {
#if CUDNN_VERSION >= 8000
      // cudnnGetConvolutionForwardAlgorithm is removed in cuDNNv8
      algo = ChooseAlgoByWorkspace<perf_t, algo_t>(
          perf_results.get(), kNUM_CUDNN_FWD_ALGS, workspace_size_limit);
#else
      VLOG(1) << "Fallback to non-v7 method to find conv "
                 "algorithm becasue the workspace size request("
              << workspace_size << ") exceeds the limit("
              << workspace_size_limit << ")";
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnGetConvolutionForwardAlgorithm(
              args.handle, args.idesc.desc(), args.wdesc.desc(),
              args.cdesc.desc(), args.odesc.desc(),
              CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
              workspace_size_limit, &algo));
#endif
    }
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetConvolutionForwardAlgorithm(
            args.handle, args.idesc.desc(), args.wdesc.desc(),
            args.cdesc.desc(), args.odesc.desc(),
            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, workspace_size_limit,
            &algo));
#endif
    VLOG(3) << "Default(Heuristic) fwd choosen algo is : " << algo;
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args, algo_t algo) {
    size_t workspace_size = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetConvolutionForwardWorkspaceSize(
            args.handle, args.idesc.desc(), args.wdesc.desc(),
            args.cdesc.desc(), args.odesc.desc(), algo, &workspace_size));
    return workspace_size;
  }
};

/* cuDNNv7 backward data-algorithm searcher, consist of three
   operation modes, namely: debug mode, heuristic mode, and
   exhaustive search mode. Specially, there are 2 pattens of
   exhaustive search mode, one for HALF type only, one for the
   rest.
   As well as one workspace size acquirsition function with
   respect to the chosen alogrithm. */
template <>
struct SearchAlgorithm<cudnnConvolutionBwdDataAlgoPerf_t> {
  using perf_t = cudnnConvolutionBwdDataAlgoPerf_t;
  using algo_t = cudnnConvolutionBwdDataAlgo_t;

  template <typename T>
  static algo_t Find(const ConvArgs& args, bool exhaustive_search,
                     bool deterministic, const phi::GPUContext& ctx) {
    algo_t algo;
    int perf_count;
    size_t workspace_size_limit = FLAGS_conv_workspace_size_limit << 20;
    auto dtype = platform::CudnnDataType<T>::type;
    SetConvMathType(ctx, dtype, args.cdesc);

    // Debug search mode.
    if (deterministic) {
      algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
      VLOG(3) << "Debug bwd-data choosen algo is : " << algo;
      return algo;
    }

    // Exhaustive search mode.
    bool open_tune = phi::autotune::AutoTuneCache::Instance().GetTuneStatus();
    if (exhaustive_search || open_tune) {
      std::string op_type("conv_bwd");
      auto key = ConvArgs::GetConvKey<T>(&args);
      auto& cache =
          phi::autotune::AutoTuneCache::Instance().RegisterOrGet(op_type);

      if (cache.Find(key) == true) {
        algo = static_cast<algo_t>(cache.Get(key));
        VLOG(3) << "Cached bwd-data choosen algo is : " << algo;
        return algo;
      }

      bool is_auto_tune =
          phi::autotune::AutoTuneStatus::Instance().UseAutoTune();
      if (exhaustive_search || is_auto_tune) {
        std::array<perf_t, kNUM_CUDNN_BWD_DATA_ALGS> perf_results;
        auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              platform::dynload::cudnnFindConvolutionBackwardDataAlgorithmEx(
                  args.handle, args.wdesc.desc(), args.w->data<T>(),
                  args.odesc.desc(), args.o->data<T>(), args.cdesc.desc(),
                  args.idesc.desc(), const_cast<T*>(args.x->data<T>()),
                  kNUM_CUDNN_BWD_DATA_ALGS, &perf_count, perf_results.data(),
                  cudnn_workspace_ptr, workspace_size_limit));
        };
        auto handle = ctx.cudnn_workspace_handle();
        handle.RunFuncSync(cudnn_find_func, workspace_size_limit);
        VLOG(3) << "BwdDataAlgo Perf result: (algo: stat, time, memory)";
        algo = ChooseAlgoByWorkspace<perf_t, algo_t>(perf_results.data(),
                                                     perf_count, 0, true);
        cache.Set(key, static_cast<int64_t>(algo));
        VLOG(3) << "Exhaustive bwd-data choosen algo is : " << algo;
        return algo;
      }
    }
// Heuristic search mode.
#if CUDNN_VERSION >= 7001
    int best_algo_idx = 0;
    size_t workspace_size = 0;
    std::unique_ptr<perf_t[]> perf_results(
        new perf_t[kNUM_CUDNN_BWD_DATA_ALGS]);
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetConvolutionBackwardDataAlgorithm_v7(
            args.handle, args.wdesc.desc(), args.odesc.desc(),
            args.cdesc.desc(), args.idesc.desc(), kNUM_CUDNN_BWD_DATA_ALGS,
            &perf_count, perf_results.get()));
    algo = (perf_results.get())[best_algo_idx].algo;

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
      algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    }
#endif
    workspace_size = GetWorkspaceSize(args, algo);
    if (workspace_size > workspace_size_limit) {
#if CUDNN_VERSION >= 8000
      // cudnnGetConvolutionBackwardDataAlgorithm is removed in CUDNN-8
      algo = ChooseAlgoByWorkspace<perf_t, algo_t>(
          perf_results.get(), kNUM_CUDNN_BWD_DATA_ALGS, workspace_size_limit);
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
              workspace_size_limit, &algo));
#endif
    }
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetConvolutionBackwardDataAlgorithm(
            args.handle, args.wdesc.desc(), args.odesc.desc(),
            args.cdesc.desc(), args.idesc.desc(),
            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            workspace_size_limit, &algo));
#endif
    VLOG(3) << "Default(Heuristic) bwd-data choosen algo is: " << algo;
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args, algo_t algo) {
    size_t workspace_size = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetConvolutionBackwardDataWorkspaceSize(
            args.handle, args.wdesc.desc(), args.odesc.desc(),
            args.cdesc.desc(), args.idesc.desc(), algo, &workspace_size));
    return workspace_size;
  }
};

/* cuDNNv7 backward filter-algorithm searcher, consist of three
   algorithm search modes, namely: debug mode, heuristic mode,
   and exhaustive search mode.
   As well as one workspace size acquirsition function with
   respect to the chosen alogrithm. */
template <>
struct SearchAlgorithm<cudnnConvolutionBwdFilterAlgoPerf_t> {
  using perf_t = cudnnConvolutionBwdFilterAlgoPerf_t;
  using algo_t = cudnnConvolutionBwdFilterAlgo_t;

  template <typename T>
  static algo_t Find(const ConvArgs& args, bool exhaustive_search,
                     bool deterministic, const phi::GPUContext& ctx) {
    platform::CUDAGraphCaptureModeGuard guard;
    algo_t algo;
    int perf_count;
    auto dtype = platform::CudnnDataType<T>::type;
    size_t workspace_size_limit = FLAGS_conv_workspace_size_limit << 20;
    SetConvMathType(ctx, dtype, args.cdesc);

    // Debug search mode.
    if (deterministic) {
      algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
      VLOG(3) << "Debug bwd-filter choosen algo is : " << algo;
      return algo;
    }

    // Exhaustive search mode.
    bool open_tune = phi::autotune::AutoTuneCache::Instance().GetTuneStatus();
    if (exhaustive_search || open_tune) {
      std::string op_type("conv_filter");
      auto key = ConvArgs::GetConvKey<T>(&args);
      auto& cache =
          phi::autotune::AutoTuneCache::Instance().RegisterOrGet(op_type);

      if (cache.Find(key) == true) {
        algo = static_cast<algo_t>(cache.Get(key));
        VLOG(3) << "Cached bwd-filter choosen algo is : " << algo;
        return algo;
      }

      bool is_auto_tune =
          phi::autotune::AutoTuneStatus::Instance().UseAutoTune();
      if (exhaustive_search || is_auto_tune) {
        // Exhaustive search mode except for HALF type
        if (dtype != CUDNN_DATA_HALF) {
          std::array<perf_t, kNUM_CUDNN_BWD_FILTER_ALGS> perf_results;
          auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
            PADDLE_ENFORCE_GPU_SUCCESS(
                platform::dynload::
                    cudnnFindConvolutionBackwardFilterAlgorithmEx(
                        args.handle, args.idesc.desc(), args.x->data<T>(),
                        args.odesc.desc(), args.o->data<T>(), args.cdesc.desc(),
                        args.wdesc.desc(), const_cast<T*>(args.w->data<T>()),
                        kNUM_CUDNN_BWD_FILTER_ALGS, &perf_count,
                        perf_results.data(), cudnn_workspace_ptr,
                        workspace_size_limit));
          };
          auto handle = ctx.cudnn_workspace_handle();
          handle.RunFuncSync(cudnn_find_func, workspace_size_limit);
          VLOG(3) << "BwdFilterAlgo Perf result: (algo: stat, time, memory)";
          algo = ChooseAlgoByWorkspace<perf_t, algo_t>(perf_results.data(),
                                                       perf_count, 0, true);
          cache.Set(key, static_cast<int64_t>(algo));
          VLOG(3) << "Exhaustive choosen algo for bwd-filter for is : " << algo;
        } else {
          // Exhaustive search mode only for HALF type
          int max_algos = 0;
          int actual_algos = 0;
#if CUDNN_VERSION_MIN(7, 0, 1)
          PADDLE_ENFORCE_GPU_SUCCESS(
              platform::dynload::
                  cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
                      args.handle, &max_algos));
#endif
          std::vector<perf_t> perf_results(max_algos);
          PADDLE_ENFORCE_GPU_SUCCESS(
              platform::dynload::cudnnFindConvolutionBackwardFilterAlgorithm(
                  args.handle, args.idesc.desc(), args.odesc.desc(),
                  args.cdesc.desc(), args.wdesc.desc(), perf_results.size(),
                  &actual_algos, perf_results.data()));
          perf_results.resize(actual_algos);
          ChooseAlgo<perf_t, algo_t>(perf_results, workspace_size_limit, &algo);
          cache.Set(key, static_cast<int64_t>(algo));
          VLOG(3)
              << "Exhaustive choosen algo for HALF type bwd-filter for is : "
              << algo;
        }
        return algo;
      }
    }
// Heuristic search mode.
#if CUDNN_VERSION >= 7001
    int best_algo_idx = 0;
    size_t workspace_size = 0;
    std::unique_ptr<perf_t[]> perf_results(
        new perf_t[kNUM_CUDNN_BWD_FILTER_ALGS]);
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetConvolutionBackwardFilterAlgorithm_v7(
            args.handle, args.idesc.desc(), args.odesc.desc(),
            args.cdesc.desc(), args.wdesc.desc(), kNUM_CUDNN_BWD_FILTER_ALGS,
            &perf_count, perf_results.get()));
    algo = (perf_results.get())[best_algo_idx].algo;
    workspace_size = (perf_results.get())[best_algo_idx].memory;

    if (workspace_size > workspace_size_limit) {
      workspace_size = workspace_size_limit;
#if CUDNN_VERSION >= 8000
      // cudnnGetConvolutionBackwardFilterAlgorithm is removed in CUDNN-8
      algo = ChooseAlgoByWorkspace<perf_t, algo_t>(
          perf_results.get(), kNUM_CUDNN_BWD_FILTER_ALGS, workspace_size_limit);
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
              workspace_size_limit, &algo));
#endif
    }
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetConvolutionBackwardFilterAlgorithm(
            args.handle, args.idesc.desc(), args.odesc.desc(),
            args.cdesc.desc(), args.wdesc.desc(),
            CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
            workspace_size_limit, &algo));
#endif
    VLOG(3) << "Default(Heuristic) bwd-filter choosen algo is: " << algo;
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args, algo_t algo) {
    platform::CUDAGraphCaptureModeGuard guard;
    size_t workspace_size = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetConvolutionBackwardFilterWorkspaceSize(
            args.handle, args.idesc.desc(), args.odesc.desc(),
            args.cdesc.desc(), args.wdesc.desc(), algo, &workspace_size));
    return workspace_size;
  }
};

}  // namespace operators
}  // namespace paddle
