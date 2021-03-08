/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/platform/miopen_desc.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
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

template <typename DeviceContext, typename T, size_t D>
static void RemovePaddingSlice(const framework::ExecutionContext& context,
                               const Tensor* input, Tensor* out,
                               const std::vector<int>& starts,
                               const std::vector<int>& axes) {
  auto& place =
      *context.template device_context<DeviceContext>().eigen_device();
  auto in_dims = input->dims();
  auto new_out_dims = out->dims();
  auto offsets = Eigen::array<int, D>();
  auto extents = Eigen::array<int, D>();
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
  out_t.device(place) = in_t.slice(offsets, extents);
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
  out << "[";
  for (auto const& tmp : v) out << tmp << ",";
  out << "]";
  return out;
}

using framework::ConvSearchCache;

struct ConvArgs {
  miopenHandle_t handle;
  platform::TensorDescriptor idesc, odesc;
  platform::FilterDescriptor wdesc;
  platform::ConvolutionDescriptor cdesc;
  const framework::Tensor *x, *w, *o;
  miopenDataType_t cudnn_dtype;

  // strides
  std::vector<int> s;
  // paddings
  std::vector<int> p;
  // dilations
  std::vector<int> d;

  ConvArgs(const framework::Tensor* x, const framework::Tensor* w,
           const framework::Tensor* o, const std::vector<int> s,
           const std::vector<int> p, const std::vector<int> d,
           miopenDataType_t dtype)
      : x(x), w(w), o(o), s(s), p(p), d(d), cudnn_dtype(dtype) {}
};

template <typename algo_t>
struct SearchAlgorithm {};

template <>
struct SearchAlgorithm<miopenConvFwdAlgorithm_t> {
  using perf_t = miopenConvAlgoPerf_t;
  using algo_t = miopenConvFwdAlgorithm_t;

  template <typename T>
  static algo_t Find(const ConvArgs& args, bool exhaustive_search,
                     bool deterministic,
                     const framework::ExecutionContext& ctx) {
    auto dtype = platform::CudnnDataType<T>::type;
    bool has_got_workspace_size = true;
    size_t workspace_size_limit = FLAGS_conv_workspace_size_limit * 1024 * 1024;
    size_t workspace_size = 0;
    algo_t algo;

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();

    auto& temp = ctx.cuda_device_context();
    AlgorithmsCache<algo_t>& algo_cache =
        *(framework::ConvSearchCache::Instance().GetForward());

    auto x_dims = framework::vectorize(args.x->dims());
    auto w_dims = framework::vectorize(args.w->dims());

    VLOG(10) << "miopenConvolutionFwdAlgoPerf_t:"
             << ", x_dims:" << x_dims << ", w_dims:" << w_dims << ", args.s"
             << args.s << ", args.p" << args.p << ", args.d" << args.d;

    algo = algo_cache.GetAlgorithm(
        x_dims, w_dims, args.s, args.p, args.d, 0,
        static_cast<int64_t>(args.cudnn_dtype), [&]() {
          int returned_algo_count;
          std::array<perf_t, kNUM_CUDNN_FWD_ALGS> perf_stat;

          auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
            PADDLE_ENFORCE_CUDA_SUCCESS(
                platform::dynload::miopenFindConvolutionForwardAlgorithm(
                    args.handle, args.idesc.desc(), args.x->data<T>(),
                    args.wdesc.desc(), args.w->data<T>(), args.cdesc.desc(),
                    args.odesc.desc(), const_cast<T*>(args.o->data<T>()),
                    kNUM_CUDNN_FWD_ALGS, &returned_algo_count, perf_stat.data(),
                    cudnn_workspace_ptr, workspace_size_limit, false));
          };
          workspace_handle.RunFuncSync(cudnn_find_func, workspace_size_limit);

          VLOG(3) << "FwdAlgo Perf result: (algo: stat, time, memory)";
          for (int i = 0; i < returned_algo_count; ++i) {
            const auto& stat = perf_stat[i];
            VLOG(3) << stat.fwd_algo;
          }
          return perf_stat[0].fwd_algo;
        });
    VLOG(3) << "choose algo " << algo;
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args, algo_t algo) {
    size_t workspace_size = 0;
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::miopenConvolutionForwardGetWorkSpaceSize(
            args.handle, args.wdesc.desc(), args.idesc.desc(),
            args.cdesc.desc(), args.odesc.desc(), &workspace_size));
    return workspace_size;
  }
};

template <>
struct SearchAlgorithm<miopenConvBwdDataAlgorithm_t> {
  using perf_t = miopenConvAlgoPerf_t;
  using algo_t = miopenConvBwdDataAlgorithm_t;

  template <typename T>
  static algo_t Find(const ConvArgs& args, bool exhaustive_search,
                     bool deterministic,
                     const framework::ExecutionContext& ctx) {
    auto dtype = platform::CudnnDataType<T>::type;
    size_t workspace_size_limit = FLAGS_conv_workspace_size_limit * 1024 * 1024;
    size_t workspace_size = 0;
    bool has_got_workspace_size = true;
    algo_t algo;

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();

    AlgorithmsCache<algo_t>& algo_cache =
        *(framework::ConvSearchCache::Instance().GetBackwardData());

    auto x_dims = framework::vectorize(args.x->dims());
    auto w_dims = framework::vectorize(args.w->dims());

    VLOG(10) << "miopenConvolutionFwdAlgoPerf_t"
             << ", x_dims:" << x_dims << ", w_dims:" << w_dims << ", args.s"
             << args.s << ", args.p" << args.p << ", args.d" << args.d;

    algo = algo_cache.GetAlgorithm(
        x_dims, w_dims, args.s, args.p, args.d, 0,
        static_cast<int64_t>(args.cudnn_dtype), [&]() {
          int returned_algo_count;
          std::array<perf_t, kNUM_CUDNN_FWD_ALGS> perf_stat;

          auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
            PADDLE_ENFORCE_CUDA_SUCCESS(
                platform::dynload::miopenFindConvolutionBackwardDataAlgorithm(
                    args.handle, args.odesc.desc(), args.o->data<T>(),
                    args.wdesc.desc(), args.w->data<T>(), args.cdesc.desc(),
                    args.idesc.desc(), const_cast<T*>(args.x->data<T>()),
                    kNUM_CUDNN_BWD_DATA_ALGS, &returned_algo_count,
                    perf_stat.data(), cudnn_workspace_ptr, workspace_size_limit,
                    false));
          };
          workspace_handle.RunFuncSync(cudnn_find_func, workspace_size_limit);

          VLOG(3) << "BwdDataAlgo Perf result: (algo: stat, time, memory)";
          for (int i = 0; i < returned_algo_count; ++i) {
            const auto& stat = perf_stat[i];
            VLOG(3) << stat.bwd_data_algo;
          }

          return perf_stat[0].bwd_data_algo;
        });
    VLOG(3) << "choose algo " << algo;
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args, algo_t algo) {
    size_t workspace_size = 0;
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::miopenConvolutionBackwardDataGetWorkSpaceSize(
            args.handle, args.odesc.desc(), args.wdesc.desc(),
            args.cdesc.desc(), args.idesc.desc(), &workspace_size));
    return workspace_size;
  }
};

template <>
struct SearchAlgorithm<miopenConvBwdWeightsAlgorithm_t> {
  using perf_t = miopenConvAlgoPerf_t;
  using algo_t = miopenConvBwdWeightsAlgorithm_t;

  template <typename T>
  static algo_t Find(const ConvArgs& args, bool exhaustive_search,
                     bool deterministic,
                     const framework::ExecutionContext& ctx) {
    auto dtype = platform::CudnnDataType<T>::type;
    size_t workspace_size_limit = FLAGS_conv_workspace_size_limit * 1024 * 1024;
    size_t workspace_size = 0;
    bool has_got_workspace_size = true;
    algo_t algo;

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();
    AlgorithmsCache<algo_t>& algo_cache =
        *(framework::ConvSearchCache::Instance().GetBackwardFilter());

    auto x_dims = framework::vectorize(args.x->dims());
    auto w_dims = framework::vectorize(args.w->dims());

    VLOG(10) << "miopenConvolutionFwdAlgoPerf_t:"
             << ", x_dims:" << x_dims << ", w_dims:" << w_dims << ", args.s"
             << args.s << ", args.p" << args.p << ", args.d" << args.d;

    algo = algo_cache.GetAlgorithm(
        x_dims, w_dims, args.s, args.p, args.d, 0,
        static_cast<int64_t>(args.cudnn_dtype), [&]() {
          int returned_algo_count;
          std::array<perf_t, kNUM_CUDNN_FWD_ALGS> perf_stat;
          auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
            PADDLE_ENFORCE_CUDA_SUCCESS(
                platform::dynload::
                    miopenFindConvolutionBackwardWeightsAlgorithm(
                        args.handle, args.odesc.desc(), args.o->data<T>(),
                        args.idesc.desc(), args.x->data<T>(), args.cdesc.desc(),
                        args.wdesc.desc(), const_cast<T*>(args.w->data<T>()),
                        kNUM_CUDNN_BWD_FILTER_ALGS, &returned_algo_count,
                        perf_stat.data(), cudnn_workspace_ptr,
                        workspace_size_limit, false));
          };
          workspace_handle.RunFuncSync(cudnn_find_func, workspace_size_limit);

          VLOG(3) << "BwdFilterAlgo Perf result: (algo: stat, time, memory)";
          for (int i = 0; i < returned_algo_count; ++i) {
            const auto& stat = perf_stat[i];
            VLOG(3) << stat.bwd_weights_algo;
          }
          return perf_stat[0].bwd_weights_algo;
        });
    VLOG(3) << "choose algo " << algo;
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args, algo_t algo) {
    size_t workspace_size = 0;
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::miopenConvolutionBackwardWeightsGetWorkSpaceSize(
            args.handle, args.odesc.desc(), args.idesc.desc(),
            args.cdesc.desc(), args.wdesc.desc(), &workspace_size));
    return workspace_size;
  }
};

}  // namespace operators
}  // namespace paddle
