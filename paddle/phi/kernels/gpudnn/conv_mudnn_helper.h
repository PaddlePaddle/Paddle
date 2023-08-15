#include "paddle/phi/kernels/conv_kernel.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/gpudnn/conv_gpudnn_base.h"
#include <mudnn.h>

namespace phi {

using ConvArgs = ConvArgsBase<dynload::Handle*, dynload::Tensor::Type>;

template <typename PerfT>
struct SearchAlgorithm {};

template <>
struct SearchAlgorithm<dynload::Convolution::Algorithm> {
  using algo_t = dynload::Convolution::Algorithm;

  static algo_t Find(const ConvArgs& args,
                     bool exhaustive_search,
                     bool deterministic,
                     size_t workspace_size,
                     const phi::GPUContext& ctx) {
    algo_t algo;

    auto workspace_handle = ctx.cudnn_workspace_handle();

    auto mudnn_find_func = [&](void* mudnn_workspace_ptr) {
          args.cdesc.desc()->GetRecommendForwardAlgorithm(
              *args.handle, 
              algo, 
              *args.odesc.desc(), 
              *args.idesc.desc(), 
              *args.wdesc.desc());
    };

    workspace_handle.RunFuncSync(mudnn_find_func, workspace_size);
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args) {
    return 0;
  }
};


void InternalMemFree(void* ptr) {
  if (!ptr) {
    return;
  }
  musaFree(ptr);
}

dynload::MemoryHandler InternalMemAlloc(size_t s) {
  void* data = nullptr;
  if (s) {
    musaMalloc(&data, s);
  }
  return dynload::MemoryHandler(data, InternalMemFree);
}


#if 0
template <>
struct SearchAlgorithm<miopenConvBwdDataAlgorithm_t> {
  using perf_t = miopenConvAlgoPerf_t;
  using algo_t = miopenConvBwdDataAlgorithm_t;

  template <typename T>
  static algo_t Find(const ConvArgs& args,
                     bool exhaustive_search,
                     bool deterministic,
                     size_t workspace_size,
                     const phi::GPUContext& ctx) {
    algo_t algo;

    auto workspace_handle = ctx.cudnn_workspace_handle();

    int find_count;
    miopenConvAlgoPerf_t find_result;
    auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::miopenFindConvolutionBackwardDataAlgorithm(
              args.handle,
              args.odesc.desc(),
              args.o->data<T>(),
              args.wdesc.desc(),
              args.w->data<T>(),
              args.cdesc.desc(),
              args.idesc.desc(),
              const_cast<T*>(args.x->data<T>()),
              kNUM_CUDNN_BWD_DATA_ALGS,
              &find_count,
              &find_result,
              cudnn_workspace_ptr,
              workspace_size,
              false));
    };

    workspace_handle.RunFuncSync(cudnn_find_func, workspace_size);
    algo = find_result.bwd_data_algo;
    VLOG(3) << "choose algo " << algo;
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args) {
    size_t workspace_size = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenConvolutionBackwardDataGetWorkSpaceSize(
            args.handle,
            args.odesc.desc(),
            args.wdesc.desc(),
            args.cdesc.desc(),
            args.idesc.desc(),
            &workspace_size));
    return workspace_size;
  }
};

template <>
struct SearchAlgorithm<miopenConvBwdWeightsAlgorithm_t> {
  using perf_t = miopenConvAlgoPerf_t;
  using algo_t = miopenConvBwdWeightsAlgorithm_t;

  template <typename T>
  static algo_t Find(const ConvArgs& args,
                     bool exhaustive_search,
                     bool deterministic,
                     size_t workspace_size,
                     const phi::GPUContext& ctx) {
    algo_t algo;

    auto workspace_handle = ctx.cudnn_workspace_handle();

    int find_count;
    miopenConvAlgoPerf_t find_result;
    auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::miopenFindConvolutionBackwardWeightsAlgorithm(
              args.handle,
              args.odesc.desc(),
              args.o->data<T>(),
              args.idesc.desc(),
              args.x->data<T>(),
              args.cdesc.desc(),
              args.wdesc.desc(),
              const_cast<T*>(args.w->data<T>()),
              kNUM_CUDNN_BWD_FILTER_ALGS,
              &find_count,
              &find_result,
              cudnn_workspace_ptr,
              workspace_size,
              false));
    };

    workspace_handle.RunFuncSync(cudnn_find_func, workspace_size);
    algo = find_result.bwd_weights_algo;
    VLOG(3) << "choose algo " << algo;
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args) {
    size_t workspace_size = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenConvolutionBackwardWeightsGetWorkSpaceSize(
            args.handle,
            args.odesc.desc(),
            args.idesc.desc(),
            args.cdesc.desc(),
            args.wdesc.desc(),
            &workspace_size));
    return workspace_size;
  }
};

#endif

} // namespace phi
