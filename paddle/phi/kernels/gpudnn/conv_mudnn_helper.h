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
    /* VLOG(3) << "choose algo " << algo; */
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args) {
    return 0;
  }
};

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

using muTensor = ::musa::dnn::Tensor;
using muHandle = ::musa::dnn::Handle;
// using BINARY_MODE = ::musa::dnn::Binary::Mode;
muTensor CreateMUTensor(const DenseTensor& tensor) {
  muTensor mu_tensor;
  mu_tensor.SetNdInfo(tensor.dims().size(), tensor.dims().Get());
  switch (tensor.dtype()) {
    case DataType::FLOAT32:
      mu_tensor.SetType(muTensor::Type::FLOAT);
      break;
    case DataType::INT32:
      mu_tensor.SetType(muTensor::Type::INT32);
      break;
    case DataType::INT64:
      mu_tensor.SetType(muTensor::Type::INT64);
      break;
    default:
      std::cerr << "=========mismatch dtype in kernel=====\n";
      throw;
  }
  mu_tensor.SetAddr(tensor.data());
  return mu_tensor;
}

void ConfigConv(
    ::musa::dnn::Convolution& c,
    const std::vector<int>& str,
    const std::vector<int>& pad ,
    const std::vector<int>& dil,
    int64_t groups) {
  auto sz = str.size();
  if (sz == 2) {
    // CHECK_MUDNN_STATUS(
        c.SetNdInfo(
            {static_cast<int>(pad[0]), static_cast<int>(pad[1])},
            {static_cast<int>(str[0]), static_cast<int>(str[1])},
            {static_cast<int>(dil[0]), static_cast<int>(dil[1])});
        // "SetNdInfo");
  } else {
    // conv3d
    // PADDLE_ENFORCE_GPU_SUCCESS(
        c.SetNdInfo(
            {static_cast<int>(pad[0]),
             static_cast<int>(pad[1]),
             static_cast<int>(pad[2])},
            {static_cast<int>(str[0]),
             static_cast<int>(str[1]),
             static_cast<int>(str[2])},
            {static_cast<int>(dil[0]),
             static_cast<int>(dil[1]),
             static_cast<int>(dil[2])});
        // "SetNdInfo");
  }
  // CHECK_MUDNN_STATUS(
      c.SetComputeMode(::musa::dnn::Convolution::ComputeMode::ALL);
      // "SetComputeMode");
  c.SetGroups(groups);
}

void InternalMemFree(void* ptr) {
  if (!ptr) {
    return;
  }
  musaFree(ptr);
}

::musa::dnn::MemoryHandler InternalMemAlloc(size_t s) {
  void* data = nullptr;
  if (s) {
    musaMalloc(&data, s);
  }
  return ::musa::dnn::MemoryHandler(data, InternalMemFree);
}


void ConfigFormat(muTensor& mt, phi::backends::gpu::DataLayout layout) {
  switch (layout) {
    case phi::backends::gpu::DataLayout::kNHWC:
      mt.SetFormat(muTensor::Format::NHWC);
      break;
    case phi::backends::gpu::DataLayout::kNCHW:
      mt.SetFormat(muTensor::Format::NCHW);
      break;
    default:
      std::cerr << "=========mismatch layout in kernel===== " << __FILE__ << std::endl;
      throw;
  }
}

} // namespace phi
