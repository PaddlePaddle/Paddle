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

#pragma once

#include <array>
#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "glog/logging.h"

#include "paddle/common/errors.h"
#include "paddle/common/macros.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/device_code.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/utils/optional.h"

#if CUDA_VERSION < 11000
// For CUDA versions less than 11.0, use a dummy type for cudaFunction_t.
using cudaFunction_t = void *;
cudaError_t cudaGetFuncBySymbol(cudaFunction_t *functionPtr,
                                const void *symbolPtr);
#endif

namespace phi {
namespace backends {
namespace gpu {

class CUDAGraphContextManager {
 public:
  using DeviceContextMap =
      std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>;

  static CUDAGraphContextManager &Instance() {
    static CUDAGraphContextManager *cuda_graph_ctx_manager =
        new CUDAGraphContextManager;
    return *cuda_graph_ctx_manager;
  }

  DeviceContext *Get(int64_t pool_id, const Place &place, int stream_priority) {
    std::lock_guard<std::mutex> lk(ctx_mtx_);
    VLOG(6) << "Get cuda graph device context for " << place;

    DeviceContextMap &ctxs = cuda_graph_ctx_pool_[pool_id];
    if (ctxs.find(place) == ctxs.end()) {
      phi::memory_utils::EmplaceDeviceContexts(
          &ctxs,
          {place},
          /*disable_setting_default_stream_for_allocator=*/true,
          stream_priority);
    }
    return ctxs[place].get().get();
  }

  void RecordCapturingDeviceContext(DeviceContext *dev_ctx) {
    capturing_ctxs_.insert(dev_ctx);
  }

  std::set<DeviceContext *> GetAllCapturingDeviceContexts() const {
    return capturing_ctxs_;
  }

  void ClearDeviceContextsRecords() { capturing_ctxs_.clear(); }

 private:
  CUDAGraphContextManager() {}
  DISABLE_COPY_AND_ASSIGN(CUDAGraphContextManager);

  std::mutex ctx_mtx_;
  std::unordered_map<int64_t, DeviceContextMap> cuda_graph_ctx_pool_;
  std::set<DeviceContext *> capturing_ctxs_;
};

class gpuKernelParams {
 public:
  explicit gpuKernelParams(void **params) : kernelParams(params) {}

  template <typename T>
  T &As(size_t idx) const {
    return *reinterpret_cast<T *>(kernelParams[idx]);
  }

  void **getParams() const { return kernelParams; }

 private:
  void **kernelParams;
};

using cudaGraphExecuterSetter_t = std::function<void(cudaGraphExec_t)>;

//  ** class CUDAGraphNodeLauncher
//
//  This class offers a interface for launching CUDA kernels in CUDA Graph, we
//  utilize the `cudaGraphExecKernelNodeSetParams` function for parameter setup.
//  Launching kernels via this class ensures proper management.
//
//  NOTE: It's essential that the first parameter for any kernel launched
//  through this class is an `unsigned int` identifier. This identifier plays a
//  crucial role in linking the CUDA kernel to its corresponding CUDA graph
//  node. We tag each kernel launch with a unique identifier to maintain
//  structured linkage with its CUDA graph node.
//
//  NOTE: This class use a singleton design pattern ensures there's only a
//  single global instance accessible via the `Instance()` method.
class CUDAGraphNodeLauncher {
 public:
  //  [Parameter Setter Callback]
  //  Sets the kernel's parameters BEFORE activating the CUDA graph. It enables
  //  dynamic determination and setup of kernel arguments.
  //
  //  parameterSetter_t parameterSetter = [saved_state](gpuKernelParams
  //  &param){
  //      // Code to compute and the parameter values from the saved_state
  //      // ...
  //      param.As<type>(idx) = calculated_value;
  //  };
  using parameterSetter_t = std::function<void(gpuKernelParams &)>;

  //  [CUDA Kernel Callback]
  //  Acts as the launcher for the kernel. It accepts an `unsigned int`
  //  identifier and uses it for the kernel launch.
  //  The `cudaGetFuncBySymbol` method can be used to fetch the `cudaFunction_t`
  //  reference of the kernel from the kernel pointer.
  //  gpuKernelCallback_t cudaKernelCallback = [=](unsigned int id) {
  //      // cudaFunction_t is REQUIRED to get here
  //      cudaFunction_t cudaFunc;
  //      PADDLE_ENFORCE_GPU_SUCCESS(cudaGetFuncBySymbol(&cudaFunc, &kernel));
  //
  //      kernel<<<>>>(id, ...);  // Launching the kernel with id
  //      return cudaFunc;
  //  };
  using gpuKernelCallback_t = std::function<cudaFunction_t(unsigned int)>;

  //  [Kernel Launch]
  //  With the callbacks defined and the CUDA function obtained, the kernel can
  //  be launched using the `KernelNodeLaunch` method.
  void KernelNodeLaunch(parameterSetter_t parameterSetter,
                        gpuKernelCallback_t cudakernelCallback);

  std::vector<cudaGraphExecuterSetter_t> GetParameterSettersForExecGraph(
      cudaGraph_t graph);

  parameterSetter_t GetParameterSetter(const gpuKernelParams &params);

  static CUDAGraphNodeLauncher &Instance() {
    static CUDAGraphNodeLauncher *launcher = new CUDAGraphNodeLauncher;
    return *launcher;
  }

 private:
  CUDAGraphNodeLauncher() : id(0) {}
  DISABLE_COPY_AND_ASSIGN(CUDAGraphNodeLauncher);

  unsigned int GenerateIdentifier() { return id++; }

  unsigned int id;
  std::unordered_map<cudaFunction_t, std::map<unsigned int, parameterSetter_t>>
      parameterSetters;
};

#if CUDA_VERSION >= 10010
static void ThrowErrorIfNotSupportCUDAGraph() {}
#else
enum gpuStreamCaptureMode {
  cudaStreamCaptureModeGlobal = 0,
  cudaStreamCaptureModeThreadLocal = 1,
  cudaStreamCaptureModeRelaxed = 2
};
static void ThrowErrorIfNotSupportCUDAGraph() {
  PADDLE_THROW(phi::errors::Unimplemented(
      "CUDA Graph is only supported when CUDA version >= 10.1"));
}
#endif

using CUDAGraphID = unsigned long long;  // NOLINT

// NOTE: Currently, we do not support to capture CUDA graph in parallel
// NOTE: Do not use this class directly because it should be used with
//       the memory pool.
class CUDAGraph {
  DISABLE_COPY_AND_ASSIGN(CUDAGraph);

  // Since the constructor would throw error is CUDA_VERSION < 10010.
  // The non-static method of CUDAGraph need not check CUDA_VERSION
  // again.
  CUDAGraph() {
    ThrowErrorIfNotSupportCUDAGraph();
    id_ = UniqueID();
  }

 public:
  static constexpr int64_t kDefaultPoolID = 0;
  static constexpr int64_t kInvalidPoolID = -1;

  ~CUDAGraph() { Reset(); }

  CUDAGraphID ID() const { return id_; }

  static int64_t SetMemoryPoolID(int64_t pool_id) {
    auto &pool_id_ = capturing_graph_->pool_id_;
    PADDLE_ENFORCE_EQ(
        pool_id_,
        kInvalidPoolID,
        phi::errors::InvalidArgument("Cannot reset memory pool id twice, the "
                                     "former memory pool id is %d.",
                                     pool_id_));
    if (pool_id <= kInvalidPoolID) {
      pool_id_ = UniqueMemoryPoolID();
    } else {
      PADDLE_ENFORCE_GE(
          pool_id,
          kDefaultPoolID,
          phi::errors::InvalidArgument("Invalid memory pool id %d.", pool_id));
      pool_id_ = pool_id;
    }
    return pool_id_;
  }

  int64_t PoolID() const { return pool_id_; }

  static int64_t CapturingPoolID() { return capturing_graph_->pool_id_; }

  void Replay();

  void Reset();

  void AddPostResetCallback(std::function<void()> callback) {
    std::lock_guard<std::mutex> guard(mtx_);
    cudagraph_post_reset_callbacks_.push_back(std::move(callback));
  }

  void AddPostCaptureCallback(std::function<void()> callback) {
    std::lock_guard<std::mutex> guard(mtx_);
    cudagraph_post_capture_callbacks_.push_back(std::move(callback));
  }

  void PrintToDotFiles(const std::string &dirname, unsigned int flags);

  static void BeginCapture(phi::GPUPlace place,
                           cudaStream_t stream,
                           gpuStreamCaptureMode mode);
  static std::unique_ptr<CUDAGraph> EndCapture();

  static void BeginSegmentCapture();
  static void EndSegmentCapture();

  static void AddPostResetCallbackDuringCapturing(
      std::function<void()> callback) {
    capturing_graph_->AddPostResetCallback(std::move(callback));
  }

  static void AddPostCaptureCallbackDuringCapturing(
      std::function<void()> callback) {
    capturing_graph_->AddPostCaptureCallback(std::move(callback));
  }

  // No need to add CUDA_VERSION macro because capturing_graph_ would
  // always be nullptr (constructor throws error)
  static bool IsCapturing() { return capturing_graph_ != nullptr; }

  static CUDAGraphID CapturingID() { return capturing_graph_->id_; }

  static phi::GPUPlace CapturingPlace() { return capturing_graph_->place_; }

  // This API can be used to debug which GPU operation is not
  // supported during capturing CUDA Graph.
  static bool IsValidCapturing();

  static bool IsThreadLocalCapturing() {
#if CUDA_VERSION >= 10010
    return IsCapturing() &&
           capturing_graph_->capture_mode_ == cudaStreamCaptureModeThreadLocal;
#else
    return false;
#endif
  }

  static bool IsThisThreadCapturing() {
    if (UNLIKELY(IsCapturing())) {
      return IsThreadLocalCapturing()
                 ? capturing_thread_id_.get() == std::this_thread::get_id()
                 : true;
    } else {
      return false;
    }
  }

  using SetSeedFunc = std::function<bool(gpuKernelParams *, bool)>;
  static void RecordRandomKernelInfo(SetSeedFunc set_seed_func) {
    std::lock_guard<std::mutex> guard(capturing_graph_->func_mtx_);
    capturing_graph_->set_seed_funcs_.emplace_back(std::move(set_seed_func));
  }

  static int64_t UniqueMemoryPoolID();

 private:
  static CUDAGraphID UniqueID();

 private:
#if CUDA_VERSION >= 10010
  std::vector<cudaGraph_t> graphs_;
  std::vector<cudaGraphExec_t> exec_graphs_;
  gpuStreamCaptureMode capture_mode_;
#endif
  cudaStream_t stream_{nullptr};
  phi::GPUPlace place_;
  CUDAGraphID id_;
  int64_t pool_id_{kInvalidPoolID};
  bool is_reset_{false};
  std::mutex mtx_;

  std::vector<SetSeedFunc> set_seed_funcs_;

  // Holds callbacks that are triggered after the CUDA graph is reset. These
  // callbacks are used for operations that need to be performed following the
  // reset of a CUDA graph.
  std::vector<std::function<void()>> cudagraph_post_reset_callbacks_;

  // Contains callbacks that are invoked after the CUDA graph has been captured.
  // These callbacks are crucial for managing memory allocations related to the
  // CUDA graph. They ensure that memory blocks not associated with a graph (as
  // detailed in cuda_malloc_async_allocator) are not erroneously released
  // during the graph's lifecycle.
  std::vector<std::function<void()>> cudagraph_post_capture_callbacks_;

  // Maintains a collection of 'pre-hooks' - functions that are executed before
  // the CUDA graph is replayed. These pre-hooks are essential for setting up
  // the necessary conditions or states required for the correct execution of
  // the CUDA graph.
  std::vector<std::vector<cudaGraphExecuterSetter_t>>
      cudagraph_pre_replay_callbacks_;

  std::mutex func_mtx_;

  bool is_first_run_{true};

  static paddle::optional<std::thread::id> capturing_thread_id_;
  static std::unique_ptr<CUDAGraph> capturing_graph_;
};

#if CUDA_VERSION >= 10010
class CUDAGraphCaptureModeGuard {
  DISABLE_COPY_AND_ASSIGN(CUDAGraphCaptureModeGuard);

 public:
  explicit CUDAGraphCaptureModeGuard(
      gpuStreamCaptureMode mode = cudaStreamCaptureModeRelaxed) {
    if (UNLIKELY(CUDAGraph::IsCapturing())) {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaThreadExchangeStreamCaptureMode(&mode));
      // After cudaThreadExchangeStreamCaptureMode is called,
      // the variable "mode" would be set to the old capturing mode.
      old_mode_ = mode;
    }
  }

  ~CUDAGraphCaptureModeGuard() PADDLE_MAY_THROW {
    if (UNLIKELY(CUDAGraph::IsCapturing())) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaThreadExchangeStreamCaptureMode(&old_mode_));
    }
  }

 private:
  gpuStreamCaptureMode old_mode_;
};
#else
class CUDAGraphCaptureModeGuard {
  DISABLE_COPY_AND_ASSIGN(CUDAGraphCaptureModeGuard);

 public:
  explicit CUDAGraphCaptureModeGuard(
      gpuStreamCaptureMode mode = cudaStreamCaptureModeRelaxed) {}
};
#endif

}  // namespace gpu
}  // namespace backends
}  // namespace phi
