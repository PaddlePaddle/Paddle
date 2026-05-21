// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_XPU

#include "paddle/phi/backends/xpu/cuda_graph.h"
#include "glog/logging.h"
#include "paddle/common/flags.h"

COMMON_DECLARE_bool(use_cuda_malloc_async_allocator);
COMMON_DECLARE_bool(auto_free_cudagraph_allocations_on_launch);

namespace phi {
namespace backends {
namespace xpu {

std::unique_ptr<CUDAGraph> CUDAGraph::capturing_graph_{nullptr};
XPUStream CUDAGraph::created_stream_{nullptr};
XPUStream CUDAGraph::original_stream_{nullptr};
bool CUDAGraph::stream_created_{false};

paddle::optional<std::thread::id> CUDAGraph::capturing_thread_id_{paddle::none};
std::vector<std::function<void()>> CUDAGraph::cudagraph_pre_capture_callbacks_;

CUDAGraphID CUDAGraph::UniqueID() {
  static std::atomic<CUDAGraphID> id;
  return id.fetch_add(1);
}

int64_t CUDAGraph::UniqueMemoryPoolID() {
  static std::atomic<int64_t> id(CUDAGraph::kDefaultPoolID + 1);
  return id.fetch_add(1);
}

void CUDAGraph::Reset() {
  if (is_reset_) return;
  for (auto graph : graphs_) {
    PADDLE_ENFORCE_XPU_SUCCESS(cudaGraphDestroy(graph));
  }
  graphs_.clear();
  for (auto exec_graph : exec_graphs_) {
    PADDLE_ENFORCE_XPU_SUCCESS(cudaGraphExecDestroy(exec_graph));
  }
  exec_graphs_.clear();

  // callback should be called in reverse order because the latter added
  // callback may rely on the former added callback.
  for (auto iter = cudagraph_post_reset_callbacks_.rbegin();
       iter != cudagraph_post_reset_callbacks_.rend();
       ++iter) {
    (*iter)(*this);
  }
  cudagraph_post_reset_callbacks_.clear();
  is_reset_ = true;
}

void CUDAGraph::Replay() {
  is_replayed_ = true;
  PADDLE_ENFORCE_EQ(is_reset_,
                    false,
                    common::errors::PermissionDenied(
                        "Cannot replay the CUDA Graph after reset is called."));
  size_t n = exec_graphs_.size();

  for (size_t i = 0; i < n; ++i) {
    if (is_first_run_ == false) {
      for (auto &hook : cudagraph_pre_replay_callbacks_[i]) {
        hook(exec_graphs_[i]);
      }
    }

    cudaError_t err =
        cudaGraphLaunch(exec_graphs_[i], static_cast<cudaStream_t>(stream_));
    PADDLE_ENFORCE_XPU_SUCCESS(err);
  }
  is_first_run_ = false;
}

void CUDAGraph::BeginSegmentCapture() {
  ThrowErrorIfNotSupportCUDAGraph();
  PADDLE_ENFORCE_EQ(
      IsCapturing(),
      true,
      common::errors::PermissionDenied("BeginSegmentCapture should be called "
                                       "when CUDA Graph is capturing."));
  if (IsThreadLocalCapturing()) {
    PADDLE_ENFORCE_EQ(IsThisThreadCapturing(),
                      true,
                      common::errors::PermissionDenied(
                          "When capturing CUDA Graph in the thread local mode, "
                          "you cannot begin segmented capturing in the thread "
                          "which is not the one that starts the capturing."));
  }

  for (auto &hook : cudagraph_pre_capture_callbacks_) {
    hook();
  }

  PADDLE_ENFORCE_XPU_SUCCESS(cudaStreamBeginCapture(
      static_cast<cudaStream_t>(capturing_graph_->stream_),
      cudaStreamCaptureModeThreadLocal));
  PADDLE_ENFORCE_EQ(IsValidCapturing(),
                    true,
                    common::errors::PermissionDenied(
                        "CUDA Graph should not be invalidated."));
  VLOG(10) << "Begin to capture CUDA Graph with ID " << capturing_graph_->id_
           << ", segment id " << capturing_graph_->graphs_.size()
           << ", memory pool id " << capturing_graph_->pool_id_;
}

void CUDAGraph::BeginCapture(phi::XPUPlace place,
                             XPUStream stream,
                             xpuStreamCaptureMode mode) {
  ThrowErrorIfNotSupportCUDAGraph();
  PADDLE_ENFORCE_EQ(IsCapturing(),
                    false,
                    common::errors::PermissionDenied(
                        "CUDA Graph can only captured one by one."));
  // Create CUDAGraph instance, which will create a new stream in constructor
  // and set it as the current device stream
  capturing_graph_.reset(new CUDAGraph());

  // Get the stream from the device context after constructor has set it
  // The constructor has already created a new stream and set it as current
  // device stream
  // Create a new stream and set it as the current device stream
  int device_id = phi::backends::xpu::GetXPUCurrentDeviceId();
  phi::backends::xpu::XPUDeviceGuard guard(device_id);

  // Get current XPUContext and save original stream
  phi::XPUContext *dev_ctx = phi::get_xpu_context(device_id);
  XPUStream current_stream = dev_ctx->stream(0);

  if (current_stream == nullptr) {
    original_stream_ = current_stream;
    // Create new stream
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_stream_create(&created_stream_));
    stream_created_ = true;
    // Set the new stream as current stream
    dev_ctx->SetStream(created_stream_, 0);
  }
  XPUStream actual_stream = dev_ctx->stream(0);
  PADDLE_ENFORCE_NOT_NULL(
      actual_stream,
      common::errors::PermissionDenied(
          "CUDA Graph cannot be captured in default CUDA stream 0."));
  capturing_graph_->place_ = place;
  capturing_graph_->stream_ = actual_stream;
  capturing_graph_->capture_mode_ = mode;
  if (mode == xpuStreamCaptureModeThreadLocal) {
    capturing_thread_id_ = std::this_thread::get_id();
    VLOG(10) << "Capturing CUDA Graph in thread local mode, thread id: "
             << capturing_thread_id_;
  }
  BeginSegmentCapture();
}

inline void sync_streams(cudaStream_t to_record, cudaStream_t to_wait) {
  if (to_record == to_wait) return;
  cudaEvent_t event = nullptr;
  PADDLE_ENFORCE_XPU_SUCCESS(
      cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  PADDLE_ENFORCE_XPU_SUCCESS(cudaEventRecord(event, to_record));
  PADDLE_ENFORCE_XPU_SUCCESS(cudaStreamWaitEvent(to_wait, event));
  PADDLE_ENFORCE_XPU_SUCCESS(cudaEventDestroy(event));
}

void CUDAGraph::EndSegmentCapture() {
  ThrowErrorIfNotSupportCUDAGraph();
  PADDLE_ENFORCE_EQ(
      IsCapturing(),
      true,
      common::errors::PermissionDenied("No CUDA Graph is capturing."));
  for (const auto &stream : capturing_graph_->streams_to_join_) {
    sync_streams(static_cast<cudaStream_t>(stream),
                 static_cast<cudaStream_t>(capturing_graph_->stream_));
  }
  capturing_graph_->streams_to_join_.clear();
  cudaGraph_t graph;
  PADDLE_ENFORCE_XPU_SUCCESS(cudaStreamEndCapture(
      static_cast<cudaStream_t>(capturing_graph_->stream_), &graph));
  auto num_nodes = static_cast<size_t>(-1);
  PADDLE_ENFORCE_XPU_SUCCESS(cudaGraphGetNodes(graph, nullptr, &num_nodes));
  if (num_nodes == 0) {
    PADDLE_ENFORCE_XPU_SUCCESS(cudaGraphDestroy(graph));
    VLOG(10) << "Skip empty CUDA Graph with ID " << capturing_graph_->id_
             << ", segment id " << capturing_graph_->graphs_.size()
             << ", memory pool id " << capturing_graph_->pool_id_;
    return;
  }

  for (auto &cudagraph_post_capture_callback :
       capturing_graph_->cudagraph_post_capture_callbacks_) {
    cudagraph_post_capture_callback();
  }
  capturing_graph_->cudagraph_post_capture_callbacks_.clear();

  capturing_graph_->cudagraph_pre_replay_callbacks_.emplace_back(
      CUDAGraphNodeLauncher::Instance().GetParameterSettersForExecGraph(graph));

  cudaGraphExec_t exec_graph;
  PADDLE_ENFORCE_XPU_SUCCESS(
      cudaGraphInstantiate(&exec_graph, graph, nullptr, nullptr, 0));
  capturing_graph_->graphs_.emplace_back(graph);
  capturing_graph_->exec_graphs_.emplace_back(exec_graph);
}

std::unique_ptr<CUDAGraph> CUDAGraph::EndCapture() {
  EndSegmentCapture();
  // Destroy the created stream before reset
  if (stream_created_ && created_stream_ != nullptr) {
    int device_id = phi::backends::xpu::GetXPUCurrentDeviceId();
    phi::backends::xpu::XPUDeviceGuard guard(device_id);

    phi::XPUContext *dev_ctx = phi::get_xpu_context(device_id);
    dev_ctx->SetStream(original_stream_, 0);

    // Destroy the created stream
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_stream_destroy(created_stream_));
    created_stream_ = nullptr;
    stream_created_ = false;
    capturing_graph_->stream_ = original_stream_;
  }
  capturing_thread_id_ = paddle::none;
  return std::move(capturing_graph_);
}

bool CUDAGraph::IsValidCapturing() {
  if (!IsCapturing()) return false;
  cudaStreamCaptureStatus status;
  CUDAGraphID id;
  PADDLE_ENFORCE_XPU_SUCCESS(cudaStreamGetCaptureInfo(
      static_cast<cudaStream_t>(capturing_graph_->stream_), &status, &id));
  return status == cudaStreamCaptureStatusActive;
}

static std::string ConcatPath(const std::string &dirname,
                              const std::string &filename) {
#ifdef _WIN32
  const std::array<char, 3> kFileSep = {"\\"};
#else
  const std::array<char, 2> kFileSep = {"/"};
#endif
  if (!dirname.empty() && dirname.back() == kFileSep[0]) {
    return dirname + filename;
  } else {
    return dirname + kFileSep.data() + filename;
  }
}

void CUDAGraph::PrintToDotFiles(const std::string &dirname,
                                unsigned int flags) {
  ThrowErrorIfNotSupportCUDAGraph();
  for (size_t i = 0; i < graphs_.size(); ++i) {
    auto filename =
        ConcatPath(dirname, "segment_" + std::to_string(i) + ".dot");
    VLOG(10) << "Save the " << i << "-th segment of graph " << id_ << " to "
             << filename;
    PADDLE_ENFORCE_XPU_SUCCESS(
        cudaGraphDebugDotPrint(graphs_[i], filename.c_str(), flags));
  }
}

void CUDAGraphNodeLauncher::KernelNodeLaunch(
    parameterSetter_t parameterSetter, gpuKernelCallback_t xpuKernelCallback) {
  if (UNLIKELY(phi::backends::xpu::CUDAGraph::IsThisThreadCapturing())) {
    unsigned int id = GenerateIdentifier();
    auto cudaFunc = xpuKernelCallback(id);

    parameterSetters[cudaFunc][id] = parameterSetter;
    VLOG(10) << "Launch kernel with cudaFunc = " << cudaFunc << " id = " << id;
  } else {
    xpuKernelCallback(0);
  }
}

std::vector<CUDAGraphExecuterSetter_t>
CUDAGraphNodeLauncher::GetParameterSettersForExecGraph(cudaGraph_t graph) {
  size_t num_nodes;
  PADDLE_ENFORCE_XPU_SUCCESS(cudaGraphGetNodes(graph, nullptr, &num_nodes));
  std::vector<cudaGraphNode_t> nodes(num_nodes);
  PADDLE_ENFORCE_XPU_SUCCESS(
      cudaGraphGetNodes(graph, nodes.data(), &num_nodes));

  std::vector<std::function<void(cudaGraphExec_t)>> hooks;
  for (auto node : nodes) {
    cudaGraphNode_t cuNode = node;
    cudaGraphNodeType pType;
    PADDLE_ENFORCE_XPU_SUCCESS(cudaGraphNodeGetType(cuNode, &pType));
    if (pType == CU_GRAPH_NODE_TYPE_KERNEL) {
      cudaKernelNodeParams cuParams;
      PADDLE_ENFORCE_XPU_SUCCESS(
          cudaGraphKernelNodeGetParams(cuNode, &cuParams));
      gpuKernelParams kernel_params(cuParams.kernelParams);
      auto kernel =
          parameterSetters.find(static_cast<cudaFunction_t>(cuParams.func));
      VLOG(10) << "[GetParameterSettersForExecGraph] cuParams.func = "
               << cuParams.func;
      // There exists a parameter setter
      if (kernel != parameterSetters.end()) {
        auto launchSequence = kernel->second;
        unsigned int id = kernel_params.As<int>(0);

        VLOG(10) << "[GetParameterSettersForExecGraph] Find launch kernel id = "
                 << id;
        auto parameterSetter = launchSequence.find(id);
        if (parameterSetter != launchSequence.end()) {
          auto setter = parameterSetter->second;
          hooks.emplace_back(
              [setter, cuNode, cuParams](cudaGraphExec_t exec_graph) {
                gpuKernelParams kernel_params(cuParams.kernelParams);
                setter(kernel_params);
                PADDLE_ENFORCE_XPU_SUCCESS(cudaGraphExecKernelNodeSetParams(
                    static_cast<CUgraphExec>(exec_graph), cuNode, &cuParams));
              });
        } else {
          PADDLE_THROW(common::errors::InvalidArgument(
              "Error: does not find launch id"));
        }
      }
    }
  }

  return hooks;
}

}  // namespace xpu
}  // namespace backends
}  // namespace phi
#endif
