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

#ifdef PADDLE_WITH_XPU

#include "paddle/phi/backends/xpu/xpu_graph.h"
#include "paddle/common/flags.h"
#include "glog/logging.h"

COMMON_DECLARE_bool(use_cuda_malloc_async_allocator);
COMMON_DECLARE_bool(auto_free_cudagraph_allocations_on_launch);

namespace phi {
namespace backends {
namespace xpu {

std::unique_ptr<XPUGraph> XPUGraph::capturing_graph_{nullptr};
paddle::optional<std::thread::id> XPUGraph::capturing_thread_id_{paddle::none};
std::vector<std::function<void()>> XPUGraph::xpu_graph_pre_capture_callbacks_;

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

XPUGraph::XPUGraph() {
  ThrowErrorIfNotSupportXPUGraph();
  id_ = UniqueID();
}

XPUGraph::~XPUGraph() {
  Reset();
}

XPUGraphID XPUGraph::ID() const {
  return id_;
}

int64_t XPUGraph::PoolID() const {
  return pool_id_;
}

void XPUGraph::Replay() {
  is_replayed_ = true;
#if CUDA_VERSION >= 10010
  PADDLE_ENFORCE_EQ(is_reset_,
                    false,
                    common::errors::PermissionDenied(
                        "Cannot replay the CUDA Graph after reset is called."));
  size_t n = exec_graphs_.size();
  for (size_t i = 0; i < n; ++i) {
    if (!is_first_run_) {
      for (auto &hook : xpu_graph_pre_replay_callbacks_[i]) {
        hook(exec_graphs_[i]);
      }
    }
    PADDLE_ENFORCE_XPU_SUCCESS(cudaGraphLaunch(exec_graphs_[i], static_cast<cudaStream_t>(stream_)));
  }
  is_first_run_ = false;
#endif
}

void XPUGraph::Reset() {
  if (is_reset_) return;
#if CUDA_VERSION >= 10010
  for (auto graph : graphs_) {
    PADDLE_ENFORCE_XPU_SUCCESS(cudaGraphDestroy(graph));
  }
  graphs_.clear();
  for (auto exec_graph : exec_graphs_) {
    PADDLE_ENFORCE_XPU_SUCCESS(cudaGraphExecDestroy(exec_graph));
  }
  exec_graphs_.clear();
#endif
  // callback should be called in reverse order because the latter added
  // callback may rely on the former added callback.
  for (auto iter = xpu_graph_post_reset_callbacks_.rbegin();
       iter != xpu_graph_post_reset_callbacks_.rend();
       ++iter) {
    (*iter)(*this);
  }
  xpu_graph_post_reset_callbacks_.clear();
  is_reset_ = true;
}

void XPUGraph::AddPostResetCallback(XPUPostResetCallback callback) {
  std::lock_guard<std::mutex> guard(mtx_);
  xpu_graph_post_reset_callbacks_.push_back(std::move(callback));
}

void XPUGraph::AddPreCaptureCallback(XPUPreCaptureCallback callback) {
  xpu_graph_pre_capture_callbacks_.push_back(std::move(callback));
}

void XPUGraph::AddPostCaptureCallback(XPUPostCaptureCallback callback) {
  std::lock_guard<std::mutex> guard(mtx_);
  xpu_graph_post_capture_callbacks_.push_back(std::move(callback));
}

void XPUGraph::AddJoiningStream(XPUStream stream) {
  streams_to_join_.insert(stream);
}

void XPUGraph::PrintToDotFiles(const std::string &dirname, unsigned int flags){
  ThrowErrorIfNotSupportXPUGraph();
#if CUDA_VERSION >= 11030
  for (size_t i = 0; i < graphs_.size(); ++i) {
    auto filename =
        ConcatPath(dirname, "segment_" + std::to_string(i) + ".dot");
    VLOG(10) << "Save the " << i << "-th segment of graph " << id_ << " to "
             << filename;
    PADDLE_ENFORCE_XPU_SUCCESS(
        cudaGraphDebugDotPrint(graphs_[i], filename.c_str(), flags));
  }
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "The print_to_dot_files() method is only supported when CUDA version >= "
      "11.3."));
#endif
}

bool XPUGraph::IsReplayed() const {
  return is_replayed_;
}

int64_t XPUGraph::SetMemoryPoolID(int64_t pool_id) {
  auto &pool_id_ = capturing_graph_->pool_id_;
  PADDLE_ENFORCE_EQ(pool_id_,
                    kInvalidPoolID,
                    common::errors::InvalidArgument(
                        "Cannot reset memory pool id twice, the "
                        "former memory pool id is %d.",
                        pool_id_));
  if (pool_id <= kInvalidPoolID) {
    pool_id_ = UniqueMemoryPoolID();
  } else {
    PADDLE_ENFORCE_GE(pool_id,
                      kDefaultPoolID,
                      common::errors::InvalidArgument(
                          "Invalid memory pool id %d.", pool_id));
    pool_id_ = pool_id;
  }
  return pool_id_;
}


void XPUGraph::BeginSegmentCapture() {
  ThrowErrorIfNotSupportXPUGraph();
#if CUDA_VERSION >= 10010
  PADDLE_ENFORCE_EQ(IsCapturing(),
                    true,
                    common::errors::PermissionDenied(
                        "BeginSegmentCapture should be called when CUDA "
                        "Graph is capturing."));
  if (IsThreadLocalCapturing()) {
    PADDLE_ENFORCE_EQ(IsThisThreadCapturing(),
                      true,
                      common::errors::PermissionDenied(
                          "When capturing CUDA Graph in the thread local mode, "
                          "you cannot begin segmented capturing in the thread "
                          "which is not the one that starts the capturing."));
  }

  for (auto &hook : xpu_graph_pre_capture_callbacks_) {
    hook();
  }

  PADDLE_ENFORCE_XPU_SUCCESS(cudaStreamBeginCapture(
      static_cast<cudaStream_t>(capturing_graph_->stream_), cudaStreamCaptureModeGlobal));
  PADDLE_ENFORCE_EQ(IsValidCapturing(),
                    true,
                    common::errors::PermissionDenied(
                        "CUDA Graph should not be invalidated."));
  VLOG(10) << "Begin to capture CUDA Graph with ID " << capturing_graph_->id_
           << ", segment id " << capturing_graph_->graphs_.size()
           << ", memory pool id " << capturing_graph_->pool_id_;
#endif
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

void XPUGraph::EndSegmentCapture() {
  ThrowErrorIfNotSupportXPUGraph();
#if CUDA_VERSION >= 10010
  PADDLE_ENFORCE_EQ(
      IsCapturing(),
      true,
      common::errors::PermissionDenied("No CUDA Graph is capturing."));

  for (const auto &stream : capturing_graph_->streams_to_join_) {
    VLOG(10) << "Joining steam when the capture is going to end stream ="
             << stream;
    sync_streams(static_cast<cudaStream_t>(stream), static_cast<cudaStream_t>(capturing_graph_->stream_));
  }
  capturing_graph_->streams_to_join_.clear();

  cudaGraph_t graph;
  PADDLE_ENFORCE_XPU_SUCCESS(
      cudaStreamEndCapture(static_cast<cudaStream_t>(capturing_graph_->stream_), &graph));
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
       capturing_graph_->xpu_graph_post_capture_callbacks_) {
    cudagraph_post_capture_callback();
  }
  capturing_graph_->xpu_graph_post_capture_callbacks_.clear();

  // TODO(huzhida): whether need this logic or not ?
//  capturing_graph_->cudagraph_pre_replay_callbacks_.emplace_back(
//      XPUGraphNodeLauncher::Instance().GetParameterSettersForExecGraph(graph));

  cudaGraphExec_t exec_graph;
  if (FLAGS_use_cuda_malloc_async_allocator &&
      FLAGS_auto_free_cudagraph_allocations_on_launch) {
#if CUDA_VERSION >= 11040
    VLOG(1) << "cudaGraphInstantiateFlagAutoFreeOnLaunch is enabled!";
    PADDLE_ENFORCE_XPU_SUCCESS(cudaGraphInstantiateWithFlags(
        &exec_graph, graph, cudaGraphInstantiateFlagAutoFreeOnLaunch));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "The cudaGraphInstantiateFlagAutoFreeOnLaunch is only supported when "
        "CUDA version >= 11.4.0"));
#endif
  } else {
    PADDLE_ENFORCE_XPU_SUCCESS(
        cudaGraphInstantiate(&exec_graph, graph, nullptr, nullptr, 0));
  }
  VLOG(10) << "End to capture CUDA Graph with ID " << capturing_graph_->id_
           << ", segment id " << capturing_graph_->graphs_.size()
           << ", memory pool id " << capturing_graph_->pool_id_;
  capturing_graph_->graphs_.emplace_back(graph);
  capturing_graph_->exec_graphs_.emplace_back(exec_graph);
#endif
}

void XPUGraph::BeginCapture(phi::XPUPlace place,
                       XPUStream stream,
                       xpuStreamCaptureMode mode) {
  ThrowErrorIfNotSupportXPUGraph();
#if CUDA_VERSION >= 10010
  PADDLE_ENFORCE_EQ(IsCapturing(),
                    false,
                    common::errors::PermissionDenied(
                        "CUDA Graph can only captured one by one."));
  PADDLE_ENFORCE_NOT_NULL(
      stream,
      common::errors::PermissionDenied(
          "CUDA Graph cannot be captured in default CUDA stream 0."));
  capturing_graph_.reset(new XPUGraph());
  capturing_graph_->place_ = place;
  capturing_graph_->stream_ = stream;
  capturing_graph_->capture_mode_ = mode;
  if (mode == xpuStreamCaptureModeThreadLocal) {
    capturing_thread_id_ = std::this_thread::get_id();
    VLOG(10) << "Capturing CUDA Graph in thread local mode, thread id: "
             << capturing_thread_id_;
  }
  BeginSegmentCapture();
#endif
}

std::unique_ptr<XPUGraph> XPUGraph::EndCapture() {
  EndSegmentCapture();
  capturing_thread_id_ = paddle::none;
  return std::move(capturing_graph_);
}

void XPUGraph::AddJoiningStreamDuringCapturing(XPUStream stream){
  capturing_graph_->AddJoiningStream(stream);
}

void XPUGraph::AddPostResetCallbackDuringCapturing(
  XPUPostResetCallback callback) {
  capturing_graph_->AddPostResetCallback(std::move(callback));
}

void XPUGraph::AddPostCaptureCallbackDuringCapturing(
  XPUPostCaptureCallback callback) {
  capturing_graph_->AddPostCaptureCallback(std::move(callback));
}

bool XPUGraph::IsCapturing() {
  return capturing_graph_ != nullptr;
}

XPUGraphID XPUGraph::CapturingID() {
  return capturing_graph_->id_;
}

phi::XPUPlace XPUGraph::CapturingPlace() {
  return capturing_graph_->place_;
}

bool XPUGraph::IsValidCapturing() {
#if CUDA_VERSION >= 10010
  if (!IsCapturing()) return false;
  cudaStreamCaptureStatus status;
  XPUGraphID id;
  PADDLE_ENFORCE_XPU_SUCCESS(
      cudaStreamGetCaptureInfo(static_cast<cudaStream_t>(capturing_graph_->stream_), &status, &id));
  return status == cudaStreamCaptureStatusActive;
#else
  return false;
#endif
}

bool XPUGraph::IsThreadLocalCapturing() {
#if CUDA_VERSION >= 10010
  return IsCapturing() &&
         capturing_graph_->capture_mode_ == xpuStreamCaptureModeThreadLocal;
#else
  return false;
#endif
}

bool XPUGraph::IsThisThreadCapturing() {
  if (UNLIKELY(IsCapturing())) {
    return !IsThreadLocalCapturing() ||
           capturing_thread_id_.get() == std::this_thread::get_id();
  }
  return false;
}

void XPUGraph::RecordRandomKernelInfo(SetSeedFunc set_seed_func) {
  std::lock_guard<std::mutex> guard(capturing_graph_->func_mtx_);
  capturing_graph_->set_seed_funcs_.emplace_back(std::move(set_seed_func));
}

int64_t XPUGraph::UniqueMemoryPoolID() {
  static std::atomic<int64_t> id(XPUGraph::kDefaultPoolID + 1);
  return id.fetch_add(1);
}

} // namespace xpu
} // namespace backends
} // namespace phi

#endif
