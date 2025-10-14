// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/custom/cuda_graph.h"
#include "paddle/phi/backends/device_manager.h"
#include "glog/logging.h"
#include "paddle/common/flags.h"

#ifdef PADDLE_WITH_CUSTOM_DEVICE

namespace phi::backends::gpu {

std::unique_ptr<CUDAGraph> CUDAGraph::capturing_graph_{nullptr};
paddle::optional<std::thread::id> CUDAGraph::capturing_thread_id_{paddle::none};
std::vector<std::function<void()>> CUDAGraph::cudagraph_pre_capture_callbacks_;

void CUDAGraph::Reset() {
  if (is_reset_) return;
  auto *device =
      phi::DeviceManager::GetDeviceWithPlace(capturing_graph_->place_);
  for (auto graph : graphs_) {
    device->CudaGraphDestroy(graph);
  }
  graphs_.clear();
  for (auto exec_graph : exec_graphs_) {
    device->CudaGraphExecDestroy(exec_graph);
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
  auto *device =
      phi::DeviceManager::GetDeviceWithPlace(capturing_graph_->place_);
  size_t n = exec_graphs_.size();
  for (size_t i = 0; i < n; ++i) {
    if (!is_first_run_) {
      for (auto &hook : cudagraph_pre_replay_callbacks_[i]) {
        hook(exec_graphs_[i]);
      }
    }
    device->CudaGraphLaunch(exec_graphs_[i], stream_);
  }
  is_first_run_ = false;
}

void CUDAGraph::BeginSegmentCapture() {
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

  for (auto &hook : cudagraph_pre_capture_callbacks_) {
    hook();
  }
  auto *device =
      phi::DeviceManager::GetDeviceWithPlace(capturing_graph_->place_);
  device->CUDAStreamBeginCapture(capturing_graph_->stream_,
                                 capturing_graph_->capture_mode_);
      PADDLE_ENFORCE_EQ(IsValidCapturing(),
                        true,
                        common::errors::PermissionDenied(
                            "CUDA Graph should not be invalidated."));
  VLOG(10) << "Begin to capture CUDA Graph with ID " << capturing_graph_->id_
           << ", segment id " << capturing_graph_->graphs_.size()
           << ", memory pool id " << capturing_graph_->pool_id_;
}

void CUDAGraph::BeginCapture(const Place &place,
                             phi::stream::stream_t stream,
                             phi::graph::streamCaptureMode mode) {
  PADDLE_ENFORCE_EQ(IsCapturing(),
                    false,
                    common::errors::PermissionDenied(
                        "CUDA Graph can only captured one by one."));
  PADDLE_ENFORCE_NOT_NULL(
      stream,
      common::errors::PermissionDenied(
          "CUDA Graph cannot be captured in default CUDA stream 0."));
  capturing_graph_.reset(new CUDAGraph());
  capturing_graph_->place_ = place;
  capturing_graph_->stream_ = stream;
  capturing_graph_->capture_mode_ = mode;
  if (mode == phi::graph::streamCaptureMode::StreamCaptureModeThreadLocal) {
    capturing_thread_id_ = std::this_thread::get_id();
    VLOG(10) << "Capturing CUDA Graph in thread local mode, thread id: "
             << capturing_thread_id_;
  }
  BeginSegmentCapture();
}

void CUDAGraph::EndSegmentCapture() {
  PADDLE_ENFORCE_EQ(
      IsCapturing(),
      true,
      common::errors::PermissionDenied("No CUDA Graph is capturing."));
  auto *device =
      phi::DeviceManager::GetDeviceWithPlace(capturing_graph_->place_);
  for (const auto &stream : capturing_graph_->streams_to_join_) {
    VLOG(10) << "Joining steam when the capture is going to end stream ="
             << stream;
    if (stream == capturing_graph_->stream_) continue;
    phi::event::Event event;
    event.Init(capturing_graph_->place_,
               phi::event::Event::Flag::DisableTiming);
    phi::stream::Stream s(capturing_graph_->place_, stream);
    event.Record(&s);
    phi::stream::Stream capture_stream(capturing_graph_->place_,
                                       capturing_graph_->stream_);
    capture_stream.WaitEvent(&event);
    event.Destroy();
  }
  capturing_graph_->streams_to_join_.clear();

  phi::graph::CUDAGraph_t graph;
  device->CudaStreamEndCapture(capturing_graph_->stream_,
                               &graph);
  auto num_nodes = static_cast<size_t>(-1);
  device->CudaGraphGetNodes(graph, nullptr, &num_nodes);
  if (num_nodes == 0) {
    device->CudaGraphDestroy(graph);
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
      CUDAGraphNodeLauncher::Instance().GetParameterSettersForExecGraph(
          capturing_graph_->place_, graph));

  phi::graph::CUDAGraphExec_t exec_graph;
  device->CudaGraphInstantiate(&exec_graph, &graph, nullptr, nullptr, 0);
  VLOG(10) << "End to capture CUDA Graph with ID " << capturing_graph_->id_
           << ", segment id " << capturing_graph_->graphs_.size()
           << ", memory pool id " << capturing_graph_->pool_id_;
  capturing_graph_->graphs_.emplace_back(graph);
  capturing_graph_->exec_graphs_.emplace_back(exec_graph);
}

std::unique_ptr<CUDAGraph> CUDAGraph::EndCapture() {
  EndSegmentCapture();
  capturing_thread_id_ = paddle::none;
  return std::move(capturing_graph_);
}

bool CUDAGraph::IsValidCapturing() {
  if (!IsCapturing()) return false;
  return true;
  auto *device =
      phi::DeviceManager::GetDeviceWithPlace(capturing_graph_->place_);
  phi::graph::streamCaptureStatus status =
      phi::graph::streamCaptureStatus::StreamCaptureStatusNone;
  device->CudaStreamGetCaptureInfo(capturing_graph_->stream_, &status);
  return status == phi::graph::streamCaptureStatus::StreamCaptureStatusActive;
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
  // TODO(zhangxiao): print.
}

void CUDAGraphNodeLauncher::KernelNodeLaunch(
    parameterSetter_t parameterSetter, gpuKernelCallback_t cudakernelCallback) {
  if (UNLIKELY(phi::backends::gpu::CUDAGraph::IsThisThreadCapturing())) {
    unsigned int id = GenerateIdentifier();
    auto cudaFunc = cudakernelCallback(id);

    parameterSetters[cudaFunc][id] = parameterSetter;
    VLOG(10) << "[KernelNodeLaunch] Launch kernel with cudaFunc = " << cudaFunc
             << " id = " << id;
  } else {
    cudakernelCallback(0);
  }
}

std::vector<GraphExecuterSetter_t>
CUDAGraphNodeLauncher::GetParameterSettersForExecGraph(
    const Place &place, phi::graph::CUDAGraph_t graph) {
  auto *device = phi::DeviceManager::GetDeviceWithPlace(place);
  phi::graph::GraphHookManager graph_hook;
  device->GetParameterSetterForExecGraph(graph, &graph_hook);
  std::vector<GraphExecuterSetter_t> hooks;
  for (int i = 0; i < graph_hook.count; i++) {
    hooks.emplace_back(graph_hook.hooks[i]);
  }
  return hooks;
}

}  // namespace phi::backends::gpu

#endif  // PADDLE_WITH_CUSTOM_DEVICE
