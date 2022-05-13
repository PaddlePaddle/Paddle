// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/device/gpu/cuda/cuda_graph.h"

namespace paddle {
namespace platform {

std::unique_ptr<CUDAGraph> CUDAGraph::capturing_graph_{nullptr};
paddle::optional<std::thread::id> CUDAGraph::capturing_thread_id_{paddle::none};

void CUDAGraph::Reset() {
  if (is_reset_) return;
#if CUDA_VERSION >= 10010
  for (auto graph : graphs_) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaGraphDestroy(graph));
  }
  graphs_.clear();
  for (auto exec_graph : exec_graphs_) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaGraphExecDestroy(exec_graph));
  }
  exec_graphs_.clear();
#endif
  // callback should be called in reverse order because the latter added
  // callback may rely on the former added callback.
  for (auto iter = callbacks_.rbegin(); iter != callbacks_.rend(); ++iter) {
    (*iter)();
  }
  callbacks_.clear();
  is_reset_ = true;
}

void CUDAGraph::Replay() {
#if CUDA_VERSION >= 10010
  PADDLE_ENFORCE_EQ(is_reset_, false,
                    errors::PermissionDenied(
                        "Cannot replay the CUDA Graph after reset is called."));
  for (auto exec_graph : exec_graphs_) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaGraphLaunch(exec_graph, stream_));
  }
#endif
}

void CUDAGraph::BeginSegmentCapture() {
  ThrowErrorIfNotSupportCUDAGraph();
#if CUDA_VERSION >= 10010
  PADDLE_ENFORCE_EQ(
      IsCapturing(), true,
      errors::PermissionDenied("BeginSegmentCapture should be called when CUDA "
                               "Graph is capturing."));
  if (IsThreadLocalCapturing()) {
    PADDLE_ENFORCE_EQ(IsThisThreadCapturing(), true,
                      platform::errors::PermissionDenied(
                          "When capturing CUDA Graph in the thread local mode, "
                          "you cannot begin segmented capturing in the thread "
                          "which is not the one that starts the capturing."));
  }
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamBeginCapture(
      capturing_graph_->stream_, capturing_graph_->capture_mode_));
  PADDLE_ENFORCE_EQ(IsValidCapturing(), true,
                    platform::errors::PermissionDenied(
                        "CUDA Graph should not be invalidated."));
  VLOG(10) << "Begin to capture CUDA Graph with ID " << capturing_graph_->id_
           << ", segment id " << capturing_graph_->graphs_.size();
#endif
}

void CUDAGraph::BeginCapture(platform::CUDAPlace place, cudaStream_t stream,
                             cudaStreamCaptureMode mode) {
  ThrowErrorIfNotSupportCUDAGraph();
#if CUDA_VERSION >= 10010
  PADDLE_ENFORCE_EQ(
      IsCapturing(), false,
      errors::PermissionDenied("CUDA Graph can only captured one by one."));
  PADDLE_ENFORCE_NOT_NULL(
      stream, errors::PermissionDenied(
                  "CUDA Graph cannot be captured in default CUDA stream 0."));
  capturing_graph_.reset(new CUDAGraph());
  capturing_graph_->place_ = place;
  capturing_graph_->stream_ = stream;
  capturing_graph_->capture_mode_ = mode;
  if (mode == cudaStreamCaptureModeThreadLocal) {
    capturing_thread_id_ = std::this_thread::get_id();
    VLOG(10) << "Capturing CUDA Graph in thread local mode, thread id: "
             << capturing_thread_id_;
  }
  BeginSegmentCapture();
#endif
}

void CUDAGraph::EndSegmentCapture() {
  ThrowErrorIfNotSupportCUDAGraph();
#if CUDA_VERSION >= 10010
  PADDLE_ENFORCE_EQ(IsCapturing(), true,
                    errors::PermissionDenied("No CUDA Graph is capturing."));
  cudaGraph_t graph;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaStreamEndCapture(capturing_graph_->stream_, &graph));
  auto num_nodes = static_cast<size_t>(-1);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaGraphGetNodes(graph, nullptr, &num_nodes));
  if (num_nodes == 0) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaGraphDestroy(graph));
    VLOG(10) << "Skip empty CUDA Graph with ID " << capturing_graph_->id_
             << ", segment id " << capturing_graph_->graphs_.size();
    return;
  }

  cudaGraphExec_t exec_graph;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaGraphInstantiate(&exec_graph, graph, nullptr, nullptr, 0));
  VLOG(10) << "End to capture CUDA Graph with ID " << capturing_graph_->id_
           << ", segment id " << capturing_graph_->graphs_.size();
  capturing_graph_->graphs_.emplace_back(graph);
  capturing_graph_->exec_graphs_.emplace_back(exec_graph);
#endif
}

std::unique_ptr<CUDAGraph> CUDAGraph::EndCapture() {
  EndSegmentCapture();
  capturing_thread_id_ = paddle::none;
  return std::move(capturing_graph_);
}

bool CUDAGraph::IsValidCapturing() {
#if CUDA_VERSION >= 10010
  if (!IsCapturing()) return false;
  cudaStreamCaptureStatus status;
  CUDAGraphID id;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaStreamGetCaptureInfo(capturing_graph_->stream_, &status, &id));
  return status == cudaStreamCaptureStatusActive;
#else
  return false;
#endif
}

static std::string ConcatPath(const std::string &dirname,
                              const std::string &filename) {
#ifdef _WIN32
  const char kFileSep[] = "\\";
#else
  const char kFileSep[] = "/";
#endif
  if (!dirname.empty() && dirname.back() == kFileSep[0]) {
    return dirname + filename;
  } else {
    return dirname + kFileSep + filename;
  }
}

void CUDAGraph::PrintToDotFiles(const std::string &dirname,
                                unsigned int flags) {
  ThrowErrorIfNotSupportCUDAGraph();
#if CUDA_VERSION >= 11030
  for (size_t i = 0; i < graphs_.size(); ++i) {
    auto filename =
        ConcatPath(dirname, "segment_" + std::to_string(i) + ".dot");
    VLOG(10) << "Save the " << i << "-th segment of graph " << id_ << " to "
             << filename;
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaGraphDebugDotPrint(graphs_[i], filename.c_str(), flags));
  }
#else
  PADDLE_THROW(platform::errors::Unimplemented(
      "The print_to_dot_files() method is only supported when CUDA version >= "
      "11.3."));
#endif
}

}  // namespace platform
}  // namespace paddle
