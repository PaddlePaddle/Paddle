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

#include "paddle/fluid/platform/cuda_graph.h"

namespace paddle {
namespace platform {

std::unique_ptr<CUDAGraph> CUDAGraph::capturing_graph_{nullptr};

void CUDAGraph::Reset() {
  if (is_reset_) return;
#if CUDA_VERSION >= 10010
  if (graph_) {
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaGraphDestroy(graph_));
    graph_ = nullptr;
  }
  if (exec_graph_) {
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaGraphExecDestroy(exec_graph_));
    exec_graph_ = nullptr;
  }
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
  PADDLE_ENFORCE_NOT_NULL(exec_graph_,
                          errors::PermissionDenied(
                              "CUDA Graph must be captured before replaying."));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaGraphLaunch(exec_graph_, stream_));
#endif
}

void CUDAGraph::BeginCapture(platform::CUDAPlace place, cudaStream_t stream,
                             cudaStreamCaptureMode mode) {
  ThrowErrorIfNotSupportCUDAGraph();
  PADDLE_ENFORCE_EQ(
      IsCapturing(), false,
      errors::PermissionDenied("CUDA Graph can only captured one by one."));
  PADDLE_ENFORCE_NOT_NULL(
      stream, errors::PermissionDenied(
                  "CUDA Graph cannot be captured in default CUDA stream 0."));
  capturing_graph_.reset(new CUDAGraph());
  capturing_graph_->place_ = place;
  capturing_graph_->stream_ = stream;

  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaStreamBeginCapture(capturing_graph_->stream_, mode));
  cudaStreamCaptureStatus status;
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamGetCaptureInfo(
      capturing_graph_->stream_, &status, &(capturing_graph_->id_)));
  PADDLE_ENFORCE_EQ(IsValidCapturing(), true,
                    platform::errors::PermissionDenied(
                        "CUDA Graph should not be invalidated."));
  VLOG(10) << "Begin to capture CUDA Graph with ID " << capturing_graph_->id_;
}

std::unique_ptr<CUDAGraph> CUDAGraph::EndCapture() {
  ThrowErrorIfNotSupportCUDAGraph();
#if CUDA_VERSION >= 10010
  PADDLE_ENFORCE_EQ(IsCapturing(), true,
                    errors::PermissionDenied("No CUDA Graph is capturing."));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamEndCapture(
      capturing_graph_->stream_, &(capturing_graph_->graph_)));
  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaGraphInstantiate(&(capturing_graph_->exec_graph_),
                           capturing_graph_->graph_, nullptr, nullptr, 0));
  VLOG(10) << "End to capture CUDA Graph with ID " << capturing_graph_->id_;
  return std::move(capturing_graph_);
#endif
}

bool CUDAGraph::IsValidCapturing() {
  if (!IsCapturing()) return false;
  cudaStreamCaptureStatus status;
  CUDAGraphID id;
  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaStreamGetCaptureInfo(capturing_graph_->stream_, &status, &id));
  return status == cudaStreamCaptureStatusActive;
}

}  // namespace platform
}  // namespace paddle
