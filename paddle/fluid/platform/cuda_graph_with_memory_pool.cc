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

#include "paddle/fluid/platform/cuda_graph_with_memory_pool.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace platform {

#ifdef PADDLE_WITH_CUDA
void BeginCUDAGraphCapture(platform::CUDAPlace place,
                           cudaStreamCaptureMode mode) {
  auto stream =
      platform::DeviceContextPool::Instance().GetByPlace(place)->stream();
  CUDAGraph::BeginCapture(place, stream, mode);
  memory::allocation::AllocatorFacade::Instance().PrepareMemoryPoolForCUDAGraph(
      CUDAGraph::CapturingID());
}

std::unique_ptr<CUDAGraph> EndCUDAGraphCapture() {
  auto graph = CUDAGraph::EndCapture();
  auto id = graph->ID();
  graph->SetResetCallback([id] {
    memory::allocation::AllocatorFacade::Instance().RemoveMemoryPoolOfCUDAGraph(
        id);
  });
  return graph;
}
#endif

}  // namespace platform
}  // namespace paddle
