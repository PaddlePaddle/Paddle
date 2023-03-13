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
#include "paddle/fluid/platform/device_event.h"
#include "paddle/phi/backends/context_pool.h"

DECLARE_bool(use_stream_safe_cuda_allocator);
DECLARE_bool(new_executor_use_cuda_graph);

namespace paddle {
namespace platform {

#ifdef PADDLE_WITH_CUDA
void InitCUDNNRelatedHandle(phi::GPUContext* dev_ctx) {
  dev_ctx->cudnn_workspace_handle().ResetWorkspace();

  // After PR(#43206), cudnn related initializations will change to lazy mode.
  // It will only be initialized when op calls them. But cuda graph not
  // support capture such kind of init, need to init all these handle before
  // cuda graph.
  dev_ctx->cublas_handle();
#if CUDA_VERSION >= 11060
  dev_ctx->cublaslt_handle();
#endif
  dev_ctx->cudnn_handle();
  dev_ctx->cusolver_dn_handle();
}

void BeginCUDAGraphCapture(phi::GPUPlace place,
                           cudaStreamCaptureMode mode,
                           int64_t pool_id) {
  auto* mutable_dev_ctx = phi::DeviceContextPool::Instance().Get(place);
  auto all_capturing_dev_ctxs =
      phi::backends::gpu::CUDAGraphContextManager::Instance()
          .GetAllCapturingDeviceContexts();
  // create_cuda_graph_stream: Whether to create a new stream to
  // capture cuda graph, usually used in multi-stream scenarios.
  // Can only be used for new executor in static mode, that is,
  // FLAGS_new_executor_use_cuda_graph needs to be set to True.
  bool create_cuda_graph_stream = false;
  if (FLAGS_new_executor_use_cuda_graph &&
      (all_capturing_dev_ctxs.size() > 1 ||
       (all_capturing_dev_ctxs.size() == 1 &&
        (*(all_capturing_dev_ctxs.begin()) != mutable_dev_ctx)))) {
    create_cuda_graph_stream = true;
  }
  if (create_cuda_graph_stream) {
    VLOG(4) << "create a new stream to capture cuda graph.";
    if (pool_id <= CUDAGraph::kInvalidPoolID) {
      pool_id = CUDAGraph::UniqueMemoryPoolID();
    }
    mutable_dev_ctx =
        phi::backends::gpu::CUDAGraphContextManager::Instance().Get(
            pool_id, place, 0);
    for (auto iter = all_capturing_dev_ctxs.begin();
         iter != all_capturing_dev_ctxs.end();
         ++iter) {
      auto* capturing_dev_ctx = reinterpret_cast<phi::GPUContext*>(*iter);
      InitCUDNNRelatedHandle(capturing_dev_ctx);
    }
  }
  auto* dev_ctx = reinterpret_cast<phi::GPUContext*>(mutable_dev_ctx);
  InitCUDNNRelatedHandle(dev_ctx);

  auto stream = dev_ctx->stream();
  CUDAGraph::BeginCapture(place, stream, mode);
  CUDAGraph::SetIsCUDAGraphStreamCreated(create_cuda_graph_stream);

  // When using cuda graph in new executor, fast GC must be used.
  // FLAGS_use_stream_safe_cuda_allocator should be true.
  auto old_value = FLAGS_use_stream_safe_cuda_allocator &&
                   !FLAGS_new_executor_use_cuda_graph;
  if (old_value) {
    FLAGS_use_stream_safe_cuda_allocator = false;
  }
  pool_id = CUDAGraph::SetMemoryPoolID(pool_id);
  memory::allocation::AllocatorFacade::Instance().PrepareMemoryPoolForCUDAGraph(
      pool_id);
  dev_ctx->SetCUDAGraphAllocator(memory::allocation::AllocatorFacade::Instance()
                                     .GetAllocator(place)
                                     .get());
  if (old_value) {
    FLAGS_use_stream_safe_cuda_allocator = true;
  }
  if (create_cuda_graph_stream) {
    // Set cuda graph allocator for all streams.
    // Establish dependencies between cuda graph stream and all other streams
    // using eventWait, so that all streams will be captured.
    std::shared_ptr<platform::DeviceEvent> cuda_graph_event =
        std::make_shared<platform::DeviceEvent>(
            dev_ctx->GetPlace(), platform::GenerateDeviceEventFlag());
    cuda_graph_event->Record(dev_ctx);

    for (auto iter = all_capturing_dev_ctxs.begin();
         iter != all_capturing_dev_ctxs.end();
         ++iter) {
      auto* capturing_dev_ctx = reinterpret_cast<phi::GPUContext*>(*iter);
      auto capturing_stream = capturing_dev_ctx->stream();
      capturing_dev_ctx->SetCUDAGraphAllocator(
          memory::allocation::AllocatorFacade::Instance()
              .GetAllocator(place, capturing_stream)
              .get());
      VLOG(4) << "set CUDAGraphAllocator for dev_ctx: " << capturing_dev_ctx
              << " with stream: " << capturing_stream;
      cuda_graph_event->Wait(platform::kCUDA, capturing_dev_ctx);
      VLOG(4) << "CUDA Graph stream eventWait. Capturing dev_ctx: "
              << capturing_dev_ctx
              << " wait for cuda graph dev_ctx: " << dev_ctx;
    }
  }
  AddResetCallbackIfCapturingCUDAGraph([pool_id] {
    memory::allocation::AllocatorFacade::Instance().RemoveMemoryPoolOfCUDAGraph(
        pool_id);
  });
}

std::unique_ptr<CUDAGraph> EndCUDAGraphCapture() {
  phi::DeviceContext* mutable_dev_ctx;
  auto place = CUDAGraph::CapturingPlace();
  bool create_cuda_graph_stream = CUDAGraph::IsCUDAGraphStreamCreated();
  if (create_cuda_graph_stream) {
    // join all other streams back to origin cuda graph stream.
    int64_t pool_id = CUDAGraph::CapturingPoolID();
    mutable_dev_ctx =
        phi::backends::gpu::CUDAGraphContextManager::Instance().Get(
            pool_id, place, 0);
    auto* cuda_graph_dev_ctx =
        reinterpret_cast<phi::GPUContext*>(mutable_dev_ctx);
    auto all_capturing_dev_ctxs =
        phi::backends::gpu::CUDAGraphContextManager::Instance()
            .GetAllCapturingDeviceContexts();
    for (auto iter = all_capturing_dev_ctxs.begin();
         iter != all_capturing_dev_ctxs.end();
         ++iter) {
      auto* capturing_dev_ctx = reinterpret_cast<phi::GPUContext*>(*iter);
      std::shared_ptr<platform::DeviceEvent> capturing_event =
          std::make_shared<platform::DeviceEvent>(
              capturing_dev_ctx->GetPlace(),
              platform::GenerateDeviceEventFlag());
      capturing_event->Record(capturing_dev_ctx);
      capturing_event->Wait(platform::kCUDA, cuda_graph_dev_ctx);
      VLOG(4) << "CUDA Graph stream eventWait. cuda graph dev_ctx: "
              << cuda_graph_dev_ctx
              << " wait for capturing dev_ctx: " << capturing_dev_ctx;
      capturing_dev_ctx->cudnn_workspace_handle().ResetWorkspace();
      capturing_dev_ctx->SetCUDAGraphAllocator(nullptr);
    }
    phi::backends::gpu::CUDAGraphContextManager::Instance()
        .ClearDeviceContextsRecords();
  } else {
    mutable_dev_ctx = phi::DeviceContextPool::Instance().Get(place);
  }
  auto* dev_ctx = reinterpret_cast<phi::GPUContext*>(mutable_dev_ctx);
  dev_ctx->cudnn_workspace_handle().ResetWorkspace();
  dev_ctx->SetCUDAGraphAllocator(nullptr);
  return CUDAGraph::EndCapture();
}
#endif

}  // namespace platform
}  // namespace paddle
