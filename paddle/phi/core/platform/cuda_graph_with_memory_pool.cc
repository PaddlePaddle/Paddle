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

#include "paddle/phi/core/platform/cuda_graph_with_memory_pool.h"

#include "paddle/common/flags.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"
#include "paddle/phi/core/platform/device_event.h"

PD_DECLARE_bool(use_stream_safe_cuda_allocator);
COMMON_DECLARE_bool(new_executor_use_cuda_graph);

namespace paddle::platform {

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
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

phi::DeviceContext* SelectCUDAGraphDeviceContext(phi::GPUPlace place,
                                                 int64_t* pool_id) {
  phi::DeviceContext* mutable_dev_ctx;
  auto all_capturing_dev_ctxs =
      phi::backends::gpu::CUDAGraphContextManager::Instance()
          .GetAllCapturingDeviceContexts();
  auto num_stream = all_capturing_dev_ctxs.size();
  if (num_stream > 0) {
    // Capturing device contexts will only be recorded in new
    // executor in temporary, that is,
    // FLAGS_new_executor_use_cuda_graph needs to be set to True.
    // This restriction can be removed if device context is
    // recorded in other modes.
    // Record method: RecordCapturingDeviceContext.
    PADDLE_ENFORCE_EQ(FLAGS_new_executor_use_cuda_graph,
                      true,
                      common::errors::InvalidArgument(
                          "FLAGS_new_executor_use_cuda_graph must be True when "
                          "capturing stream is recorded."));
    if (num_stream > 1) {
      VLOG(4) << "Use a new stream to capture cuda graph. Used in multi-stream "
                 "scenarios with new executor.";
      if (*pool_id <= CUDAGraph::kInvalidPoolID) {
        *pool_id = CUDAGraph::UniqueMemoryPoolID();
      }
      mutable_dev_ctx =
          phi::backends::gpu::CUDAGraphContextManager::Instance().Get(
              *pool_id, place, 0);
    } else {
      VLOG(4) << "Use recorded stream to capture cuda graph. Used in "
                 "single-stream scenarios with new executor.";
      mutable_dev_ctx = *(all_capturing_dev_ctxs.begin());
    }
  } else {
    VLOG(4) << "Use default stream to capture cuda graph.";
    mutable_dev_ctx = phi::DeviceContextPool::Instance().Get(place);
  }
  return mutable_dev_ctx;
}

void BeginCUDAGraphCapture(phi::GPUPlace place,
                           gpuStreamCaptureMode mode,
                           int64_t pool_id) {
  auto* mutable_dev_ctx = SelectCUDAGraphDeviceContext(place, &pool_id);
  auto* dev_ctx = reinterpret_cast<phi::GPUContext*>(mutable_dev_ctx);
  InitCUDNNRelatedHandle(dev_ctx);

  auto all_capturing_dev_ctxs =
      phi::backends::gpu::CUDAGraphContextManager::Instance()
          .GetAllCapturingDeviceContexts();
  auto num_stream = all_capturing_dev_ctxs.size();
  if (num_stream > 1) {
    for (auto all_capturing_dev_ctx : all_capturing_dev_ctxs) {
      auto* capturing_dev_ctx =
          reinterpret_cast<phi::GPUContext*>(all_capturing_dev_ctx);
      InitCUDNNRelatedHandle(capturing_dev_ctx);
    }
  }

  auto stream = dev_ctx->stream();
  CUDAGraph::BeginCapture(place, stream, mode);

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
  if (num_stream > 1) {
    // Set cuda graph allocator for all streams.
    // Establish dependencies between cuda graph stream and all other streams
    // using eventWait, so that all streams will be captured.
    std::shared_ptr<platform::DeviceEvent> cuda_graph_event =
        std::make_shared<platform::DeviceEvent>(
            dev_ctx->GetPlace(), platform::GenerateDeviceEventFlag());
    cuda_graph_event->Record(dev_ctx);

    for (auto all_capturing_dev_ctx : all_capturing_dev_ctxs) {
      auto* capturing_dev_ctx =
          reinterpret_cast<phi::GPUContext*>(all_capturing_dev_ctx);
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
  AddPostResetCallbackIfCapturingCUDAGraph(
      [=](paddle::optional<const CUDAGraph&> graph) {
        memory::allocation::AllocatorFacade::Instance()
            .RemoveMemoryPoolOfCUDAGraph(pool_id);
      });
}

std::unique_ptr<CUDAGraph> EndCUDAGraphCapture() {
  auto place = CUDAGraph::CapturingPlace();
  auto pool_id = CUDAGraph::CapturingPoolID();
  auto* mutable_dev_ctx = SelectCUDAGraphDeviceContext(place, &pool_id);
  auto* dev_ctx = reinterpret_cast<phi::GPUContext*>(mutable_dev_ctx);

  auto all_capturing_dev_ctxs =
      phi::backends::gpu::CUDAGraphContextManager::Instance()
          .GetAllCapturingDeviceContexts();
  auto num_stream = all_capturing_dev_ctxs.size();
  if (num_stream > 1) {
    // join all other streams back to origin cuda graph stream.
    for (auto all_capturing_dev_ctx : all_capturing_dev_ctxs) {
      auto* capturing_dev_ctx =
          reinterpret_cast<phi::GPUContext*>(all_capturing_dev_ctx);
      std::shared_ptr<platform::DeviceEvent> capturing_event =
          std::make_shared<platform::DeviceEvent>(
              capturing_dev_ctx->GetPlace(),
              platform::GenerateDeviceEventFlag());
      capturing_event->Record(capturing_dev_ctx);
      capturing_event->Wait(platform::kCUDA, dev_ctx);
      VLOG(4) << "CUDA Graph stream eventWait. cuda graph dev_ctx: " << dev_ctx
              << " wait for capturing dev_ctx: " << capturing_dev_ctx;
      capturing_dev_ctx->cudnn_workspace_handle().ResetWorkspace();
      capturing_dev_ctx->SetCUDAGraphAllocator(nullptr);
    }
  }

  phi::backends::gpu::CUDAGraphContextManager::Instance()
      .ClearDeviceContextsRecords();
  dev_ctx->cudnn_workspace_handle().ResetWorkspace();
  dev_ctx->SetCUDAGraphAllocator(nullptr);
  return CUDAGraph::EndCapture();
}
#endif

}  // namespace paddle::platform
