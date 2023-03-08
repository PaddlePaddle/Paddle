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
#include "paddle/phi/backends/context_pool.h"
// #include "paddle/phi/backends/context_pool_utils.h"
#include "paddle/fluid/platform/device_event.h"

DECLARE_bool(use_stream_safe_cuda_allocator);
DECLARE_bool(new_executor_use_cuda_graph);

namespace paddle {
namespace platform {

#ifdef PADDLE_WITH_CUDA
void BeginCUDAGraphCapture(phi::GPUPlace place,
                           cudaStreamCaptureMode mode,
                           int64_t pool_id,
                           bool create_cuda_graph_stream) {
  phi::DeviceContext* mutable_dev_ctx;
  if (create_cuda_graph_stream) {
    PADDLE_ENFORCE_EQ(FLAGS_new_executor_use_cuda_graph,
                      true,
                      platform::errors::InvalidArgument(
                          "FLAGS_new_executor_use_cuda_graph must be True when "
                          "create_cuda_graph_stream=True"));
    if (pool_id <= kInvalidPoolID) {
      pool_id = UniqueMemoryPoolID();
    }
    mutable_dev_ctx = phi::backends::gpu::CUDAGraphContextManager::Instance()
                          .Get(pool_id, place, 0)
                          .get()
                          .get();
    auto dev_ctxs = phi::backends::gpu::CUDAGraphContextManager::Instance()
                        .GetAllDeviceContexts();
    VLOG(4) << "yoki2";
    for (auto iter = dev_ctxs.begin(); iter != dev_ctxs.end(); ++iter) {
      VLOG(4) << "yoki3";
      auto* stream_dev_ctx = reinterpret_cast<phi::GPUContext*>(*iter);
      VLOG(4) << "yoki4: stream_dev_ctx: " << stream_dev_ctx;
      stream_dev_ctx->cudnn_workspace_handle().ResetWorkspace();
      VLOG(4) << "yoki2";
      // After PR(#43206), cudnn related initializations will change to lazy
      // mode. It will only be initialized when op calls them. But cuda graph
      // not support capture such kind of init, need to init all these handle
      // before cuda graph.
      stream_dev_ctx->cublas_handle();
      VLOG(4) << "yoki3";
#if CUDA_VERSION >= 11060
      stream_dev_ctx->cublaslt_handle();
      VLOG(4) << "yoki4";
#endif
      stream_dev_ctx->cudnn_handle();
      VLOG(4) << "yoki5";
      stream_dev_ctx->cusolver_dn_handle();
    }
  } else {
    mutable_dev_ctx = phi::DeviceContextPool::Instance().Get(place);
  }
  auto* dev_ctx = reinterpret_cast<phi::GPUContext*>(mutable_dev_ctx);

  /*std::unique_ptr<phi::DeviceContext> dev_ctx_unique_ptr;
  // std::map<phi::Place,
  // std::shared_future<std::unique_ptr<phi::DeviceContext>>> ctxs; if
  // (create_cuda_graph_stream) {
  dev_ctx_unique_ptr =
      phi::CreateDeviceContext<phi::GPUContext>(place, true, 0);
  dev_ctx = reinterpret_cast<phi::GPUContext*>(dev_ctx_unique_ptr.get());*/
  /*phi::EmplaceDeviceContexts(&ctxs, {place}, true, 0);
  dev_ctx_unique_ptr = std::move(ctxs[place].get());
  dev_ctx = reinterpret_cast<phi::GPUContext*>(dev_ctx_unique_ptr.get());*/
  /*} else {
    auto* mutable_dev_ctx = phi::DeviceContextPool::Instance().Get(place);
    dev_ctx = reinterpret_cast<phi::GPUContext*>(mutable_dev_ctx);
  }*/
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

  auto stream = dev_ctx->stream();
  CUDAGraph::BeginCapture(place, stream, mode);
  CUDAGraph::SetCreateCUDAGraphStream(create_cuda_graph_stream);
  // CUDAGraph::SetCapturingDeviceContext(std::move(dev_ctx_unique_ptr));
  // CUDAGraph::SetCapturingDeviceContext(dev_ctx);

  // When using cuda graph in new executor, fast GC must be used.
  // FLAGS_use_stream_safe_cuda_allocator should be true.
  auto old_value = FLAGS_use_stream_safe_cuda_allocator &&
                   !FLAGS_new_executor_use_cuda_graph;
  if (old_value) {
    FLAGS_use_stream_safe_cuda_allocator = false;
  }
  // int64_t old_pool_id = pool_id;
  pool_id = CUDAGraph::SetMemoryPoolID(pool_id);
  // phi::backends::gpu::CUDAGraphContextManager::Instance().UpdatePoolId(
  //     old_pool_id, pool_id);
  memory::allocation::AllocatorFacade::Instance().PrepareMemoryPoolForCUDAGraph(
      pool_id);
  dev_ctx->SetCUDAGraphAllocator(memory::allocation::AllocatorFacade::Instance()
                                     .GetAllocator(place)
                                     .get());
  if (create_cuda_graph_stream) {
    std::shared_ptr<platform::DeviceEvent> cuda_graph_event =
        std::make_shared<platform::DeviceEvent>(
            dev_ctx->GetPlace(), platform::GenerateDeviceEventFlag());
    cuda_graph_event->Record(dev_ctx);

    auto dev_ctxs = phi::backends::gpu::CUDAGraphContextManager::Instance()
                        .GetAllDeviceContexts();
    VLOG(4) << "yoki2";
    for (auto iter = dev_ctxs.begin(); iter != dev_ctxs.end(); ++iter) {
      VLOG(4) << "yoki3";
      auto* stream_dev_ctx = reinterpret_cast<phi::GPUContext*>(*iter);
      VLOG(4) << "yoki4: stream_dev_ctx: " << stream_dev_ctx;
      auto stream_cap = stream_dev_ctx->stream();
      stream_dev_ctx->SetCUDAGraphAllocator(
          memory::allocation::AllocatorFacade::Instance()
              .GetAllocator(place, stream_cap)
              .get());
      VLOG(4) << "set CUDAGraphAllocator. dev_ctx: " << stream_dev_ctx
              << "  stream: " << stream_cap;

      VLOG(4) << "yoki4: stream_dev_ctx: " << stream_dev_ctx;
      cuda_graph_event->Wait(platform::kCUDA, stream_dev_ctx);
      VLOG(4) << "CUDA Graph stream eventWait. stream: " << stream_dev_ctx
              << " wait for cuda graph stream: " << dev_ctx;
    }
  }
  if (old_value) {
    FLAGS_use_stream_safe_cuda_allocator = true;
  }
  AddResetCallbackIfCapturingCUDAGraph([pool_id] {
    memory::allocation::AllocatorFacade::Instance().RemoveMemoryPoolOfCUDAGraph(
        pool_id);
  });
}

std::unique_ptr<CUDAGraph> EndCUDAGraphCapture() {
  // auto* dev_ctx = CUDAGraph::CapturingDeviceContext();
  //  auto place = CUDAGraph::CapturingPlace();
  //  auto* mutable_dev_ctx = phi::DeviceContextPool::Instance().Get(place);
  //  auto* dev_ctx = reinterpret_cast<phi::GPUContext*>(mutable_dev_ctx);
  phi::DeviceContext* mutable_dev_ctx;
  auto place = CUDAGraph::CapturingPlace();
  bool create_cuda_graph_stream = CUDAGraph::CreateCUDAGraphStream();
  int64_t pool_id = CUDAGraph::CapturingPoolID();
  if (create_cuda_graph_stream) {
    mutable_dev_ctx = phi::backends::gpu::CUDAGraphContextManager::Instance()
                          .Get(pool_id, place, 0)
                          .get()
                          .get();
    auto* cuda_graph_dev_ctx =
        reinterpret_cast<phi::GPUContext*>(mutable_dev_ctx);
    auto dev_ctxs = phi::backends::gpu::CUDAGraphContextManager::Instance()
                        .GetAllDeviceContexts();
    VLOG(4) << "yoki2";
    for (auto iter = dev_ctxs.begin(); iter != dev_ctxs.end(); ++iter) {
      VLOG(4) << "yoki3";
      auto* stream_dev_ctx = reinterpret_cast<phi::GPUContext*>(*iter);
      VLOG(4) << "yoki4: stream_dev_ctx: " << stream_dev_ctx;
      std::shared_ptr<platform::DeviceEvent> stream_event =
          std::make_shared<platform::DeviceEvent>(
              stream_dev_ctx->GetPlace(), platform::GenerateDeviceEventFlag());
      stream_event->Record(stream_dev_ctx);
      stream_event->Wait(platform::kCUDA, cuda_graph_dev_ctx);
      VLOG(4) << "CUDA Graph stream eventWait. cuda graph stream: "
              << cuda_graph_dev_ctx << " wait for stream: " << stream_dev_ctx;
      stream_dev_ctx->cudnn_workspace_handle().ResetWorkspace();
      stream_dev_ctx->SetCUDAGraphAllocator(nullptr);
    }
    phi::backends::gpu::CUDAGraphContextManager::Instance()
        .ClearDeviceContext();
  } else {
    mutable_dev_ctx = phi::DeviceContextPool::Instance().Get(place);
  }
  auto* dev_ctx = reinterpret_cast<phi::GPUContext*>(mutable_dev_ctx);
  dev_ctx->cudnn_workspace_handle().ResetWorkspace();
  dev_ctx->SetCUDAGraphAllocator(nullptr);
  // std::unique_ptr<CUDAGraph> end_capture_graph = CUDAGraph::EndCapture();
  // if (create_cuda_graph_stream) {
  //   VLOG(4) << "yoki clear1";
  //   phi::backends::gpu::CUDAGraphContextManager::Instance().ClearCUDAGraphContext(pool_id);
  //   VLOG(4) << "yoki clear2";
  // }
  // return end_capture_graph;
  return CUDAGraph::EndCapture();
}
#endif

}  // namespace platform
}  // namespace paddle
