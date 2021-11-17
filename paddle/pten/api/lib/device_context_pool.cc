//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pten/api/lib/device_context_pool.h"

#include <memory>
#include <set>
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/allocator.h"
#include "paddle/pten/core/backends/gpu/cuda_context.h"
#include "paddle/pten/core/context.h"
#include "paddle/pten/core/device_context.h"

namespace paddle {
namespace experimental {

using CPUPlace = paddle::platform::CPUPlace;
using CUDAPlace = paddle::platform::CUDAPlace;

DeviceContextPool* DeviceContextPool::pool = nullptr;

pten::DeviceContext* DeviceContextPool::Get(const Place& place) {
  VLOG(6) << "DeviceContextPool Get: " << place;
  auto it = device_contexts_.find(place);
  if (it == device_contexts_.end()) {
    PADDLE_THROW(paddle::platform::errors::Unimplemented(
        "Place %s is not supported. Please check that your paddle compiles "
        "with WITH_GPU, WITH_XPU or WITH_ASCEND_CL option or check that "
        "your train process set the correct device id if you use Executor.",
        place));
  }
  return it->second.get().get();
}

template <typename DevCtx, typename PlaceType>
inline void EmplaceDeviceContext(
    std::map<Place, std::shared_future<std::unique_ptr<pten::DeviceContext>>>*
        map_ptr,
    std::map<Place, std::unique_ptr<pten::Allocator>>* alloc_ptr,
    Place p) {
  using PtrType = std::unique_ptr<pten::DeviceContext>;
  LOG(INFO) << "explace device context " << p;
  map_ptr->emplace(p, std::async(std::launch::deferred, [=] {
                     // lazy evaluation. i.e., only create device context at
                     // first `Get`
                     return PtrType(new DevCtx(BOOST_GET_CONST(PlaceType, p)));
                   }));
}

template <>
inline void EmplaceDeviceContext<pten::CPUContext, CPUPlace>(
    std::map<Place, std::shared_future<std::unique_ptr<pten::DeviceContext>>>*
        map_ptr,
    std::map<Place, std::unique_ptr<pten::Allocator>>* alloc_ptr,
    Place p) {
  using PtrType = std::unique_ptr<pten::DeviceContext>;
  map_ptr->emplace(
      p, std::async(std::launch::deferred, [=] {
        // lazy evaluation. i.e., only create device context at
        // first `Get`
        auto ctx = PtrType(new pten::CPUContext(BOOST_GET_CONST(CPUPlace, p)));
        ctx->SetAllocator(alloc_ptr->at(BOOST_GET_CONST(CPUPlace, p)).get());
        return ctx;
      }));
}

template <>
inline void EmplaceDeviceContext<pten::CUDAContext, CUDAPlace>(
    std::map<Place, std::shared_future<std::unique_ptr<pten::DeviceContext>>>*
        map_ptr,
    std::map<Place, std::unique_ptr<pten::Allocator>>* alloc_ptr,
    Place p) {
  // using PtrType = std::unique_ptr<pten::DeviceContext>;
  map_ptr->emplace(
      p, std::async(std::launch::deferred, [=] {
        // lazy evaluation. i.e., only create device context at
        // first `Get`
        auto ctx = std::unique_ptr<pten::DeviceContext>(
            reinterpret_cast<pten::DeviceContext*>(
                new pten::CUDAContext(BOOST_GET_CONST(CUDAPlace, p))));
        ctx->SetAllocator(alloc_ptr->at(BOOST_GET_CONST(CUDAPlace, p)).get());
        auto* ptr = dynamic_cast<pten::CUDAContext*>(ctx.get());
        auto& pool = paddle::platform::DeviceContextPool::Instance();
        auto device_ctx = pool.GetByPlace(BOOST_GET_CONST(CUDAPlace, p));

        ptr->SetCUDAMaxGridDimX(device_ctx->GetCUDAMaxGridDimSize().x);
        ptr->SetCUDAMaxGridDimY(device_ctx->GetCUDAMaxGridDimSize().y);
        ptr->SetCUDAMaxGridDimZ(device_ctx->GetCUDAMaxGridDimSize().z);
        ptr->SetSMCount(device_ctx->GetSMCount());
        ptr->SetTensorCoreAvailable(device_ctx->tensor_core_available());
        ptr->SetComputeCapability(device_ctx->GetComputeCapability());
        ptr->SetMaxThreadsPerBlock(device_ctx->GetMaxThreadsPerBlock());

        // need to set 3 cublas handle?
        ptr->SetCublasHandle(device_ctx->cublas_handle());
        //  device_ctx->cublas_handle()

        // Fluid now only support one stream.
        ptr->SetStream(device_ctx->stream());
        ptr->SetHostToDeviceStream(device_ctx->stream());
        ptr->SetDeviceToHostStream(device_ctx->stream());

#ifdef PADDLE_WITH_CUDNN
        ptr->SetCudnnHandle(device_ctx->cudnn_handle());
#endif

#ifdef PADDLE_WITH_NCCL
        ptr->SetNcclComm(device_ctx->nccl_comm());
#endif

#ifdef PADDLE_WITH_EIGEN
        ptr->SetEigenDevice(device_ctx->eigen_device());
#endif
// device_ctx->cusolver_dn_handle()
// device_ctx->cudnn_workspace_handle()

#ifdef PADDLE_WITH_EIGEN
        ptr->SetEigenDevice(device_ctx->eigen_device());
#endif

        return ctx;
      }));
}

DeviceContextPool::DeviceContextPool(const std::vector<Place>& places) {
  PADDLE_ENFORCE_GT(places.size(),
                    0,
                    paddle::platform::errors::InvalidArgument(
                        "The number of platform places should "
                        "be larger than 0. But received %d.",
                        places.size()));
  std::set<Place> set;
  for (auto& p : places) {
    set.insert(p);
  }
  for (auto& p : set) {
    if (paddle::platform::is_cpu_place(p)) {
      CPUPlace place = BOOST_GET_CONST(CPUPlace, p);
      allocators_.emplace(place,
                          new paddle::experimental::DefaultAllocator(place));
      // #ifdef PADDLE_WITH_MKLDNN
      // EmplaceDeviceContext<MKLDNNDeviceContext, CPUPlace>(&device_contexts_,
      // p);
      // #else
      EmplaceDeviceContext<pten::CPUContext, CPUPlace>(
          &device_contexts_, &allocators_, p);
      // #endif
    } else if (paddle::platform::is_gpu_place(p)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      CUDAPlace place = BOOST_GET_CONST(CUDAPlace, p);
      allocators_.emplace(place,
                          new paddle::experimental::DefaultAllocator(place));
      EmplaceDeviceContext<pten::CUDAContext, CUDAPlace>(
          &device_contexts_, &allocators_, p);
#else
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "CUDAPlace is not supported. Please "
          "re-compile with WITH_GPU option."));
#endif
    } else if (paddle::platform::is_cuda_pinned_place(p)) {
      // #if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      //       EmplaceDeviceContext<CUDAPinnedDeviceContext, CUDAPinnedPlace>(
      //           &device_contexts_, p);
      // #else
      //       PADDLE_THROW(paddle::platform::errors::Unimplemented(
      //           "CUDAPlace is not supported. Please re-compile with WITH_GPU
      //           "
      //           "option."));
      // #endif
    } else if (paddle::platform::is_xpu_place(p)) {
      // #ifdef PADDLE_WITH_XPU
      //       EmplaceDeviceContext<XPUDeviceContext,
      //       XPUPlace>(&device_contexts_, p);
      // #else
      //       PADDLE_THROW(
      //           paddle::platform::errors::Unimplemented("XPUPlace is not
      //           supported. Please "
      //                                           "re-compile with WITH_XPU
      //                                           option."));
      // #endif
    } else if (paddle::platform::is_npu_place(p)) {
      // #ifdef PADDLE_WITH_ASCEND_CL
      //       EmplaceDeviceContext<NPUDeviceContext,
      //       NPUPlace>(&device_contexts_, p);
      // #else
      //       PADDLE_THROW(paddle::platform::errors::Unimplemented(
      //           "NPUPlace is not supported. Please "
      //           "re-compile with WITH_ASCEND_CL option."));
      // #endif
    } else if (paddle::platform::is_npu_pinned_place(p)) {
      // #ifdef PADDLE_WITH_ASCEND_CL
      //       EmplaceDeviceContext<NPUPinnedDeviceContext, NPUPinnedPlace>(
      //           &device_contexts_, p);
      // #else
      //       PADDLE_THROW(paddle::platform::errors::Unimplemented(
      //           "NPUPinnedPlace is not supported. Please re-compile with "
      //           "WITH_ASCEND_CL "
      //           "option."));
      // #endif
    }
  }
}

}  // namespace experimental
}  // namespace paddle
