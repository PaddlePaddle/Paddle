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

#include "paddle/pten/core/device_context_pool.h"

#include <set>

namespace pten {

DeviceContextPool* DeviceContextPool::pool = nullptr;

DeviceContext* DeviceContextPool::Get(const Place& place) {
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
    std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>*
        map_ptr,
    Place p) {
  using PtrType = std::unique_ptr<DeviceContext>;
  LOG(INFO) << "explace device context " << p;
  map_ptr->emplace(p, std::async(std::launch::deferred, [=] {
                     // lazy evaluation. i.e., only create device context at
                     // first `Get`
                     return PtrType(new DevCtx(BOOST_GET_CONST(PlaceType, p)));
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
      LOG(INFO) << "is_cpu_place, insert....";
      // #ifdef PADDLE_WITH_MKLDNN
      // EmplaceDeviceContext<MKLDNNDeviceContext, CPUPlace>(&device_contexts_,
      // p);
      // #else
      EmplaceDeviceContext<CPUContext, CPUPlace>(&device_contexts_, p);
      // #endif
    } else if (paddle::platform::is_gpu_place(p)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      EmplaceDeviceContext<CUDAContext, CUDAPlace>(&device_contexts_, p);
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

}  // namespace pten
