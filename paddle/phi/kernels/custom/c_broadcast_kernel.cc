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

#include <glog/logging.h>
#include "paddle/phi/api/backward/backward_api.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/distributed/collective/process_group.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/xccl_comm_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#ifdef PADDLE_WITH_CUSTOM_DEVICE
namespace phi {

template <typename T, typename Context>
void CBroadcastKernel(const Context& dev_ctx,
                      const DenseTensor& x_in,
                      int ring_id,
                      int root,
                      bool use_calc_stream,
                      DenseTensor* out) {
  auto x = &x_in;
  const auto& place = dev_ctx.GetPlace();
  dev_ctx.template Alloc<T>(out);
  int rid = ring_id;
  auto comm = reinterpret_cast<phi::distributed::XCCLCommContext*>(
      phi::distributed::CommContextManager::GetInstance().Get(
          std::to_string(rid)));

  std::shared_ptr<phi::stream::Stream> stream;
  if (use_calc_stream) {
    stream = dev_ctx.GetStream();
  } else {
    stream = comm->GetStream();
  }

  int numel = x->numel();
  auto dtype = x->dtype();
  if (root == comm->GetRank()) {
    phi::DeviceManager::CCLBroadcast(place.GetDeviceType(),
                                     const_cast<void*>(x->data()),
                                     numel,
                                     dtype,
                                     root,
                                     comm->GetXcclComm(),
                                     *stream);
    VLOG(3) << "rank " << comm->GetRank() << " invoke Bcast. sent "
            << x->numel();
    if (out != x) {
      phi::Copy(dev_ctx,
                *static_cast<const phi::DenseTensor*>(x),
                place,
                false,
                static_cast<phi::DenseTensor*>(out));
    }
  } else {
    phi::DeviceManager::CCLBroadcast(place.GetDeviceType(),
                                     out->data(),
                                     numel,
                                     dtype,
                                     root,
                                     comm->GetXcclComm(),
                                     *stream);
    VLOG(3) << "rank " << comm->GetRank() << " invoke Bcast. received "
            << common::product(out->dims());
  }
  out->set_lod(x->lod());
}
}  // namespace phi

PD_REGISTER_KERNEL(c_broadcast,
                   Custom,
                   ALL_LAYOUT,
                   phi::CBroadcastKernel,
                   float,
                   double,
                   int32_t,
                   int64_t,
                   phi::dtype::float16) {}
#endif
