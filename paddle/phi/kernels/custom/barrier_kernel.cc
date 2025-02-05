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
void BarrierKernel(const Context& dev_ctx,
                   const DenseTensor& x_in,
                   int ring_id,
                   bool use_calc_stream,
                   DenseTensor* out) {
  auto in = &x_in;

  auto place = dev_ctx.GetPlace();
  int64_t numel = in->numel();
  const void* sendbuff = in->data();
  void* recvbuff = dev_ctx.template Alloc<T>(out);
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
  phi::DeviceManager::CCLAllReduce(place.GetDeviceType(),
                                   const_cast<void*>(sendbuff),
                                   recvbuff,
                                   numel,
                                   in->dtype(),
                                   phi::ccl::CCLReduceOp::SUM,
                                   comm->GetXcclComm(),
                                   *stream);
}
}  // namespace phi

PD_REGISTER_KERNEL(barrier, Custom, ALL_LAYOUT, phi::BarrierKernel, int) {}
#endif
