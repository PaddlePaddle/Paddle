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

#pragma once

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/distributed/collective/process_group.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/xccl_comm_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#ifdef PADDLE_WITH_CUSTOM_DEVICE
namespace phi {

template <typename T, typename Context, phi::ccl::CCLReduceOp red_type>
void CAllReduceKernel(const Context& dev_ctx,
                      const DenseTensor& x_in,
                      int ring_id,
                      bool use_calc_stream,
                      bool use_model_parallel,
                      DenseTensor* out) {
  auto in = &x_in;
  int rid = ring_id;

  auto place = dev_ctx.GetPlace();
  auto dtype = in->dtype();
  int64_t numel = in->numel();
  const void* sendbuff = in->data<T>();
  out->Resize(in->dims());
  void* recvbuff = dev_ctx.template Alloc<T>(out);

  auto map = phi::distributed::ProcessGroupMapFromGid::getInstance();
  if (map->has(rid)) {
    // Use ProcessGroup
    phi::distributed::ProcessGroup* pg = map->get(rid);
    std::vector<phi::DenseTensor> in_tensor;
    std::vector<phi::DenseTensor> out_tensor;
    in_tensor.push_back(*in);
    out_tensor.push_back(*out);

    phi::distributed::AllreduceOptions opts;
    switch (red_type) {
      case phi::ccl::CCLReduceOp::SUM:
        opts.reduce_op = phi::distributed::ReduceOp::SUM;
        break;

      case phi::ccl::CCLReduceOp::MAX:
        opts.reduce_op = phi::distributed::ReduceOp::MAX;
        break;

      case phi::ccl::CCLReduceOp::MIN:
        opts.reduce_op = phi::distributed::ReduceOp::MIN;
        break;

      case phi::ccl::CCLReduceOp::PRODUCT:
        opts.reduce_op = phi::distributed::ReduceOp::PRODUCT;
        break;

      default:
        PADDLE_THROW(common::errors::InvalidArgument("Invalid reduce type: %d",
                                                     red_type));
    }

    auto task = pg->AllReduce(in_tensor, out_tensor, opts);
    task->Wait();
    return;
  }

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
                                   dtype,
                                   red_type,
                                   comm->GetXcclComm(),
                                   *stream);
}
}  // namespace phi

#endif
