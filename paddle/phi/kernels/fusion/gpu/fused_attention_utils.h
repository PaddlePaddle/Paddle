// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif
#include "paddle/common/errors.h"
#include "paddle/phi/core/platform/collective_helper.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {
namespace fusion {

template <typename T>
static void AllReduce(phi::DenseTensor &tensor,  // NOLINT
                      const int ring_id,
                      const phi::GPUContext &dev_ctx) {
  if (ring_id == -1) return;
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  auto dtype = phi::ToNCCLDataType(tensor.dtype());
  int64_t numel = tensor.numel();
  const void *sendbuff = tensor.data<T>();
  auto place = dev_ctx.GetPlace();
  void *recvbuff =
      dev_ctx.template Alloc<T>(&tensor, tensor.numel() * sizeof(T));

  gpuStream_t stream = dev_ctx.stream();
  paddle::platform::NCCLComm *comm = nullptr;
  phi::distributed::NCCLCommContext *comm_ctx = nullptr;
  comm_ctx = static_cast<phi::distributed::NCCLCommContext *>(
      dev_ctx.GetCommContext());
  if (comm_ctx) {
    comm_ctx->AllReduce(&tensor, tensor, ncclSum, stream);
  } else {
    comm = paddle::platform::NCCLCommContext::Instance().Get(ring_id, place);
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclAllReduce(
        sendbuff, recvbuff, numel, dtype, ncclSum, comm->comm(), stream));
  }
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "PaddlePaddle should compile with NCCL or RCCL when used tensor model "
      "parallel op."));
#endif
}

template <typename T>
static void AllReduce(phi::DenseTensor &tensor,  // NOLINT
                      const int ring_id,
                      const int count UNUSED,
                      const phi::GPUContext &ctx) {
  AllReduce<T>(tensor, ring_id, ctx);
}

}  // namespace fusion
}  // namespace phi
