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

#include "paddle/common/ddim.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_array.h"

namespace phi {

#if (defined(PADDLE_WITH_RCCL) || defined(PADDLE_WITH_NCCL)) && \
        NCCL_VERSION_CODE >= 2703 ||                            \
    defined(PADDLE_WITH_XPU_BKCL)
template <typename Context, typename CommContext, typename StreamType>
void send_shape_info(const Context& dev_ctx,
                     const DenseTensor& x,
                     CommContext* comm_ctx,
                     int peer,
                     StreamType stream) {
#if (defined(PADDLE_WITH_RCCL) || defined(PADDLE_WITH_NCCL)) && \
    NCCL_VERSION_CODE >= 2703
  PADDLE_ENFORCE_EQ((stream != nullptr && comm_ctx != nullptr),
                    true,
                    errors::InvalidArgument(
                        "NCCLComm and Stream should be provided if use NCCL "
                        "to send the shape info."));
#elif defined(PADDLE_WITH_XPU_BKCL)
  PADDLE_ENFORCE_EQ(
      (comm_ctx != nullptr),
      true,
      errors::InvalidArgument("BKCLComm should be provided if use BKCL "
                              "to send the shape info."));
#endif
  paddle::DataType shape_dtype = paddle::DataType::INT32;
  auto dims = x.dims();
  int shape_size = dims.size();

  // step1: send the shape size
  phi::DenseTensor cpu_shape_size_tensor(shape_dtype);
  cpu_shape_size_tensor.Resize({1});
  dev_ctx.HostAlloc(&cpu_shape_size_tensor, shape_dtype);
  auto* cpu_data = cpu_shape_size_tensor.data<int>();
  cpu_data[0] = shape_size;

  // copy the shape size tensor to gpu/xpu and send
  phi::DenseTensor* shape_size_tensor = new phi::DenseTensor(shape_dtype);
  shape_size_tensor->Resize({1});
  dev_ctx.Alloc(shape_size_tensor, shape_dtype);
  const auto& cpu_place = phi::CPUPlace();
  memory_utils::Copy(dev_ctx.GetPlace(),
                     shape_size_tensor->data(),
                     cpu_place,
                     cpu_shape_size_tensor.data(),
                     cpu_shape_size_tensor.numel() * sizeof(int),
                     stream);

  comm_ctx->Send(*shape_size_tensor, shape_size_tensor->numel(), peer, stream);

  // step2: send the shape
  phi::DenseTensor cpu_shape_tensor(shape_dtype);
  cpu_shape_tensor.Resize({shape_size});
  dev_ctx.HostAlloc(&cpu_shape_tensor, shape_dtype);
  auto* cpu_shape_data = cpu_shape_tensor.data<int>();
  for (int i = 0; i < shape_size; ++i) {
    cpu_shape_data[i] = dims[i];
  }

  // copy the shape tensor to gpu and send
  phi::DenseTensor* shape_tensor = new phi::DenseTensor(shape_dtype);
  shape_tensor->Resize({shape_size});
  dev_ctx.Alloc(shape_tensor, shape_dtype);
  memory_utils::Copy(dev_ctx.GetPlace(),
                     shape_tensor->data(),
                     cpu_place,
                     cpu_shape_tensor.data(),
                     cpu_shape_tensor.numel() * sizeof(int),
                     stream);
  comm_ctx->Send(*shape_tensor, shape_tensor->numel(), peer, stream);
}
#endif

#if (defined(PADDLE_WITH_RCCL) || defined(PADDLE_WITH_NCCL)) && \
        NCCL_VERSION_CODE >= 2703 ||                            \
    defined(PADDLE_WITH_XPU_BKCL)
template <typename Context, typename CommContext, typename StreamType>
DDim recv_shape_info(const Context& dev_ctx,
                     phi::DenseTensor* out,
                     CommContext* comm_ctx,
                     int peer) {
  StreamType stream = dev_ctx.stream();
#if (defined(PADDLE_WITH_RCCL) || defined(PADDLE_WITH_NCCL)) && \
    NCCL_VERSION_CODE >= 2703
  PADDLE_ENFORCE_EQ((stream != nullptr && comm_ctx != nullptr),
                    true,
                    errors::InvalidArgument(
                        "NCCLComm and Stream should be provided if use NCCL "
                        "to send the shape info."));
#elif defined(PADDLE_WITH_XPU_BKCL)
  PADDLE_ENFORCE_EQ(
      (comm_ctx != nullptr),
      true,
      errors::InvalidArgument("BKCLComm should be provided if use BKCL "
                              "to send the shape info."));
#endif
  paddle::DataType shape_dtype = paddle::DataType::INT32;

  // phi::DenseTensor shape_size_tensortensor(shape_dtype);
  phi::DenseTensor* shape_size_tensortensor = new phi::DenseTensor(shape_dtype);
  shape_size_tensortensor->Resize({1});
  dev_ctx.Alloc(shape_size_tensortensor, shape_dtype);
  comm_ctx->Recv(
      shape_size_tensortensor, shape_size_tensortensor->numel(), peer, stream);

  // copy the shape size tensor to cpu
  phi::DenseTensor* cpu_shape_size_tensor = new phi::DenseTensor(shape_dtype);
  cpu_shape_size_tensor->Resize({1});
  dev_ctx.HostAlloc(cpu_shape_size_tensor, shape_dtype);

  memory_utils::Copy(phi::CPUPlace(),
                     cpu_shape_size_tensor->data(),
                     dev_ctx.GetPlace(),
                     shape_size_tensortensor->data(),
                     shape_size_tensortensor->numel() * sizeof(int),
                     stream);

  auto* cpu_data = cpu_shape_size_tensor->data<int>();
  int shape_size = cpu_data[0];

  // step2: send the shape
  // phi::DenseTensor shape_tensor(shape_dtype);
  phi::DenseTensor* shape_tensor = new phi::DenseTensor(shape_dtype);
  shape_tensor->Resize({shape_size});
  dev_ctx.Alloc(shape_tensor, shape_dtype);
  comm_ctx->Recv(shape_tensor, shape_tensor->numel(), peer, stream);

  // copy the shape tensor to cpu
  phi::DenseTensor* cpu_shape_tensor = new phi::DenseTensor(shape_dtype);
  cpu_shape_tensor->Resize({shape_size});
  dev_ctx.HostAlloc(cpu_shape_tensor, shape_dtype);

  memory_utils::Copy(phi::CPUPlace(),
                     cpu_shape_tensor->data(),
                     dev_ctx.GetPlace(),
                     shape_tensor->data(),
                     shape_tensor->numel() * sizeof(int),
                     stream);
  auto* cpu_shape_data = cpu_shape_tensor->data<int>();
  std::vector<int> all_shape;
  for (int i = 0; i < shape_size; ++i) {
    all_shape.emplace_back(cpu_shape_data[i]);
  }
  DDim new_dim;
  new_dim = new_dim.reshape(all_shape);

  return new_dim;
}

template <typename Context, typename CommContext>
CommContext* GetCommContext(const Context& dev_ctx, int peer) {
  PADDLE_ENFORCE_GE(
      peer,
      0,
      errors::InvalidArgument("The peer (%d) for send op must be non-negative.",
                              peer));

  auto comm_ctx = static_cast<CommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(
      comm_ctx,
      nullptr,
      errors::Unavailable(
          "NCCLCommContext/BKCLCommContext is nullptr, collective op should "
          "has ring_id attr."));

  PADDLE_ENFORCE_LT(
      peer,
      comm_ctx->GetSize(),
      errors::InvalidArgument("The value of peer (%d) you set must "
                              "be less than comm->nranks (%d).",
                              peer,
                              comm_ctx->GetSize()));
  return comm_ctx;
}
#endif

}  // namespace phi
