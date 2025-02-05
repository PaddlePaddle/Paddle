// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/p_recv_kernel.h"

#include "glog/logging.h"

#include "paddle/common/ddim.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_NCCL) || \
    defined(PADDLE_WITH_RCCL) && NCCL_VERSION_CODE >= 2703
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif

namespace phi {

#if (defined(PADDLE_WITH_RCCL) || defined(PADDLE_WITH_NCCL)) && \
    NCCL_VERSION_CODE >= 2703
template <typename Context>
DDim recv_shape_info(const Context& dev_ctx,
                     phi::DenseTensor* out,
                     distributed::NCCLCommContext* comm_ctx,
                     int peer) {
  gpuStream_t stream = dev_ctx.stream();
  PADDLE_ENFORCE_EQ((stream != nullptr && comm_ctx != nullptr),
                    true,
                    errors::InvalidArgument(
                        "NCCLComm and Stream should be provided if use NCCL "
                        "to send the shape info."));
  paddle::DataType shape_dtype = paddle::DataType::INT32;
  ncclDataType_t nccl_dtype = ncclInt;

  // phi::DenseTensor gpu_shape_size_tensor(shape_dtype);
  phi::DenseTensor* gpu_shape_size_tensor = new phi::DenseTensor(shape_dtype);
  gpu_shape_size_tensor->Resize({1});
  dev_ctx.Alloc(gpu_shape_size_tensor, shape_dtype);
  comm_ctx->Recv(
      gpu_shape_size_tensor, gpu_shape_size_tensor->numel(), peer, stream);

  // copy the shape size tensor to cpu
  phi::DenseTensor* cpu_shape_size_tensor = new phi::DenseTensor(shape_dtype);
  cpu_shape_size_tensor->Resize({1});
  dev_ctx.HostAlloc(cpu_shape_size_tensor, shape_dtype);

  memory_utils::Copy(phi::CPUPlace(),
                     cpu_shape_size_tensor->data(),
                     dev_ctx.GetPlace(),
                     gpu_shape_size_tensor->data(),
                     gpu_shape_size_tensor->numel() * sizeof(int),
                     stream);

  auto* cpu_data = cpu_shape_size_tensor->data<int>();
  int shape_size = cpu_data[0];
  VLOG(3) << "recv the shape size: " << shape_size << " from peer: " << peer;

  // step2: send the shape
  // phi::DenseTensor gpu_shape_tensor(shape_dtype);
  phi::DenseTensor* gpu_shape_tensor = new phi::DenseTensor(shape_dtype);
  gpu_shape_tensor->Resize({shape_size});
  dev_ctx.Alloc(gpu_shape_tensor, shape_dtype);
  comm_ctx->Recv(gpu_shape_tensor, gpu_shape_tensor->numel(), peer, stream);

  // copy the shape tensor to cpu
  phi::DenseTensor* cpu_shape_tensor = new phi::DenseTensor(shape_dtype);
  cpu_shape_tensor->Resize({shape_size});
  dev_ctx.HostAlloc(cpu_shape_tensor, shape_dtype);

  memory_utils::Copy(phi::CPUPlace(),
                     cpu_shape_tensor->data(),
                     dev_ctx.GetPlace(),
                     gpu_shape_tensor->data(),
                     gpu_shape_tensor->numel() * sizeof(int),
                     stream);
  auto* cpu_shape_data = cpu_shape_tensor->data<int>();
  std::vector<int> all_shape;
  for (int i = 0; i < shape_size; ++i) {
    all_shape.emplace_back(cpu_shape_data[i]);
  }
  DDim new_dim;
  new_dim = new_dim.reshape(all_shape);
  VLOG(3) << "recv the shape: (" << new_dim << ") from peer";

  return new_dim;
}

template <typename Context>
distributed::NCCLCommContext* GetCommContext(const Context& dev_ctx, int peer) {
  PADDLE_ENFORCE_GE(
      peer,
      0,
      errors::InvalidArgument("The peer (%d) for send op must be non-negative.",
                              peer));

  auto comm_ctx =
      static_cast<distributed::NCCLCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(
      comm_ctx,
      nullptr,
      errors::Unavailable("NCCLCommContext is nullptr, collective op should "
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

template <typename T, typename Context>
void PRecvKernel(const Context& dev_ctx,
                 int peer,
                 DataType dtype,
                 const std::vector<int>& out_shape,
                 bool dynamic_shape,
                 DenseTensor* out) {
#if defined(PADDLE_WITH_NCCL) || \
    defined(PADDLE_WITH_RCCL) && NCCL_VERSION_CODE >= 2703

  auto comm_ctx = GetCommContext(dev_ctx, peer);
  gpuStream_t stream = dev_ctx.stream();

  // auto data_type = phi::TransToPhiDataType(dtype);
  if (dynamic_shape) {
    DDim new_dim = recv_shape_info<Context>(dev_ctx, out, comm_ctx, peer);
    out->Resize(new_dim);
  }
  dev_ctx.Alloc(out, dtype);
  comm_ctx->Recv(out, out->numel(), peer, stream);
#else
  PADDLE_THROW(
      errors::PreconditionNotMet("PaddlePaddle should compile with GPU."
                                 "and NCCL version >= 2.7.3 is needed."));
#endif
}

template <typename T, typename Context>
void PRecvArrayKernel(const Context& dev_ctx,
                      int peer,
                      DataType dtype,
                      const std::vector<int>& out_shape,
                      TensorArray* out_array) {
#if defined(PADDLE_WITH_NCCL) || \
    defined(PADDLE_WITH_RCCL) && NCCL_VERSION_CODE >= 2703

  auto comm_ctx = GetCommContext(dev_ctx, peer);
  gpuStream_t stream = dev_ctx.stream();
  for (size_t idx = 0; idx < out_shape.size(); ++idx) {
    VLOG(3) << "DenseTensorArray: idx(" << idx << ")";
    auto out = out_array->at(idx);
    auto out_dims = out.dims();
    dev_ctx.Alloc(&out, dtype);
    comm_ctx->Recv(&out, out.numel(), peer, stream);
    VLOG(3) << "rank " << comm_ctx->GetRank() << " recv "
            << common::product(out_dims) << " from " << peer;
  }
#else
  PADDLE_THROW(
      errors::PreconditionNotMet("PaddlePaddle should compile with GPU."
                                 "and NCCL version >= 2.7.3 is needed."));
#endif
}

}  // namespace phi

#if NCCL_VERSION_CODE >= 21000
PD_REGISTER_KERNEL(p_recv,
                   GPU,
                   ALL_LAYOUT,
                   phi::PRecvKernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int64_t,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(p_recv_array,
                   GPU,
                   ALL_LAYOUT,
                   phi::PRecvArrayKernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int64_t,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(p_recv,
                   GPU,
                   ALL_LAYOUT,
                   phi::PRecvKernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int64_t,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(p_recv_array,
                   GPU,
                   ALL_LAYOUT,
                   phi::PRecvArrayKernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int64_t,
                   phi::dtype::float16) {}
#endif
