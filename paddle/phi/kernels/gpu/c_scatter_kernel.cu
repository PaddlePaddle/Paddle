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

#include "glog/logging.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void CScatterOpCUDAKernel(const Context& dev_ctx,
                          const DenseTensor& input,
                          int ring_id,
                          int root,
                          int nranks,
                          bool use_calc_stream,
                          DenseTensor* out) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  auto x = &input;
  int numel = x->numel();
  ncclDataType_t dtype = phi::ToNCCLDataType(x->dtype());

  int root_id = root;
  auto place = dev_ctx.GetPlace();
  gpuStream_t stream = nullptr;
  phi::distributed::NCCLCommContext* comm_ctx = nullptr;
  PADDLE_ENFORCE_GE(
      root_id,
      0,
      common::errors::InvalidArgument(
          "The root_id (%d) for c_scatter_op must be non-negative.", root_id));
  PADDLE_ENFORCE_GE(
      ring_id,
      0,
      common::errors::InvalidArgument(
          "The ring_id (%d) for c_scatter_op must be non-negative.", ring_id));

  comm_ctx =
      static_cast<phi::distributed::NCCLCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(comm_ctx,
                    nullptr,
                    common::errors::Unavailable(
                        "NCCLCommContext is nullptr, collective op should "
                        "has ring_id attr."));
  PADDLE_ENFORCE_EQ(nranks,
                    comm_ctx->GetSize(),
                    common::errors::InvalidArgument(
                        "The number of ranks (%d) you set of must "
                        "be equal to comm_ctx->GetSize() (%d).",
                        nranks,
                        comm_ctx->GetSize()));

  stream = comm_ctx->GetStream();
  VLOG(3) << "new comm_context_manager has ring_id " << ring_id;

  if (use_calc_stream) {
    // should ExecutionContext for calc stream.
    stream = dev_ctx.stream();
  }

  phi::DDim x_dims = x->dims();
  phi::DDim out_dims(x_dims);
  phi::DenseTensor temp;
  temp.Resize(out_dims);
  auto out_ptr = dev_ctx.template Alloc<T>(&temp);

  if (root_id == comm_ctx->GetRank()) {
    comm_ctx->Broadcast(const_cast<phi::DenseTensor*>(x), *x, root_id, stream);
    phi::Copy(dev_ctx,
              *static_cast<const phi::DenseTensor*>(x),
              place,
              false,
              static_cast<phi::DenseTensor*>(&temp));
  } else {
    comm_ctx->Broadcast(&temp, temp, root_id, stream);
  }

  out_dims[0] = out_dims[0] / nranks;
  auto start_index = out_dims[0] * comm_ctx->GetRank();
  auto end_index = start_index + out_dims[0];
  temp = temp.Slice(start_index, end_index);
  temp.Resize(out_dims);
  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);
  phi::Copy(dev_ctx,
            *static_cast<const phi::DenseTensor*>(&temp),
            place,
            true,
            static_cast<phi::DenseTensor*>(out));
  out->Resize(out_dims);
#else
  PADDLE_ENFORCE_EQ(
      true,
      false,
      common::errors::Unavailable("PaddlePaddle should compile with GPU."));
#endif
}
}  // namespace phi

PD_REGISTER_KERNEL(c_scatter,
                   GPU,
                   ALL_LAYOUT,
                   phi::CScatterOpCUDAKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
