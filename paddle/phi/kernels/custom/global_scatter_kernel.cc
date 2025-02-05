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
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/distributed/collective/process_group.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/xccl_comm_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#ifdef PADDLE_WITH_CUSTOM_DEVICE
namespace phi {

template <typename T, typename Context>
void GlobalScatterKernel(const Context& dev_ctx,
                         const DenseTensor& x_in,
                         const DenseTensor& local_count_in,
                         const DenseTensor& global_count_in,
                         int ring_id,
                         bool use_calc_stream,
                         DenseTensor* out) {
  auto x = &x_in;
  auto local_count = &local_count_in;
  auto global_count = &global_count_in;

  const int rid = ring_id;
  auto place = dev_ctx.GetPlace();

  PADDLE_ENFORCE_EQ(
      local_count->dtype(),
      phi::DataType::INT64,
      common::errors::InvalidArgument("Please use int64 type in local_count."));
  PADDLE_ENFORCE_EQ(global_count->dtype(),
                    phi::DataType::INT64,
                    common::errors::InvalidArgument(
                        "Please use int64 type in global_count."));

  auto map = distributed::ProcessGroupMapFromGid::getInstance();
  const int64_t* cpu_local_count_data;
  const int64_t* cpu_global_count_data;
  phi::DenseTensor cpu_local_count;
  if (local_count->place().GetType() == phi::AllocationType::CPU) {
    cpu_local_count_data = local_count->data<int64_t>();
  } else {
    phi::Copy(dev_ctx, *local_count, phi::CPUPlace(), true, &cpu_local_count);
    cpu_local_count_data = cpu_local_count.data<int64_t>();
  }
  auto global_count_len = 0;
  phi::DenseTensor cpu_global_count;
  if (global_count->place().GetType() == phi::AllocationType::CPU) {
    cpu_global_count_data = global_count->data<int64_t>();
    global_count_len = global_count->numel();
  } else {
    phi::Copy(dev_ctx, *global_count, phi::CPUPlace(), true, &cpu_global_count);
    cpu_global_count_data = cpu_global_count.data<int64_t>();
    global_count_len = cpu_global_count.numel();
  }

  if (map->has(rid)) {
    distributed::ProcessGroup* pg = map->get(rid);
    auto stream =
        reinterpret_cast<phi::CustomContext*>(pg->GetDeviceContext(place))
            ->GetStream();
    int nranks = pg->GetSize();
    int rank = pg->GetRank();
    auto in_feat = x->dims()[1];
    auto n_expert = local_count->dims()[0] / nranks;
    int64_t fwd_count = 0;

    for (auto i = 0; i < global_count_len; ++i) {
      fwd_count += cpu_global_count_data[i];
    }
    phi::DDim out_dims = common::make_ddim({fwd_count, in_feat});
    int64_t* expert_ptr = new int64_t[n_expert * nranks];
    expert_ptr[0] = 0;
    auto tot_experts = n_expert * nranks;
    for (auto i = 1; i < tot_experts; ++i) {
      expert_ptr[i] = expert_ptr[i - 1] + cpu_local_count_data[i - 1];
    }

    auto recv_ptr = 0;
    out->Resize(out_dims);
    dev_ctx.template Alloc<T>(out);

    for (auto i = 0; i < n_expert; ++i) {
      for (auto j = 0; j < rank; ++j) {
        int idx = i + j * n_expert;
        if (cpu_global_count_data[idx]) {
          pg->Recv(out,
                   j,
                   recv_ptr * in_feat,
                   cpu_global_count_data[idx] * in_feat,
                   /*sync_op*/ true);
          recv_ptr += cpu_global_count_data[idx];
        }
      }
      for (auto j = 0; j < nranks; ++j) {
        if (j != rank) {
          int idx = i + j * n_expert;
          if (cpu_local_count_data[idx]) {
            phi::DenseTensor tmp = *x;
            pg->Send(tmp,
                     j,
                     expert_ptr[idx] * in_feat,
                     cpu_local_count_data[idx] * in_feat,
                     /*sync_op*/ true);
          }
        }
      }
      if (cpu_local_count_data[i + rank * n_expert]) {
        phi::DeviceManager::GetDeviceWithPlace(place)->MemoryCopyD2D(
            reinterpret_cast<void*>(out->data<T>() + recv_ptr * in_feat),
            reinterpret_cast<const void*>(x->data<T>() +
                                          expert_ptr[rank] * in_feat),
            (cpu_local_count_data[rank] * in_feat) * phi::SizeOf(x->dtype()),
            stream.get());
        recv_ptr += cpu_global_count_data[rank];
      }
      for (auto j = rank + 1; j < nranks; ++j) {
        int idx = i + j * n_expert;
        if (cpu_global_count_data[idx]) {
          pg->Recv(out,
                   j,
                   recv_ptr * in_feat,
                   cpu_global_count_data[idx] * in_feat,
                   /*sync_op*/ true);
          recv_ptr += cpu_global_count_data[idx];
        }
      }
    }
  } else {
    auto comm = reinterpret_cast<phi::distributed::XCCLCommContext*>(
        phi::distributed::CommContextManager::GetInstance().Get(
            std::to_string(rid)));

    std::shared_ptr<phi::stream::Stream> stream;
    if (use_calc_stream) {
      stream = dev_ctx.GetStream();
    } else {
      stream = comm->GetStream();
    }

    int nranks = comm->GetSize();
    int rank = comm->GetRank();
    auto in_feat = x->dims()[1];
    auto n_expert = local_count->dims()[0] / nranks;
    int64_t fwd_count = 0;

    for (auto i = 0; i < global_count_len; ++i) {
      fwd_count += cpu_global_count_data[i];
    }
    phi::DDim out_dims = common::make_ddim({fwd_count, in_feat});
    int64_t* expert_ptr = new int64_t[n_expert * nranks];
    expert_ptr[0] = 0;
    auto tot_experts = n_expert * nranks;
    for (auto i = 1; i < tot_experts; ++i) {
      expert_ptr[i] = expert_ptr[i - 1] + cpu_local_count_data[i - 1];
    }

    auto recv_ptr = 0;
    auto send_buf = x->data<T>();
    out->Resize(out_dims);
    auto recv_buf = dev_ctx.template Alloc<T>(out);

    for (auto i = 0; i < n_expert; ++i) {
      for (auto j = 0; j < rank; ++j) {
        int idx = i + j * n_expert;
        if (cpu_global_count_data[idx]) {
          phi::DeviceManager::CCLRecv(
              place.GetDeviceType(),
              reinterpret_cast<void*>(recv_buf + recv_ptr * in_feat),
              cpu_global_count_data[idx] * in_feat,
              x->dtype(),
              j,
              comm->GetXcclComm(),
              *stream);
          recv_ptr += cpu_global_count_data[idx];
        }
      }
      for (auto j = 0; j < nranks; ++j) {
        if (j != rank) {
          int idx = i + j * n_expert;
          if (cpu_local_count_data[idx]) {
            phi::DeviceManager::CCLSend(
                place.GetDeviceType(),
                const_cast<void*>(reinterpret_cast<const void*>(
                    send_buf + expert_ptr[idx] * in_feat)),
                cpu_local_count_data[idx] * in_feat,
                x->dtype(),
                j,
                comm->GetXcclComm(),
                *stream);
          }
        }
      }
      if (cpu_local_count_data[i + rank * n_expert]) {
        phi::DeviceManager::GetDeviceWithPlace(place)->MemoryCopyD2D(
            reinterpret_cast<void*>(recv_buf + recv_ptr * in_feat),
            reinterpret_cast<const void*>(send_buf +
                                          expert_ptr[rank] * in_feat),
            (cpu_local_count_data[rank] * in_feat) * phi::SizeOf(x->dtype()),
            stream.get());
        recv_ptr += cpu_global_count_data[rank];
      }
      for (auto j = rank + 1; j < nranks; ++j) {
        int idx = i + j * n_expert;
        if (cpu_global_count_data[idx]) {
          phi::DeviceManager::CCLRecv(
              place.GetDeviceType(),
              reinterpret_cast<void*>(recv_buf + recv_ptr * in_feat),
              cpu_global_count_data[idx] * in_feat,
              x->dtype(),
              j,
              comm->GetXcclComm(),
              *stream);
          recv_ptr += cpu_global_count_data[idx];
        }
      }
    }
  }

  phi::DeviceManager::SynchronizeDevice(dev_ctx.GetPlace());
}
}  // namespace phi

PD_REGISTER_KERNEL(global_scatter,
                   Custom,
                   ALL_LAYOUT,
                   phi::GlobalScatterKernel,
                   float,
                   double,
                   int32_t,
                   int64_t,
                   phi::dtype::float16) {}
#endif
