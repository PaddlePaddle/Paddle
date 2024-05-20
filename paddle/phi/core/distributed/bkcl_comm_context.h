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
#pragma once

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/distributed/comm_context.h"

namespace phi {
class DenseTensor;
namespace distributed {

class BKCLCommContext final : public CommContext {
 public:
  BKCLCommContext(int rank, int size, BKCLUniqueId BKCL_id);
  ~BKCLCommContext() override = default;

  BKCLContext_t GetBKCLComm();

  XPUStream GetStream();

  XPUEvent GetComputeEvent();

  void SetComputeEvent(
      std::shared_ptr<std::remove_pointer<XPUEvent>::type>&& compute_event);

  XPUEvent GetCommEvent();

  void SetCommEvent(
      std::shared_ptr<std::remove_pointer<XPUEvent>::type>&& comm_event);

  phi::XPUContext* GetDevContext();

  void SetDevContext(std::unique_ptr<phi::XPUContext>&& dev_ctx);

  void Broadcast(phi::DenseTensor* out_tensor,
                 const phi::DenseTensor& in_tensor,
                 int root,
                 XPUStream stream);

  void Send(const phi::DenseTensor& in_tensor,
            const int64_t& count,
            const int& peer,
            XPUStream stream);

  void Recv(phi::DenseTensor* out_tensor,
            const int64_t& count,
            const int& peer,
            XPUStream stream);

  void ReduceScatter(phi::DenseTensor* out_tensor,
                     const phi::DenseTensor& in_tensor,
                     BKCLOp reduce_type,
                     XPUStream stream);

  void AllGather(phi::DenseTensor* out_tensor,
                 const phi::DenseTensor& in_tensor,
                 XPUStream stream);

  void AllReduce(phi::DenseTensor* out_tensor,
                 const phi::DenseTensor& in_tensor,
                 BKCLOp reduce_type,
                 XPUStream stream);

  void Reduce(phi::DenseTensor* out_tensor,
              const phi::DenseTensor& in_tensor,
              BKCLOp reduce_type,
              int root,
              XPUStream stream);

  void GroupStart();

  void GroupEnd();

 private:
  DISABLE_COPY_AND_ASSIGN(BKCLCommContext);

  BKCLContext_t bkcl_comm_;

  std::unique_ptr<phi::XPUContext> dev_ctx_;

  // used for comm wait compute, compute_stream-->event-->comm_stream
  std::shared_ptr<std::remove_pointer<XPUEvent>::type> compute_event_;

  // used for compute wait comm, comm_stream-->event-->compute_stream
  std::shared_ptr<std::remove_pointer<XPUEvent>::type> comm_event_;
};

}  // namespace distributed
}  // namespace phi
