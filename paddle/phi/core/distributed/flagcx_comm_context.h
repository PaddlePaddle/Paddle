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

#include "paddle/common/macros.h"
#include "paddle/phi/backends/dynload/flagcx.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/core/distributed/comm_context.h"

namespace phi {
class DenseTensor;
namespace distributed {

class FlagcxCommContext final : public CommContext {
 public:
  FlagcxCommContext(int rank, int size, flagcxHandlerGroup_t flagcx_handler);
  ~FlagcxCommContext() override = default;

  int GetFlagcxVersion();

  flagcxComm_t GetFlagcxComm();

  void Broadcast(phi::DenseTensor* out_tensor,
                 const phi::DenseTensor& in_tensor,
                 int root,
                 flagcxStream_t stream);

  void Send(const phi::DenseTensor& in_tensor,
            const int64_t& count,
            const int& peer,
            flagcxStream_t stream);

  void Recv(phi::DenseTensor* out_tensor,
            const int64_t& count,
            const int& peer,
            flagcxStream_t stream);

  void ReduceScatter(phi::DenseTensor* out_tensor,
                     const phi::DenseTensor& in_tensor,
                     flagcxRedOp_t reduce_type,
                     flagcxStream_t stream);

  void AllGather(phi::DenseTensor* out_tensor,
                 const phi::DenseTensor& in_tensor,
                 flagcxStream_t stream);

  void AllReduce(phi::DenseTensor* out_tensor,
                 const phi::DenseTensor& in_tensor,
                 flagcxRedOp_t reduce_type,
                 flagcxStream_t stream);

  void Reduce(phi::DenseTensor* out_tensor,
              const phi::DenseTensor& in_tensor,
              flagcxRedOp_t reduce_type,
              int root,
              flagcxStream_t stream);

  void GroupStart();

  void GroupEnd();

 private:
  DISABLE_COPY_AND_ASSIGN(FlagcxCommContext);

  int flagcx_version_;

  std::unique_ptr<phi::GPUContext> dev_ctx_;

  // used for comm wait compute, compute_stream-->event-->comm_stream
  std::shared_ptr<std::remove_pointer<phi::gpuEvent_t>::type> compute_event_;

  // used for compute wait comm, comm_stream-->event-->compute_stream
  std::shared_ptr<std::remove_pointer<phi::gpuEvent_t>::type> comm_event_;

 public:
  flagcxHandlerGroup_t flagcx_handler_;
};

}  // namespace distributed
}  // namespace phi
