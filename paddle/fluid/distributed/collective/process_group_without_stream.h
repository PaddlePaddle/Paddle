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

#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

namespace paddle {
namespace distributed {

class ProcessGroupWithoutStream : public ProcessGroup {
 public:
  ProcessGroupWithoutStream(int rank, int size, int gid)
      : ProcessGroup(rank, size, gid) {}

  virtual ~ProcessGroupWithoutStream() = default;

  // methods from base class
  using ProcessGroup::AllGather;
  using ProcessGroup::Recv;
  using ProcessGroup::Send;

  std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      bool sync_op) override {
    return AllGather(out_tensor,
                     in_tensor,
                     /*offset*/ 0,
                     /*numel*/ -1,  // -1 indicates the whole tensor
                     sync_op);
  }

  std::shared_ptr<ProcessGroup::Task> Recv(phi::DenseTensor* tensor,
                                           int src_rank,
                                           bool sync_op) override {
    return Recv(tensor,
                src_rank,
                /*offset*/ 0,
                /*numel*/ -1,  // -1 indicates the whole tensor
                sync_op);
  }

  std::shared_ptr<ProcessGroup::Task> Send(const phi::DenseTensor& tensor,
                                           int dst_rank,
                                           bool sync_op) override {
    return Send(tensor,
                dst_rank,
                /*offset*/ 0,
                /*numel*/ -1,  // -1 indicates the whole tensor
                sync_op);
  }
};

}  // namespace distributed
}  // namespace paddle
