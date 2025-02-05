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

#include "paddle/common/errors.h"
#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace distributed {

// NOTE: Notice that some backends use `stream` as an abstract conception of
// hardware resource. We provide this base class allowing users to put
// communications on calculation stream. In some scenarios, we found this will
// save the time of switching streams.
class ProcessGroupWithStream : public ProcessGroup {
 public:
  class TaskStream : public ProcessGroup::Task {
   public:
    TaskStream(int rank, CommType comm_type, bool sync_op, bool use_calc_stream)
        : Task(rank, comm_type, sync_op), use_calc_stream_(use_calc_stream) {}

    virtual ~TaskStream() = default;

    // TODO(liyurui): This constructor is temporary here for compatible reason,
    // will be deleted soon.
    TaskStream(int rank,
               const std::vector<phi::DenseTensor>& inputs,
               CommType comm_type)
        : Task(rank, inputs, comm_type) {}

    TaskStream(int rank,
               const std::vector<phi::DenseTensor>& inputs,
               CommType comm_type,
               bool sync_op,
               bool use_calc_stream)
        : Task(rank, inputs, comm_type, sync_op),
          use_calc_stream_(use_calc_stream) {}

   protected:
    bool UseCalcStream() const { return use_calc_stream_; }

   private:
    bool use_calc_stream_{false};
  };

 public:
  ProcessGroupWithStream(int rank, int size, int gid)
      : ProcessGroup(rank, size, gid) {}

  virtual ~ProcessGroupWithStream() = default;

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

  std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      int64_t offset,
      int64_t numel,
      bool sync_op) override {
    return AllGather(out_tensor,
                     in_tensor,
                     offset,
                     numel,
                     sync_op,
                     /*use_calc_stream*/ false);
  }

  std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      bool sync_op,
      bool use_calc_stream) override {
    return AllGather(out_tensor,
                     in_tensor,
                     /*offset*/ 0,
                     /*numel*/ -1,  // -1 indicates the whole tensor
                     sync_op,
                     use_calc_stream);
  }

  std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      int64_t offset UNUSED,
      int64_t numel UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) override {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroupWithStream (%s) does not support all_gather.",
        GetBackendName()));
  }

  std::shared_ptr<ProcessGroup::Task> AllReduce(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const AllreduceOptions& opts,
      bool sync_op) override {
    return AllReduce(out_tensor,
                     in_tensor,
                     opts,
                     sync_op,
                     /*use_calc_stream*/ false);
  }

  std::shared_ptr<ProcessGroup::Task> AllReduce(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const AllreduceOptions& opts UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) override {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroupWithStream (%s) does not support all_reduce.",
        GetBackendName()));
  }

  std::shared_ptr<ProcessGroup::Task> AllToAll(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const std::vector<int64_t>& out_size_each_rank,
      const std::vector<int64_t>& in_size_each_rank,
      bool sync_op) override {
    return AllToAll(out_tensor,
                    in_tensor,
                    out_size_each_rank,
                    in_size_each_rank,
                    sync_op,
                    /*use_calc_stream*/ false);
  }

  std::shared_ptr<ProcessGroup::Task> AllToAll(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const std::vector<int64_t>& out_size_each_rank UNUSED,
      const std::vector<int64_t>& in_size_each_rank UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) override {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroupWithStream (%s) does not support all_to_all.",
        GetBackendName()));
  }

  std::shared_ptr<ProcessGroup::Task> Broadcast(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const BroadcastOptions& opts,
      bool sync_op) override {
    return Broadcast(out_tensor,
                     in_tensor,
                     opts,
                     sync_op,
                     /*use_calc_stream*/ false);
  }

  std::shared_ptr<ProcessGroup::Task> Broadcast(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const BroadcastOptions& opts UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) override {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroupWithStream (%s) does not support broadcast.",
        GetBackendName()));
  }

  std::shared_ptr<ProcessGroup::Task> Reduce(phi::DenseTensor* out_tensor,
                                             const phi::DenseTensor& in_tensor,
                                             const ReduceOptions& opts,
                                             bool sync_op) override {
    return Reduce(out_tensor,
                  in_tensor,
                  opts,
                  sync_op,
                  /*use_calc_stream*/ false);
  }

  std::shared_ptr<ProcessGroup::Task> Reduce(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const ReduceOptions& opts UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) override {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroupWithStream (%s) does not support reduce.",
        GetBackendName()));
  }

  std::shared_ptr<ProcessGroup::Task> ReduceScatter(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const ReduceScatterOptions& opts,
      bool sync_op) override {
    return ReduceScatter(out_tensor,
                         in_tensor,
                         opts,
                         sync_op,
                         /*use_calc_stream*/ false);
  }

  std::shared_ptr<ProcessGroup::Task> ReduceScatter(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const ReduceScatterOptions& opts UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) override {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroupWithStream (%s) does not support reduce_scatter.",
        GetBackendName()));
  }

  std::shared_ptr<ProcessGroup::Task> Scatter(phi::DenseTensor* out_tensor,
                                              const phi::DenseTensor& in_tensor,
                                              const ScatterOptions& opts,
                                              bool sync_op) override {
    return Scatter(out_tensor,
                   in_tensor,
                   opts,
                   sync_op,
                   /*use_calc_stream*/ false);
  }

  std::shared_ptr<ProcessGroup::Task> Scatter(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const ScatterOptions& opts UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) override {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroupWithStream (%s) does not support scatter.",
        GetBackendName()));
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

  std::shared_ptr<ProcessGroup::Task> Recv(phi::DenseTensor* tensor,
                                           int src_rank,
                                           int64_t offset,
                                           int64_t numel,
                                           bool sync_op) override {
    return Recv(tensor,
                src_rank,
                offset,
                numel,
                sync_op,
                /*use_calc_stream*/ false);
  }

  std::shared_ptr<ProcessGroup::Task> Recv(phi::DenseTensor* tensor,
                                           int src_rank,
                                           bool sync_op,
                                           bool use_calc_stream) override {
    return Recv(tensor,
                src_rank,
                /*offset*/ 0,
                /*numel*/ -1,  // -1 indicates sending the whole tensor
                sync_op,
                use_calc_stream);
  }

  std::shared_ptr<ProcessGroup::Task> Recv(phi::DenseTensor* tensor UNUSED,
                                           int src_rank UNUSED,
                                           int64_t offset UNUSED,
                                           int64_t numel UNUSED,
                                           bool sync_op UNUSED,
                                           bool use_calc_stream
                                               UNUSED) override {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroupWithStream (%s) does not support recv.",
        GetBackendName()));
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

  std::shared_ptr<ProcessGroup::Task> Send(const phi::DenseTensor& tensor,
                                           int dst_rank,
                                           int64_t offset,
                                           int64_t numel,
                                           bool sync_op) override {
    return Send(tensor,
                dst_rank,
                offset,
                numel,
                sync_op,
                /*use_calc_stream*/ false);
  }

  std::shared_ptr<ProcessGroup::Task> Send(const phi::DenseTensor& tensor,
                                           int dst_rank,
                                           bool sync_op,
                                           bool use_calc_stream) override {
    return Send(tensor,
                dst_rank,
                /*offset*/ 0,
                /*numel*/ -1,  // -1 indicates receiving the whole tensor
                sync_op,
                use_calc_stream);
  }

  std::shared_ptr<ProcessGroup::Task> Send(
      const phi::DenseTensor& tensor UNUSED,
      int dst_rank UNUSED,
      int64_t offset UNUSED,
      int64_t numel UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) override {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroupWithStream (%s) does not support send.",
        GetBackendName()));
  }
};

}  // namespace distributed
}  // namespace paddle
