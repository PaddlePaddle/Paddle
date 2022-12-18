// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/collective/ProcessGroup.h"

namespace paddle {
namespace distributed {

// NOTE(liyurui): Notice that some backends use `stream` as an abstract
// conception of hardward resource. We provide this base class allowing users to
// put communications on calculation stream. In some scenorios, we found this
// will save the time of switching streams.
class ProcessGroupStream : public ProcessGroup {
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
  ProcessGroupStream(int rank, int size, int gid);
  virtual ~ProcessGroupStream() = default;
  using ProcessGroup::GetDeviceContext;

  virtual phi::DeviceContext* GetDeviceContext(const Place& place,
                                               bool use_calc_stream) const;

  std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      int64_t offset,
      int64_t numel,
      bool sync_op) override;

  virtual std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      bool sync_op,
      bool use_calc_stream);

  virtual std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      int64_t offset,
      int64_t numel,
      bool sync_op,
      bool use_calc_stream);

  std::shared_ptr<ProcessGroup::Task> AllReduce(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const AllreduceOptions& opts,
      bool sync_op) override;

  virtual std::shared_ptr<ProcessGroup::Task> AllReduce(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const AllreduceOptions& opts,
      bool sync_op,
      bool use_calc_stream);

  std::shared_ptr<ProcessGroup::Task> AllToAll(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const std::vector<int64_t>& out_size_each_rank,
      const std::vector<int64_t>& in_size_each_rank,
      bool sync_op) override;

  virtual std::shared_ptr<ProcessGroup::Task> AllToAll(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const std::vector<int64_t>& out_size_each_rank,
      const std::vector<int64_t>& in_size_each_rank,
      bool sync_op,
      bool use_calc_stream);

  std::shared_ptr<ProcessGroup::Task> Broadcast(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const BroadcastOptions& opts,
      bool sync_op) override;

  virtual std::shared_ptr<ProcessGroup::Task> Broadcast(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const BroadcastOptions& opts,
      bool sync_op,
      bool use_calc_stream);

  std::shared_ptr<ProcessGroup::Task> Reduce(phi::DenseTensor* out_tensor,
                                             const phi::DenseTensor& in_tensor,
                                             const ReduceOptions& opts,
                                             bool sync_op) override;

  virtual std::shared_ptr<ProcessGroup::Task> Reduce(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const ReduceOptions& opts,
      bool sync_op,
      bool use_calc_stream);

  std::shared_ptr<ProcessGroup::Task> ReduceScatter(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const ReduceScatterOptions& opts,
      bool sync_op) override;

  virtual std::shared_ptr<ProcessGroup::Task> ReduceScatter(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const ReduceScatterOptions& opts,
      bool sync_op,
      bool use_calc_stream);

  std::shared_ptr<ProcessGroup::Task> Scatter(phi::DenseTensor* out_tensor,
                                              const phi::DenseTensor& in_tensor,
                                              const ScatterOptions& opts,
                                              bool sync_op) override;

  virtual std::shared_ptr<ProcessGroup::Task> Scatter(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const ScatterOptions& opts,
      bool sync_op,
      bool use_calc_stream);

  std::shared_ptr<ProcessGroup::Task> Recv(phi::DenseTensor* tensor,
                                           int src_rank,
                                           int64_t offset,
                                           int64_t numel,
                                           bool sync_op) override;

  virtual std::shared_ptr<ProcessGroup::Task> Recv(phi::DenseTensor* tensor,
                                                   int src_rank,
                                                   bool sync_op,
                                                   bool use_calc_stream);

  virtual std::shared_ptr<ProcessGroup::Task> Recv(phi::DenseTensor* tensor,
                                                   int src_rank,
                                                   int64_t offset,
                                                   int64_t numel,
                                                   bool sync_op,
                                                   bool use_calc_stream);

  std::shared_ptr<ProcessGroup::Task> Send(const phi::DenseTensor& tensor,
                                           int dst_rank,
                                           int64_t offset,
                                           int64_t numel,
                                           bool sync_op) override;

  std::shared_ptr<ProcessGroup::Task> Send(const phi::DenseTensor& tensor,
                                           int dst_rank,
                                           bool sync_op,
                                           bool use_calc_stream);

  virtual std::shared_ptr<ProcessGroup::Task> Send(
      const phi::DenseTensor& tensor,
      int dst_rank,
      int64_t offset,
      int64_t numel,
      bool sync_op,
      bool use_calc_stream);
};

}  // namespace distributed
}  // namespace paddle
