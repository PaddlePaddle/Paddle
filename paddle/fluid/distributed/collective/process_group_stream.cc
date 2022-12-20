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

#include "paddle/fluid/distributed/collective/process_group_stream.h"

namespace paddle {
namespace distributed {

ProcessGroupStream::ProcessGroupStream(int rank, int size, int gid)
    : ProcessGroup(rank, size, gid) {}

phi::DeviceContext* ProcessGroupStream::GetDeviceContext(
    const Place& place, bool use_calc_stream) const {
  PADDLE_THROW(platform::errors::Unimplemented(
      "ProcessGroup%s does not support get device_context.", GetBackendName()));
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::AllGather(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    int64_t offset,
    int64_t numel,
    bool sync_op) {
  return AllGather(out_tensor,
                   in_tensor,
                   offset,
                   numel,
                   sync_op,
                   /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::AllGather(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    bool sync_op,
    bool use_calc_stream) {
  return AllGather(out_tensor,
                   in_tensor,
                   /*offset*/ 0,
                   /*numel*/ -1,  // -1 indicates the whole tensor
                   sync_op,
                   use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::AllGather(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    int64_t offset,
    int64_t numel,
    bool sync_op,
    bool use_calc_stream) {
  PADDLE_THROW(platform::errors::Unimplemented(
      "ProcessGroup%s does not support all_gather.", GetBackendName()));
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::AllReduce(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const AllreduceOptions& opts,
    bool sync_op) {
  return AllReduce(out_tensor,
                   in_tensor,
                   opts,
                   sync_op,
                   /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::AllReduce(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const AllreduceOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  PADDLE_THROW(platform::errors::Unimplemented(
      "ProcessGroup%s does not support all_reduce.", GetBackendName()));
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::AllToAll(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const std::vector<int64_t>& out_size_each_rank,
    const std::vector<int64_t>& in_size_each_rank,
    bool sync_op) {
  return AllToAll(out_tensor,
                  in_tensor,
                  out_size_each_rank,
                  in_size_each_rank,
                  sync_op,
                  /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::AllToAll(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const std::vector<int64_t>& out_size_each_rank,
    const std::vector<int64_t>& in_size_each_rank,
    bool sync_op,
    bool use_calc_stream) {
  PADDLE_THROW(platform::errors::Unimplemented(
      "ProcessGroup%s does not support all_to_all.", GetBackendName()));
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Broadcast(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const BroadcastOptions& opts,
    bool sync_op) {
  return Broadcast(out_tensor,
                   in_tensor,
                   opts,
                   sync_op,
                   /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Broadcast(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const BroadcastOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  PADDLE_THROW(platform::errors::Unimplemented(
      "ProcessGroup%s does not support broadcast.", GetBackendName()));
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Reduce(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ReduceOptions& opts,
    bool sync_op) {
  return Reduce(out_tensor,
                in_tensor,
                opts,
                sync_op,
                /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Reduce(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ReduceOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  PADDLE_THROW(platform::errors::Unimplemented(
      "ProcessGroup%s does not support reduce.", GetBackendName()));
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::ReduceScatter(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ReduceScatterOptions& opts,
    bool sync_op) {
  return ReduceScatter(out_tensor,
                       in_tensor,
                       opts,
                       sync_op,
                       /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::ReduceScatter(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ReduceScatterOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  PADDLE_THROW(platform::errors::Unimplemented(
      "ProcessGroup%s does not support reduce_scatter.", GetBackendName()));
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Scatter(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ScatterOptions& opts,
    bool sync_op) {
  return Scatter(out_tensor,
                 in_tensor,
                 opts,
                 sync_op,
                 /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Scatter(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ScatterOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  PADDLE_THROW(platform::errors::Unimplemented(
      "ProcessGroup%s does not support scatter.", GetBackendName()));
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Recv(
    phi::DenseTensor* tensor,
    int src_rank,
    int64_t offset,
    int64_t numel,
    bool sync_op) {
  return Recv(tensor,
              src_rank,
              offset,
              numel,
              sync_op,
              /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Recv(
    phi::DenseTensor* tensor,
    int src_rank,
    bool sync_op,
    bool use_calc_stream) {
  return Recv(tensor,
              src_rank,
              /*offset*/ 0,
              /*numel*/ -1,  // -1 indicates sending the whole tensor
              sync_op,
              use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Recv(
    phi::DenseTensor* tensor,
    int src_rank,
    int64_t offset,
    int64_t numel,
    bool sync_op,
    bool use_calc_stream) {
  PADDLE_THROW(platform::errors::Unimplemented(
      "ProcessGroup%s does not support recv.", GetBackendName()));
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Send(
    const phi::DenseTensor& tensor,
    int dst_rank,
    int64_t offset,
    int64_t numel,
    bool sync_op) {
  return Send(tensor,
              dst_rank,
              offset,
              numel,
              sync_op,
              /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Send(
    const phi::DenseTensor& tensor,
    int dst_rank,
    bool sync_op,
    bool use_calc_stream) {
  return Send(tensor,
              dst_rank,
              /*offset*/ 0,
              /*numel*/ -1,  // -1 indicates receiving the whole tensor
              sync_op,
              use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Send(
    const phi::DenseTensor& tensor,
    int dst_rank,
    int64_t offset,
    int64_t numel,
    bool sync_op,
    bool use_calc_stream) {
  PADDLE_THROW(platform::errors::Unimplemented(
      "ProcessGroup%s does not support send.", GetBackendName()));
}

}  // namespace distributed
}  // namespace paddle
