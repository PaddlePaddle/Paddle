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

#include "paddle/fluid/distributed/collective/ProcessGroupStream.h"

namespace paddle {
namespace distributed {

ProcessGroupStream::ProcessGroupStream(int rank, int size, int gid)
    : ProcessGroup(rank, size, gid) {}

const phi::DeviceContext& ProcessGroupStream::GetDeviceContext(
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
    int64_t offset,
    int64_t numel,
    bool sync_op,
    bool use_calc_stream) {
  PADDLE_THROW(platform::errors::Unimplemented(
      "ProcessGroup%s does not support recv.", GetBackendName()));
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Send(
    phi::DenseTensor* tensor,
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
    phi::DenseTensor*,
    int dst_rank,
    int64_t offset,
    int64_t numel,
    bool sync_op,
    bool use_calc_stream) {
  PADDLE_THROW(platform::errors::Unimplemented(
      "ProcessGroup%s does not support send.", GetBackendName()));
}

// TODO(sunyilun): methods below will be removed later
std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::AllToAll(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    bool sync_op) {
  return AllToAll(in_tensors,
                  out_tensors,
                  sync_op,
                  /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::AllToAll(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    bool sync_op,
    bool use_calc_stream) {
  PADDLE_THROW(platform::errors::InvalidArgument(
      "ProcessGroup%s does not support do alltoall", GetBackendName()));
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::AllToAllSingle(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    std::vector<int64_t>& in_sizes,
    std::vector<int64_t>& out_sizes,
    bool sync_op) {
  return AllToAllSingle(in_tensors,
                        out_tensors,
                        in_sizes,
                        out_sizes,
                        sync_op,
                        /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::AllToAllSingle(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    std::vector<int64_t>& in_sizes,
    std::vector<int64_t>& out_sizes,
    bool sync_op,
    bool use_calc_stream) {
  PADDLE_THROW(platform::errors::InvalidArgument(
      "ProcessGroup%s does not support do alltoall_single", GetBackendName()));
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Reduce(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const ReduceOptions& opts,
    bool sync_op) {
  return Reduce(in_tensors,
                out_tensors,
                opts,
                sync_op,
                /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Reduce(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const ReduceOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  PADDLE_THROW(platform::errors::InvalidArgument(
      "ProcessGroup%s does not support do reduce", GetBackendName()));
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::ReduceScatter(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const ReduceScatterOptions& opts,
    bool sync_op) {
  return ReduceScatter(in_tensors,
                       out_tensors,
                       opts,
                       sync_op,
                       /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::ReduceScatter(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const ReduceScatterOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  PADDLE_THROW(platform::errors::InvalidArgument(
      "ProcessGroup%s does not support do reduce_scatter", GetBackendName()));
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Scatter(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const ScatterOptions& opts,
    bool sync_op) {
  return Scatter(in_tensors,
                 out_tensors,
                 opts,
                 sync_op,
                 /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Scatter(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const ScatterOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  PADDLE_THROW(platform::errors::InvalidArgument(
      "ProcessGroup%s does not support do scatter", GetBackendName()));
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Recv(
    std::vector<phi::DenseTensor>& tensors, int src_rank, bool sync_op) {
  return Recv(tensors,
              src_rank,
              sync_op,
              /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Recv(
    std::vector<phi::DenseTensor>& tensors,
    int src_rank,
    bool sync_op,
    bool use_calc_stream) {
  PADDLE_THROW(platform::errors::InvalidArgument(
      "ProcessGroup%s does not support do recv", GetBackendName()));
}

}  // namespace distributed
}  // namespace paddle
