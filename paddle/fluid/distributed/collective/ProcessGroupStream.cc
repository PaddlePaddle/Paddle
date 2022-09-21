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

ProcessGroupStream::ProcessGroupStream(int rank,
                                       int size,
                                       const platform::Place& place,
                                       int gid)
    : ProcessGroup(rank, size, place, gid) {}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::AllReduce(
    std::vector<phi::DenseTensor>& input_tensors,   // NOLINT
    std::vector<phi::DenseTensor>& output_tensors,  // NOLINT
    const AllreduceOptions& options,
    bool sync_op) {
  return AllReduce(input_tensors,
                   output_tensors,
                   options,
                   sync_op,
                   /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::AllReduce(
    std::vector<phi::DenseTensor>& input_tensors,   // NOLINT
    std::vector<phi::DenseTensor>& output_tensors,  // NOLINT
    const AllreduceOptions& options,
    bool sync_op,
    bool use_calc_stream) {
  PADDLE_THROW(platform::errors::InvalidArgument(
      "ProcessGroup%s does not support do allreduce", GetBackendName()));
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Send(
    std::vector<phi::DenseTensor>& tensors, int dst_rank, bool sync_op) {
  return Send(tensors,
              dst_rank,
              sync_op,
              /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Send(
    std::vector<phi::DenseTensor>& tensors,
    int dst_rank,
    bool sync_op,
    bool use_calc_stream) {
  PADDLE_THROW(platform::errors::InvalidArgument(
      "ProcessGroup%s does not support do send", GetBackendName()));
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Send_Partial(
    phi::DenseTensor& tensors,
    int dst_rank,
    int offset,
    int length,
    bool sync_op) {
  return Send_Partial(tensors,
                      dst_rank,
                      offset,
                      length,
                      sync_op,
                      /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Send_Partial(
    phi::DenseTensor& tensors,
    int dst_rank,
    int offset,
    int length,
    bool sync_op,
    bool use_calc_stream) {
  PADDLE_THROW(platform::errors::InvalidArgument(
      "ProcessGroup%s does not support do send_partial", GetBackendName()));
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

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Recv_Partial(
    phi::DenseTensor& tensors,
    int src_rank,
    int offset,
    int length,
    bool sync_op) {
  return Recv_Partial(tensors,
                      src_rank,
                      offset,
                      length,
                      sync_op,
                      /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::Recv_Partial(
    phi::DenseTensor& tensors,
    int src_rank,
    int offset,
    int length,
    bool sync_op,
    bool use_calc_stream) {
  PADDLE_THROW(platform::errors::InvalidArgument(
      "ProcessGroup%s does not support do recv_partial", GetBackendName()));
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::AllGather_Partial(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    int offset,
    int length,
    bool sync_op) {
  return AllGather_Partial(in_tensors,
                           out_tensors,
                           offset,
                           length,
                           sync_op,
                           /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupStream::AllGather_Partial(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    int offset,
    int length,
    bool sync_op,
    bool use_calc_stream) {
  PADDLE_THROW(platform::errors::InvalidArgument(
      "ProcessGroup%s does not support do recv_partial", GetBackendName()));
}

}  // namespace distributed
}  // namespace paddle
