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

#include "paddle/phi/core/distributed/gloo_comm_context.h"
#include "paddle/phi/core/distributed/gloo_utils.h"

#include <gloo/allgather.h>
#include <gloo/allreduce.h>
#include <gloo/broadcast.h>
#include <gloo/reduce.h>
#include <gloo/types.h>

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/distributed/check/static_check.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace distributed {

GlooCommContext::GlooCommContext(
    int rank,
    int size,
    std::shared_ptr<gloo::rendezvous::Store> store,
    std::shared_ptr<gloo::transport::Device> device)
    : CommContext(rank, size) {
  gloo_context_ = std::make_shared<gloo::rendezvous::Context>(rank, size);
  gloo_context_->connectFullMesh(*store, device);
}

void GlooCommContext::Broadcast(phi::DenseTensor* out_tensor,
                                const phi::DenseTensor& in_tensor,
                                int root) {
  // gloo only uses CPU now
  CommStaticCheck::SameShape(*out_tensor,
                             in_tensor,
                             /*dst_rank*/ rank_,
                             /*cur_rank*/ rank_,
                             size_,
                             phi::AllocationType::CPU);
  gloo::BroadcastOptions opts(gloo_context_);
  const auto& dtype = in_tensor.dtype();
  GENERATE_FUNC(dtype, SetOutput, &opts, out_tensor);
  if (rank_ == root) {
    GENERATE_FUNC(dtype, SetInput, &opts, in_tensor);
  }
  opts.setRoot(root);
  gloo::broadcast(opts);
}

void GlooCommContext::AllGather(phi::DenseTensor* out_tensor,
                                const phi::DenseTensor& in_tensor) {
  // gloo only uses CPU now

  gloo::AllgatherOptions opts(gloo_context_);
  const auto& dtype = in_tensor.dtype();
  GENERATE_FUNC(dtype, SetInput, &opts, in_tensor);
  GENERATE_FUNC(dtype, SetOutput, &opts, out_tensor);
  gloo::allgather(opts);
}

void GlooCommContext::AllReduce(phi::DenseTensor* out_tensor,
                                const phi::DenseTensor& in_tensor,
                                int reduce_type) {
  gloo::AllreduceOptions opts(gloo_context_);
  const auto& dtype = in_tensor.dtype();
  GENERATE_FUNC(dtype, SetInput, &opts, in_tensor);
  GENERATE_FUNC(dtype, SetOutput, &opts, out_tensor);
  GENERATE_FUNC(dtype, SetReduceFunc, &opts, reduce_type);
  gloo::allreduce(opts);
}

void GlooCommContext::Reduce(phi::DenseTensor* out_tensor,
                             const phi::DenseTensor& in_tensor,
                             int reduce_type,
                             int root) {
  gloo::ReduceOptions opts(gloo_context_);
  opts.setRoot(root);
  const auto& dtype = in_tensor.dtype();
  GENERATE_FUNC(dtype, SetInput, &opts, in_tensor);
  GENERATE_FUNC(dtype, SetOutput, &opts, out_tensor);
  GENERATE_FUNC(dtype, SetReduceFunc, &opts, reduce_type);
  gloo::reduce(opts);
}

}  // namespace distributed
}  // namespace phi
