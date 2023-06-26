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
#include <gloo/gather.h>
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

typedef void (*reduce_func)(void*, const void*, const void*, size_t);
enum class ReduceOp : std::uint8_t { SUM = 0, AVG, MAX, MIN, PRODUCT };
template <typename T>
reduce_func get_function(const int& r) {
  switch (r) {
    case 0:
      return reduce_func(&::gloo::sum<T>);
    case 4:
      return reduce_func(&::gloo::product<T>);
    case 3:
      return reduce_func(&::gloo::min<T>);
    case 2:
      return reduce_func(&::gloo::max<T>);
    case 1:
      VLOG(0) << "Error: Unsupported ReduceOp::AVG.";
      exit(-1);
  }

  VLOG(0) << "Error: Unknown ReduceOp.";
  exit(-1);
}

template <typename T>
void _get_function_impl(gloo::AllreduceOptions::Func& fn,  // NOLINT
                        const int op) {
  fn = get_function<T>(op);
}

gloo::AllreduceOptions::Func _get_function(const phi::DataType type,
                                           const int op) {
  gloo::AllreduceOptions::Func fn;
  GENERATE_FUNC(type, _get_function_impl, fn, op);
  return fn;
}

void GlooCommContext::Broadcast(phi::DenseTensor* out_tensor,
                                const phi::DenseTensor& in_tensor,
                                int root,
                                uint32_t tag) {
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
  opts.setTag(tag);
  gloo::broadcast(opts);
}

void GlooCommContext::AllGather(phi::DenseTensor* out_tensor,
                                const phi::DenseTensor& in_tensor,
                                uint32_t tag) {
  // gloo only uses CPU now

  gloo::AllgatherOptions opts(gloo_context_);
  const auto& dtype = in_tensor.dtype();
  opts.setTag(tag);
  GENERATE_FUNC(dtype, SetInput, &opts, in_tensor);
  GENERATE_FUNC(dtype, SetOutput, &opts, out_tensor);
  gloo::allgather(opts);
}

void GlooCommContext::Gather(phi::DenseTensor* out_tensor,
                             const phi::DenseTensor& in_tensor,
                             int src,
                             uint32_t tag) {
  gloo::GatherOptions opts(gloo_context_);
  const auto& dtype = in_tensor.dtype();
  opts.setTag(tag);
  opts.setRoot(src);
  GENERATE_FUNC(dtype, SetInput, &opts, in_tensor);
  if (rank_ == src) {
    GENERATE_FUNC(dtype, SetOutput, &opts, out_tensor);
  }
  gloo::gather(opts);
}

void GlooCommContext::AllReduce(phi::DenseTensor* out_tensor,
                                const phi::DenseTensor& in_tensor,
                                int reduce_type,
                                uint32_t tag) {
  gloo::AllreduceOptions opts(gloo_context_);
  const auto& dtype = in_tensor.dtype();
  opts.setTag(tag);
  GENERATE_FUNC(dtype, SetInput, &opts, in_tensor);
  GENERATE_FUNC(dtype, SetOutput, &opts, out_tensor);
  GENERATE_FUNC(dtype, SetReduceFunc, &opts, reduce_type);
  opts.setReduceFunction(_get_function(dtype, reduce_type));
  gloo::allreduce(opts);
}

void GlooCommContext::Reduce(phi::DenseTensor* out_tensor,
                             const phi::DenseTensor& in_tensor,
                             int reduce_type,
                             int root,
                             uint32_t tag) {
  gloo::ReduceOptions opts(gloo_context_);
  opts.setTag(tag);
  opts.setRoot(root);
  const auto& dtype = in_tensor.dtype();
  GENERATE_FUNC(dtype, SetInput, &opts, in_tensor);
  GENERATE_FUNC(dtype, SetOutput, &opts, out_tensor);
  GENERATE_FUNC(dtype, SetReduceFunc, &opts, reduce_type);
  gloo::reduce(opts);
}

void GlooCommContext::Send(const phi::DenseTensor& in_tensor,
                           int dst,
                           uint32_t tag) {
  SendRecvOptions opts(gloo_context_);
  const auto& dtype = in_tensor.dtype();
  GENERATE_FUNC(dtype, SetInput, &opts, in_tensor);
  opts.setSrc(gloo_context_.get()->rank);
  opts.setDst(dst);
  opts.setTag(tag);
  send_recv(&opts);
}
void GlooCommContext::Recv(phi::DenseTensor* out_tensor,
                           int src,
                           const phi::DenseTensor& in_tensor,
                           uint32_t tag) {
  SendRecvOptions opts(gloo_context_);
  const auto& dtype = in_tensor.dtype();
  // const auto& dtype = out_tensor->dtype();
  GENERATE_FUNC(dtype, SetOutput, &opts, out_tensor);
  opts.setTag(tag);
  // opts.setSrc(rank);
  opts.setSrc(gloo_context_.get()->rank);
  opts.setDst(src);
  send_recv(&opts);
}

}  // namespace distributed
}  // namespace phi
