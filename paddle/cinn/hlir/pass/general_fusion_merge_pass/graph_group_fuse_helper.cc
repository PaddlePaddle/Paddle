// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/graph_group_fuse_helper.h"

namespace cinn {
namespace hlir {
namespace pass {
template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::AllOutputsSameSize(
    const OpGroupPtr& first, const OpGroupPtr& second) const {
  return is_same_size(
      &ctx_->graph_group_fusion_helper(), first.GetGroup(), second.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::HorizontalElementwiseFuseReduce(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return honrizontal_elementwise_fuse_reduce(
      &ctx_->graph_group_fusion_helper(), src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::ElementwiseFuseBroadcast(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return elementwise_fuse_broadcast(
      &ctx_->graph_group_fusion_helper(), src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::HorizontalWithInjective(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return horizontal_with_injective(
      &ctx_->graph_group_fusion_helper(), src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::ElementwiseFuseReduce(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return elementwise_fuse_reduce(
      &ctx_->graph_group_fusion_helper(), src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::BroadcastFuseReduce(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return broadcast_fuse_reduce(
      &ctx_->graph_group_fusion_helper(), src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::InjectiveHorizontalWithReduce(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return injective_horizontal_with_reduce(
      &ctx_->graph_group_fusion_helper(), src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::ReduceFuseElementwise(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return reduce_fuse_elementwise(
      &ctx_->graph_group_fusion_helper(), src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::ReduceFuseBroadcast(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return reduce_fuse_broadcast(
      &ctx_->graph_group_fusion_helper(), src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::ReduceFuseReduce(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return reduce_fuse_reduce(
      &ctx_->graph_group_fusion_helper(), src.GetGroup(), dst.GetGroup());
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn
