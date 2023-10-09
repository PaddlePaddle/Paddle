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

#include "paddle/cinn/adt/schedule_descriptor.h"
#include "paddle/cinn/adt/equation_solver.h"
#include "paddle/cinn/adt/igroup.h"
#include "paddle/cinn/adt/index_expr_infer_context.h"
#include "paddle/cinn/adt/kgroup.h"
#include "paddle/cinn/adt/schedule_dim.h"

namespace cinn::adt {

namespace {

const std::vector<int32_t>& GetTensorShape(const Tensor& tensor) {
  CHECK(tensor.Has<adapter::Tensor>());
  return tensor.Get<adapter::Tensor>().GetShape();
}

ScheduleDescriptor CreateOptimizedScheduleDescriptor(
    const List<ScheduleDim>& loop_sizes) {
  ADT_TODO();
}

}  // namespace

ScheduleDescriptor MakeOptimizedScheduleDescriptor(
    const std::shared_ptr<KGroup>& kgroup,
    const std::shared_ptr<IGroup>& igroup) {
  const auto& schedule_dim = igroup->anchor_schedule_dims();

  return CreateOptimizedScheduleDescriptor(schedule_dim);
}

ScheduleDescriptor MakeNaiveScheduleDescriptor(
    const std::shared_ptr<KGroup>& kgroup,
    const std::shared_ptr<IGroup>& igroup) {
  const Tensor& tensor = igroup->anchor_tensor();

  List<LoopDescriptor> ret{};
  const auto tensor_shape = GetTensorShape(tensor);
  for (int32_t dim : tensor_shape) {
    ret->emplace_back(LoopDescriptor{Temporal{}, dim});
  }
  return ret;
}

List<LoopSize> GenerateLoopSizeFromSd(const ScheduleDescriptor& sd) {
  List<LoopSize> sd_sizes{};
  for (const auto& loop_descriptor : *sd) {
    const auto& [loop_type, loop_size] = loop_descriptor.tuple();
    sd_sizes->emplace_back(loop_size);
  }
  return sd_sizes;
}

std::string DebugStringImpl(const LoopDescriptor& loop_descriptor) {
  const auto& [loop_type, loop_size] = loop_descriptor.tuple();
  std::string ret{};
  auto* string = &ret;
  loop_type >>
      match{[&](const S0x&) { *string += "blockIdx.x"; },
            [&](const S0y&) { *string += "blockIdx.y"; },
            [&](const S0z&) { *string += "blockIdx.z"; },
            [&](const S1x&) { *string += "threadIdx.x"; },
            [&](const S1y&) { *string += "threadIdx.y"; },
            [&](const S1z&) { *string += "threadIdx.z"; },
            [&](const Temporal& temporal) {
              *string += temporal.iter_var_name();
            },
            [&](const Vectorize& vectorize) {
              *string += vectorize.iter_var_name();
            },
            [&](const Unroll& unroll) { *string += unroll.iter_var_name(); }};
  CHECK(loop_size.Has<std::int64_t>());
  *string += "=0.." + std::to_string(loop_size.Get<std::int64_t>());
  return ret;
}

}  // namespace cinn::adt
