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
#include "paddle/common/enforce.h"
namespace cinn::adt {

LoopDescriptors CreateScheduleDescriptor(const ScheduleMesh& sched_mesh,
                                         const List<LoopType>& loop_types) {
  const auto& sched_dims = GetOutputDimValues(sched_mesh);
  PADDLE_ENFORCE_EQ(
      sched_dims->size(),
      loop_types->size(),
      ::common::errors::InvalidArgument(
          "The size of sched_dims and loop_types should be equal, but got "
          "sched_dims size = %d, loop_types size = %d.",
          sched_dims->size(),
          loop_types->size()));
  LoopDescriptors ret{};
  for (std::size_t i = 0; i < sched_dims->size(); ++i) {
    ret->emplace_back(LoopDescriptor{loop_types->at(i), sched_dims->at(i)});
  }
  return ret;
}

List<LoopSize> GenerateLoopSizeFromSd(const LoopDescriptors& sd) {
  List<LoopSize> sd_sizes{};
  for (const auto& loop_descriptor : *sd) {
    const auto& [loop_type, loop_size] = loop_descriptor.tuple();
    sd_sizes->emplace_back(loop_size);
  }
  return sd_sizes;
}

}  // namespace cinn::adt
