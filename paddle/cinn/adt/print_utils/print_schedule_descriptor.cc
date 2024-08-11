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

#include "paddle/cinn/adt/print_utils/print_schedule_descriptor.h"
#include "paddle/cinn/adt/schedule_descriptor.h"

namespace cinn::adt {

std::string ToTxtString(const LoopDescriptor& loop_descriptor) {
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
  PADDLE_ENFORCE_EQ(loop_size.Has<std::int64_t>(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The loop_size should have type int64_t."));
  *string += "=0.." + std::to_string(loop_size.Get<std::int64_t>());
  return ret;
}

}  // namespace cinn::adt
