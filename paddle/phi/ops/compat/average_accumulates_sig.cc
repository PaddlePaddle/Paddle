/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {
KernelSignature AverageAccumulatesOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "average_accumulates",
      {"param",
       "in_sum_1",
       "in_sum_2",
       "in_sum_3",
       "in_num_accumulates",
       "in_old_num_accumulates",
       "in_num_updates"},
      {"average_window", "max_average_window", "min_average_window"},
      {"out_sum_1",
       "out_sum_2",
       "out_sum_3",
       "out_num_accumulates",
       "out_old_num_accumulates",
       "out_num_updates"});
}
}  // namespace phi
PD_REGISTER_ARG_MAPPING_FN(average_accumulates,
                           phi::AverageAccumulatesOpArgumentMapping);
