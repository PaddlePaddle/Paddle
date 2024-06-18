// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/cinn/utils/external_func_names.h"

namespace cinn::utils {

const std::unordered_set<std::string>& GetProhibitScheduleExternalFuncNames() {
  static const std::unordered_set<std::string>
      prohibit_schedule_external_func_names = {
#define CINN_FUNC2STRING(str) #str
#define CINN_NVGPU_FUNC_TYPE(FUNC, TYPE)     \
  CINN_FUNC2STRING(cinn_nvgpu_##FUNC##TYPE), \
      CINN_FUNC2STRING(cinn_host_##FUNC##TYPE)

#define GEN_FUNC_NAME(_, impl) \
  _(impl, gt_num)              \
  _(impl, lt_num)              \
  _(impl, index_add)           \
  _(impl, next_smallest)

#define GEN_FUNC_NAME_WITH_TYPE(_, ...)                                     \
  _(__VA_ARGS__, _bool), _(__VA_ARGS__, _fp16), _(__VA_ARGS__, _fp32),      \
      _(__VA_ARGS__, _fp64), _(__VA_ARGS__, _uint8), _(__VA_ARGS__, _int8), \
      _(__VA_ARGS__, _int16), _(__VA_ARGS__, _int32), _(__VA_ARGS__, _int64),

          GEN_FUNC_NAME(GEN_FUNC_NAME_WITH_TYPE, CINN_NVGPU_FUNC_TYPE)
#undef GEN_FUNC_NAME
#undef GEN_FUNC_NAME_WITH_TYPE
#undef CINN_NVGPU_FUNC_TYPE
#undef CINN_FUNC2STRING
      };
  return prohibit_schedule_external_func_names;
}

}  // namespace cinn::utils
