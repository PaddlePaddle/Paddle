// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/cost_model.h"

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device_tracer.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"
// #include "paddle/fluid/platform/profiler_helper.h"

namespace paddle {
namespace framework {

CostData CostModel::ProfileMeasure(const ProgramDesc& program,
                                   const std::string& device) {
  CostData cost_data;

  SetTracerOption(platform::TracerOption::kAllOpDetail);
  // TODO(zhhsplendid): handle the case that Profiler is already enabled
  platform::ProfilerState profiler_state;
  platform::Place place;
  // TODO(zhhsplendid): add code to transform string to lower case
  std::string device_lower_case = device;
  if (device_lower_case == "cpu") {
    profiler_state = platform::ProfilerState::kCPU;
    place = platform::CPUPlace();
  } else if (device_lower_case == "gpu") {
    profiler_state = platform::ProfilerState::kAll;
    place = platform::CUDAPlace();
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Not support %s in CostModel now", device));
  }

  Executor executor(place);
  Scope scope;

  EnableProfiler(profiler_state);
  executor.Run(program, &scope, /*block_id = */ 0);

  // TODO(zhhsplendid): modify from print to cost data
  DisableProfiler(platform::EventSortingKey::kDefault,
                  "/tmp/huihuang_profile_tmp");

  return cost_data;
}

}  // namespace framework
}  // namespace paddle
