// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/auto_schedule/measure/measure.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/graph_compiler_util.h"

namespace cinn {
namespace auto_schedule {

// This class utilize the GraphCompiler bound to the graph to build
// the input schedule as executable objects
class SimpleBuilder : public ScheduleBuilder {
 public:
  explicit SimpleBuilder(hlir::framework::GraphCompiler* graph_compiler);

  // Build and pack the result
  BuildResult Build(const MeasureInput& input) override;

 private:
  hlir::framework::GraphCompiler* graph_compiler_;
};

}  // namespace auto_schedule
}  // namespace cinn
