// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include <chrono>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "paddle/common/performance_statistician.h"
#include "paddle/fluid/framework/new_executor/interpretercore.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/common/place.h"
#include "paddle/pir/include/core/program.h"

namespace cinn {
namespace ir {
namespace search {

struct MeasureResult {
  ::common::TimeDuration compile_time;
  ::common::TimeDuration avg_kernel_execute_time;
  ::common::TimeDuration avg_total_execute_time;
  std::string err_msg;
};

class Measurer {
 public:
  explicit Measurer(::pir::Program* program);

  void Compile();

  void Run(const std::unordered_map<std::string, std::vector<int64_t>>&
               input_name_and_shape,
           int repeat);

  MeasureResult Result() const;

 private:
  std::string compile_label_;
  std::string execute_label_;
  ::pir::Program* program_;
  phi::Place place_ = phi::GPUPlace(0);
  std::unique_ptr<pir::Program> kernel_program_;
  std::unique_ptr<paddle::framework::Scope> exe_scope_ =
      std::make_unique<paddle::framework::Scope>();
  std::unique_ptr<paddle::framework::InterpreterCore> executor_;
};

}  // namespace search
}  // namespace ir
}  // namespace cinn
