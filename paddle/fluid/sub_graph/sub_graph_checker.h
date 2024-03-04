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

#pragma once

#include "paddle/fluid/framework/scope.h"
#include "paddle/pir/include/core/program.h"

namespace paddle {
namespace test {

constexpr char kInputPrefix[] = "pt_input_";
constexpr char kOutputPrefix[] = "pt_output_";
constexpr char kFetchSuffix[] = "@fetch";

class SubGraphChecker {
 public:
  SubGraphChecker(std::shared_ptr<pir::Program> orig_program,
                  std::shared_ptr<pir::Program> prim_program);

  bool CheckResult();

  std::vector<double> CheckSpeed();

 private:
  void InitInputs(const std::vector<pir::Value>& input_values,
                  pir::Block* block,
                  paddle::framework::Scope* scope);
  void AppendGetParameter(const std::vector<pir::Value>& input_values,
                          pir::Block* block);
  void AppendFetchOp(pir::Block* block,
                     std::vector<std::string>* names,
                     const std::string& prefix);

  void RemoveFetchOp(pir::Block* block);

  std::vector<phi::DenseTensor> RunPhiResult();
  std::vector<phi::DenseTensor> RunCinnResult();

  double RunPhiSpeed();
  double RunCinnSpeed();
  std::shared_ptr<pir::Program> phi_program_;
  std::shared_ptr<pir::Program> prim_program_;

  std::unique_ptr<pir::Program> phi_kernel_program_;

  paddle::framework::Scope inner_scope_;

  std::vector<pir::Value> phi_input_values_;
  std::vector<std::string> phi_fetch_names_;

  std::vector<pir::Value> cinn_input_values_;
  std::vector<std::string> cinn_fetch_names_;
};

}  // namespace test
}  // namespace paddle
