// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

/*
 * This file defines the class Argument, which is the input and output of the
 * analysis module. All the fields that needed either by Passes or PassManagers
 * are contained in Argument.
 *
 * TODO(Superjomn) Find some way better to contain the fields when it grow too
 * big.
 */

#pragma once

#include <string>
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/inference/analysis/data_flow_graph.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * The argument definition of both Pass and PassManagers.
 *
 * All the fields should be registered here for clearness.
 */
struct Argument {
  Argument() = default;
  explicit Argument(const std::string& fluid_model_dir)
      : fluid_model_dir(new std::string(fluid_model_dir)) {}
  // The directory of the trained model.
  std::unique_ptr<std::string> fluid_model_dir;
  // The path of `__model__` and `param`, this is used when the file name of
  // model and param is changed.
  std::unique_ptr<std::string> fluid_model_program_path;
  std::unique_ptr<std::string> fluid_model_param_path;

  // The graph that process by the Passes or PassManagers.
  std::unique_ptr<DataFlowGraph> main_dfg;

  // The original program desc.
  std::unique_ptr<framework::proto::ProgramDesc> origin_program_desc;

  // The processed program desc.
  std::unique_ptr<framework::proto::ProgramDesc> transformed_program_desc;

  // The output storage path of ModelStorePass.
  std::unique_ptr<std::string> model_output_store_path;
};

#define UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)
#define ANALYSIS_ARGUMENT_CHECK_FIELD(field__)               \
  if (UNLIKELY(!(field__))) {                                \
    LOG(ERROR) << "field " << #field__ << " should be set."; \
    return false;                                            \
  }

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
