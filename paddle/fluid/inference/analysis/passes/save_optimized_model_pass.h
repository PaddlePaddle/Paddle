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

#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/data_type.h"

namespace paddle {
namespace inference {
namespace analysis {

class SaveOptimizedModelPass {
 public:
  explicit SaveOptimizedModelPass(
      const std::string& model_file,
      const std::string& params_file,
      const std::string& output_model_file,
      const std::string& output_params_file,
      phi::Backend backend,
      const std::unordered_set<std::string>& black_list);

  void Run();

 private:
  void LoadModel();
  void SaveModel();

 private:
  std::string model_file_;
  std::string params_file_;
  std::string output_model_file_;
  std::string output_params_file_;
  phi::Backend backend_;
  std::unordered_set<std::string> black_list_;

  framework::Scope scope_;
  std::unique_ptr<framework::ir::Graph> main_graph_{nullptr};
};

void SaveOptimizedModel(const std::string& model_file,
                        const std::string& params_file,
                        const std::string& output_model_file,
                        const std::string& output_params_file,
                        phi::Backend backend,
                        const std::unordered_set<std::string>& black_list);

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
