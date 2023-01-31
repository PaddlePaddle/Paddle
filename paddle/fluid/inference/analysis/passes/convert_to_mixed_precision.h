// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/data_type.h"

namespace paddle {
namespace inference {
namespace analysis {

class ConvertToMixedPrecisionPass {
 public:
  explicit ConvertToMixedPrecisionPass(
      const std::string& model_file,
      const std::string& params_file,
      const std::string& mixed_model_file,
      const std::string& mixed_params_file,
      phi::DataType mixed_precision,
      phi::Backend backend,
      bool keep_io_types,
      const std::unordered_set<std::string>& black_list);

  void Run();

 private:
  void LoadModel();
  void SaveMixedModel();

 private:
  std::string model_file_;
  std::string params_file_;
  std::string mixed_model_file_;
  std::string mixed_params_file_;
  phi::DataType mixed_precision_;
  phi::Backend backend_;
  bool keep_io_types_;
  std::unordered_set<std::string> black_list_;

  framework::Scope scope_;
  std::unique_ptr<framework::ir::Graph> main_graph_{nullptr};
};

bool OpSupportPrecision(const std::string& op_type,
                        phi::Backend backend,
                        phi::DataType precision,
                        const std::unordered_set<std::string>& black_list);

void InsertCastOp(
    framework::ir::Graph* graph,
    framework::ir::Node* var_node,
    framework::ir::Node* op_node,
    framework::proto::VarType::Type from_type,
    framework::proto::VarType::Type to_type,
    framework::BlockDesc* block_desc,
    int* suffix,
    std::unordered_map<framework::ir::Node*, framework::ir::Node*>* visited);

void ConvertToMixedPrecision(const std::string& model_file,
                             const std::string& params_file,
                             const std::string& mixed_model_file,
                             const std::string& mixed_params_file,
                             phi::DataType mixed_precision,
                             phi::Backend backend,
                             bool keep_io_types,
                             const std::unordered_set<std::string>& black_list);

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
