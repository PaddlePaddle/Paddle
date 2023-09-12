// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/op_mapper_registry.h"
#include "paddle/cinn/frontend/paddle/cpp/block_desc.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/hlir/framework/scope.h"
#include "paddle/cinn/utils/type_defs.h"

namespace cinn {
namespace frontend {

// Transform paddle model to CINN fronted::Program object.
// The paddle model is readed from __model__ file in model_dir, the
// PaddleModelConvertor will run each op's kernel registered in OpMapper, each
// kernel will add instruction in NetBuilder, after running all op of model, it
// will invoke its Build function and finally return the complete
// fronted::Program object. Note that if anyone op not registered, the program
// will failed and aborted.
class PaddleModelConvertor {
 public:
  PaddleModelConvertor();

  PaddleModelConvertor(const common::Target& target,
                       std::shared_ptr<NetBuilder> builder = nullptr,
                       std::shared_ptr<hlir::framework::Scope> scope = nullptr);

  // prepare feed variable before run CINN op
  void PrepareRun(const paddle::cpp::BlockDesc& block_desc,
                  OpMapperContext* ctx);

  // RunOp accept OpDesc and global run context then run it's kernel registered
  // in OpMapper.
  static void RunOp(const paddle::cpp::OpDesc& op_desc,
                    const OpMapperContext& ctx);

  static void RunOp(
      const std::string& op_type,
      const std::map<std::string, std::vector<std::string>>& inputs,
      const std::map<std::string, std::vector<std::string>>& outputs,
      const std::map<std::string, cinn::utils::Attribute>& attrs,
      const OpMapperContext& ctx);

  void RunOp(const std::string& op_type,
             const std::map<std::string, std::vector<std::string>>& inputs,
             const std::map<std::string, std::vector<std::string>>& outputs,
             const std::map<std::string, cinn::utils::Attribute>& attrs);

  void CreateInput(const std::string& dtype,
                   const cinn::utils::ShapeType& shape,
                   const std::string& name);

  Program operator()();

  // operator() accept the modle's directory, and return the fronted::Program
  // object.
  Program LoadModel(
      const std::string& model_dir,
      bool is_combined = false,
      const std::unordered_map<std::string, std::vector<int64_t>>& feed = {});

  // return the internal variable map
  const std::unordered_map<std::string, Variable>& var_map() const {
    return var_map_;
  }

  // return the map from the variable name in paddle model to cinn program.
  const std::unordered_map<std::string, std::string>& var_model_to_program_map()
      const {
    return var_model_to_program_map_;
  }

  // return the map the paddle variable name to cinn variable object
  std::unordered_map<std::string, Variable> GetFetchList(
      const std::unordered_set<std::string>& fetch_name_list = {}) const;

 private:
  std::unordered_map<std::string, Variable> var_map_;
  // map from var in Paddle model to var name in program.
  std::unordered_map<std::string, std::string> var_model_to_program_map_;
  // fetch var names used in Paddle
  std::unordered_set<std::string> fetch_var_names_;

  std::unique_ptr<OpMapperContext> ctx_;
  std::shared_ptr<NetBuilder> builder_;
  const common::Target& target_;
  std::shared_ptr<hlir::framework::Scope> scope_;
};

}  // namespace frontend
}  // namespace cinn
