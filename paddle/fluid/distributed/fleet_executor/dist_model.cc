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

#include <glog/logging.h>

#include "paddle/fluid/distributed/fleet_executor/dist_model.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace distributed {

namespace {
bool IsPersistable(const framework::VarDesc *var) {
  if (var->Persistable() &&
      var->GetType() != framework::proto::VarType::FEED_MINIBATCH &&
      var->GetType() != framework::proto::VarType::FETCH_LIST &&
      var->GetType() != framework::proto::VarType::RAW) {
    return true;
  }
  return false;
}
}  // namespace

bool DistModel::Init() {
  /* TODO(fleet exe dev): implement this funct */
  place_ = paddle::platform::CUDAPlace(config_.device_id);
  if (!PrepareScope()) {
    return false;
  }
  if (!PrepareProgram()) {
    return false;
  }
  if (!CommInit()) {
    return false;
  }
  return true;
}

bool DistModel::CommInit() {
  // TODO(fleet executor): init the comm
  return true;
}

bool DistModel::PrepareScope() {
  scope_.reset(new framework::Scope());
  return true;
}

bool DistModel::PrepareProgram() {
  if (!LoadProgram()) {
    return false;
  }
  if (!LoadParameters()) {
    return false;
  }
  return true;
}

bool DistModel::LoadProgram() {
  VLOG(3) << "Loading program from " << config_.model_dir;
  PADDLE_ENFORCE_NE(config_.model_dir, "", platform::errors::InvalidArgument(
                                               "Model dir must be provided."));
  std::string model_path = config_.model_dir + ".pdmodel";
  framework::proto::ProgramDesc program_proto;
  std::string pb_content;
  // Read binary
  std::ifstream fin(model_path, std::ios::in | std::ios::binary);
  PADDLE_ENFORCE_EQ(
      static_cast<bool>(fin.is_open()), true,
      platform::errors::NotFound(
          "Cannot open file %s, please confirm whether the file is normal.",
          model_path));
  fin.seekg(0, std::ios::end);
  pb_content.resize(fin.tellg());
  fin.seekg(0, std::ios::beg);
  fin.read(&(pb_content.at(0)), pb_content.size());
  fin.close();
  program_proto.ParseFromString(pb_content);
  VLOG(5) << pb_content;
  program_.reset(new framework::ProgramDesc(program_proto));
  return true;
}

bool DistModel::LoadParameters() {
  VLOG(3) << "Loading parameters from " << config_.model_dir;
  PADDLE_ENFORCE_NOT_NULL(program_.get(),
                          platform::errors::PreconditionNotMet(
                              "The program should be loaded first."));
  const auto &global_block = program_->MutableBlock(0);

  // create a temporary program to load parameters.

  std::unique_ptr<framework::ProgramDesc> load_program(
      new framework::ProgramDesc());
  framework::BlockDesc *load_block = load_program->MutableBlock(0);
  std::vector<std::string> params;

  for (auto *var : global_block->AllVars()) {
    if (IsPersistable(var)) {
      VLOG(3) << "persistable variable's name: " << var->Name();
      framework::VarDesc *new_var = load_block->Var(var->Name());
      new_var->SetShape(var->GetShape());
      new_var->SetDataType(var->GetDataType());
      new_var->SetType(var->GetType());
      new_var->SetLoDLevel(var->GetLoDLevel());
      new_var->SetPersistable(true);
      params.push_back(new_var->Name());
    }
  }

  std::string param_path = config_.model_dir + ".pdiparams";
  // sort paramlist to have consistent ordering
  std::sort(params.begin(), params.end());
  // append just the load_combine op
  framework::OpDesc *op = load_block->AppendOp();
  op->SetType("load_combine");
  op->SetOutput("Out", params);
  op->SetAttr("file_path", {param_path});
  op->CheckAttrs();

  framework::NaiveExecutor e(place_);
  // Create all persistable variables in root scope to load them from ckpt.
  // Other non-persistable variables will be created in the micro scope
  // managed by fleet executor.
  e.CreateVariables(*program_, 0, true, scope_.get());
  e.Prepare(scope_.get(), *load_program, 0, false);
  e.Run();
  VLOG(3) << "After loading there are " << scope_->LocalVarNames().size()
          << " vars.";

  return true;
}

void DistModel::Run(const std::vector<framework::Tensor> &input_data,
                    std::vector<framework::Tensor> *output_data) {
  /* TODO(fleet exe dev): implement this funct */
}

}  // namespace distributed
}  // namespace paddle
