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
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace distributed {

bool DistModel::Init() {
  /* TODO(fleet exe dev): implement this funct */
  if (!PrepareScope()) {
    return false;
  }
  if (!PrepareProgram()) {
    return false;
  }
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
  program_.reset(new framework::ProgramDesc(program_proto));
  return true;
}

void DistModel::Run(const std::vector<framework::Tensor> &input_data,
                    std::vector<framework::Tensor> *output_data) {
  /* TODO(fleet exe dev): implement this funct */
}

}  // namespace distributed
}  // namespace paddle
