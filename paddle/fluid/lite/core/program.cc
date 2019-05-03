// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/core/program.h"
#include "paddle/fluid/lite/core/optimizer.h"

namespace paddle {
namespace lite {

void RuntimeProgram::PersistModel(const std::string &path,
                                  const framework::proto::ProgramDesc &desc) {
  // Persist model.
  const std::string model_path = path + "/__model__";
  std::ofstream model_ostream(model_path, std::ios_base::binary);
  CHECK(model_ostream.is_open());
  const std::string pb_str = SerializeModelTopology(desc);
  model_ostream.write(pb_str.c_str(), pb_str.size());

  // Persist params.
  const std::string params_path = path + "/params";
  CHECK(!IsFileExists(params_path)) << "file " << params_path
                                    << " exists, can't overwrite";
  std::ofstream params_ostream(params_path, std::ios_base::binary);
  CHECK(params_ostream.is_open());
  framework::proto::ProgramDesc latest_program;
  latest_program.ParseFromString(pb_str);
  SerializeParams(params_ostream, latest_program);
}

std::string RuntimeProgram::SerializeModelTopology(
    const framework::proto::ProgramDesc &desc) {
  const std::string kKernelTypeAttr = "__@kernel_type_attr@__";
  auto program_dummy = desc;
  program_dummy.mutable_blocks(0)->clear_ops();
  for (auto &node : instructions_) {
    auto desc_dummy = node.op()->op_info()->desc();
    OpDesc desc(desc_dummy);
    desc.SetAttr(kKernelTypeAttr, node.kernel()->SerializeKernelType());
    // append new opdesc
    *program_dummy.mutable_blocks(0)->add_ops() = *desc.Proto();
  }
  return program_dummy.SerializeAsString();
}

void RuntimeProgram::SerializeParams(
    std::ostream &os, const framework::proto::ProgramDesc &desc) {
  std::vector<std::string> ws;
  for (auto &item : desc.blocks(0).vars()) {
    if (item.name() == "feed" || item.name() == "fetch") continue;
    if (item.persistable()) {
      ws.push_back(item.name());
    }
  }

  CHECK(exec_scope_);
  SerializeTensors(os, *exec_scope_, ws);
}

}  // namespace lite
}  // namespace paddle
