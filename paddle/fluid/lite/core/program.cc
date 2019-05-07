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

void RuntimeProgram::PersistModel(const std::string &dir,
                                  const framework::proto::ProgramDesc &desc) {
  // Persist model.
  const std::string model_path = dir + "/__model__";
  std::ofstream model_ostream(model_path, std::ios_base::binary);
  CHECK(model_ostream.is_open());
  const std::string pb_str = SerializeProgram(desc);
  model_ostream.write(pb_str.c_str(), pb_str.size());
  model_ostream.close();

  // Persist params.
  framework::proto::ProgramDesc latest_program;
  latest_program.ParseFromString(pb_str);
  SaveParams(dir, latest_program);
}

std::string RuntimeProgram::SerializeProgram(
    const framework::proto::ProgramDesc &desc) {
  auto program_dummy = desc;
  program_dummy.mutable_blocks(0)->clear_ops();
  for (auto &node : instructions_) {
    auto desc_dummy = node.op()->op_info()->desc();
    OpDesc desc(desc_dummy);
    desc.SetAttr(kKernelTypeAttr, node.kernel()->SerializedKernelType());
    // append new opdesc
    *program_dummy.mutable_blocks(0)->add_ops() = *desc.Proto();
  }
  return program_dummy.SerializeAsString();
}

void RuntimeProgram::SaveParams(const std::string &dir,
                                const framework::proto::ProgramDesc &desc) {
  CHECK(exec_scope_);
  for (auto &item : desc.blocks(0).vars()) {
    const std::string path = dir + "/" + item.name();
    if (item.name() == "feed" || item.name() == "fetch") continue;
    if (item.persistable()) {
      std::ofstream file(path, std::ios::binary);
      SerializeTensor(file, *exec_scope_, item.name());
      file.close();
    }
  }
}

}  // namespace lite
}  // namespace paddle
