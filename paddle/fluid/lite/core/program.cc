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
    pb::OpDesc pb_desc;
    TransformOpDescCppToPb(*node.op()->op_info(), &pb_desc);
    pb_desc.SetAttr(kKernelTypeAttr, node.kernel()->SerializedKernelType());
    // append new opdesc
    *program_dummy.mutable_blocks(0)->add_ops() = *pb_desc.Proto();
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

void Program::Build(const framework::proto::ProgramDesc &program) {
  CHECK(ops_.empty()) << "Executor duplicate Build found";
  // Create operators.
  for (const auto &proto_op_desc : program.blocks(0).ops()) {
    lite::OpDesc op_desc_dummy(proto_op_desc);
    cpp::OpDesc op_desc;
    TransformOpDescPbToCpp(op_desc_dummy, &op_desc);
    auto op_type = op_desc.Type();
    // if (op_type == "feed" || op_type == "fetch") continue;
    VLOG(4) << "create Op [" << op_type << "]";
    LOG(INFO) << "create Op [" << op_type << "]";
    auto op = LiteOpRegistry::Global().Create(op_type);
    CHECK(op) << "no Op found for " << op_type;
    ops_.emplace_back(std::move(op));
    ops_.back()->Attach(op_desc, exec_scope_);
  }
}

void Program::PrepareWorkspace(const framework::proto::ProgramDesc &program) {
  CHECK(!exec_scope_) << "Duplicate PrepareWorkspace found";
  exec_scope_ = &scope_->NewScope();
  // Create Feed and Fetch var.
  scope_->Var("feed")->GetMutable<std::vector<lite::Tensor>>();
  scope_->Var("fetch")->GetMutable<std::vector<lite::Tensor>>();

  CHECK(!program.blocks().empty());

  // Add feed/fetch vars
  vars_.emplace_back(new VarInfo);
  vars_.back()->SetName("feed");
  vars_.back()->SetPersistable(false);
  vars_.emplace_back(new VarInfo);
  vars_.back()->SetName("fetch");
  vars_.back()->SetPersistable(false);

  for (auto proto_var_desc : program.blocks(0).vars()) {
    pb::VarDesc pb_var_desc(proto_var_desc);
    auto name = pb_var_desc.Name();
    // Ignore feed/fetch vars in origin program
    if (name == "feed" || name == "fetch") continue;
    cpp::VarDesc cpp_var_desc;
    TransformVarDescPbToCpp(pb_var_desc, &cpp_var_desc);
    vars_.emplace_back(new VarInfo(cpp_var_desc));

    if (!vars_.back()->Persistable()) {
      exec_scope_->Var(name);
    }
  }
}

}  // namespace lite
}  // namespace paddle
