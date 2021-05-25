/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_def.pb.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include "google/protobuf/text_format.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

void OpProtoAndCheckerMaker::Validate() {
  validated_ = true;
  CheckNoDuplicatedInOutAttrs();
}

OpProtoAndCheckerMaker::VariableBuilder OpProtoAndCheckerMaker::AddInput(
    const std::string& name, const std::string& comment) {
  auto* input = proto_->add_inputs();
  input->set_name(name);
  input->set_comment(comment);
  return OpProtoAndCheckerMaker::VariableBuilder{input};
}

OpProtoAndCheckerMaker::VariableBuilder OpProtoAndCheckerMaker::AddOutput(
    const std::string& name, const std::string& comment) {
  auto* output = proto_->add_outputs();
  output->set_name(name);
  output->set_comment(comment);
  return OpProtoAndCheckerMaker::VariableBuilder{output};
}

void OpProtoAndCheckerMaker::CheckNoDuplicatedInOutAttrs() {
  std::unordered_set<std::string> names;
  auto checker = [&](const std::string& name) {
    PADDLE_ENFORCE_EQ(
        names.count(name), 0,
        platform::errors::AlreadyExists("Attribute [%s] is duplicated.", name));
    names.insert(name);
  };
  for (auto& attr : proto_->attrs()) {
    checker(attr.name());
  }
  for (auto& input : proto_->inputs()) {
    checker(input.name());
  }
  for (auto& output : proto_->outputs()) {
    checker(output.name());
  }
}

bool WriteProtoToTextFile(const google::protobuf::Message& proto, const std::string& filename)
{
    int fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd == -1)
        return false;

    google::protobuf::io::FileOutputStream* output = new google::protobuf::io::FileOutputStream(fd);
    bool success = google::protobuf::TextFormat::Print(proto, output);

    delete output;
    close(fd);
    return success;
}

void PrintSample() {
  static bool a{false};
  if(!a) { 
    proto::OpDef op_def;
    op_def.set_type("while");
    op_def.mutable_def()->add_inputs()->set_name("X");
    op_def.mutable_def()->add_inputs()->set_name("Condition");
    op_def.mutable_def()->add_outputs()->set_name("Out");
    op_def.mutable_def()->add_outputs()->set_name("StepScopes");

    auto* sub_block = op_def.mutable_def()->add_attrs();
    sub_block->set_type(proto::AttrType::BLOCK);
    sub_block->set_name("sub_block");

    auto* is_test = op_def.mutable_extra()->add_attrs();
    is_test->set_type(proto::AttrType::BOOLEAN);
    is_test->set_name("is_test");

    auto* skip_eager_deletion_vars = op_def.mutable_extra()->add_attrs();
    skip_eager_deletion_vars->set_type(proto::AttrType::STRINGS);
    skip_eager_deletion_vars->set_name("skip_eager_deletion_vars");

    auto* op_role = op_def.mutable_extra()->add_attrs();
    op_role->set_type(proto::AttrType::INT);
    op_role->set_name("op_role");

    auto* op_role_var = op_def.mutable_extra()->add_attrs();
    op_role_var->set_type(proto::AttrType::STRINGS);
    op_role_var->set_name("op_role_var");

    auto* op_namescope = op_def.mutable_extra()->add_attrs();
    op_namescope->set_type(proto::AttrType::STRING);
    op_namescope->set_name("op_namescope");

    auto* op_callstack = op_def.mutable_extra()->add_attrs();
    op_callstack->set_type(proto::AttrType::STRINGS);
    op_callstack->set_name("op_callstack");

    auto* op_device = op_def.mutable_extra()->add_attrs();
    op_device->set_type(proto::AttrType::STRING);
    op_device->set_name("op_device");

    WriteProtoToTextFile(op_def, "hello.pbtxt");
    a = !a;
  }
}

void OpProtoAndCheckerMaker::operator()(proto::OpProto* proto,
                                        OpAttrChecker* attr_checker) {
  proto_ = proto;
  op_checker_ = attr_checker;
  Make();
  op_checker_->RecordExplicitCheckerNum();
  PrintSample();

  AddAttr<int>(OpRoleAttrName(), "The role of this operator")
      .InEnum(
          {static_cast<int>(OpRole::kForward),
           static_cast<int>(OpRole::kBackward),
           static_cast<int>(OpRole::kOptimize), static_cast<int>(OpRole::kRPC),
           static_cast<int>(OpRole::kDist), static_cast<int>(OpRole::kLRSched),
           static_cast<int>(OpRole::kLoss) | static_cast<int>(OpRole::kForward),
           static_cast<int>(OpRole::kLoss) |
               static_cast<int>(OpRole::kBackward),
           static_cast<int>(OpRole::kOptimize) |
               static_cast<int>(OpRole::kLRSched),
           static_cast<int>(OpRole::kNotSpecified)})
      .SetDefault(static_cast<int>(OpRole::kNotSpecified));
  AddAttr<std::vector<std::string>>(OpRoleVarAttrName(),
                                    "Optimized for variable")
      .SetDefault({});

  AddAttr<std::string>(OpNamescopeAttrName(), "Operator name with namesope.")
      .SetDefault("");

  AddAttr<std::vector<std::string>>(OpCreationCallstackAttrName(),
                                    "Callstack for Op Creatation.")
      .SetDefault({});
  AddAttr<std::string>(OpDeviceAttrName(), "Device type of this operator.")
      .SetDefault("");
  Validate();

  proto::OpDef op_def;

  op_def.set_type(proto_->type());
  auto* extra = op_def.mutable_extra();
  extra->add_inputs();
  extra->add_outputs();
  extra->add_attrs();

  auto* def = op_def.mutable_def();
  for (auto& i : proto_->inputs()) {
    auto* input = def->add_inputs();
    input->set_name(i.name());
  }
  for (auto& i : proto_->outputs()) {
    auto* output = def->add_outputs();
    output->set_name(i.name());
  }
  for (auto& i : proto_->attrs()) {
    auto* attr = def->add_attrs();
    attr->set_name(i.name());
    attr->set_type(i.type());
  }


  std::string path{std::string(proto_->type()) + ".pbtxt"};
  WriteProtoToTextFile(op_def, path);
  std::cout << "========= hello!! 123" << path << "=========\n";
}

}  // namespace framework
}  // namespace paddle
