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

#include "paddle/fluid/lite/model_parser/compatible_pb.h"
#include <string>
#include <vector>

namespace paddle {
namespace lite {

void OpInputsPbToCpp(const pb::OpDesc &pb_desc, cpp::OpDesc *cpp_desc) {
  for (const std::string &param : pb_desc.InputArgumentNames()) {
    cpp_desc->SetInput(param, pb_desc.Input(param));
  }
}

void OpInputsCppToPb(const cpp::OpDesc &cpp_desc, pb::OpDesc *pb_desc) {
  for (const std::string &param : cpp_desc.InputArgumentNames()) {
    pb_desc->SetInput(param, cpp_desc.Input(param));
  }
}

void OpOutputsPbToCpp(const pb::OpDesc &pb_desc, cpp::OpDesc *cpp_desc) {
  for (const std::string &param : pb_desc.OutputArgumentNames()) {
    cpp_desc->SetOutput(param, pb_desc.Output(param));
  }
}

void OpOutputsCppToPb(const cpp::OpDesc &cpp_desc, pb::OpDesc *pb_desc) {
  for (const std::string &param : cpp_desc.OutputArgumentNames()) {
    pb_desc->SetOutput(param, cpp_desc.Output(param));
  }
}

void OpAttrsPbToCpp(const pb::OpDesc &pb_desc, cpp::OpDesc *cpp_desc) {
  using AttrType = OpDescAPI::AttrType;
  auto set_attr = [&](const std::string &name, AttrType type) {
    switch (type) {
      case AttrType::INT:
        cpp_desc->SetAttr<int32_t>(name, pb_desc.GetAttr<int32_t>(name));
        break;
      case AttrType::FLOAT:
        cpp_desc->SetAttr<float>(name, pb_desc.GetAttr<float>(name));
        break;
      case AttrType::STRING:
        cpp_desc->SetAttr<std::string>(name,
                                       pb_desc.GetAttr<std::string>(name));
        break;
      case AttrType::INTS:
        cpp_desc->SetAttr<std::vector<int>>(
            name, pb_desc.GetAttr<std::vector<int>>(name));
        break;
      case AttrType::FLOATS:
        cpp_desc->SetAttr<std::vector<float>>(
            name, pb_desc.GetAttr<std::vector<float>>(name));
        break;
      case AttrType::BOOLEAN:
        cpp_desc->SetAttr<bool>(name, pb_desc.GetAttr<bool>(name));
        break;
      case AttrType::STRINGS:
        cpp_desc->SetAttr<std::vector<std::string>>(
            name, pb_desc.GetAttr<std::vector<std::string>>(name));
        break;
      default:
        LOG(FATAL) << "Unsupported attr type found " << static_cast<int>(type);
    }
  };

  for (const auto &attr_name : pb_desc.AttrNames()) {
    auto type = pb_desc.GetAttrType(attr_name);
    set_attr(attr_name, type);
  }
}

void OpAttrsCppToPb(const cpp::OpDesc &cpp_desc, pb::OpDesc *pb_desc) {
  using AttrType = OpDescAPI::AttrType;
  auto set_attr = [&](const std::string &name, AttrType type) {
    switch (type) {
#define IMPL_ONE(type__, T)                               \
  case AttrType::type__:                                  \
    pb_desc->SetAttr<T>(name, cpp_desc.GetAttr<T>(name)); \
    break;
      IMPL_ONE(INT, int32_t);
      IMPL_ONE(FLOAT, float);
      IMPL_ONE(STRING, std::string);
      IMPL_ONE(STRINGS, std::vector<std::string>);
      IMPL_ONE(FLOATS, std::vector<float>);
      IMPL_ONE(INTS, std::vector<int>);
      IMPL_ONE(BOOLEAN, bool);
      default:
        LOG(FATAL) << "Unsupported attr type found: " << static_cast<int>(type);
    }
  };
#undef IMPL_ONE
  for (const auto &attr_name : cpp_desc.AttrNames()) {
    auto type = cpp_desc.GetAttrType(attr_name);
    set_attr(attr_name, type);
  }
}

void TransformOpDescPbToCpp(const pb::OpDesc &pb_desc, cpp::OpDesc *cpp_desc) {
  cpp_desc->SetType(pb_desc.Type());
  OpInputsPbToCpp(pb_desc, cpp_desc);
  OpOutputsPbToCpp(pb_desc, cpp_desc);
  OpAttrsPbToCpp(pb_desc, cpp_desc);
}

void TransformOpDescCppToPb(const cpp::OpDesc &cpp_desc, pb::OpDesc *pb_desc) {
  pb_desc->SetType(cpp_desc.Type());
  OpInputsCppToPb(cpp_desc, pb_desc);
  OpOutputsCppToPb(cpp_desc, pb_desc);
  OpAttrsCppToPb(cpp_desc, pb_desc);
}

void TransformVarDescPbToCpp(const pb::VarDesc &pb_desc,
                             cpp::VarDesc *cpp_desc) {
  using VarType = lite::VarDescAPI::VarDataType;

#define VarDescCopyOnce(name__) cpp_desc->Set##name__(pb_desc.name__())
  VarDescCopyOnce(Name);
  VarDescCopyOnce(Persistable);
  VarDescCopyOnce(Type);
  auto type = cpp_desc->Type();
  if (type == VarType::SELECTED_ROWS || type == VarType::LOD_TENSOR ||
      type == VarType::LOD_TENSOR_ARRAY) {
    VarDescCopyOnce(Shape);
    VarDescCopyOnce(DataType);
  } else {
    cpp_desc->SetShape(std::vector<int64_t>({1}));
    cpp_desc->SetDataType(VarType::UNK);
  }
#undef VarDescCopyOnce
}

void TransformVarDescCppToPb(const cpp::VarDesc &cpp_desc,
                             pb::VarDesc *pb_desc) {
  LOG(FATAL) << "Not supported now.";
}

}  // namespace lite
}  // namespace paddle
