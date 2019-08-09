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

void InputsPbToCpp(const pb::OpDesc &pb_desc, cpp::OpDesc *cpp_desc) {
  for (const std::string &param : pb_desc.InputArgumentNames()) {
    cpp_desc->SetInput(param, pb_desc.Input(param));
  }
}

void InputsCppToPb(const cpp::OpDesc &cpp_desc, pb::OpDesc *pb_desc) {
  for (const std::string &param : cpp_desc.InputArgumentNames()) {
    pb_desc->SetInput(param, cpp_desc.Input(param));
  }
}

void OutputsPbToCpp(const pb::OpDesc &pb_desc, cpp::OpDesc *cpp_desc) {
  for (const std::string &param : pb_desc.OutputArgumentNames()) {
    cpp_desc->SetOutput(param, pb_desc.Output(param));
  }
}

void OutputsCppToPb(const cpp::OpDesc &cpp_desc, pb::OpDesc *pb_desc) {
  for (const std::string &param : cpp_desc.OutputArgumentNames()) {
    pb_desc->SetOutput(param, cpp_desc.Output(param));
  }
}

void AttrsPbToCpp(const pb::OpDesc &pb_desc, cpp::OpDesc *cpp_desc) {
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
      case AttrType::LONGS:
        cpp_desc->SetAttr<std::vector<int64_t>>(
            name, pb_desc.GetAttr<std::vector<int64_t>>(name));
        break;
      case AttrType::LONG:
        cpp_desc->SetAttr<int64_t>(name, pb_desc.GetAttr<int64_t>(name));
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

void AttrsCppToPb(const cpp::OpDesc &cpp_desc, pb::OpDesc *pb_desc) {
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
  InputsPbToCpp(pb_desc, cpp_desc);
  OutputsPbToCpp(pb_desc, cpp_desc);
  AttrsPbToCpp(pb_desc, cpp_desc);
}

void TransformOpDescCppToPb(const cpp::OpDesc &cpp_desc, pb::OpDesc *pb_desc) {
  pb_desc->SetType(cpp_desc.Type());
  InputsCppToPb(cpp_desc, pb_desc);
  OutputsCppToPb(cpp_desc, pb_desc);
  AttrsCppToPb(cpp_desc, pb_desc);
}

}  // namespace lite
}  // namespace paddle
