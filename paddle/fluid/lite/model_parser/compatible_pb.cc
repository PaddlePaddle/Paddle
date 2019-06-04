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
#include "compatible_pb.h"

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
      default:
        LOG(FATAL) << "Unsupported attr type found";
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
      case AttrType::INT:
        pb_desc->SetAttr<int32_t>(name, cpp_desc.GetAttr<int32_t>(name));
        break;
      case AttrType::FLOAT:
        pb_desc->SetAttr<float>(name, cpp_desc.GetAttr<float>(name));
        break;
      default:
        LOG(FATAL) << "Unsupported attr type found";
    }
  };
  for (const auto &attr_name : cpp_desc.AttrNames()) {
    auto type = cpp_desc.GetAttrType(attr_name);
    set_attr(attr_name, type);
  }
}

void TransformOpDescPbToCpp(const pb::OpDesc &pb_desc, cpp::OpDesc *cpp_desc) {
  InputsPbToCpp(pb_desc, cpp_desc);
  OutputsPbToCpp(pb_desc, cpp_desc);
  AttrsPbToCpp(pb_desc, cpp_desc);
}

void TransformOpDescCppToPb(const cpp::OpDesc &cpp_desc, pb::OpDesc *pb_desc) {
  InputsCppToPb(cpp_desc, pb_desc);
  OutputsCppToPb(cpp_desc, pb_desc);
  AttrsCppToPb(cpp_desc, pb_desc);
}

}  // namespace lite
}  // namespace paddle
