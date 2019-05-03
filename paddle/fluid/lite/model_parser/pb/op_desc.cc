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

#include "paddle/fluid/lite/model_parser/pb/op_desc.h"

namespace paddle {
namespace lite {
namespace pb {

template <>
void OpDesc::SetAttr<std::string>(const std::string &name,
                                  const std::string &v) {
  auto &xs = *desc_.mutable_attrs();
  auto it = std::find_if(
      xs.begin(), xs.end(),
      [&](const framework::proto::OpDesc_Attr &x) { return x.name() == name; });
  if (it == xs.end()) {
    auto *attr = xs.Add();
    attr->set_name(name);
    it = std::find_if(xs.begin(), xs.end(),
                      [&](const framework::proto::OpDesc_Attr &x) {
                        return x.name() == name;
                      });
  }

  it->set_type(framework::proto::STRING);
  it->set_s(v.c_str());
}

}  // namespace pb
}  // namespace lite
}  // namespace paddle
