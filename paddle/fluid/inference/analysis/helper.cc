// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/analysis/helper.h"

namespace paddle::inference::analysis {

template <>
void SetAttr<std::string>(framework::proto::OpDesc *op,
                          const std::string &name,
                          const std::string &data) {
  auto *attr = op->add_attrs();
  attr->set_name(name);
  attr->set_type(paddle::framework::proto::AttrType::STRING);
  attr->set_s(data);
}
template <>
void SetAttr<int>(framework::proto::OpDesc *op,
                  const std::string &name,
                  const int &data) {
  auto *attr = op->add_attrs();
  attr->set_name(name);
  attr->set_type(paddle::framework::proto::AttrType::INT);
  attr->set_i(data);
}
template <>
void SetAttr<bool>(framework::proto::OpDesc *op,
                   const std::string &name,
                   const bool &data) {
  auto *attr = op->add_attrs();
  attr->set_name(name);
  attr->set_type(paddle::framework::proto::AttrType::BOOLEAN);
  attr->set_b(data);
}
template <>
void SetAttr<int64_t>(framework::proto::OpDesc *op,
                      const std::string &name,
                      const int64_t &data) {
  auto *attr = op->add_attrs();
  attr->set_name(name);
  attr->set_type(paddle::framework::proto::AttrType::LONG);
  attr->set_l(data);
}
template <>
void SetAttr<std::vector<std::string>>(framework::proto::OpDesc *op,
                                       const std::string &name,
                                       const std::vector<std::string> &data) {
  auto *attr = op->add_attrs();
  attr->set_name(name);
  attr->set_type(paddle::framework::proto::AttrType::STRINGS);
  for (const auto &s : data) {
    attr->add_strings(s.c_str());
  }
}

template <>
void SetAttr<std::vector<int>>(framework::proto::OpDesc *op,
                               const std::string &name,
                               const std::vector<int> &data) {
  auto *attr = op->add_attrs();
  attr->set_name(name);
  attr->set_type(paddle::framework::proto::AttrType::INTS);
  for (const auto i : data) {
    attr->add_ints(i);
  }
}

template <>
void SetAttr<std::vector<int64_t>>(framework::proto::OpDesc *op,
                                   const std::string &name,
                                   const std::vector<int64_t> &data) {
  auto *attr = op->add_attrs();
  attr->set_name(name);
  attr->set_type(paddle::framework::proto::AttrType::LONGS);
  for (const auto i : data) {
    attr->add_longs(i);
  }
}

}  // namespace paddle::inference::analysis
