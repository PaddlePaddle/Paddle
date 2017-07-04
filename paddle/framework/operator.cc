/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/operator.h"

namespace paddle {
namespace framework {

Operator::Operator(const OpDesc &desc): op_desc_(desc) {
  inputs_.reserve(desc.inputs_size());
  for(const std::string& input: op_desc_.inputs()) {
    inputs_.push_back(input);
  }
  outputs_.reserve(op_desc_.outputs_size());
  for(const std::string& output: op_desc_.outputs()) {
    outputs_.push_back(output);
  }
  std::vector<AttrDesc> attrs;
  attrs.reserve(desc.attrs_size());
  for(const AttrDesc& attr: op_desc_.attrs()) {
    attrs.push_back(attr);
  }
  InitializeAttrs(attrs);
}

} // namespace framework
} // namespace paddle