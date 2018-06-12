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
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/contrib/dynamic/variable.h"

namespace paddle {
namespace dynamic {

using VariableHandleMap = std::map<std::string, std::vector<VariableHandle>>;

struct OpHandle {
  OpHandle(const std::string &type,
           const VariableHandleMap &in_vars,
           const VariableHandleMap &out_vars,
           const framework::AttributeMap &attrs)
      : type_(type), inputs_(in_vars), outputs_(out_vars), attrs_(attrs) {}

  std::string type_;
  VariableHandleMap inputs_;
  VariableHandleMap outputs_;
  framework::AttributeMap attrs_;
};

class Tape {
 public:
  void AddOp(const std::string &type,
             const VariableHandleMap &in_vars,
             VariableHandleMap out_vars,
             const framework::AttributeMap &attrs);
  void Forward();
  void Backward(VariableHandle target);

 private:
  bool has_been_backwarded_ = false;
  size_t current_position_ = 0;

  std::vector<OpHandle> tape_;
  std::shared_ptr<Tape> backward_tape_;
};

Tape &get_global_tape();

void reset_global_tape();
}
}
