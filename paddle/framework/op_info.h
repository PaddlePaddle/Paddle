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

#pragma once
#include <functional>
#include <map>
#include <string>
#include <unordered_map>

#include "paddle/framework/attribute.h"

namespace paddle {
namespace framework {
class OperatorBase;
using VariableNameMap = std::map<std::string, std::vector<std::string>>;

using OpCreator = std::function<OperatorBase*(
    const std::string& /*type*/, const VariableNameMap& /*inputs*/,
    const VariableNameMap& /*outputs*/, const AttributeMap& /*attrs*/)>;

struct OpInfo {
  OpCreator creator_;
  std::string grad_op_type_;
  OpProto* proto_;
  OpAttrChecker* checker_;
};

extern std::unordered_map<std::string, const OpInfo>& OpInfoMap();

}  // namespace framework
}  // namespace paddle
