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
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/platform/variant.h"

namespace paddle {
namespace framework {
class OperatorBase;
class OpDescBind;
class BlockDescBind;
class BlockDesc;
class InferShapeContext;
class BlockDescBind;

using VariableNameMap = std::map<std::string, std::vector<std::string>>;

// The order should be as same as framework.proto
using Attribute =
    boost::variant<boost::blank, int, float, std::string, std::vector<int>,
                   std::vector<float>, std::vector<std::string>, bool,
                   std::vector<bool>, BlockDescBind*>;

using AttributeMap = std::unordered_map<std::string, Attribute>;

using OpCreator = std::function<OperatorBase*(
    const std::string& /*type*/, const VariableNameMap& /*inputs*/,
    const VariableNameMap& /*outputs*/, const AttributeMap& /*attrs*/)>;

using GradOpMakerFN = std::function<std::vector<std::unique_ptr<OpDescBind>>(
    const OpDescBind&, const std::unordered_set<std::string>& /*no_grad_set*/,
    std::unordered_map<std::string, std::string>* /*grad_to_var*/,
    const std::vector<BlockDescBind*>& grad_block)>;

using InferVarTypeFN = std::function<void(const OpDescBind& /*op_desc*/,
                                          BlockDescBind* /*block*/)>;

using InferShapeFN = std::function<void(InferShapeContext*)>;

}  // namespace framework
}  // namespace paddle
