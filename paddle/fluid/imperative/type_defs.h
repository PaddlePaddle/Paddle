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

#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace paddle {
namespace imperative {

class VarBase;
class OpBase;

typedef std::map<std::string, std::vector<std::shared_ptr<VarBase>>>
    VarBasePtrMap;
typedef std::vector<std::weak_ptr<VarBase>> VarBaseWeakPtrList;
typedef std::map<std::string, std::vector<OpBase*>> OpBasePtrMap;
typedef std::unordered_map<
    const VarBase*,
    std::pair<platform::Place,
              std::vector<std::pair<int, std::shared_ptr<VarBase>>>>>
    BackwardSumMap;  // var_grad -> {place, {id -> var_grad@rename}}
typedef std::unordered_map<const VarBase*, std::pair<int, bool>> GradientRef;
// var_grad -> {ref_times, is_first_to_be_accumulate}

}  // namespace imperative
}  // namespace paddle
