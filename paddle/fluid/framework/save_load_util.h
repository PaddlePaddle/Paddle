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

#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/imperative/type_defs.h"

namespace paddle {
namespace framework {

class Scope;

bool SaveStaticNameListToDisk(
    const std::string& file_name,
    const std::vector<std::string>& vec_tensor_name_list, const Scope& scope);

bool LoadStaticNameListFromDisk(
    const std::string& file_name,
    const std::vector<std::string>& vec_tensor_name_list, const Scope& scope);

bool SaveDygraphVarBaseListToDisk(
    const std::string& file_name,
    const std::vector<std::shared_ptr<imperative::VarBase>>& vec_var_base_list);

const std::vector<std::shared_ptr<imperative::VarBase>>
LoadDygraphVarBaseListFromDisk(const std::string& file_name);

bool SaveTensorToDisk(const std::string& file_name,
                      const std::map<std::string, Tensor*>& map_tensor);

bool LoadTensorFromDisk(
    const std::string& file_name,
    std::map<std::string, std::shared_ptr<Tensor>>* map_tensor);

}  // namespace framework
}  // namespace paddle
