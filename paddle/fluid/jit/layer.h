// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/jit/ast.h"
#include "paddle/fluid/jit/base_function.h"
#include "paddle/fluid/jit/compilation_unit.h"
#include "paddle/fluid/jit/exector_function.h"
#include "paddle/fluid/jit/ivalue.h"
#include "paddle/fluid/jit/object.h"
#include "paddle/fluid/jit/pe_function.h"

namespace paddle {
namespace jit {

class Layer {
 public:
  // TODO(dev): Make vector<string>, num_slot as in argument
  // Layer(const std::shared_ptr<ClassType>& type) : obj_(type, /*num_slot*/ 0U)
  // {}
  Layer(
      const std::vector<std::string>& all_func_name,
      const std::vector<framework::ProgramDesc>& all_program_desc,
      const std::vector<std::vector<std::string>>& param_name_for_each_program,
      const std::map<std::string, IValue>& param_dict) {
    VLOG(3) << "program size: " << all_program_desc.size();
    // Layer manage the life time of all parameter.
    // params_ = params;
    for (size_t i = 0; i < all_func_name.size(); ++i) {
      function_dict[all_func_name[i]] = std::make_shared<ExectorFunction>(
          all_program_desc[i], param_name_for_each_program[i], param_dict);
    }
  }

  // TODO(dev): make it as const function
  std::shared_ptr<BaseFunction> GetFunction(const std::string& name) {
    VLOG(3) << "funcs_ size: " << function_dict.size();
    return function_dict[name];
  }

  std::vector<IValue> forward(const std::vector<IValue>& args) {
    auto func = GetFunction("forward");
    return (*func)(args);
  }

 private:
  // internal::Object obj_;
  // TODO(dev): we should class them.
  // std::vector<framework::ProgramDesc> all_program_desc_;
  // std::vector<std::vector<std::string>> param_name_for_each_program_;
  // std::vector<IValue> all_param_;
  std::map<std::string, std::shared_ptr<BaseFunction>> function_dict;
};

}  // namespace jit
}  // namespace paddle
