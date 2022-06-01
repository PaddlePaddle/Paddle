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
  Layer(const std::vector<std::string>& func_names,
        const std::vector<framework::ProgramDesc>& progs,
        const std::vector<IValue>& params)
      : progs_(progs), params_(params) {
    VLOG(3) << "program size: " << progs.size();
    // Layer manage the life time of all parameter.
    // params_ = params;
    for (size_t i = 0; i < func_names.size(); ++i) {
      funcs_.insert(std::make_pair(
          func_names[i], std::make_shared<PEFunction>(progs_[i], params_)));
    }
  }

  // TODO(dev): make it as const function
  std::shared_ptr<BaseFunction> GetFunction(const std::string& name) {
    VLOG(3) << "funcs_ size: " << funcs_.size();
    return funcs_[name];
  }

  std::vector<IValue> forward(const std::vector<IValue>& args) {
    auto func = GetFunction("forward");
    return (*func)(args);
  }

 private:
  // internal::Object obj_;
  // TODO(dev): we should class them.
  std::vector<framework::ProgramDesc> progs_;
  // std::vector<std::string> param_names_;
  std::vector<IValue> params_;
  // std::vector<std::shared_ptr<BaseFunction>> funcs_;

  // std::vector<std::string> func_names;
  std::map<std::string, std::shared_ptr<BaseFunction>> funcs_;
};

}  // namespace jit
}  // namespace paddle
