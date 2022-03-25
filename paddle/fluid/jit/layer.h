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
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/jit/ast.h"
#include "paddle/fluid/jit/compilation_unit.h"
#include "paddle/fluid/jit/function.h"
#include "paddle/fluid/jit/ivalue.h"
#include "paddle/fluid/jit/object.h"

namespace paddle {
namespace jit {

class Layer {
 public:
  // TODO(dev): Make vector<string>, num_slot as in argument
  // Layer(const std::shared_ptr<ClassType>& type) : obj_(type, /*num_slot*/ 0U)
  // {}
  Layer(const std::vector<framework::ProgramDesc>& progs,
        const std::vector<IValue>& params) {
    cout << "program size: " << progs.size() << endl;
    // Layer manage the life time of all parameter.
    params_ = params;
    for (size_t i = 0; i < progs.size(); ++i) {
      funcs_.emplace_back(std::make_shared<Function>(progs[i], params_));
    }
  }

  // TODO: make it as const function
  std::shared_ptr<Function> GetFunction(const std::string& name) {
    cout << "funcs_ size: " << funcs_.size() << endl;
    return funcs_[0];
  }

  std::vector<IValue> forward(const std::vector<IValue>& args) {
    auto func = GetFunction("forward");
    return (*func)(args);
  }

 private:
  // internal::Object obj_;
  // TODO: we should class them.
  std::vector<framework::ProgramDesc> progs_;
  std::vector<std::string> param_names_;
  std::vector<IValue> params_;
  std::vector<std::shared_ptr<Function>> funcs_;
};

}  // namespace jit
}  // namespace paddle
