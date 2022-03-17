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
#include "paddle/fluid/jit/serializer.h"

namespace paddle {
namespace jit {

class Layer {
 public:
  // TODO(dev): Make vector<string>, num_slot as in argument
  Layer(std::shared_ptr<CompilationUnit> cu, const ClassType& type)
      : obj_(ClassType::Create({}, cu), /*num_slot*/ 0U) {}

  void save(const std::string& file_path) const { Export(*this, file_path); }

  Function* get_mothod(const std::string& name) const;

  std::vector<IValue> forward(const std::vector<IValue>& args);

 private:
  internal::Object obj_;
};

}  // namespace jit
}  // namespace paddle
