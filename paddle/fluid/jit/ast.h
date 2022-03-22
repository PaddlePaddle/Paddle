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

namespace paddle {
namespace jit {
class IValue;
class Function;
class CompilationUnit;

class ClassType {
 public:
  ClassType(const std::vector<std::string>& names,
            std::weak_ptr<CompilationUnit> cu)
      : const_names_(names), compilation_unit_(cu) {}

  static std::shared_ptr<ClassType> Create(
      const std::vector<std::string>& names,
      std::weak_ptr<CompilationUnit> cu) {
    return std::make_shared<ClassType>(names, cu);
  }

  // const std::vector<Function*> Methods() const;

  // const IValue& GetAttribute(size_t slot) const;
  // const IValue& GetAttribute(const std::string& name) const;

  // size_t AddAttribute(const std::string& name, IValue val);

 private:
  // TODO(dev): disingwish parameter and buffer
  std::vector<std::string> const_names_;
  std::vector<IValue> const_value_;

  std::vector<Function*> methods_;
  std::vector<Function*> static_method_;
  std::weak_ptr<CompilationUnit> compilation_unit_;
};

}  // namespace jit
}  // namespace paddle
