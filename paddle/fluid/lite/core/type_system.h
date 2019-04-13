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
// This file contains the file system of the lite system. Every data type in
// Variable should be registered here, and the analysis phase will check the
// data type correction.
// This mechanism is made for keeping our system simpler and more stable, for
// the dubious typed Variables in the Operators' inputs and outputs are disaster
// for analysis and runtime.

#include <glog/logging.h>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/lite/core/tensor.h"

namespace paddle {
namespace lite {

// NOTE TypeSystem has some overhead, and better to be used in analysis phase.
class TypeSystem {
 private:
  // Put all valid types for Variables here!
  TypeSystem() {
    // Tensor is a valid data type for Variable.
    Register<Tensor>("tensor");
  }

 public:
  static TypeSystem& Global() {
    static TypeSystem x;
    return x;
  }

  template <typename T>
  void Register(const std::string& type) {
    size_t hash = typeid(T).hash_code();
    CHECK(!types_.count(hash)) << "duplicate register type " << type
                               << " found!";
    types_[hash] = type;
    names_.insert(type);
  }

  template <typename T>
  bool Contains() const {
    return types_.count(typeid(T).hash_code());
  }

  bool Contains(size_t hash) const { return types_.count(hash); }

  bool Contains(const std::string& type) { return names_.count(type); }

  std::string DebugInfo() const {
    std::stringstream ss;
    for (const auto& it : types_) {
      ss << it.second << "\n";
    }
    return ss.str();
  }

 private:
  std::unordered_map<size_t /*hash*/, std::string /*name*/> types_;
  TypeSystem(const TypeSystem&) = delete;
  std::unordered_set<std::string> names_;
};

}  // namespace lite
}  // namespace paddle
