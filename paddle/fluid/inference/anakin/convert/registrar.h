// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <functional>
#include <map>
#include <memory>
#include <string>

namespace paddle {
namespace inference {
namespace anakin {

class AnakinOpConverter;

class Register {
 public:
  Register() = default;
  std::shared_ptr<AnakinOpConverter> Create(const std::string &name);
  static Register *instance();
  void RegisterFn(const std::string &name,
                  std::function<std::shared_ptr<AnakinOpConverter>()> fn) {
    registry_[name] = fn;
  }

 private:
  std::map<std::string, std::function<std::shared_ptr<AnakinOpConverter>()>>
      registry_;
};

std::shared_ptr<AnakinOpConverter> Register::Create(const std::string &name) {
  auto it = registry_.find(name);
  if (it == registry_.end()) return nullptr;
  return it->second();
}

Register *Register::instance() {
  static Register factory;
  return &factory;
}

template <typename T, typename... Args>
class Registrar {
 public:
  Registrar(const std::string &name, Args... args) {
    std::shared_ptr<AnakinOpConverter> converter =
        std::make_shared<T>(std::move(args)...);
    Register::instance()->RegisterFn(name, [converter]() { return converter; });
  }
};

}  // namespace anakin
}  // namespace inference
}  // namespace paddle
