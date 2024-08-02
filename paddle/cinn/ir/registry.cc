// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/registry.h"

#include <map>
#include <mutex>  // NOLINT
#include "paddle/common/enforce.h"

namespace cinn::ir {
struct Registry::Manager {
  static Manager *Global() {
    static Manager manager;
    return &manager;
  }

  std::mutex mu;
  std::map<std::string, Registry *> functions;

 private:
  Manager() = default;
  Manager(const Manager &) = delete;
  void operator=(Manager &) = delete;
};

Registry &Registry::SetBody(lang::PackedFunc f) {
  func_ = f;
  return *this;
}

Registry &Registry::SetBody(lang::PackedFunc::body_t f) {
  func_ = lang::PackedFunc(f);
  return *this;
}

Registry::Registry(const std::string &name) : name_(name) {}

/*static*/ Registry &Registry::Register(const std::string &name,
                                        bool can_override) {
  auto *manager = Registry::Manager::Global();
  std::lock_guard<std::mutex> lock(manager->mu);
  if (manager->functions.count(name)) {
    PADDLE_ENFORCE_EQ(
        can_override,
        true,
        ::common::errors::AlreadyExists(
            "Global PackedFunc[%s] already exists and cannot be overridden.",
            name));
  }

  auto *r = new Registry(name);
  manager->functions[name] = r;
  return *r;
}

/*static*/ bool Registry::Remove(const std::string &name) {
  auto manager = Manager::Global();
  std::lock_guard<std::mutex> lock(manager->mu);
  auto it = manager->functions.find(name);
  if (it != manager->functions.end()) {
    manager->functions.erase(it);
    return true;
  }
  return false;
}

/*static*/ const lang::PackedFunc *Registry::Get(const std::string &name) {
  auto *manager = Manager::Global();
  std::lock_guard<std::mutex> lock(manager->mu);
  auto *r = manager->functions[name];
  if (r) {
    return &r->func_;
  }
  return nullptr;
}

/*static*/ std::vector<std::string> Registry::ListNames() {
  auto *manager = Manager::Global();
  std::lock_guard<std::mutex> lock(manager->mu);
  std::vector<std::string> keys;
  for (const auto &_k_v_ : manager->functions) {
    auto &k = std::get<0>(_k_v_);
    auto &v = std::get<1>(_k_v_);
    keys.push_back(k);
  }
  return keys;
}

}  // namespace cinn::ir
