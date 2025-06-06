// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <mutex>
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/registry/registry.h"

namespace ap::registry {

struct RegistrySingleton {
  static adt::Result<Registry> Singleton() {
    std::unique_lock<std::mutex> lock(*SingletonMutex());
    ADT_CHECK(MutOptSingleton()->has_value())
        << adt::errors::NotImplementedError{
               std::string() + "Registry singleton not initialized. "};
    return MutOptSingleton()->value();
  }

  static void Add(const AbstractDrrPassRegistryItem& item) {
    auto registry = MutSingleton();
    const auto& abstract_drr_pass_name = item->abstract_drr_pass_name;
    int64_t nice = item->nice;
    std::unique_lock<std::mutex> lock(*SingletonMutex());
    registry->abstract_drr_pass_registry_items[abstract_drr_pass_name][nice]
        .emplace_back(item);
  }

  static void Add(const ClassicDrrPassRegistryItem& item) {
    auto registry = MutSingleton();
    const auto& classic_drr_pass_name = item->classic_drr_pass_name;
    int64_t nice = item->nice;
    std::unique_lock<std::mutex> lock(*SingletonMutex());
    registry->classic_drr_pass_registry_items[classic_drr_pass_name][nice]
        .emplace_back(item);
  }

  static void Add(const AccessTopoDrrPassRegistryItem& item) {
    auto registry = MutSingleton();
    const auto& access_topo_drr_pass_name = item->access_topo_drr_pass_name;
    int64_t nice = item->nice;
    std::unique_lock<std::mutex> lock(*SingletonMutex());
    registry
        ->access_topo_drr_pass_registry_items[access_topo_drr_pass_name][nice]
        .emplace_back(item);
  }

  static std::mutex* SingletonMutex() {
    static std::mutex mutex;
    return &mutex;
  }

 private:
  static Registry MutSingleton() {
    std::unique_lock<std::mutex> lock(*SingletonMutex());
    if (!MutOptSingleton()->has_value()) {
      *MutOptSingleton() = Registry{};
    }
    return MutOptSingleton()->value();
  }

  static std::optional<Registry>* MutOptSingleton() {
    static std::optional<Registry> ctx{};
    return &ctx;
  }
};

}  // namespace ap::registry
