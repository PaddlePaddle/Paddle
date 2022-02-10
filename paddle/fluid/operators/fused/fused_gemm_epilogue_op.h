/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>
#include <unordered_map>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {

class EpilogueMetaData {
 public:
  memory::allocation::AllocationPtr auxiliary = nullptr;
};

class EpilogueSingleton {
 public:
  static EpilogueSingleton& Instance() {
    static EpilogueSingleton instance;
    return instance;
  }

  EpilogueSingleton(const EpilogueSingleton&) = delete;
  void operator=(const EpilogueSingleton&) = delete;

  EpilogueMetaData& Data(const std::string& str) { return map_[str]; }

 private:
  EpilogueSingleton() {}
  std::unordered_map<std::string, EpilogueMetaData> map_;
};

}  // namespace operators
}  // namespace paddle
