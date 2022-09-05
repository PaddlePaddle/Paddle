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

#include <Python.h>
#include "paddle/fluid/eager/hooks.h"
#include "paddle/phi/core/enforce.h"

namespace egr {

class SavedTensorsHooks {
 public:
  SavedTensorsHooks() = default;

  ~SavedTensorsHooks() {}

  void SetHooks(PyObject* pack_hook, PyObject* unpack_hook);

  void ResetHooks() {
    pack_hook_ = nullptr;
    unpack_hook_ = nullptr;
    is_enable_ = false;
  }

  bool IsEnable() { return is_enable_; }

  std::shared_ptr<PackHookBase> GetPackHook() { return pack_hook_; }
  std::shared_ptr<UnPackHookBase> GetUnPackHook() { return unpack_hook_; }

  static SavedTensorsHooks& GetInstance() {
    static SavedTensorsHooks instance;
    return instance;
  }

 private:
  std::shared_ptr<PackHookBase> pack_hook_;
  std::shared_ptr<UnPackHookBase> unpack_hook_;
  bool is_enable_{false};
};

}  // namespace egr
