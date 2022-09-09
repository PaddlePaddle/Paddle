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
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/hooks.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

namespace egr {
#if !(defined(PADDLE_NO_PYTHON) && defined(PADDLE_ON_INFERENCE))
class PackHook : public PackHookBase {
 public:
  explicit PackHook(PyObject* hook);

  ~PackHook();

  void* operator()(const paddle::experimental::Tensor& tensor) override;

  void* operator()(void* py_tensor) override;

 private:
  PyObject* hook_;
};

class UnPackHook : public UnPackHookBase {
 public:
  explicit UnPackHook(PyObject* hook);

  ~UnPackHook();

  paddle::experimental::Tensor operator()(void* packed_value) override;

  void* operator()(void* packed_value, void* other) override;

 private:
  PyObject* hook_;
};
#endif

class SavedTensorsHooks {
 public:
  SavedTensorsHooks() = default;

  ~SavedTensorsHooks() {}

  void SetHooks(PyObject* pack_hook, PyObject* unpack_hook) {
#if !(defined(PADDLE_NO_PYTHON) && defined(PADDLE_ON_INFERENCE))
    PADDLE_ENFORCE_EQ(pack_hook_ == nullptr && unpack_hook_ == nullptr,
                      true,
                      paddle::platform::errors::InvalidArgument(
                          "paddle.autograd.saved_tensors_hooks only one pair "
                          "of hooks is allowed at a time."));
    pack_hook_ = std::make_shared<PackHook>(pack_hook);
    unpack_hook_ = std::make_shared<UnPackHook>(unpack_hook);
    is_enable_ = true;
#endif
  }

  void ResetHooks() {
#if !(defined(PADDLE_NO_PYTHON) && defined(PADDLE_ON_INFERENCE))
    pack_hook_ = nullptr;
    unpack_hook_ = nullptr;
    is_enable_ = false;
#endif
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
