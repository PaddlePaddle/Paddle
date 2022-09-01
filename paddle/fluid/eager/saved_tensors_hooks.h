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
#include "paddle/phi/core/enforce.h"

namespace egr {

class SavedTensorsHooks {
 public:
  SavedTensorsHooks() = default;

  ~SavedTensorsHooks() {}

  void set_hooks(PyObject* pack_hook, PyObject* unpack_hook) {
    PADDLE_ENFORCE_EQ(pack_hook_ == nullptr && unpack_hook_ == nullptr,
                      true,
                      paddle::platform::errors::InvalidArgument(
                          "paddle.autograd.saved_tensors_hooks only one pair "
                          "of hooks is allowed at a time."));

    pack_hook_ = pack_hook;
    unpack_hook_ = unpack_hook;
    Py_XINCREF(pack_hook_);
    Py_XINCREF(unpack_hook_);
    is_enable_ = true;
  }

  void reset_hooks() {
    Py_XDECREF(pack_hook_);
    Py_XDECREF(unpack_hook_);
    pack_hook_ = nullptr;
    unpack_hook_ = nullptr;
    is_enable_ = false;
  }

  bool is_enable() { return is_enable_; }

  PyObject* get_pack_hook() { return pack_hook_; }
  PyObject* get_unpack_hook() { return unpack_hook_; }

  static SavedTensorsHooks& GetInstance() {
    static SavedTensorsHooks instance;
    return instance;
  }

 private:
  PyObject* pack_hook_{nullptr};
  PyObject* unpack_hook_{nullptr};
  bool is_enable_{false};
};

}  // namespace egr
