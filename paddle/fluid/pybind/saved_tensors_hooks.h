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

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/pybind/pyobject_holder.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

namespace paddle {
namespace pybind {

class PackHook {
 public:
  virtual ~PackHook() = default;
  virtual std::shared_ptr<PyObjectHolder> operator()(
      const paddle::experimental::Tensor& tensor) = 0;
  virtual void* operator()(void* py_tensor) = 0;
};

class UnPackHook {
 public:
  virtual ~UnPackHook() = default;
  virtual paddle::experimental::Tensor operator()(
      std::shared_ptr<PyObjectHolder> packed_value) = 0;
  virtual void* operator()(void* packed_value, void* other) = 0;
};

class SavedTensorsHooks {
 public:
  SavedTensorsHooks() = default;

  ~SavedTensorsHooks() {}

  void SetHooks(std::shared_ptr<PackHook> pack_hook,
                std::shared_ptr<UnPackHook> unpack_hook) {
    PADDLE_ENFORCE_EQ(pack_hook_ == nullptr && unpack_hook_ == nullptr,
                      true,
                      paddle::platform::errors::InvalidArgument(
                          "paddle.autograd.saved_tensors_hooks only one pair "
                          "of hooks is allowed at a time."));
    pack_hook_ = pack_hook;
    unpack_hook_ = unpack_hook;
    is_enable_ = true;
  }

  void ResetHooks() {
    pack_hook_ = nullptr;
    unpack_hook_ = nullptr;
    is_enable_ = false;
  }

  bool IsEnable() { return is_enable_; }

  std::shared_ptr<PackHook> GetPackHook() { return pack_hook_; }
  std::shared_ptr<UnPackHook> GetUnPackHook() { return unpack_hook_; }

  static SavedTensorsHooks& GetInstance() {
    static SavedTensorsHooks instance;
    return instance;
  }

 private:
  std::shared_ptr<PackHook> pack_hook_;
  std::shared_ptr<UnPackHook> unpack_hook_;
  bool is_enable_{false};
};

}  // namespace pybind
}  // namespace paddle
