// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <memory>
#include <utility>
#include <vector>

#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/api/include/tensor.h"
namespace egr {

class TensorHook {
 public:
  virtual ~TensorHook() = default;
  virtual paddle::experimental::Tensor operator()(
      const paddle::experimental::Tensor& var) = 0;
};

class VoidHook {
 public:
  virtual ~VoidHook() = default;
  virtual void operator()() = 0;
};

class CppTensorHook : public TensorHook {
 public:
  explicit CppTensorHook(const std::function<paddle::experimental::Tensor(
                             const paddle::experimental::Tensor&)>& fn)
      : fn_(std::move(fn)) {}

  paddle::experimental::Tensor operator()(
      const paddle::experimental::Tensor& var) override {
    return fn_(var);
  }

 private:
  std::function<paddle::experimental::Tensor(
      const paddle::experimental::Tensor&)>
      fn_;
};

class CppVoidHook : public VoidHook {
 public:
  explicit CppVoidHook(const std::function<void()>& fn) : fn_(std::move(fn)) {}

  void operator()() override { return fn_(); }

 private:
  std::function<void()> fn_;
};

class PyObjectHolderBase {
 public:
  virtual ~PyObjectHolderBase() = default;
  virtual void* get() = 0;
  virtual void reset(void* ptr) = 0;
  virtual void inc_ref() = 0;
  virtual void dec_ref() = 0;
};

class PackHookBase {
 public:
  virtual ~PackHookBase() = default;
  virtual std::shared_ptr<PyObjectHolderBase> operator()(
      const paddle::experimental::Tensor& tensor) = 0;
  virtual void* operator()(void* py_tensor) = 0;
};

class UnPackHookBase {
 public:
  virtual ~UnPackHookBase() = default;
  virtual paddle::experimental::Tensor operator()(
      std::shared_ptr<PyObjectHolderBase> packed_value) = 0;
  virtual void* operator()(void* packed_value, void* other) = 0;
};

class SavedTensorsHooks {
 public:
  SavedTensorsHooks() = default;

  ~SavedTensorsHooks() {}

  void SetHooks(std::shared_ptr<PackHookBase> pack_hook,
                std::shared_ptr<UnPackHookBase> unpack_hook) {
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
