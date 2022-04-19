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
#include "paddle/phi/api/include/tensor.h"
namespace egr {

class TensorHook {
 public:
  virtual ~TensorHook() = default;
  virtual paddle::experimental::Tensor operator()(
      const paddle::experimental::Tensor& var) = 0;
};

class TensorVoidHook {
 public:
  virtual ~TensorVoidHook() = default;
  virtual void operator()() = 0;
};

class CppTensorHook : public TensorHook {
 public:
  explicit CppTensorHook(std::function<paddle::experimental::Tensor(
                             const paddle::experimental::Tensor&)>&& fn)
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

class CppTensorVoidHook : public TensorVoidHook {
 public:
  explicit CppTensorVoidHook(std::function<void()>&& fn) : fn_(std::move(fn)) {}

  void operator()() override { return fn_(); }

 private:
  std::function<void()> fn_;
};
}  // namespace egr
