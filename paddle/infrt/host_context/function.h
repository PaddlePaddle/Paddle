// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <llvm/ADT/ArrayRef.h>

#include <string>

namespace infrt {
namespace host_context {

struct Value;
struct ValueRef;

/**
 * Base class of all executable Function.
 *
 * This is used by `infrt.call` op, to execute a function.
 */
class Function {
 public:
  Function(Function&& other)
      : name_(other.name_),
        num_arguments_(other.num_arguments_),
        num_results_(other.num_results_) {}

  Function() = delete;

  std::string name() const { return name_; }

  size_t num_arguments() const { return num_arguments_; }
  size_t num_results() const { return num_results_; }

  virtual void Execute(llvm::ArrayRef<Value*> arguments,
                       llvm::MutableArrayRef<ValueRef> results,
                       bool is_region = false) const {}

  virtual ~Function() = default;

 protected:
  Function(std::string name, size_t num_arguments, size_t num_results)
      : name_(name), num_arguments_(num_arguments), num_results_(num_results) {}

 private:
  std::string name_;
  size_t num_arguments_{};
  size_t num_results_{};
};

}  // namespace host_context
}  // namespace infrt
