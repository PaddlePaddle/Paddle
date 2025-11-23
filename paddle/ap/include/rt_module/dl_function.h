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

#include <memory>
#include "paddle/ap/include/adt/adt.h"

namespace ap::rt_module {

class DlHandle;

// dynamic link function
class DlFunction {
 public:
  DlFunction(const std::shared_ptr<const DlHandle>& dl_handle,
             void* func,
             void (*wrapper)(void* ret, void* func, void** args))
      : dl_handle_(dl_handle), func_(func), api_wrapper_(wrapper) {}

  DlFunction(const DlFunction&) = default;
  DlFunction(DlFunction&&) = default;

  bool operator==(const DlFunction& other) const {
    // It's correct to ignore dl_handle_
    return this->func_ == other.func_ &&
           this->api_wrapper_ == other.api_wrapper_;
  }

  adt::Result<adt::Ok> Apply(void* ret, void** args) const {
    ADT_LET_CONST_REF(dl_handle_guard, adt::WeakPtrLock(dl_handle_));
    api_wrapper_(ret, func_, args);
    (void)dl_handle_guard;
    return adt::Ok{};
  }

 private:
  std::weak_ptr<const DlHandle> dl_handle_;
  void* func_;
  void (*api_wrapper_)(void* ret, void* func, void** args);
};

}  // namespace ap::rt_module
