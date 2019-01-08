/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <exception>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace paddle {
namespace platform {

template <typename T>
class DebugSupport {
 public:
  // Returns the singleton of DebugSupport.
  static DebugSupport* GetInstance() {
    static thread_local std::unique_ptr<DebugSupport> debugSupport_(nullptr);
    static thread_local std::once_flag init_flag_;

    std::call_once(init_flag_,
                   [&]() { debugSupport_.reset(new DebugSupport<T>()); });
    return debugSupport_.get();
  }

  T GetInformation() const { return info; }

  void SetInformation(const T& v) { info = v; }

  std::string Format() const;

 private:
  T info;
};

using PythonDebugSupport = DebugSupport<std::vector<std::string>>;

template <>
std::string PythonDebugSupport::Format() const;

}  // namespace platform
}  // namespace paddle
