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

#include <dlfcn.h>
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/rt_module/dl_function.h"
#include "paddle/ap/include/rt_module/dl_handle.h"

namespace ap::rt_module {

class NaiveDlHandle : public DlHandle {
 public:
  NaiveDlHandle(const NaiveDlHandle&) = default;
  NaiveDlHandle(NaiveDlHandle&&) = default;
  ~NaiveDlHandle() {
    dlclose(main_handle_);
    dlclose(api_wrappers_handle_);
  }

  adt::Result<DlFunction> DlSym(const std::string& name) const override {
    void* function = dlsym(main_handle_, name.c_str());
    void* api_wrapper = dlsym(api_wrappers_handle_, name.c_str());
    ADT_CHECK(function != nullptr)
        << adt::errors::ValueError{std::string() + "main so '" + name +
                                   "' not found in '" + main_so_path_ + "'"};
    ADT_CHECK(api_wrapper != nullptr) << adt::errors::ValueError{
        std::string() + "api_wrapper so '" + name + "' not found in '" +
        api_wrappers_so_path_ + "'"};
    std::shared_ptr<const DlHandle> self = shared_from_this();
    ADT_CHECK(self != nullptr);
    using ApiWrapperT = void (*)(void* ret, void* func, void** args);
    DlFunction ret{self, function, reinterpret_cast<ApiWrapperT>(api_wrapper)};
    return ret;
  }

  static adt::Result<std::shared_ptr<const DlHandle>> DlOpen(
      const std::string& main_so_path,
      const std::string& api_wrappers_so_path) {
    void* main_handle = dlopen(main_so_path.c_str(), RTLD_LAZY);
    if (!main_handle) {
      return adt::errors::RuntimeError{
          std::string() + "dlopen failed. error message: " + dlerror() +
          ". path: " + main_so_path};
    }
    void* api_wrappers_handle = dlopen(api_wrappers_so_path.c_str(), RTLD_LAZY);
    if (!api_wrappers_handle) {
      dlclose(main_handle);
      return adt::errors::RuntimeError{
          std::string() + "dlopen failed. error message: " + dlerror() +
          ". path: " + api_wrappers_so_path};
    }
    return std::shared_ptr<const DlHandle>(new NaiveDlHandle(
        main_handle, api_wrappers_handle, main_so_path, api_wrappers_so_path));
  }

 private:
  NaiveDlHandle(void* main_handle,
                void* api_wrappers_handle,
                const std::string& main_so_path,
                const std::string& api_wrappers_so_path)
      : main_handle_(main_handle),
        api_wrappers_handle_(api_wrappers_handle),
        main_so_path_(main_so_path),
        api_wrappers_so_path_(api_wrappers_so_path) {}

  void* main_handle_;
  void* api_wrappers_handle_;
  std::string main_so_path_;
  std::string api_wrappers_so_path_;
};

}  // namespace ap::rt_module
