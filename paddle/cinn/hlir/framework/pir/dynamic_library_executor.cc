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

#include "paddle/cinn/hlir/framework/pir/dynamic_library_executor.h"
#include "paddle/common/enforce.h"
#include <iostream>

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {

DynamicLibraryExecutor::~DynamicLibraryExecutor() {
  UnloadLibrary();
}

bool DynamicLibraryExecutor::LoadLibrary(const std::string& library_path) {
  if (library_handle_) {
    LOG(WARNING) << "Library already loaded: " << library_path_;
    return true;
  }
  
  library_handle_ = dlopen(library_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (!library_handle_) {
    LOG(ERROR) << "Failed to load dynamic library: " << dlerror();
    return false;
  }
  
  library_path_ = library_path;
  LOG(INFO) << "Successfully loaded dynamic library: " << library_path_;
  return true;
}

void* DynamicLibraryExecutor::GetFunction(const std::string& function_name) {
  if (!library_handle_) {
    LOG(ERROR) << "Library not loaded, cannot get function: " << function_name;
    return nullptr;
  }
  
  void* func_ptr = dlsym(library_handle_, function_name.c_str());
  if (!func_ptr) {
    LOG(ERROR) << "Failed to get function " << function_name << ": " << dlerror();
    return nullptr;
  }
  
  LOG(INFO) << "Successfully got function pointer for: " << function_name;
  return func_ptr;
}

void DynamicLibraryExecutor::ExecuteKernel(void* function_ptr, 
                                          const std::vector<void*>& args,
                                          const std::vector<int>& arg_sizes) {
  if (!function_ptr) {
    LOG(ERROR) << "Invalid function pointer";
    return;
  }
  
  // 将函数指针转换为适当的类型并执行
  // 这里需要根据实际的函数签名进行调整
  typedef void (*KernelFunc)(void**);
  KernelFunc kernel = reinterpret_cast<KernelFunc>(function_ptr);
  
  // 执行内核函数
  kernel(const_cast<void**>(args.data()));
  
  LOG(INFO) << "Successfully executed kernel function";
}

void DynamicLibraryExecutor::UnloadLibrary() {
  if (library_handle_) {
    dlclose(library_handle_);
    library_handle_ = nullptr;
    library_path_.clear();
    LOG(INFO) << "Unloaded dynamic library";
  }
}

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn