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
#include <string>
#include <vector>
#include <memory>
#include "paddle/cinn/hlir/framework/pir/utils.h"

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {

class DynamicLibraryExecutor {
 public:
  DynamicLibraryExecutor() = default;
  ~DynamicLibraryExecutor();
  
  // 加载动态链接库
  bool LoadLibrary(const std::string& library_path);
  
  // 获取函数指针
  void* GetFunction(const std::string& function_name);
  
  // 执行内核函数
  void ExecuteKernel(void* function_ptr, 
                     const std::vector<void*>& args,
                     const std::vector<int>& arg_sizes);
  
  // 卸载动态链接库
  void UnloadLibrary();
  
  // 检查库是否已加载
  bool IsLibraryLoaded() const { return library_handle_ != nullptr; }
  
  // 获取库路径
  const std::string& GetLibraryPath() const { return library_path_; }

 private:
  void* library_handle_{nullptr};
  std::string library_path_;
};

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn