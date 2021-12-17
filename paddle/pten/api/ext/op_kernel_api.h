/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/pten/api/ext/dll_decl.h"
#include "paddle/pten/api/ext/exception.h"
#include "paddle/pten/api/ext/op_meta_info.h"
#include "paddle/pten/api/include/tensor.h"
#include "paddle/utils/any.h"

namespace paddle {
namespace framework {
class PADDLE_API OpKernelInfoHelper;
}  // namespace framework

////////////////////// Kernel Function (PD_KERNEL) ////////////////////////
typedef struct PD_KernelBuilder PD_KernelBuilder;
typedef struct PD_ExecutionContext PD_ExecutionContext;

// using KernelFunc = void (*)(const PD_ExecutionContext*);

////////////////////// Op Kernel Info //////////////////////
class PADDLE_API OpKernelInfo {
 public:
  explicit OpKernelInfo(const std::string& op_name) : name_(op_name) {}

  // format: PD_KERNEL(...)
  OpKernelInfo& SetKernelFn(KernelFunc&& func);

 private:
  friend class framework::OpKernelInfoHelper;

  // 1. desc info
  std::string name_;
  //   std::vector<std::string> inputs_;
  //   std::vector<std::string> outputs_;
  //   std::vector<std::string> attrs_;

  // 2. func info
  KernelFunc kernel_fn_{nullptr};
  //   InferShapeFunc infer_shape_fn_{nullptr};
  //   InferDtypeFunc infer_dtype_fn_{nullptr};
};

//////////////// Op Kernel Info Map /////////////////

class PADDLE_API OpKernelInfoMap {
 public:
  // this function's impl should keep in header file.
  // if move to cc file, meta info can not be added
  // into map
  static OpKernelInfoMap& Instance() {
    static OpKernelInfoMap g_custom_kernel_info_map;
    return g_custom_kernel_info_map;
  }

  std::vector<OpKernelInfo>& operator[](const std::string& name);

  const std::unordered_map<std::string, std::vector<OpKernelInfo>>& GetMap()
      const;

 private:
  OpKernelInfoMap() = default;
  std::unordered_map<std::string, std::vector<OpKernelInfo>> map_;

  PD_DISABLE_COPY_AND_ASSIGN(OpKernelInfoMap);
};

//////////////// Op Kernel Info Builder /////////////////

class PADDLE_API OpKernelInfoBuilder {
 public:
  explicit OpKernelInfoBuilder(std::string&& name);
  OpKernelInfoBuilder& SetKernelFn(KernelFunc func);

 private:
  // Forward Op name
  std::string name_;
  // ref current info ptr
  OpKernelInfo* info_ptr_;
};

#define PD_BUILD_KERNEL(op_name)                                        \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                       \
      __reg_op__##op_name,                                              \
      "PD_BUILD_KERNEL must be called in global namespace.");           \
  static ::paddle::OpKernelInfoBuilder __op_kernel_info_##op_name##__ = \
      ::paddle::OpKernelInfoBuilder(#op_name)

}  // namespace paddle

#ifdef __cplusplus
extern "C" {
#endif

paddle::OpKernelInfoMap& PD_GetOpKernelInfoMap();

// PADDLE_API int PD_NumInputs(const paddle::PD_ExecutionContext* ctx);

#ifdef __cplusplus
}  // end extern "C"
#endif
