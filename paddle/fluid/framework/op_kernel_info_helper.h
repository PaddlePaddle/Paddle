/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/pten/api/ext/op_kernel_info.h"
#include "paddle/pten/core/kernel_factory.h"

namespace paddle {
namespace framework {

class OpKernelInfoHelper {
 public:
  static const std::string& GetOpName(const paddle::OpKernelInfo& info) {
    return info.op_name_;
  }

  static const pten::Backend& GetBackend(const paddle::OpKernelInfo& info) {
    return info.backend_;
  }

  static const pten::DataLayout& GetDataLayout(
      const paddle::OpKernelInfo& info) {
    return info.layout_;
  }

  static const pten::DataType& GetDataType(const paddle::OpKernelInfo& info) {
    return info.dtype_;
  }

  static pten::KernelKey GetKernelKey(const paddle::OpKernelInfo& info) {
    return pten::KernelKey(info.backend_, info.layout_, info.dtype_);
  }

  static const CustomKernelFunc& GetKernelFn(const paddle::OpKernelInfo& info) {
    return info.kernel_fn_;
  }

  static void* GetVariadicKernelFn(const paddle::OpKernelInfo& info) {
    return info.variadic_kernel_fn_;
  }

  static const paddle::SmallVector<TensorArgDef>& GetInputDefs(
      const paddle::OpKernelInfo& info) {
    return info.input_defs_;
  }

  static const paddle::SmallVector<TensorArgDef>& GetOutputDefs(
      const paddle::OpKernelInfo& info) {
    return info.output_defs_;
  }

  static const paddle::SmallVector<AttributeArgDef>& GetAttributeDefs(
      const paddle::OpKernelInfo& info) {
    return info.attribute_defs_;
  }
};

}  // namespace framework
}  // namespace paddle
