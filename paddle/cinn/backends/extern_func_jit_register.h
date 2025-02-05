// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

/**
 * \file This file defines some functions and macros to help register the extern
 * functions into JIT.
 */
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/cinn/backends/extern_func_emitter.h"
#include "paddle/cinn/backends/extern_func_emitter_builtin.h"
#include "paddle/cinn/backends/extern_func_protos.h"
#include "paddle/cinn/backends/function_prototype.h"
#include "paddle/cinn/backends/llvm/codegen_llvm.h"
#include "paddle/cinn/backends/llvm/ir_builder_mixin.h"
#include "paddle/cinn/backends/llvm/llvm_util.h"
#include "paddle/cinn/backends/llvm/runtime_symbol_registry.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/common/enforce.h"
/**
 * Helper to register an external function into CINN, including the prototype,
 * the function address.
 * @param fn__: name of the function
 * @param target__: the Target.
 */
#define REGISTER_EXTERN_FUNC_HELPER(fn__, target__) \
  ::cinn::backends::RegisterExternFunction(         \
      #fn__, target__, reinterpret_cast<void*>(fn__))

#define REGISTER_FACKED_EXTERN_FUNC_HELPER(fn__, target__) \
  ::cinn::backends::RegisterExternFunction(#fn__, target__)

/**
 * Register an external function with one input and one output.
 */
#define REGISTER_EXTERN_FUNC_1_IN_1_OUT(fn__, target__, in_type__, out_type__) \
  REGISTER_EXTERN_FUNC_HELPER(fn__, target__)                                  \
      .SetRetType<out_type__>()                                                \
      .AddInputType<in_type__>()                                               \
      .End()

/**
 * Register an external function with one input and one output.
 */
#define REGISTER_EXTERN_FUNC_2_IN_1_OUT(                \
    fn__, target__, in_type1__, in_type2__, out_type__) \
  REGISTER_EXTERN_FUNC_HELPER(fn__, target__)           \
      .SetRetType<out_type__>()                         \
      .AddInputType<in_type1__>()                       \
      .AddInputType<in_type2__>()                       \
      .End()

/**
 * Register a sourced function(No function address, called in generated source
 * code).
 */
#define REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(      \
    fn__, target__, in_type__, out_type__)           \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(fn__, target__) \
      .SetRetType<out_type__>()                      \
      .AddInputType<in_type__>()                     \
      .End()

/**
 * Register a sourced function(No function address, called in generated source
 * code).
 */
#define REGISTER_EXTERN_SOURCE_FUNC_2_IN_1_OUT(         \
    fn__, target__, in_type1__, in_type2__, out_type__) \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(fn__, target__)    \
      .SetRetType<out_type__>()                         \
      .AddInputType<in_type1__>()                       \
      .AddInputType<in_type2__>()                       \
      .End()

namespace cinn {
namespace backends {

static const char* TargetToBackendRepr(Target target) {
  return target.arch.Match(
      [&](common::UnknownArch) -> const char* { CINN_NOT_IMPLEMENTED; },
      [&](common::X86Arch) -> const char* { return backend_llvm_host; },
      [&](common::ARMArch) -> const char* { CINN_NOT_IMPLEMENTED; },
      [&](common::NVGPUArch) -> const char* { return backend_nvgpu; },
      [&](common::HygonDCUArchHIP) -> const char* {
        return backend_hygondcu_hip;
      },
      [&](common::HygonDCUArchSYCL) -> const char* {
        return backend_hygondcu_sycl;
      });
}

/**
 * Helper class to register an external function.
 */
struct RegisterExternFunction {
  /**
   * Constructor.
   * @param fn_name Name of the function.
   * @param target Target of the function.
   * @param fn_ptr Address of the function, not valid if leave as null.
   */
  RegisterExternFunction(const std::string& fn_name,
                         Target target,
                         void* fn_ptr = nullptr)
      : fn_name_(fn_name),
        target_(target),
        fn_ptr_(fn_ptr),
        fn_proto_builder_(fn_name) {}

  /**
   * Add an input type.
   * @tparam T The input type.
   * @return itself.
   */
  template <typename T>
  RegisterExternFunction& AddInputType() {
    fn_proto_builder_.AddInputType<T>();
    return *this;
  }

  /**
   * Add an output type.
   * @tparam T The output type.
   * @return itself.
   */
  template <typename T>
  RegisterExternFunction& AddOutputType() {
    fn_proto_builder_.AddOutputType<T>();
    return *this;
  }

  /**
   * Add an return type.
   * @tparam T The return type.
   * @return itself.
   */
  template <typename T>
  RegisterExternFunction& SetRetType() {
    fn_proto_builder_.SetRetType<T>();
    return *this;
  }

  /**
   * Add an shape inference.
   * @param handle The handle to help inference the shape.
   * @return itself.
   */
  RegisterExternFunction& SetShapeInference(
      FunctionProto::shape_inference_t handle) {
    fn_proto_builder_.SetShapeInference(handle);
    return *this;
  }

  /**
   * End the register, once end, further modification is disallowed.
   */
  void End();

 private:
  const std::string& fn_name_;
  Target target_;
  void* fn_ptr_{};
  FunctionProto::Builder fn_proto_builder_;
};

}  // namespace backends
}  // namespace cinn
