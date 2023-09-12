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

#pragma once

#include <string>
#include <vector>

#include "paddle/cinn/backends/codegen_c.h"
#include "paddle/cinn/backends/extern_func_emitter.h"
#include "paddle/cinn/backends/extern_func_protos.h"
#include "paddle/cinn/backends/llvm/codegen_llvm.h"
#include "paddle/cinn/backends/llvm/llvm_util.h"

namespace cinn {
namespace backends {

//! Function names

static const char* extern_tanh_host_repr = "__cinn_host_tanh_fp32";
static const char* extern_tanh_v_host_repr = "__cinn_host_tanh_v";

/**
 * A bridge for the Emitters to access CodeGenLLVM's internal members.
 */
class CodeGenLLVMforEmitter : public CodeGenLLVM {
 public:
  explicit CodeGenLLVMforEmitter(CodeGenLLVM* x)
      : CodeGenLLVM(x->m(), x->b(), x->named_vars()) {}
};

class ExternFunctionLLVMEmitter : public ExternFunctionEmitter {
 public:
  explicit ExternFunctionLLVMEmitter(const std::string& fn_name)
      : fn_name_(fn_name) {}

  void BindCodeGen(void* codegen) override;
  const char* func_name() const override;
  bool RetValuePacked() const override;
  const char* backend_kind() const override;

 protected:
  void EmitImpl(const ir::Call* op) override;
  FunctionProto& fn_proto() const;
  llvm::FunctionType* llvm_fn_type() const;

  CodeGenLLVM* codegen_{};
  std::string fn_name_;
};

}  // namespace backends
}  // namespace cinn
