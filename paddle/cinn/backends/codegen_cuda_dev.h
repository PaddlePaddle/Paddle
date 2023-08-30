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
#include <unordered_set>
#include <vector>

#include "paddle/cinn/backends/codegen_c.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/module.h"
#include "paddle/cinn/ir/utils/ir_printer.h"
#include "paddle/cinn/lang/packed_func.h"
#include "paddle/cinn/runtime/cinn_runtime.h"

namespace cinn::ir {
class Module;
}  // namespace cinn::ir

namespace cinn {
namespace backends {

/**
 * CUDA device code generator.
 *
 * It generates the device function, e.g, the function called "myadd" will have
 * a __global__ functon called "myadd_kernel", different from codegen_c, the
 * declaration of the "myadd_kernel" function has an expanded argument list,
 * which finally similar to `__global__ void myadd(float* __restrict__ A, float*
 * __restrict__ B, int n);`
 */
class CodeGenCUDA_Dev : public CodeGenC {
 public:
  explicit CodeGenCUDA_Dev(Target target);

  /**
   * Compile the \p module to \p outputs.
   */
  void Compile(const ir::Module& module, const Outputs& outputs);

  //! Compile on NVRTC.
  std::string Compile(const ir::Module& module, bool for_nvrtc = true);

  void Compile(const ir::LoweredFunc& func);

  /**
   * \brief Print a function argument in CUDA syntax. Currently, just some
   * decoration of __restrict__.
   * @param arg the argument.
   * @return the representation in CUDA syntax.
   *
   * We make it a static to make the test easier.
   */
  void PrintFuncArg(const ir::Argument& arg);

  std::string Compile(const ir::Module& module, OutputKind output_kind);

  static const std::string& GetSourceHeader();

 protected:
  void Visit(const ir::_Var_* op) override;
  void Visit(const ir::_LoweredFunc_* op) override;
  void Visit(const ir::Min* op) override;
  void Visit(const ir::Max* op) override;
  void Visit(const ir::Alloc* op) override;
  void Visit(const ir::Call* op) override;
  void Visit(const ir::Load* op) override;
  void Visit(const ir::Store* op) override;
  void Visit(const ir::Let* op) override;

  // Print element access at a cuda built-in vector on a load/store node
  bool PrintBuiltinVectorAccess(const ir::LoadStoreAddrMnger* op,
                                ir::Expr index,
                                bool is_store);

  void PrintBuiltinCodes();

  void PrintIncludes() override;

  void PrintTempBufferCreation(const ir::Buffer& buffer);

  void PrintTempBufferAliasDefinition(const ir::Buffer& buffer);

  std::vector<Expr> GenerateBufferAliasExprs(
      const ir::_LoweredFunc_* op, const std::vector<ir::Buffer>& temp_buffers);

  /**
   * Print the function declaration, this is different from C, we expand the
   * arguments and get something like
   * `__global__ void myadd(float* __restrict__ A, float* __restrict__ B, int
   * n);`
   */
  void PrintFunctionDeclaration(const ir::_LoweredFunc_* op);

 private:
  Target target_;
  bool for_nvrtc_{false};
  // names of vectorized tensors from `Let` statments where dtypes of the
  // tensors are customized_type with customized_type::kcuda_builtin_vector_t
  // prefix
  std::unordered_set<std::string> vectorized_tensor_names_;
  static const std::string source_header_;
};

}  // namespace backends
}  // namespace cinn
