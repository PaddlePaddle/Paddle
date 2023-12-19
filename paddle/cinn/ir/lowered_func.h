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
#include <map>
#include <string>
#include <vector>

#include "paddle/cinn/ir/buffer.h"
#include "paddle/cinn/ir/ir_base.h"

namespace cinn {
namespace ir {

class _LoweredFunc_;

/**
 * A struct representing an argument to a lowered function. Used for specifying
 * the function signature of generated code.
 */
struct Argument {
  //! kInput: arg is input
  //! kOutput: arg is output
  //! kUnknown: arg maybe input or output
  enum class IO { kInput = 0, kOutput = 1, kUnknown = 2 };

  IO io{IO::kInput};

  Argument() = default;
  explicit Argument(const ir::Buffer& buffer, IO io = IO::kInput);
  explicit Argument(const ir::Var& var, IO io = IO::kInput);

  //! Set the buffer argument, all the buffer information are stored in
  //! ir::Buffer.
  void set_buffer(const ir::Buffer& x);

  //! Set the var argument.
  void set_var(const ir::Var& x);

  bool is_input() const { return io == IO::kInput; }
  bool is_output() const { return io == IO::kOutput; }

  bool is_var() const { return var_arg_.defined(); }
  bool is_buffer() const { return buffer_arg_.defined(); }
  bool defined() const { return is_var() || is_buffer(); }

  ir::Buffer buffer_arg() const;
  ir::Var var_arg() const;

  //! The type of the buffer or scalar.
  Type type() const;

  std::string name() const;

  std::string human_readable() const;

 private:
  //! The buffer field.
  ir::Buffer buffer_arg_;
  //! The scalar field.
  ir::Var var_arg_;
};

//! Wrapper for _LoweredFunc_
class LoweredFunc : public IrNodeRef {
 public:
  LoweredFunc() = default;
  explicit LoweredFunc(IrNode* n) : IrNodeRef(n) {}

  operator Expr() const { return Expr(ptr()); }

  const _LoweredFunc_* operator->() const;
  _LoweredFunc_* operator->();
};

using dim3_t = std::array<int, 3>;
struct CudaAxisInfo {
  CudaAxisInfo() {
    for (int& v : grid_dims_) v = 1;
    for (int& v : block_dims_) v = 1;
    set_valid(false);
  }

  void set_grid_dim(int offset, int x);
  void set_block_dim(int offset, int x);

  int grid_dim(int offset) const;
  int block_dim(int offset) const;

  void CopyGridDimsTo(std::vector<int>* dest) const;
  void CopyBlockDimsTo(std::vector<int>* dest) const;

  inline void set_valid(bool x = false) { valid_ = x; }
  inline bool valid() const { return valid_; }

  //! Extend the axis dims and keep the larger dims.
  void ExtendWith(const CudaAxisInfo& other);

 private:
  // the three dimensions represents x, y, z
  dim3_t grid_dims_;
  // the three dimensions represents x, y, z
  dim3_t block_dims_;
  bool valid_{false};
};

std::ostream& operator<<(std::ostream& os, const CudaAxisInfo& x);

/**
 * Definition of a lowered function. Note that, it should be functional.
 *
 * Arguments of the function:
 *
 * both the input and output arguments, the output arguments are in the tail.
 */
struct _LoweredFunc_ : ExprNode<_LoweredFunc_> {
  //! The name of this function.
  std::string name;

  //! The Arguments used in the body of the function.
  std::vector<Argument> args;

  //! Temporary buffers(as output), these buffers will not appear in the
  //! function's argument list, but will be used in the body.
  std::vector<Buffer> temp_bufs;

  //! Body of this function.
  Expr body;

  DeviceAPI device_api{DeviceAPI::UNK};

  CudaAxisInfo cuda_axis_info;

  /**
   * The output buffer will be resized to the size required, we leave all the
   * expression here. The allocation and deallocation expressions will insert
   * into the head and tail of the function's body. It supports lazy
   * allocation/deallocation if the corresponding intristic methods support.
   *
   * Currently, we assume that all the input and output buffers should locate in
   * heap, no other memory type is allowed.
   */
  // @{
  std::vector<Expr> alloc_output_buffer_exprs;
  std::vector<Expr> dealloc_output_buffer_exprs;
  // @}

  //! something like: float* A_data = (float*)(A->memory);
  std::vector<Expr> buffer_data_cast_exprs;

  std::vector<Expr> argument_prepare_exprs;

  static LoweredFunc Make(const std::string& name,
                          const std::vector<Argument>& args,
                          const Expr& body,
                          const std::vector<ir::Buffer>& temp_bufs);

  // A simple version of the make function method,
  // regardless of the argument buffer information and IO information of
  // Argument, after building the function to optimize the buffer through pass
  static LoweredFunc Make(const std::string& name,
                          const std::vector<Argument>& args,
                          const Expr& body);

  bool is_gpu_host() const { return cuda_axis_info.valid(); }

  void Verify() const override {}

  std::vector<Expr*> expr_fields() override;
  std::vector<const Expr*> expr_fields() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::_LoweredFunc_;

  std::vector<Expr> PrepareCreateTempBufferExprs() const;
  //! Prepare the expressions for `alloc_tmp_buffer_exprs`.
  std::vector<Expr> PrepareAllocTempBufferExprs() const;
  std::vector<Expr> PrepareDeallocTempBufferExprs() const;
  std::vector<Expr> CudaPrepareAllocTempBufferExprs() const;
  std::vector<Expr> CudaAliasVarExprs() const;
  void PrepareBufferCastExprs(bool with_expr_gen_tensor = true);
  void PrepareCudaAxisInfoFromBody();

 private:
  void CheckValid() const;
  //! Prepare the expressions for `alloc_output_buffer_exprs`.
  void PrepareAllocOutputBufferExprs();
  //! Prepare the expressions for `dealloc_output_buffer_exprs`.
  void PrepareDeallocOutputBufferExprs();
  //! Insert the allocation expr for temporary variables.
  void AllocTempBuffer();

  void PrepareArgumentExprs();
  //! Get all the Buffers the function body references.
  //! NOTE it will return the buffers with duplicates removed(by comparing their
  //! name).
  std::vector<Tensor> CollectAllTensorReference(
      bool with_expr_gen_tensor = true) const;
};

}  // namespace ir
}  // namespace cinn
