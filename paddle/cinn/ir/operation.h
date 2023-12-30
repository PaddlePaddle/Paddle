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
#include <utility>
#include <vector>

#include "paddle/cinn/ir/buffer.h"
#include "paddle/cinn/ir/dim.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/tensor.h"

namespace cinn {
namespace ir {

/**
 * @brief A placeholder op represents an input placeholder.
 */
struct PlaceholderOp : public _Operation_ {
  //! The symbolic shape of the input.
  std::vector<Dim> sym_shape;
  //! The shape of the input.
  std::vector<Expr> shape;
  //! The data type of the input.
  Type dtype;

  static Operation Make(const std::string &name,
                        const std::vector<Expr> &shape,
                        Type dtype);

  static Operation Make(const std::string &name,
                        const std::vector<Dim> &sym_shape,
                        Type dtype);

  const char *func_type() const override;

  static char const *__func_type__;
};

struct CallOp : public _Operation_ {
  const std::string &target() const;

  Expr call_expr;

  std::vector<Expr> &read_args();
  std::vector<Expr> &write_args();
  const std::vector<Expr> &read_args() const;
  const std::vector<Expr> &write_args() const;
  std::vector<Expr> args() const;

  //! A reference to the target LoweredFunc if this CallOp calls an generated
  //! LoweredFunc.
  Expr func;

  // the offset int the tuple of return values.
  int value_slot{-1};

  bool is_tuple_get{false};

  //! Number of the value slots.
  int num_value_slots{0};

  CallOp() = default;

  static Operation Make(const std::string &call_target, Expr call_op);

  const char *func_type() const override;

  static char const *__func_type__;
};

/**
 * The operation of the preceding view of a tensor.
 */
struct PrecedingViewOp : public _Operation_ {
  Expr tensor;

  int preceding_axis{-1};

  static Operation Make(const Tensor &tensor, int preceding_axis);

  const char *func_type() const override;

  static char const *__func_type__;
};

/**
 * Share the same buffer.
 */
struct BufferShareOp : public _Operation_ {
  const char *func_type() const override;
  static Operation Make() { return Operation(new BufferShareOp); }
  static char const *__func_type__;
};

/**
 * @brief A Compute op that compute a tensor on certain domain.
 */
struct ComputeOp : public _Operation_ {
  using handle_t = std::function<Expr(const std::vector<Expr> &)>;
  //! Var on each dimension
  std::vector<Var> axis;
  //! Var on each reduction axis, if the body is a Reduction.
  std::vector<Var> reduce_axis;
  //! Shape of the output.
  std::vector<Expr> shape;
  //! The compute expression.
  std::vector<Expr> body;
  //! The functor to generate the body, used to inline the expression if needed.
  handle_t producer_fn;

  ComputeOp() = default;

  static Operation Make(const std::string &name,
                        ComputeOp::handle_t handle,
                        const std::vector<Expr> &shape,
                        const std::vector<Expr> &domain,
                        const std::vector<Var> &reduce_axis = {},
                        const std::map<std::string, IrNodeRef> &attrs = {},
                        const std::string &tag = "");

  const char *func_type() const override;

  static const char *__func_type__;
};

}  // namespace ir
}  // namespace cinn
