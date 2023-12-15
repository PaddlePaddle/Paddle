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

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/ir/buffer.h"
#include "paddle/cinn/ir/dim.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/operation.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/runtime/intrinsic.h"

namespace cinn {
namespace lang {

using ir::Expr;

/**
 * Placeholder
 * @tparam T
 */
template <typename T>
class Placeholder {
 public:
  Placeholder(const std::string &name, const std::vector<int> &shape);
  Placeholder(const std::string &name, const std::vector<Expr> &shape);
  Placeholder(const std::string &name, const std::vector<ir::Dim> &shape);

  //! Get a slice.
  // @{
  Expr operator()(Expr a) const { return Call({a}); }
  Expr operator()(Expr a, Expr b) const { return Call({a, b}); }
  Expr operator()(Expr a, Expr b, Expr c) const { return Call({a, b, c}); }
  Expr operator()(Expr a, Expr b, Expr c, Expr d) const {
    return Call({a, b, c, d});
  }
  Expr operator()(const std::vector<Expr> &indices) const;
  // @}

  Type type() const { return tensor_->type(); }

  operator ir::Tensor() { return tensor_; }
  operator ir::Expr() { return Expr(tensor_); }

  ir::Tensor &operator->() { return tensor_; }
  const ir::Tensor &operator->() const { return tensor_; }

  ir::Tensor tensor() const { return tensor_; }

 private:
  Expr Call(const std::vector<Expr> &indices) const;

  void Init(const std::string &name, const std::vector<Expr> &shape);
  void Init(const std::string &name, const std::vector<ir::Dim> &shape);

  ir::Tensor tensor_;
};

template <typename T>
Expr Placeholder<T>::operator()(const std::vector<Expr> &indices) const {
  return tensor_(indices);
}

template <typename T>
Expr Placeholder<T>::Call(const std::vector<Expr> &indices) const {
  return tensor_(indices);
}

template <typename T>
Placeholder<T>::Placeholder(const std::string &name,
                            const std::vector<int> &shape) {
  std::vector<Expr> _shape;
  for (int v : shape) _shape.push_back(Expr(v));
  Init(name, _shape);
}

template <typename T>
Placeholder<T>::Placeholder(const std::string &name,
                            const std::vector<Expr> &shape) {
  Init(name, shape);
}

template <typename T>
Placeholder<T>::Placeholder(const std::string &name,
                            const std::vector<ir::Dim> &shape) {
  Init(name, shape);
}

ir::Tensor CreatePlaceHolder(const std::vector<int> &shape,
                             Type type,
                             const std::string &name);

ir::Tensor CreatePlaceHolder(const std::vector<Expr> &shape,
                             Type type,
                             const std::string &name);

ir::Tensor CreatePlaceHolder(const std::vector<ir::Dim> &shape,
                             Type type,
                             const std::string &name);

/// ------- details -------
template <typename T>
void Placeholder<T>::Init(const std::string &name,
                          const std::vector<Expr> &shape) {
  ir::Var buffer_ptr(Context::Global().NewName("buffer"));
  buffer_ptr->set_type(type_of<T>());

  std::vector<Expr> strides(shape.size(), Expr(1));
  Expr offset(0);

  std::vector<ir::Var> axis;
  for (int i = 0; i < shape.size(); i++)
    axis.emplace_back(cinn::common::axis_name(i));

  auto op = ir::PlaceholderOp::Make(name, shape, type_of<T>());

  tensor_ = ir::Tensor(name, type_of<T>(), shape, shape, op, {});
  Buffer buffer(tensor_->type());
  tensor_->Bind(buffer);
}

template <typename T>
void Placeholder<T>::Init(const std::string &name,
                          const std::vector<ir::Dim> &shape) {
  ir::Var buffer_ptr(Context::Global().NewName("buffer"));
  buffer_ptr->set_type(type_of<T>());

  std::vector<Expr> strides(shape.size(), Expr(1));
  Expr offset(0);

  std::vector<ir::Var> axis;
  for (int i = 0; i < shape.size(); i++)
    axis.emplace_back(cinn::common::axis_name(i));

  auto op = ir::PlaceholderOp::Make(name, shape, type_of<T>());

  tensor_ = ir::Tensor(name, type_of<T>(), shape, shape, op, {});
  Buffer buffer(tensor_->type());
  tensor_->Bind(buffer);
}

}  // namespace lang
}  // namespace cinn
