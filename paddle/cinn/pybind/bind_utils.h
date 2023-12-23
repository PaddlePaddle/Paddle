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

#include <pybind11/pybind11.h>

#include <string>

#include "paddle/cinn/common/cinn_value.h"
#include "paddle/cinn/common/shared.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/pybind/bind.h"
#include "paddle/cinn/runtime/cinn_runtime.h"

namespace py = pybind11;

namespace cinn::pybind {
using cinn::common::CINNValue;
using cinn::common::Shared;
using cinn::common::Type;
using ir::Expr;
using ir::ExprNode;

using ExprOp = absl::variant<ir::IntImm,
                             ir::UIntImm,
                             ir::FloatImm,
                             ir::StringImm,
                             ir::Cast,
                             ir::Let,
                             ir::Reduce,
                             ir::Call,
                             ir::_Var_,
                             ir::Select,
                             ir::Load,
                             ir::Store,
                             ir::Alloc,
                             ir::Free,
                             ir::IfThenElse,
                             ir::For,
                             ir::PolyFor,
                             ir::Ramp,
                             ir::Broadcast,
                             ir::Product,
                             ir::Sum,
                             ir::Block,
                             ir::_Module_>;
using BinaryOp = absl::variant<>;
using UnaryOp = absl::variant<>;

// hold CINNValue
using ValueVar =
    absl::variant<int32_t, int64_t, float, ir::Var, ir::Expr, std::nullptr_t>;

inline ValueVar ConvertToVar(const CINNValue &value) {
  auto type_code = value.type_code();
  ValueVar var;
  if (type_code == ::cinn_type_code<int32_t>()) {
    var = static_cast<int32_t>(value);
  } else if (type_code == ::cinn_type_code<int64_t>()) {
    var = static_cast<int64_t>(value);
  } else if (type_code == ::cinn_type_code<float>()) {
    var = static_cast<float>(value);
  } else if (type_code == CINNValue::TypeCode<ir::Var>()) {
    var = value.operator ir::Var();
  } else if (type_code == CINNValue::TypeCode<ir::Expr>()) {
    var = ir::Expr(value.operator ir::Expr());
  } else {
    var = nullptr;
  }

  return var;
}

template <typename T>
auto DefineShared(py::module *m, absl::string_view obj_name) {
  std::string name = "Shared" + std::string(obj_name);
  py::class_<Shared<T>> shared(*m, name.c_str());

  shared.def(py::init<>())
      .def(py::init<T *>())
      .def(py::init<const Shared<T> &>());
  return shared;
}

template <typename NodeType>
void DefineExprNode(py::module *m, absl::string_view node_name) {
  using ExprNodeT = ExprNode<NodeType>;

  std::string prefix{"ExprNode"};
  std::string name = prefix + std::string(node_name);
  py::class_<ExprNodeT, ir::IrNode> expr_node(
      *m, name.c_str(), py::module_local());
  expr_node.def(py::init<>())
      .def(py::init<Type>())
      .def(py::init<int>())
      .def("operands_mutable", py::overload_cast<>(&ExprNodeT::operands))
      .def("operands_const",
           py::overload_cast<>(&ExprNodeT::operands, py::const_))
      .def("operand_mutable",
           py::overload_cast<int>(&ExprNodeT::operand),
           py::return_value_policy::reference)
      .def("operand_const",
           py::overload_cast<int>(&ExprNodeT::operand, py::const_),
           py::return_value_policy::reference)
      .def("copy", &ExprNodeT::Copy)
      .def("node_type", &ExprNodeT::node_type);
}

template <typename NodeType>
void DefineBinaryOpNode(py::module *m, absl::string_view node_name) {
  DefineExprNode<NodeType>(m, node_name);
  std::string prefix{"BinaryOpNode"};
  std::string name = prefix + std::string(node_name);
  using BinaryOpNodeT = ir::BinaryOpNode<NodeType>;
  py::class_<BinaryOpNodeT, ir::ExprNode<NodeType>> binary_op_node(
      *m, name.c_str());
  binary_op_node.def(py::init<>())
      .def(py::init<Type, Expr, Expr>())
      .def("a_mutable",
           py::overload_cast<>(&BinaryOpNodeT::a),
           py::return_value_policy::reference)
      .def("a_const",
           py::overload_cast<>(&BinaryOpNodeT::a, py::const_),
           py::return_value_policy::reference)
      .def("b_mutable",
           py::overload_cast<>(&BinaryOpNodeT::b),
           py::return_value_policy::reference)
      .def("b_const",
           py::overload_cast<>(&BinaryOpNodeT::b, py::const_),
           py::return_value_policy::reference)
      .def("type", &BinaryOpNodeT::type)
      .def("expr_fields_mutable",
           py::overload_cast<>(&BinaryOpNodeT::expr_fields))
      .def("expr_fields_const",
           py::overload_cast<>(&BinaryOpNodeT::expr_fields, py::const_));
}

template <typename NodeType>
void DefineUnaryOpNode(py::module *m, absl::string_view node_name) {
  using UnaryOpNodeT = ir::UnaryOpNode<NodeType>;
  DefineExprNode<NodeType>(m, node_name);

  std::string name = "UnaryOpNode" + std::string(node_name);
  py::class_<UnaryOpNodeT, ir::ExprNode<NodeType>> unary_op_node(*m,
                                                                 name.c_str());
  unary_op_node.def(py::init<>())
      .def(py::init<Type, Expr>())
      .def("type", &UnaryOpNodeT::type)
      .def("v_mutable",
           py::overload_cast<>(&UnaryOpNodeT::v),
           py::return_value_policy::reference)
      .def("v_const",
           py::overload_cast<>(&UnaryOpNodeT::v, py::const_),
           py::return_value_policy::reference)
      .def("expr_fields_mutable",
           py::overload_cast<>(&UnaryOpNodeT::expr_fields))
      .def("expr_fields_const",
           py::overload_cast<>(&UnaryOpNodeT::expr_fields, py::const_))
      .def("operands_mutable",
           py::overload_cast<>(&UnaryOpNodeT::operands),
           py::return_value_policy::reference)
      .def("operands_const",
           py::overload_cast<>(&UnaryOpNodeT::operands, py::const_),
           py::return_value_policy::reference);
}

class IrNodeWrapper : ir::IrNode {
  using ir::IrNode::IrNode;
};

}  // namespace cinn::pybind
