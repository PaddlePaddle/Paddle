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

#include <llvm/Support/FormatVariadic.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <string>
#include <type_traits>

#include "paddle/cinn/common/shared.h"
#include "paddle/cinn/ir/dim.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/operation.h"
#include "paddle/cinn/ir/registry.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_compare.h"
#include "paddle/cinn/lang/packed_func.h"
#include "paddle/cinn/poly/stage.h"
#include "paddle/cinn/pybind/bind.h"
#include "paddle/cinn/pybind/bind_utils.h"
#include "paddle/cinn/pybind/ir/ir.h"
#include "paddle/cinn/pybind/ir/ir_context.h"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, cinn::common::Shared<T>);

namespace cinn::pybind {
using ir::IrNode;
using ir::IrNodeRef;
using ir::IrNodeTy;

// lowered_func.h
using ir::Argument;
using ir::Expr;
using ir::LoweredFunc;
using ir::Var;

namespace {
void BindLoweredFunc(py::module *);
void BindNode(py::module *);
void BindIrVisitor(py::module *);
void BindIrIr(py::module *);
void BindOperation(py::module *);
void BindPackedFunc(py::module *);
void BindRegistry(py::module *);

void BindLoweredFunc(py::module *m) {
  py::class_<Argument> argument(*m, "Argument");

  py::enum_<Argument::IO> io(argument, "IO");
  io.value("kInput", Argument::IO::kInput)
      .value("kOutput", Argument::IO::kOutput)
      .value("kUnknown", Argument::IO::kUnknown);

  argument
      .def(py::init<const ir::Buffer &, Argument::IO>(),
           py::arg("buffer"),
           py::arg("io") = Argument::IO::kInput)
      .def(py::init<const ir::Var &, Argument::IO>(),
           py::arg("var"),
           py::arg("io") = Argument::IO::kInput)
      .def("set_buffer", &Argument::set_buffer)
      .def("set_var", &Argument::set_var)
      .def("is_input", &Argument::is_input)
      .def("is_output", &Argument::is_output)
      .def("is_var", &Argument::is_var)
      .def("is_buffer", &Argument::is_buffer)
      .def("defined", &Argument::defined)
      .def("buffer_arg", &Argument::buffer_arg)
      .def("type", &Argument::type)
      .def("name", &Argument::name)
      .def("human_readable", &Argument::human_readable);

  py::class_<LoweredFunc> lowered_func(*m, "LoweredFunc");
  lowered_func.def(py::init<>())
      .def(py::init<IrNode *>())
      .def(
          "name",
          [](const ir::LoweredFunc &self) -> std::string { return self->name; })
      .def("__str__",
           [](const ir::LoweredFunc &self) -> std::string {
             return utils::GetStreamCnt(self);
           })
      .def("__repr__",
           [](const ir::LoweredFunc &self) -> std::string {
             return llvm::formatv(
                 "<LoweredFunc {0}>", self.get(), self->name.c_str());
           })
      .def("body", [](const ir::LoweredFunc &self) { return self->body; });
}

void BindNode(py::module *m) {
  // enum class IrNodeTy
  py::enum_<ir::IrNodeTy> ir_node_ty(*m, "IrNodeTy");
  ir_node_ty.value("kUnk", ir::IrNodeTy::kUnk);
#define DECLARE_IR_NODE_TY(__ty) ir_node_ty.value(#__ty, ir::IrNodeTy::__ty);
  NODETY_FORALL(DECLARE_IR_NODE_TY)
#undef DECLARE_IR_NODE_TY

  // class IrNode
  py::class_<ir::IrNode, IrNodeWrapper> ir_node(
      *m, "IrNode", py::module_local());
  ir_node.def(py::init<>())
      .def(py::init<ir::Type>())
      .def_readwrite("operands", &ir::IrNode::operands)
      .def("node_type", &ir::IrNode::node_type)
      .def("type", &ir::IrNode::type)
      .def("set_type", &ir::IrNode::set_type)
      .def("expr_fields_mutable", py::overload_cast<>(&ir::IrNode::expr_fields))
      .def("expr_fields_const",
           py::overload_cast<>(&ir::IrNode::expr_fields, py::const_))
      .def("type_info", &ir::IrNode::type_info);

  // class Shared<IrNode>
  DefineShared<IrNode>(m, "IrNode");

  // class IrNodeRef : public Shared<IrNode>
  py::class_<ir::IrNodeRef, cinn::common::Shared<IrNode>> ir_node_ref(
      *m, "IrNodeRef");
  ir_node_ref.def(py::init<>())
      .def(py::init<const ir::IrNodeRef &>())
      .def(py::init<ir::IrNode *>())
      .def("node_type", &ir::IrNodeRef::node_type);

  // struct IntImm : ExprNode<IntImm>
  DefineExprNode<ir::IntImm>(m, "IntImm");
  py::class_<ir::IntImm, ir::ExprNode<ir::IntImm>> int_imm(*m, "IntImm");
  int_imm.def_readwrite("value", &ir::IntImm::value)
      .def(py::init<Type, int64_t>())
      .def("__str__",
           [](const ir::IntImm &self) { return std::to_string(self.value); })
      .def("__repr__", [](ir::IntImm &self) -> std::string {
        return llvm::formatv("<IntImm {0}>", self.self(), self.value);
      });

  // struct UIntImm : ExprNode<UIntImm>
  DefineExprNode<ir::UIntImm>(m, "UIntImm");
  py::class_<ir::UIntImm, ir::ExprNode<ir::UIntImm>> uint_imm(*m, "UIntImm");
  uint_imm.def_readwrite("value", &ir::UIntImm::value)
      .def(py::init<Type, uint64_t>());

  // struct FloatImm : ExprNode<FloatImm>
  DefineExprNode<ir::FloatImm>(m, "FloatImm");
  py::class_<ir::FloatImm, ir::ExprNode<ir::FloatImm>> float_imm(*m,
                                                                 "FloatImm");
  float_imm.def_readwrite("value", &ir::FloatImm::value)
      .def(py::init<Type, double>());

  // struct StringImm : ExprNode<StringImm>
  DefineExprNode<ir::StringImm>(m, "StringImm");
  py::class_<ir::StringImm, ir::ExprNode<ir::StringImm>> string_imm(
      *m, "StringImm");
  string_imm.def_readwrite("value", &ir::StringImm::value)
      .def(py::init<const std::string &>());

  auto expr = py::class_<ir::Expr, ir::IrNodeRef>(*m, "Expr");

  expr.def(py::init<ir::Expr &>());
  expr.def(py::init<ir::IrNode *>());
  expr.def(py::init<const ir::Var &>());
  expr.def(py::init<int32_t>());
  expr.def(py::init<uint32_t>());
  expr.def(py::init<int64_t>());
  expr.def(py::init<uint64_t>());
  expr.def(py::init<float>());
  expr.def(py::init<double>());
  expr.def(py::init<const std::string &>());

  expr.def("as_int32", &ir::Expr::as_int32)
      .def("as_int64", &ir::Expr::as_int64)
      .def("as_float", &ir::Expr::as_float)
      .def("as_double", &ir::Expr::as_double)
      .def("int", [](ir::Expr &self) { return self.As<ir::IntImm>()->value; })
      .def("float",
           [](ir::Expr &self) { return self.As<ir::FloatImm>()->value; })

      .def("__str__",
           [](const Expr &self) { return utils::GetStreamCnt(self); })
      .def("__repr__", [](const Expr &self) -> std::string {
        std::string content = self.get() ? utils::GetStreamCnt(self) : "";
        return llvm::formatv("<cinn.ir.Expr {0}>", content);
      });

  expr.def("as_var_mutable",
           py::overload_cast<>(&ir::Expr::as_var),
           py::return_value_policy::reference)
      .def("as_var_const",
           py::overload_cast<>(&ir::Expr::as_var, py::const_),
           py::return_value_policy::reference)
      .def("as_var_ref", &ir::Expr::as_var_ref);

  expr.def("as_buffer_mutable",
           py::overload_cast<>(&ir::Expr::as_buffer),
           py::return_value_policy::reference)
      .def("as_buffer_const",
           py::overload_cast<>(&ir::Expr::as_buffer, py::const_),
           py::return_value_policy::reference)
      .def("as_buffer_ref", &ir::Expr::as_buffer_ref);

  expr.def("is_constant", &ir::Expr::is_constant)
      .def("get_constant", &ir::Expr::get_constant)
      .def("is_var", &ir::Expr::is_var)
      .def("type", &ir::Expr::type);

  // operators

#define BIND_POD_BINARY_OP(otype__) \
  .def(py::self + otype__)          \
      .def(py::self - otype__)      \
      .def(py::self *otype__)       \
      .def(py::self / otype__)      \
      .def(py::self % otype__)      \
      .def(py::self < otype__)      \
      .def(py::self <= otype__)     \
      .def(py::self > otype__)      \
      .def(py::self >= otype__)     \
      .def(otype__ + py::self)      \
      .def(otype__ - py::self)      \
      .def(otype__ *py::self)       \
      .def(otype__ / py::self)      \
      .def(otype__ % py::self)      \
      .def(otype__ < py::self)      \
      .def(otype__ <= py::self)     \
      .def(otype__ > py::self)      \
      .def(otype__ >= py::self)

  expr                              //
      BIND_POD_BINARY_OP(py::self)  //
      BIND_POD_BINARY_OP(int())     //
      BIND_POD_BINARY_OP(float());

  expr.def("__add__",
           [](const Expr &self, const Var &other) -> Expr {
             return self + other;
           })
      .def("__sub__",
           [](const Expr &self, const Var &other) -> Expr {
             return self - other;
           })
      .def("__mul__",
           [](const Expr &self, const Var &other) -> Expr {
             return self * other;
           })
      .def("__div__", [](const Expr &self, const Var &other) -> Expr {
        return self / other;
      });
}

// empty visitor
void BindIrVisitor(py::module *m) {
  py::class_<ir::ir_utils::IrEqualVisitor> ir_compare(*m, "IrCompare");
  ir_compare.def(py::init<bool, bool>())
      .def("compare",
           [](ir::ir_utils::IrEqualVisitor &self,
              const cinn::ir::Expr &lhs,
              const cinn::ir::Expr &rhs) { return self.Compare(lhs, rhs); });

  py::class_<ir::IRVisitor> ir_visitor(*m, "IRVisitor");
  ir_visitor.def(py::init<>())
      .def("visit", py::overload_cast<const ir::Expr *>(&ir::IRVisitor::Visit));
#define DEFINE_VISIT_FN(__ty) \
  ir_visitor.def("visit",     \
                 py::overload_cast<const ir::__ty *>(&ir::IRVisitor::Visit));
  NODETY_FORALL(DEFINE_VISIT_FN)
#undef DEFINE_VISIT_FN
}

void BindIrIr(py::module *m) {
  using ir::Expr;
  using ir::IrNode;
  using ir::IrNodeRef;
  using ir::Var;
  using py::arg;

  // struct Cast : ExprNode<Cast>
  DefineExprNode<ir::Cast>(m, "Cast");
  py::class_<ir::Cast, ExprNode<ir::Cast>> cast(*m, "Cast");
  cast.def(py::init<>())
      .def("v_mutable",
           py::overload_cast<>(&ir::Cast::v),
           py::return_value_policy::reference)
      .def("v_const",
           py::overload_cast<>(&ir::Cast::v, py::const_),
           py::return_value_policy::reference);

  // struct Let : ExprNode<Let>
  DefineExprNode<ir::Let>(m, "Let");
  py::class_<ir::Let, ExprNode<ir::Let>> let(*m, "Let");
  let.def(py::init<>())
      .def_readwrite("symbol", &ir::Let::symbol)
      .def_readwrite("body", &ir::Let::body)
      .def_static("make", &ir::Let::Make)
      .def("type", &ir::Let::type)
      .def("expr_fields_mutable", py::overload_cast<>(&ir::Let::expr_fields))
      .def("expr_fields_const",
           py::overload_cast<>(&ir::Let::expr_fields, py::const_));

  // struct Reduce : ExprNode<Reduce>
  DefineExprNode<ir::Reduce>(m, "Reduce");
  py::class_<ir::Reduce, ExprNode<ir::Reduce>> reduce(*m, "Reduce");
  py::enum_<ir::Reduce::ReduceType> reduce_type(reduce, "ReduceType");
  reduce_type  //
      .value("kSum", ir::Reduce::ReduceType::kSum)
      .value("kSub", ir::Reduce::ReduceType::kSub)
      .value("kMul", ir::Reduce::ReduceType::kMul)
      .value("kDiv", ir::Reduce::ReduceType::kDiv)
      .value("kMax", ir::Reduce::ReduceType::kMax)
      .value("kMin", ir::Reduce::ReduceType::kMin)
      .value("kAll", ir::Reduce::ReduceType::kAll)
      .value("kAny", ir::Reduce::ReduceType::kAny);

  reduce.def_readwrite("init", &ir::Reduce::init)
      .def_readwrite("body", &ir::Reduce::body)
      .def_readwrite("reduce_type", &ir::Reduce::reduce_type)
      .def_static("make", &ir::Reduce::Make)
      .def("type", &ir::Reduce::type)
      .def("expr_fields_mutable", py::overload_cast<>(&ir::Reduce::expr_fields))
      .def("expr_fields_const",
           py::overload_cast<>(&ir::Reduce::expr_fields, py::const_));

  // enum class CallType
  py::enum_<ir::CallType> call_type(*m, "CallType");
  call_type.value("Extern", ir::CallType::Extern)
      .value("CINN", ir::CallType::CINN)
      .value("Intrinsic", ir::CallType::Intrinsic)
      .value("ISL", ir::CallType::ISL);

  // struct Call : ExprNode<Call>
  DefineExprNode<ir::Call>(m, "Call");
  py::class_<ir::Call, ExprNode<ir::Call>> call(*m, "Call");
  call.def(py::init<Type>())
      .def_readwrite("name", &ir::Call::name)
      .def_readwrite("read_args", &ir::Call::read_args)
      .def_readwrite("write_args", &ir::Call::write_args)
      .def_readwrite("call_type", &ir::Call::call_type)
      .def_readwrite("func", &ir::Call::func)
      .def_readwrite("value_index", &ir::Call::value_index)
      .def_static("make", &ir::Call::Make)
      .def("total_args_count", &ir::Call::total_args_count)
      .def("is_extern_call", &ir::Call::is_extern_call)
      .def("is_cinn_call", &ir::Call::is_cinn_call)
      .def("is_intrinsic_call", &ir::Call::is_intrinsic_call)
      .def("is_isl_call", &ir::Call::is_isl_call)
      .def("expr_fields_mutable", py::overload_cast<>(&ir::Call::expr_fields))
      .def("expr_fields_const",
           py::overload_cast<>(&ir::Call::expr_fields, py::const_));

  // struct _Var_ : ExprNode<_Var_>
  DefineExprNode<ir::_Var_>(m, "_Var_");
  py::class_<ir::_Var_, ExprNode<ir::_Var_>> _var_(*m, "_Var_");
  _var_.def_readwrite("name", &ir::_Var_::name)
      .def_readwrite("is_reduce_axis", &ir::_Var_::is_reduce_axis)
      .def_readwrite("lower_bound", &ir::_Var_::lower_bound)
      .def_readwrite("upper_bound", &ir::_Var_::upper_bound)
      .def_readwrite("tag", &ir::_Var_::tag)
      .def(py::init<>())
      .def(py::init<const std::string &, Type>())
      .def_static("make",
                  py::overload_cast<const std::string &, const Type &>(
                      &ir::_Var_::Make))
      .def_static("make",
                  py::overload_cast<ir::Expr,
                                    ir::Expr,
                                    const std::string &,
                                    bool,
                                    bool,
                                    bool>(&ir::_Var_::Make))
      .def("copy", &ir::_Var_::Copy);

  // struct Select
  DefineExprNode<ir::Select>(m, "Select");
  py::class_<ir::Select, ExprNode<ir::Select>> select(*m, "Select");
  select.def_readwrite("condition", &ir::Select::condition)
      .def_readwrite("true_value", &ir::Select::true_value)
      .def_readwrite("false_value", &ir::Select::false_value)
      .def(py::init<ir::Expr, ir::Expr, ir::Expr>())
      .def_static("make", &ir::Select::Make)
      .def("type", &ir::Select::type)
      .def("expr_fields_mutable", py::overload_cast<>(&ir::Select::expr_fields))
      .def("expr_fields_const",
           py::overload_cast<>(&ir::Select::expr_fields, py::const_));

  // struct LoadStoreAddrMnger
  py::class_<ir::LoadStoreAddrMnger> load_store_addr_manager(
      *m, "LoadStoreAddrMnger");
  load_store_addr_manager
      .def_readwrite("tensor", &ir::LoadStoreAddrMnger::tensor)
      .def("is_addr_tensor", &ir::LoadStoreAddrMnger::is_addr_tensor)
      .def("is_addr_scalar", &ir::LoadStoreAddrMnger::is_addr_scalar);

  // struct Load : ExprNode<Load>, LoadStoreAddrMnger
  DefineExprNode<ir::Load>(m, "Load");
  py::class_<ir::Load, ExprNode<ir::Load>, ir::LoadStoreAddrMnger> load(*m,
                                                                        "Load");
  load.def_readwrite("indices", &ir::Load::indices)
      .def("index", &ir::Load::index)
      .def_static("make", &ir::Load::Make)
      .def("expr_fields_mutable", py::overload_cast<>(&ir::Load::expr_fields))
      .def("expr_fields_const",
           py::overload_cast<>(&ir::Load::expr_fields, py::const_))
      .def("name", &ir::Load::name)
      .def("type", &ir::Load::type);

  // struct Store : ExprNode<Store>, LoadStoreAddrMnger
  DefineExprNode<ir::Store>(m, "Store");
  py::class_<ir::Store, ExprNode<ir::Store>, ir::LoadStoreAddrMnger> store(
      *m, "Store");
  store.def_readwrite("value", &ir::Store::value)
      .def_readwrite("indices", &ir::Store::indices)
      .def_static("make", &ir::Store::Make)
      .def("expr_fields_mutable", py::overload_cast<>(&ir::Store::expr_fields))
      .def("expr_fields_const",
           py::overload_cast<>(&ir::Store::expr_fields, py::const_))
      .def("type", &ir::Store::type)
      .def("index", &ir::Store::index);

#define DEFINE_BINARY_NODE(__node)                                           \
  DefineBinaryOpNode<ir::__node>(m, #__node);                                \
  py::class_<ir::__node, ir::BinaryOpNode<ir::__node>> py_##__node(*m,       \
                                                                   #__node); \
  py_##__node.def(py::init<ir::Expr, ir::Expr>())                            \
      .def_static("make",                                                    \
                  py::overload_cast<ir::Expr, ir::Expr>(&ir::__node::Make))  \
      .def("type", &ir::__node::type)

  DEFINE_BINARY_NODE(Add);
  DEFINE_BINARY_NODE(Sub);
  DEFINE_BINARY_NODE(Mul);
  DEFINE_BINARY_NODE(Div);
  DEFINE_BINARY_NODE(Mod);
  DEFINE_BINARY_NODE(Min);
  DEFINE_BINARY_NODE(Max);
  DEFINE_BINARY_NODE(EQ);
  DEFINE_BINARY_NODE(NE);
  DEFINE_BINARY_NODE(LT);
  DEFINE_BINARY_NODE(LE);
  DEFINE_BINARY_NODE(GT);
  DEFINE_BINARY_NODE(GE);
  DEFINE_BINARY_NODE(And);
  DEFINE_BINARY_NODE(Or);

#undef DEFINE_BINARY_NODE

  // FracOp
  DefineBinaryOpNode<ir::FracOp>(m, "FracOp");
  py::class_<ir::FracOp, ir::BinaryOpNode<ir::FracOp>> frac_op(*m, "FracOp");
  frac_op.def(py::init<>())
      .def_static("make", &ir::FracOp::Make)
      .def("type", &ir::FracOp::type);

#define DEFINE_UNARY_NODE(__node)                                           \
  DefineUnaryOpNode<ir::__node>(m, #__node);                                \
  py::class_<ir::__node, ir::UnaryOpNode<ir::__node>> py_##__node(*m,       \
                                                                  #__node); \
  py_##__node.def(py::init<ir::Expr>()).def_static("make", &ir::__node::Make)

  DEFINE_UNARY_NODE(Minus);
  DEFINE_UNARY_NODE(Not);
#undef DEFINE_UNARY_NODE

  py::class_<Var, IrNodeRef> var(*m, "Var");
  var.def(py::init<>())
      .def(py::init<IrNode *>())
      .def(py::init<const std::string &, cinn::common::Type>(),
           arg("name_hint"),
           arg("t") = cinn::common::type_of<int>())
      .def(py::init<Expr, Expr, const std::string &>())
      .def(py::init<int, const std::string &>())
      .def(py::init<Expr, const std::string &>())
      .def("rename", [](Var &self, std::string &name) { self->name = name; })
      .def("get_mutable",
           py::overload_cast<>(&Var::get),
           py::return_value_policy::reference)
      .def("get_const",
           py::overload_cast<>(&Var::get, py::const_),
           py::return_value_policy::reference)
      .def("to_expr_mutable", py::overload_cast<>(&Var::operator ir::Expr))
      .def("to_expr_const",
           py::overload_cast<>(&Var::operator ir::Expr, py::const_))
      .def("__repr__",
           [](Var &self) -> std::string {
             return llvm::formatv("<cinn.ir.Var {0}>", self->name);
           })
      .def("expr", [](Var &self) -> Expr { return Expr(self->self()); })

          BIND_POD_BINARY_OP(int())  //
      BIND_POD_BINARY_OP(int32_t())  //
      BIND_POD_BINARY_OP(float())

#define BINARY_OP(type__)                                                   \
  .def("__add__", [](Var &self, type__ v) -> Expr { return self + v; })     \
      .def("__sub__", [](Var &self, type__ v) -> Expr { return self - v; }) \
      .def("__truediv__",                                                   \
           [](Var &self, type__ v) -> Expr { return self / v; })            \
      .def("__mul__", [](Var &self, type__ v) -> Expr { return self * v; }) \
      .def("__mod__", [](Var &self, type__ v) -> Expr { return self % v; })

          BINARY_OP(int32_t)  //
      BINARY_OP(int64_t)      //
      BINARY_OP(float)        //
      BINARY_OP(double);
#undef BINARY_OP

  DefineExprNode<ir::Product>(m, "Product");
  py::class_<ir::Product, ir::ExprNode<ir::Product>> product(*m, "Product");
  product.def_static("make", &ir::Product::Make)
      .def("type", &ir::Product::type)
      .def("operand_mutable",
           py::overload_cast<int>(&ir::Product::operand),
           py::return_value_policy::reference)
      .def("operand_const",
           py::overload_cast<int>(&ir::Product::operand, py::const_),
           py::return_value_policy::reference);

  DefineExprNode<ir::Sum>(m, "Sum");
  py::class_<ir::Sum, ir::ExprNode<ir::Sum>> sum(*m, "Sum");
  sum.def_static("make", &ir::Sum::Make)
      .def("operand_mutable",
           py::overload_cast<int>(&ir::Sum::operand),
           py::return_value_policy::reference)
      .def("operand_const",
           py::overload_cast<int>(&ir::Sum::operand, py::const_),
           py::return_value_policy::reference)
      .def("type", &ir::Sum::type);

  DefineExprNode<ir::Block>(m, "Block");
  py::class_<ir::Block, ir::ExprNode<ir::Block>> block(*m, "Block");
  block.def_readwrite("stmts", &ir::Block::stmts)
      .def(py::init<>())
      .def_static("make", &ir::Block::Make)
      .def("expr_fields_mutable", py::overload_cast<>(&ir::Block::expr_fields))
      .def("expr_fields_const",
           py::overload_cast<>(&ir::Block::expr_fields, py::const_));

  py::class_<ir::_Module_, ir::IrNode> _module_(*m, "_Module_");
  _module_.def_readwrite("name", &ir::_Module_::name)
      .def_readwrite("target", &ir::_Module_::target)
      .def_readwrite("buffers", &ir::_Module_::buffers)
      .def_readwrite("functions", &ir::_Module_::functions)
      .def_readwrite("submodules", &ir::_Module_::submodules);

  DefineExprNode<ir::_Buffer_>(m, "_Buffer_");
  py::class_<ir::_Buffer_, ir::ExprNode<ir::_Buffer_>> _buffer_(*m, "_Buffer_");
  _buffer_
      .def_static(
          "make",
          py::overload_cast<const std::string &, Type>(&ir::_Buffer_::Make))
      .def_static(
          "make",
          py::overload_cast<const std::string &, const std::vector<Expr> &>(
              &ir::_Buffer_::Make));
  py::class_<ir::Buffer> buffer(*m, "Buffer");
  buffer.def(py::init<>());

  py::class_<ir::ModuleExpr> module_expr(*m, "ModuleExpr");
  module_expr.def(py::init<const std::vector<Expr> &>());

  DefineExprNode<ir::IfThenElse>(m, "IfThenElse");
  py::class_<ir::IfThenElse> if_then_else(*m, "IfThenElse");
  if_then_else.def_static(
      "make",
      py::overload_cast<Expr, Expr, Expr>(&ir::IfThenElse::Make),
      py::arg("condition"),
      py::arg("true_case"),
      py::arg("false_case") = ir::Expr());
}

void BindOperation(py::module *m) {
  py::class_<ir::PlaceholderOp> placeholder_op(*m, "PlaceholderOp");
  placeholder_op.def_readwrite("shape", &ir::PlaceholderOp::shape)
      .def_readwrite("dtype", &ir::PlaceholderOp::dtype)
      .def_static("make",
                  py::overload_cast<const std::string &,
                                    const std::vector<Expr> &,
                                    Type>(&ir::PlaceholderOp::Make))
      .def_static("make",
                  py::overload_cast<const std::string &,
                                    const std::vector<ir::Dim> &,
                                    Type>(&ir::PlaceholderOp::Make))
      .def("func_type", &ir::PlaceholderOp::func_type);

  py::class_<ir::CallOp> call_op(*m, "CallOp");
  call_op.def("target", &ir::CallOp::target)
      .def_readwrite("call_expr", &ir::CallOp::call_expr)
      .def("read_args_mutable", py::overload_cast<>(&ir::CallOp::read_args))
      .def("read_args_const",
           py::overload_cast<>(&ir::CallOp::read_args, py::const_))
      .def("write_args_mutable", py::overload_cast<>(&ir::CallOp::write_args))
      .def("write_args_const",
           py::overload_cast<>(&ir::CallOp::write_args, py::const_))
      .def("args", &ir::CallOp::args)
      .def_readwrite("func", &ir::CallOp::func)
      .def_readwrite("value_slot", &ir::CallOp::value_slot)
      .def_readwrite("is_tuple_get", &ir::CallOp::is_tuple_get)
      .def_readwrite("num_value_slots", &ir::CallOp::num_value_slots)
      .def(py::init<>())
      .def_static("make", &ir::CallOp::Make)
      .def("func_type", &ir::CallOp::func_type);

  py::class_<ir::ComputeOp> compute_op(*m, "ComputeOp");
  compute_op.def_readwrite("reduce_axis", &ir::ComputeOp::reduce_axis)
      .def_readwrite("shape", &ir::ComputeOp::shape)
      .def_readwrite("body", &ir::ComputeOp::body)
      .def_readwrite("producer_fn", &ir::ComputeOp::producer_fn)
      .def(py::init<>())
      .def_static("make", &ir::ComputeOp::Make)
      .def("func_type", &ir::ComputeOp::func_type);
}

void BindIrTensor(py::module *m) {
  py::class_<ir::Tensor, ir::IrNodeRef> tensor(*m, "Tensor");
  tensor.def(py::init<>())
      .def(py::init<ir::IrNode *>())
      .def("ndims", &ir::Tensor::ndims)
      .def("__call__", [](ir::Tensor &self, Expr a) { return self(a); })
      .def("__call__",
           [](ir::Tensor &self, Expr a, Expr b) { return self(a, b); })
      .def("__call__",
           [](ir::Tensor &self, Expr a, Expr b, Expr c) {
             return self(a, b, c);
           })
      .def("__call__",
           [](ir::Tensor &self, Expr a, Expr b, Expr c, Expr d) {
             return self(a, b, c, d);
           })
      .def("__getitem__", [](ir::Tensor &self, Expr a) { return self(a); })
      .def("__getitem__",
           [](ir::Tensor &self, Expr a, Expr b) { return self(a, b); })
      .def("__getitem__",
           [](ir::Tensor &self, Expr a, Expr b, Expr c) {
             return self(a, b, c);
           })
      .def("__getitem__",
           [](ir::Tensor &self, Expr a, Expr b, Expr c, Expr d) {
             return self(a, b, c, d);
           })
      .def("__getitem__",
           [](ir::Tensor &self, std::vector<Expr> idx) { return self(idx); })
      .def("Expr", [](ir::Tensor &self) { return self.operator Expr(); });

  DefineExprNode<ir::_Tensor_>(m, "_Tensor_");
  py::class_<ir::_Tensor_, ir::ExprNode<ir::_Tensor_>> _tensor_(*m, "_Tensor_");
  _tensor_.def_readwrite("shape", &ir::_Tensor_::shape)
      .def_readwrite("reduce_axis", &ir::_Tensor_::reduce_axis)
      .def_readwrite("operation", &ir::_Tensor_::operation)
      .def_readwrite("name", &ir::_Tensor_::name)
      .def_readwrite("buffer", &ir::_Tensor_::buffer)
      .def("domain_with_reduce_axis", &ir::_Tensor_::domain_without_reduce_axis)
      .def("domain_without_reduce_axis",
           &ir::_Tensor_::domain_without_reduce_axis)
      .def_static(
          "make",
          py::overload_cast<const std::string &,
                            Type,
                            const std::vector<Expr> &,
                            const std::vector<Expr> &,
                            const std::vector<Var> &>(&ir::_Tensor_::Make),
          py::arg("name"),
          py::arg("dtype"),
          py::arg("shape"),
          py::arg("domain"),
          py::arg("reduce_axis") = std::vector<Var>({}))
      .def("is_tuple", &ir::_Tensor_::is_tuple)
      .def("is_tuple_get", &ir::_Tensor_::is_tuple_get)
      .def("tuple_get", &ir::_Tensor_::TupleGet)
      .def("get_depend_tensor_names", &ir::_Tensor_::GetDependTensorNames)
      .def("is_depend_on_statement", &ir::_Tensor_::IsDependOnStatement)
      .def("depending_tensor_names", &ir::_Tensor_::DependingTensorNames)
      .def("same_shape_with", &ir::_Tensor_::HasSameShapeWith)
      .def("is_compute_node", &ir::_Tensor_::is_compute_node)
      .def("is_placeholder_node", &ir::_Tensor_::is_placeholder_node)
      .def("is_call_node", &ir::_Tensor_::is_call_node)
      .def("is_extern_call_node", &ir::_Tensor_::is_extern_call_node)
      .def("is_preceding_view_node", &ir::_Tensor_::is_preceding_view_node)
      .def("is_buffer_shared_node", &ir::_Tensor_::is_buffer_shared_node)
      .def("operation_type", &ir::_Tensor_::operation_type)
      .def("get_compute_op", &ir::_Tensor_::get_compute_op)
      .def("get_placeholder_op", &ir::_Tensor_::get_placeholder_op)
      .def("body", &ir::_Tensor_::body)
      .def("tensor_store_expanded_body",
           &ir::_Tensor_::tensor_store_expanded_body)
      .def("inline_expanded", &ir::_Tensor_::inline_expanded)
      .def("contains_reduce_axis", &ir::_Tensor_::contains_reduce_axis)
      .def("expr_fields_mutable",
           py::overload_cast<>(&ir::_Tensor_::expr_fields))
      .def("expr_fields_const",
           py::overload_cast<>(&ir::_Tensor_::expr_fields, py::const_))
      .def("axis", &ir::_Tensor_::axis)
      .def("axis_with_reduce", &ir::_Tensor_::axis_with_reduce)
      .def("buffer_depended_tensor_names",
           &ir::_Tensor_::buffer_depended_tensor_names)
      .def(py::init<>())
      .def("has_expression", &ir::_Tensor_::has_expression)
      .def("reshape", &ir::_Tensor_::Reshape)
      .def("reshape_copied", &ir::_Tensor_::ReshapeCopied)
      .def("with_buffer",
           py::overload_cast<const ir::Type &>(&ir::_Tensor_::WithBuffer),
           py::arg("type") = Type::type_t::Void)
      .def("with_buffer",
           py::overload_cast<const std::string &,
                             const std::string &,
                             const ir::Type &>(&ir::_Tensor_::WithBuffer),
           py::arg("memory_type"),
           py::arg("buffer_name") = "",
           py::arg("type") = Type::type_t::Void)
      .def("bind", py::overload_cast<lang::Buffer &>(&ir::_Tensor_::Bind))
      .def("bind", py::overload_cast<const ir::Buffer &>(&ir::_Tensor_::Bind))
      .def("__str__", [](const ir::Tensor &self) {
        return "<Tensor " + self->name + ">";
      });

  py::class_<ir::Operation /*, ir::FunctionDef*/> operation(*m, "Operation");
  operation.def(py::init<>())
      .def(py::init<ir::IrNode *>())
      .def_readwrite("name", &ir::Operation::name);
}

auto PackedFuncCall(lang::PackedFunc &self, py::args args) {  // NOLINT
  lang::Args cinn_args;
  using cinn::common::CINNValue;
  for (auto handle : args) {
    if (py::isinstance<py::int_>(handle)) {
      cinn_args.Append(CINNValue(py::cast<int64_t>(handle)));
    } else if (py::isinstance<py::float_>(handle)) {
      cinn_args.Append(CINNValue(py::cast<float>(handle)));
    } else if (py::isinstance<ir::Var>(handle)) {
      cinn_args.Append(CINNValue(py::cast<ir::Var>(handle)));
    } else if (py::isinstance<ir::Expr>(handle)) {
      cinn_args.Append(CINNValue(py::cast<ir::Expr>(handle)));
    } else {
      std::stringstream ss;
      ss << "unsupported type: " << std::string(py::str(handle.get_type()));
      PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
    }
  }
  lang::RetValue ret_value;
  self.body()(cinn_args, &ret_value);
  return ConvertToVar(ret_value);
}

void BindPackedFunc(py::module *m) {
  py::class_<lang::Args> args(*m, "Args");
  args.def(py::init<>())
      .def(py::init<cinn_value_t *, int *, int>())
      .def("append", &lang::Args::Append)
      .def("size", &lang::Args::size)
      .def("__len__", &lang::Args::size)
      .def(
          "__getitem__",
          [](lang::Args &self, int i) { return self[i]; },
          py::return_value_policy::reference)
      .def("__setitem__",
           [](lang::Args &self, int i, cinn::common::CINNValue &v) {
             self[i] = v;
           });

  py::class_<lang::PackedFunc> packed_func(*m, "PackedFunc");
  packed_func.def(py::init<>())
      .def(py::init<const std::string &>())
      .def(py::init<lang::PackedFunc::body_t>())
      .def("body", &lang::PackedFunc::body)
      .def("__call__", &PackedFuncCall);
}

void BindRegistry(py::module *m) {
  py::class_<ir::Registry> registry(*m, "Registry");
  registry
      .def_static("register",
                  &ir::Registry::Register,
                  py::arg("name"),
                  py::arg("override") = false,
                  py::return_value_policy::reference)
      .def_static("register",
                  &ir::Registry::Register,
                  py::return_value_policy::reference)
      .def_static("remove", &ir::Registry::Remove)
      .def_static("get", &ir::Registry::Get, py::return_value_policy::reference)
      .def_static("list_names", &ir::Registry::ListNames)
      .def("set_body",
           py::overload_cast<lang::PackedFunc>(&ir::Registry::SetBody),
           py::return_value_policy::reference);

#ifdef CINN_WITH_TEST
  ir::Registry::Register("test_add_int64")
      .SetBody([](lang::Args args, lang::RetValue *rv) {
        int64_t x = args[0];
        int64_t y = args[1];
        *rv = x + y;
      });

  ir::Registry::Register("test_add_expr")
      .SetBody([](lang::Args args, lang::RetValue *rv) {
        ir::Expr x = args[0];
        ir::Expr y = args[1];
        *rv = x + y;
      });

  ir::Registry::Register("test_mul_float")
      .SetBody([](lang::Args args, lang::RetValue *rv) {
        float x = args[0];
        float y = args[1];
        *rv = x * y;
      });
#endif
}

void BindIrContext(py::module *m) {
  using ir::Expr;
  using ir::IrNode;
  using ir::IrNodeRef;
  using ir::Var;
  using py::arg;

  py::class_<IRContext> ir_ctx(*m, "IRContext");
  ir_ctx.def(py::init<>())
      .def(py::init<IRContextNode *>())
      .def("EnterWithContext",
           [](IRContext &self) { self.data_->EnterWithContext(); })
      .def("ExitWithContext",
           [](IRContext &self) { self.data_->ExitWithContext(); })
      .def("get_for_loop_var",
           [](IRContext &self) {
             return self.data_->safe_as<ForContextNode>()->loop_var;
           })
      .def_static("MakeLowerFunctionContext",
                  [](std::string &name) {
                    return IRContext(new LowerFuncContextNode(name));
                  })
      .def_static("MakeScheduleBlockContext",
                  [](std::string &name) {
                    return IRContext(new ScheduleBlockContextNode(name));
                  })
      .def_static("MakeIfContext",
                  [](Expr expr) { return IRContext(new IfContextNode(expr)); })
      .def_static("MakeElseContext",
                  []() { return IRContext(new ElseContextNode()); })
      .def_static("MakeThenContext",
                  []() { return IRContext(new ThenContextNode()); });

  m->def("link_to_parent_context", &pybind::LinkToParentContext);

  py::class_<IRBuilder> ir_builder(*m, "IRBuilder");
  ir_builder.def(py::init<>())
      .def("EnterWithContext", &IRBuilder::EnterWithContext)
      .def("ExitWithContext", &IRBuilder::ExitWithContext)
      .def("get_result",
           [](IRBuilder &self) { return self.data_->GetResult(); });

  m->def("AxisMap", &AxisMap);
  m->def("TensorStore", &TensorStore);
  m->def("Arg", py::overload_cast<const std::string &, Var>(&Arg));
  m->def("Arg", py::overload_cast<const std::string &, ir::Buffer>(&Arg));
  m->def("Sequential", py::overload_cast<Expr, Expr>(&Sequential));
}
}  // namespace

void BindIr(py::module *m) {
  BindOperation(m);
  BindLoweredFunc(m);
  BindNode(m);
  BindIrVisitor(m);
  BindIrIr(m);
  BindIrTensor(m);
  BindIrContext(m);
  BindPackedFunc(m);
  BindRegistry(m);
}
}  // namespace cinn::pybind
