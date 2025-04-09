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

#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/common/object.h"
#include "paddle/cinn/common/shared.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/common/type.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/pybind/bind.h"
#include "paddle/cinn/pybind/bind_utils.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/utils/string.h"

namespace py = pybind11;

namespace cinn::pybind {

using cinn::common::Arch;
using cinn::common::ARMArch;
using cinn::common::bfloat16;
using cinn::common::CINNValue;
using cinn::common::float16;
using cinn::common::HygonDCUArchHIP;
using cinn::common::HygonDCUArchSYCL;
using cinn::common::NVGPUArch;
using cinn::common::Target;
using cinn::common::Type;
using cinn::common::UnknownArch;
using cinn::common::X86Arch;
using utils::GetStreamCnt;
using utils::StringFormat;

namespace {
void BindTarget(py::module *);
void BindType(py::module *);
void BindShared(py::module *);
void BindCinnValue(py::module *);

void ResetGlobalNameID() { cinn::common::Context::Global().ResetNameId(); }

void BindTarget(py::module *m) {
  py::class_<Arch>(*m, "Arch")
      .def("IsX86Arch",
           [](const common::Arch &arch) {
             return std::holds_alternative<common::X86Arch>(arch);
           })
      .def("IsNVGPUArch",
           [](const common::Arch &arch) {
             return std::holds_alternative<common::NVGPUArch>(arch);
           })
      .def("IsHygonDCUArchHIP", [](const common::Arch &arch) {
        return std::holds_alternative<common::HygonDCUArchHIP>(arch);
      });

  py::class_<Target> target(*m, "Target");
  target.def_readwrite("os", &Target::os)
      .def_readwrite("arch", &Target::arch)
      .def_static("X86Arch", []() -> common::Arch { return common::X86Arch{}; })
      .def_static("NVGPUArch",
                  []() -> common::Arch { return common::NVGPUArch{}; })
      .def_static("HygonDCUArchHIP",
                  []() -> common::Arch { return common::HygonDCUArchHIP{}; })
      .def_readwrite("bits", &Target::bits)
      .def_readwrite("features", &Target::features)
      .def(py::init<>())
      .def(py::init<Target::OS,
                    Arch,
                    Target::Bit,
                    const std::vector<Target::Feature> &>())
      .def("defined", &Target::defined)
      .def("runtime_arch", &Target::runtime_arch);

  m->def("DefaultHostTarget", &cinn::common::DefaultHostTarget)
      .def("DefaultNVGPUTarget", &cinn::common::DefaultNVGPUTarget)
      .def("DefaultHygonDcuHipTarget", &cinn::common::DefaultHygonDcuHipTarget)
      .def("DefaultTarget", &cinn::common::DefaultTarget);

  m->def("get_target", &cinn::runtime::CurrentTarget::GetCurrentTarget);
  m->def("set_target",
         &cinn::runtime::CurrentTarget::SetCurrentTarget,
         py::arg("target"));

  py::enum_<Target::OS> os(target, "OS");
  os.value("Unk", Target::OS::Unk)
      .value("Linux", Target::OS::Linux)
      .value("Windows", Target::OS::Windows);

  py::enum_<Target::Bit> bit(target, "Bit");
  bit.value("Unk", Target::Bit::Unk)
      .value("k32", Target::Bit::k32)
      .value("k64", Target::Bit::k64);

  py::enum_<Target::Feature> feature(target, "Feature");
  feature.value("JIT", Target::Feature::JIT)
      .value("Debug", Target::Feature::Debug);

  m->def("is_compiled_with_cuda", cinn::runtime::IsCompiledWithCUDA);
  m->def("is_compiled_with_cudnn", cinn::runtime::IsCompiledWithCUDNN);
  m->def("reset_name_id", ResetGlobalNameID);
}

void BindType(py::module *m) {
  py::class_<Type> type(*m, "Type");
  type.def(py::init<>())
      .def(py::init<Type &>())
      .def(py::init<Type::type_t, int, int, Type::specific_type_t>());
#define DEFINE_TYPE_METHOD(__name) (type = type.def(#__name, &Type::__name))
  DEFINE_TYPE_METHOD(is_primitive);
  DEFINE_TYPE_METHOD(is_unk);
  DEFINE_TYPE_METHOD(is_void);
  DEFINE_TYPE_METHOD(is_bool);
  DEFINE_TYPE_METHOD(is_vector);
  DEFINE_TYPE_METHOD(is_scalar);
  DEFINE_TYPE_METHOD(is_float);
  DEFINE_TYPE_METHOD(is_float16);
  DEFINE_TYPE_METHOD(is_bfloat16);
  DEFINE_TYPE_METHOD(is_int);
  DEFINE_TYPE_METHOD(is_uint);
  DEFINE_TYPE_METHOD(is_string);
  DEFINE_TYPE_METHOD(set_cpp_handle);
  DEFINE_TYPE_METHOD(is_cpp_handle);
  DEFINE_TYPE_METHOD(set_cpp_handle2);
  DEFINE_TYPE_METHOD(is_cpp_handle2);
  DEFINE_TYPE_METHOD(set_cpp_const);
  DEFINE_TYPE_METHOD(is_cpp_const);
  DEFINE_TYPE_METHOD(set_customized_type);
  DEFINE_TYPE_METHOD(customized_type);
  DEFINE_TYPE_METHOD(is_customized_type);
  DEFINE_TYPE_METHOD(with_bits);
  DEFINE_TYPE_METHOD(with_type);
  DEFINE_TYPE_METHOD(with_cpp_const);
#undef DEFINE_TYPE_METHOD
  type.def("vector_of", &Type::VectorOf)
      .def("element_of", &Type::ElementOf)
      .def("pointer_of", &Type::PointerOf)
      .def("__str__", [](const Type &self) { return GetStreamCnt(self); })
      .def("__repr__", [](const Type &self) {
        return StringFormat("<Type: %s>", GetStreamCnt(self).c_str());
      });

  py::enum_<Type::type_t> type_t(type, "type_t");
  type_t.value("unk", Type::type_t::Unk)
      .value("int", Type::type_t::Int)
      .value("uInt", Type::type_t::UInt)
      .value("float", Type::type_t::Float)
      .value("string", Type::type_t::String)
      .value("void", Type::type_t::Void)
      .value("customized", Type::type_t::Customized)
      .export_values();

  py::enum_<Type::specific_type_t> specific_type_t(type, "specific_type_t");
  specific_type_t.value("UNK", Type::specific_type_t::None)
      .value("FP16", Type::specific_type_t::FP16)
      .value("BF16", Type::specific_type_t::BF16)
      .export_values();

  py::enum_<Type::cpp_type_t> cpp_type_t(type, "cpp_type_t");
  cpp_type_t.value("None", Type::cpp_type_t::None)
      .value("Const", Type::cpp_type_t::Const)
      .value("Handle", Type::cpp_type_t::Handle)
      .value("HandleHandle", Type::cpp_type_t::HandleHandle)
      .export_values();

  m->def("Void", &cinn::common::Void)
      .def("Int", &cinn::common::Int, py::arg("bits"), py::arg("lanes") = 1)
      .def("UInt", &cinn::common::UInt, py::arg("bits"), py::arg("lanes") = 1)
      .def("Float",
           &cinn::common::Float,
           py::arg("bits"),
           py::arg("lanes") = 1,
           py::arg("st") = Type::specific_type_t::None)
      .def("Float16", &cinn::common::Float16, py::arg("lanes") = 1)
      .def("BFloat16", &cinn::common::BFloat16, py::arg("lanes") = 1)
      .def("Bool", &cinn::common::Bool, py::arg("lanes") = 1)
      .def("String", &cinn::common::String);

  m->def(
       "make_const",
       [](const Type &type, int32_t val) -> Expr {
         return cinn::common::make_const(type, val);
       },
       py::arg("type"),
       py::arg("val"))
      .def(
          "make_const",
          [](const Type &type, int64_t val) -> Expr {
            return cinn::common::make_const(type, val);
          },
          py::arg("type"),
          py::arg("val"))
      .def(
          "make_const",
          [](const Type &type, float val) -> Expr {
            return cinn::common::make_const(type, val);
          },
          py::arg("type"),
          py::arg("val"))
      .def(
          "make_const",
          [](const Type &type, double val) -> Expr {
            return cinn::common::make_const(type, val);
          },
          py::arg("type"),
          py::arg("val"))
      .def(
          "make_const",
          [](const Type &type, bool val) -> Expr {
            return cinn::common::make_const(type, val);
          },
          py::arg("type"),
          py::arg("val"));

  m->def("type_of", [](absl::string_view dtype) {
    return cinn::common::Str2Type(dtype.data());
  });
}

void BindShared(py::module *m) {
  py::class_<cinn::common::RefCount> ref_count(*m, "RefCount");
  ref_count.def(py::init<>())
      .def("inc", &cinn::common::RefCount::Inc)
      .def("dec", &cinn::common::RefCount::Dec)
      .def("is_zero", &cinn::common::RefCount::is_zero)
      .def("to_string", &cinn::common::RefCount::to_string)
      .def("val", &cinn::common::RefCount::val);
}

// TODO(wanghaipeng03) using true_type or false_type as tag dispatcher losses
// semantic context
template <typename T1, typename T2, typename F>
inline auto __binary_op_fn_dispatch(T1 x, T2 y, F fn, std::true_type) {
  return fn(ir::Expr(x), ir::Expr(y)).as_var_ref();
}
template <typename T1, typename T2, typename F>
inline auto __binary_op_fn_dispatch(T1 x, T2 y, F fn, std::false_type) {
  return fn(x, y);
}

template <typename T1, typename T2, typename F>
inline void __binary_op_visitor_dispatch(
    CINNValue &v, T1 lhs, T2 rhs, F fn, std::true_type) {  // NOLINT
  v = CINNValue();
}
template <typename T1, typename T2, typename F>
inline void __binary_op_visitor_dispatch(
    CINNValue &v, T1 lhs, T2 rhs, F fn, std::false_type) {  // NOLINT
  v.Set(fn(lhs, rhs));
}

void BindCinnValue(py::module *m) {
  using cinn::common::_CINNValuePack_;
  using cinn::common::CINNValuePack;

  DefineShared<_CINNValuePack_>(m, "_CINNValuePack_");

  py::class_<_CINNValuePack_> cinn_value_pack(*m, "_CINNValuePack_");
  cinn_value_pack.def_static("make", &_CINNValuePack_::Make)
      .def("__getitem__",
           [](_CINNValuePack_ &self, int offset) { return self[offset]; })
      .def("__setitem__",
           [](_CINNValuePack_ &self, int offset, CINNValue &v) {
             self[offset] = v;
           })
      .def("add_value", &_CINNValuePack_::AddValue)
      .def("clear", &_CINNValuePack_::Clear)
      .def("size", &_CINNValuePack_::size)
      .def("__len__", &_CINNValuePack_::size)
      .def("type_info", &_CINNValuePack_::type_info);

  py::class_<CINNValuePack, cinn::common::Shared<_CINNValuePack_>>
      cinn_value_pack_shared(*m, "CINNValuePack");
  cinn_value_pack_shared.def(py::init<_CINNValuePack_ *>())
      .def("__getitem__",
           [](CINNValuePack &self, int offset) { return self[offset]; })
      .def("__setitem__", [](CINNValuePack &self, int offset, CINNValue &v) {
        self[offset] = v;
      });

  py::class_<CINNValue, cinn_pod_value_t> cinn_value(*m, "CINNValue");
  cinn_value.def(py::init<>())
      .def(py::init<cinn_value_t, int>())
      .def(py::init<bool>())
      .def(py::init<int8_t>())
      .def(py::init<int32_t>())
      .def(py::init<int64_t>())
      .def(py::init<float>())
      .def(py::init<double>())
      .def(py::init<char *>())
      .def(py::init<cinn_buffer_t *>())
      .def(py::init<void *>())
      .def(py::init<const char *>())
      .def(py::init<const ir::Var &>())
      .def(py::init<const ir::Expr &>())
      .def(py::init<const CINNValuePack &>())
      .def("defined", &CINNValue::defined)
      .def("to_double",
           [](CINNValue &self) { return static_cast<double>(self); })
      .def("to_float", [](CINNValue &self) { return static_cast<float>(self); })
      .def("to_int8", [](CINNValue &self) { return static_cast<int8_t>(self); })
      .def("to_int32",
           [](CINNValue &self) { return static_cast<int32_t>(self); })
      .def("to_int64",
           [](CINNValue &self) { return static_cast<int64_t>(self); })
      .def("to_void_p",
           [](CINNValue &self) { return static_cast<void *>(self); })
      .def("to_cinn_buffer_p",
           [](CINNValue &self) { return static_cast<cinn_buffer_t *>(self); })
      .def("to_str", [](CINNValue &self) { return static_cast<char *>(self); })
      .def("to_var", [](CINNValue &self) { return self.operator ir::Var(); })
      .def("to_expr",
           [](CINNValue &self) { return ir::Expr(self.operator ir::Expr()); })
      .def("set", &CINNValue::Set<int32_t>)
      .def("set", &CINNValue::Set<int64_t>)
      .def("set", &CINNValue::Set<float>)
      .def("set", &CINNValue::Set<double>)
      .def("set", &CINNValue::Set<char *>)
      .def("set", &CINNValue::Set<const ir::Var &>)
      .def("set", &CINNValue::Set<const ir::Expr &>)
      .def("set", &CINNValue::Set<cinn_buffer_t *>)
      .def("set", &CINNValue::Set<const CINNValuePack &>)
      .def("set", &CINNValue::Set<const char *>)
      .def("set", &CINNValue::Set<const CINNValue &>);

  auto binary_op_visitor = [](CINNValue &v, auto lhs, auto rhs, auto fn) {
    using lhs_t = decltype(lhs);
    using rhs_t = decltype(rhs);
    using tag_t =
        std::conditional_t<std::is_same<lhs_t, std::nullptr_t>::value ||
                               std::is_same<rhs_t, std::nullptr_t>::value ||
                               !std::is_same<lhs_t, rhs_t>::value,
                           std::true_type,
                           std::false_type>;
    __binary_op_visitor_dispatch(v, lhs, rhs, fn, tag_t{});
  };

#define DEFINE_BINARY_OP(__op, __fn)                                          \
  auto __op##_fn = [&](auto x, auto y) {                                      \
    constexpr auto is_var_x =                                                 \
        std::is_same<std::decay_t<decltype(x)>, ir::Var>::value;              \
    constexpr auto is_var_y =                                                 \
        std::is_same<std::decay_t<decltype(y)>, ir::Var>::value;              \
    using tag_t = std::                                                       \
        conditional_t<is_var_x && is_var_y, std::true_type, std::false_type>; \
    return __binary_op_fn_dispatch(x, y, __fn, tag_t{});                      \
  };                                                                          \
  cinn_value.def(#__op, [&](CINNValue &self, CINNValue &other) {              \
    auto visitor = [&](auto x, auto y) {                                      \
      return binary_op_visitor(self, x, y, __op##_fn);                        \
    };                                                                        \
    absl::visit(visitor, ConvertToVar(self), ConvertToVar(other));            \
    return self;                                                              \
  })

  DEFINE_BINARY_OP(__add__, [](auto x, auto y) { return x + y; });
  DEFINE_BINARY_OP(__sub__, [](auto x, auto y) { return x - y; });
  DEFINE_BINARY_OP(__mul__, [](auto x, auto y) { return x * y; });
  DEFINE_BINARY_OP(__truediv__, [](auto x, auto y) { return x / y; });
  DEFINE_BINARY_OP(__and__, [](auto x, auto y) { return x && y; });
  DEFINE_BINARY_OP(__or__, [](auto x, auto y) { return x || y; });
  DEFINE_BINARY_OP(__lt__, [](auto x, auto y) { return x < y; });
  DEFINE_BINARY_OP(__le__, [](auto x, auto y) { return x <= y; });
  DEFINE_BINARY_OP(__gt__, [](auto x, auto y) { return x > y; });
  DEFINE_BINARY_OP(__ge__, [](auto x, auto y) { return x >= y; });

#undef DEFINE_BINARY_OP
}
}  // namespace

void BindCommon(py::module *m) {
  BindTarget(m);
  BindType(m);
  BindShared(m);
  BindCinnValue(m);
}
}  // namespace cinn::pybind
