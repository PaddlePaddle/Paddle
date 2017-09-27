/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pybind/protobuf.h"
#include <deque>
#include <iostream>
#include "paddle/framework/block_desc.h"
#include "paddle/framework/op_desc.h"
#include "paddle/framework/program_desc.h"
#include "paddle/framework/var_desc.h"

// Cast boost::variant for PyBind.
// Copy from
// https://github.com/pybind/pybind11/issues/576#issuecomment-269563199
namespace pybind11 {
namespace detail {

// Can be replaced by a generic lambda in C++14
struct variant_caster_visitor : public boost::static_visitor<handle> {
  return_value_policy policy;
  handle parent;

  variant_caster_visitor(return_value_policy policy, handle parent)
      : policy(policy), parent(parent) {}

  template <class T>
  handle operator()(T const &src) const {
    return make_caster<T>::cast(src, policy, parent);
  }
};

template <class Variant>
struct variant_caster;

template <template <class...> class V, class... Ts>
struct variant_caster<V<Ts...>> {
  using Type = V<Ts...>;

  template <typename T>
  typename std::enable_if<
      !std::is_same<T, boost::detail::variant::void_>::value, bool>::type
  try_load(handle src, bool convert) {
    auto caster = make_caster<T>();
    if (!load_success_ && caster.load(src, convert)) {
      load_success_ = true;
      value = cast_op<T>(caster);
      return true;
    }
    return false;
  }

  template <typename T>
  typename std::enable_if<std::is_same<T, boost::detail::variant::void_>::value,
                          bool>::type
  try_load(handle src, bool convert) {
    return false;
  }

  bool load(handle src, bool convert) {
    auto unused = {false, try_load<Ts>(src, convert)...};
    (void)(unused);
    return load_success_;
  }

  static handle cast(Type const &src, return_value_policy policy,
                     handle parent) {
    variant_caster_visitor visitor(policy, parent);
    return boost::apply_visitor(visitor, src);
  }

  PYBIND11_TYPE_CASTER(Type, _("Variant"));
  bool load_success_{false};
};

// Add specialization for concrete variant type
template <class... Args>
struct type_caster<boost::variant<Args...>>
    : variant_caster<boost::variant<Args...>> {};

}  // namespace detail
}  // namespace pybind11

namespace paddle {
namespace pybind {

// Bind Methods
void BindProgramDesc(py::module &m) {
  py::class_<framework::ProgramDescBind>(m, "ProgramDesc", "")
      .def_static(
          "instance",
          []() -> framework::ProgramDescBind * {
            return &framework::ProgramDescBind::Instance(&GetProgramDesc());
          },
          py::return_value_policy::reference)
      .def_static("__create_program_desc__",
                  []() -> framework::ProgramDescBind * {
                    // Only used for unit-test
                    auto *prog_desc = new ProgramDesc;
                    auto *block = prog_desc->mutable_blocks()->Add();
                    block->set_idx(0);
                    block->set_parent_idx(-1);
                    return &framework::ProgramDescBind::Instance(prog_desc);
                  },
                  py::return_value_policy::reference)
      .def("append_block", &framework::ProgramDescBind::AppendBlock,
           py::return_value_policy::reference)
      .def("block", &framework::ProgramDescBind::Block,
           py::return_value_policy::reference)
      .def("__str__", &framework::ProgramDescBind::DebugString)
      .def("num_blocks", &framework::ProgramDescBind::Size);
}

void BindBlockDesc(py::module &m) {
  py::class_<framework::BlockDescBind>(m, "BlockDesc", "")
      .def_property_readonly("id", &framework::BlockDescBind::ID)
      .def_property_readonly("parent", &framework::BlockDescBind::Parent)
      .def("append_op", &framework::BlockDescBind::AppendOp,
           py::return_value_policy::reference)
      .def("prepend_op", &framework::BlockDescBind::PrependOp,
           py::return_value_policy::reference)
      .def("new_var",
           [](framework::BlockDescBind &self, py::bytes byte_name) {
             std::string name = byte_name;
             return self.NewVar(name);
           },
           py::return_value_policy::reference)
      .def("var",
           [](framework::BlockDescBind &self, py::bytes byte_name) {
             std::string name = byte_name;
             return self.Var(name);
           },
           py::return_value_policy::reference)
      .def("all_vars", &framework::BlockDescBind::AllVars,
           py::return_value_policy::reference)
      .def("all_ops", &framework::BlockDescBind::AllOps,
           py::return_value_policy::reference);
}

void BindVarDsec(py::module &m) {
  py::enum_<framework::DataType>(m, "DataType", "")
      .value("BOOL", DataType::BOOL)
      .value("INT16", DataType::INT16)
      .value("INT32", DataType::INT32)
      .value("INT64", DataType::INT64)
      .value("FP16", DataType::FP16)
      .value("FP32", DataType::FP32)
      .value("FP64", DataType::FP64);

  py::class_<framework::VarDescBind>(m, "VarDesc", "")
      .def("name",
           [](const framework::framework::VarDescBind &self) {
             py::bytes name = self.Name();
             return name;
           },
           py::return_value_policy::reference)
      .def("set_shape", &framework::VarDescBind::SetShape)
      .def("set_data_type", &framework::VarDescBind::SetDataType)
      .def("shape", &framework::VarDescBind::Shape,
           py::return_value_policy::reference)
      .def("data_type", &framework::VarDescBind::DataType);
}

void BindOpDesc(py::module &m) {
  py::enum_<framework::AttrType>(m, "AttrType", "")
      .value("INT", AttrType::INT)
      .value("INTS", AttrType::INTS)
      .value("FLOAT", AttrType::FLOAT)
      .value("FLOATS", AttrType::FLOATS)
      .value("STRING", AttrType::STRING)
      .value("STRINGS", AttrType::STRINGS)
      .value("BOOL", AttrType::BOOLEAN)
      .value("BOOLS", AttrType::BOOLEANS)
      .value("BLOCK", AttrType::BLOCK);

  py::class_<framework::OpDescBind> op_desc(m, "OpDesc", "");
  op_desc.def("type", &framework::OpDescBind::Type)
      .def("set_type", &framework::OpDescBind::SetType)
      .def("input", &framework::OpDescBind::Input)
      .def("input_names", &framework::OpDescBind::InputNames)
      .def("set_input", &framework::OpDescBind::SetInput)
      .def("output", &framework::OpDescBind::Output)
      .def("output_names", &framework::OpDescBind::OutputNames)
      .def("set_output", &framework::OpDescBind::SetOutput)
      .def("__str__", &framework::OpDescBind::DebugString)
      .def("__repr__", &framework::OpDescBind::DebugString)
      .def("has_attr", &framework::OpDescBind::HasAttr)
      .def("attr_type", &framework::OpDescBind::GetAttrType)
      .def("attr_names", &framework::OpDescBind::AttrNames)
      .def("set_attr", &framework::OpDescBind::SetAttr)
      .def("attr", &framework::OpDescBind::GetAttr)
      .def("set_block_attr", &framework::OpDescBind::SetBlockAttr)
      .def("get_block_attr", &framework::OpDescBind::GetBlockAttr);
}

}  // namespace pybind
}  // namespace paddle
