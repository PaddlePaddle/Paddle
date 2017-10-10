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

using namespace paddle::framework;  // NOLINT

// Bind Methods
void BindProgramDesc(py::module &m) {
  py::class_<ProgramDescBind>(m, "ProgramDesc", "")
      .def_static("instance",
                  []() -> ProgramDescBind * {
                    return &ProgramDescBind::Instance(&GetProgramDesc());
                  },
                  py::return_value_policy::reference)
      .def_static("__create_program_desc__",
                  []() -> ProgramDescBind * {
                    // Only used for unit-test
                    auto *prog_desc = new ProgramDesc;
                    auto *block = prog_desc->mutable_blocks()->Add();
                    block->set_idx(0);
                    block->set_parent_idx(-1);
                    return &ProgramDescBind::Instance(prog_desc);
                  },
                  py::return_value_policy::reference)
      .def("append_block", &ProgramDescBind::AppendBlock,
           py::return_value_policy::reference)
      .def("block", &ProgramDescBind::Block, py::return_value_policy::reference)
      .def("num_blocks", &ProgramDescBind::Size);
}

void BindBlockDesc(py::module &m) {
  py::class_<BlockDescBind>(m, "BlockDesc", "")
      .def_property_readonly("id", &BlockDescBind::ID)
      .def_property_readonly("parent", &BlockDescBind::Parent)
      .def("append_op", &BlockDescBind::AppendOp,
           py::return_value_policy::reference)
      .def("prepend_op", &BlockDescBind::PrependOp,
           py::return_value_policy::reference)
      .def("new_var",
           [](BlockDescBind &self, py::bytes byte_name) {
             std::string name = byte_name;
             return self.NewVar(name);
           },
           py::return_value_policy::reference)
      .def("var",
           [](BlockDescBind &self, py::bytes byte_name) {
             std::string name = byte_name;
             return self.Var(name);
           },
           py::return_value_policy::reference)
      .def("all_vars", &BlockDescBind::AllVars,
           py::return_value_policy::reference)
      .def("all_ops", &BlockDescBind::AllOps,
           py::return_value_policy::reference);
}

void BindVarDsec(py::module &m) {
  py::enum_<DataType>(m, "DataType", "")
      .value("BOOL", DataType::BOOL)
      .value("INT16", DataType::INT16)
      .value("INT32", DataType::INT32)
      .value("INT64", DataType::INT64)
      .value("FP16", DataType::FP16)
      .value("FP32", DataType::FP32)
      .value("FP64", DataType::FP64);

  py::class_<VarDescBind>(m, "VarDesc", "")
      .def("name",
           [](const VarDescBind &self) {
             py::bytes name = self.Name();
             return name;
           },
           py::return_value_policy::reference)
      .def("set_shape", &VarDescBind::SetShape)
      .def("set_data_type", &VarDescBind::SetDataType)
      .def("shape", &VarDescBind::Shape, py::return_value_policy::reference)
      .def("data_type", &VarDescBind::GetDataType);
}

void BindOpDesc(py::module &m) {
  py::enum_<AttrType>(m, "AttrType", "")
      .value("INT", AttrType::INT)
      .value("INTS", AttrType::INTS)
      .value("FLOAT", AttrType::FLOAT)
      .value("FLOATS", AttrType::FLOATS)
      .value("STRING", AttrType::STRING)
      .value("STRINGS", AttrType::STRINGS)
      .value("BOOL", AttrType::BOOLEAN)
      .value("BOOLS", AttrType::BOOLEANS)
      .value("BLOCK", AttrType::BLOCK);

  py::class_<OpDescBind> op_desc(m, "OpDesc", "");
  op_desc.def("type", &OpDescBind::Type)
      .def("set_type", &OpDescBind::SetType)
      .def("input", &OpDescBind::Input)
      .def("input_names", &OpDescBind::InputNames)
      .def("set_input", &OpDescBind::SetInput)
      .def("output", &OpDescBind::Output)
      .def("output_names", &OpDescBind::OutputNames)
      .def("set_output", &OpDescBind::SetOutput)
      .def("has_attr", &OpDescBind::HasAttr)
      .def("attr_type", &OpDescBind::GetAttrType)
      .def("attr_names", &OpDescBind::AttrNames)
      .def("set_attr", &OpDescBind::SetAttr)
      .def("attr", &OpDescBind::GetAttr)
      .def("set_block_attr", &OpDescBind::SetBlockAttr)
      .def("get_block_attr", &OpDescBind::GetBlockAttr);
}

}  // namespace pybind
}  // namespace paddle
