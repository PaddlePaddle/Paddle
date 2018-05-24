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
#include "paddle/framework/backward.h"
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

template <typename T>
static py::bytes SerializeMessage(T &self) {
  // Check IsInitialized in Python
  std::string retv;
  PADDLE_ENFORCE(self.Proto()->SerializePartialToString(&retv),
                 "Cannot serialize message");
  return retv;
}

// Bind Methods
void BindProgramDesc(py::module &m) {
  py::class_<ProgramDesc>(m, "ProgramDesc", "")
      .def(py::init<>())
      .def("__init__",
           [](ProgramDesc &self, const ProgramDesc &other) {
             new (&self) ProgramDesc(other);
           })
      .def("__init__",
           [](ProgramDesc &self, const py::bytes &binary_str) {
             std::string str(binary_str);
             new (&self) ProgramDesc(str);
           })
      .def("append_block", &ProgramDesc::AppendBlock,
           py::return_value_policy::reference)
      .def("append_backward",
           [](ProgramDesc &program_desc, const VarDesc &target,
              const std::unordered_set<std::string> &no_grad_vars) {
             ParamGradInfoMap param_grad_map =
                 AppendBackward(program_desc, target, no_grad_vars);
             std::unordered_map<
                 std::string, std::tuple<std::string /* grad_var_name */,
                                         int /* block_idx */, int /* op_idx */>>
                 retv;
             for (auto it = param_grad_map.begin(); it != param_grad_map.end();
                  ++it) {
               const auto &grad_info = it->second;
               retv[it->first] = std::make_tuple(
                   grad_info.name_, grad_info.block_idx_, grad_info.op_idx_);
             }
             return retv;
           })
      .def("block", &ProgramDesc::MutableBlock,
           py::return_value_policy::reference)
      .def("num_blocks", &ProgramDesc::Size)
      .def("serialize_to_string", SerializeMessage<ProgramDesc>)
      .def("parse_from_string",
           [](ProgramDesc &program_desc, const std::string &data) {
             proto::ProgramDesc *desc = program_desc.Proto();
             PADDLE_ENFORCE(desc->ParseFromString(data),
                            "Fail to parse ProgramDesc from string. This could "
                            "be a bug of Paddle.");
           });
}

void BindBlockDesc(py::module &m) {
  py::class_<BlockDesc>(m, "BlockDesc", "")
      .def_property_readonly("id", &BlockDesc::ID)
      .def_property_readonly("parent", &BlockDesc::Parent)
      .def("append_op", &BlockDesc::AppendOp,
           py::return_value_policy::reference)
      .def("prepend_op", &BlockDesc::PrependOp,
           py::return_value_policy::reference)
      .def("remove_op", &BlockDesc::RemoveOp)
      .def("var",
           [](BlockDesc &self, py::bytes byte_name) {
             std::string name = byte_name;
             return self.Var(name);
           },
           py::return_value_policy::reference)
      .def("has_var",
           [](BlockDesc &self, py::bytes byte_name) {
             std::string name = byte_name;
             return self.HasVar(name);
           })
      .def("find_var",
           [](BlockDesc &self, py::bytes byte_name) {
             std::string name = byte_name;
             return self.FindVar(name);
           },
           py::return_value_policy::reference)
      .def("all_vars", &BlockDesc::AllVars, py::return_value_policy::reference)
      .def("op_size", &BlockDesc::OpSize)
      .def("op", &BlockDesc::Op, py::return_value_policy::reference)
      .def("serialize_to_string", SerializeMessage<BlockDesc>);
}

void BindVarDsec(py::module &m) {
  py::enum_<proto::DataType>(m, "DataType", "")
      .value("BOOL", proto::DataType::BOOL)
      .value("INT16", proto::DataType::INT16)
      .value("INT32", proto::DataType::INT32)
      .value("INT64", proto::DataType::INT64)
      .value("FP16", proto::DataType::FP16)
      .value("FP32", proto::DataType::FP32)
      .value("FP64", proto::DataType::FP64);

  py::class_<VarDesc> var_desc(m, "VarDesc", "");
  var_desc
      .def("name",
           [](const VarDesc &self) {
             py::bytes name = self.Name();
             return name;
           },
           py::return_value_policy::reference)
      .def("set_shape", &VarDesc::SetShape)
      .def("set_dtype", &VarDesc::SetDataType)
      .def("shape", &VarDesc::Shape, py::return_value_policy::reference)
      .def("dtype", &VarDesc::GetDataType)
      .def("lod_level", &VarDesc::GetLodLevel)
      .def("set_lod_level", &VarDesc::SetLoDLevel)
      .def("type", &VarDesc::GetType)
      .def("set_type", &VarDesc::SetType)
      .def("serialize_to_string", SerializeMessage<VarDesc>)
      .def("persistable", &VarDesc::Persistable)
      .def("set_persistable", &VarDesc::SetPersistable);

  py::enum_<proto::VarDesc::VarType>(var_desc, "VarType", "")
      .value("LOD_TENSOR", proto::VarDesc::LOD_TENSOR)
      .value("SELECTED_ROWS", proto::VarDesc::SELECTED_ROWS)
      .value("FEED_MINIBATCH", proto::VarDesc::FEED_MINIBATCH)
      .value("FETCH_LIST", proto::VarDesc::FETCH_LIST)
      .value("STEP_SCOPES", proto::VarDesc::STEP_SCOPES)
      .value("LOD_RANK_TABLE", proto::VarDesc::LOD_RANK_TABLE)
      .value("LOD_TENSOR_ARRAY", proto::VarDesc::LOD_TENSOR_ARRAY);
}

void BindOpDesc(py::module &m) {
  py::enum_<proto::AttrType>(m, "AttrType", "")
      .value("INT", proto::AttrType::INT)
      .value("INTS", proto::AttrType::INTS)
      .value("FLOAT", proto::AttrType::FLOAT)
      .value("FLOATS", proto::AttrType::FLOATS)
      .value("STRING", proto::AttrType::STRING)
      .value("STRINGS", proto::AttrType::STRINGS)
      .value("BOOL", proto::AttrType::BOOLEAN)
      .value("BOOLS", proto::AttrType::BOOLEANS)
      .value("BLOCK", proto::AttrType::BLOCK);

  py::class_<OpDesc> op_desc(m, "OpDesc", "");
  op_desc.def("type", &OpDesc::Type)
      .def("set_type", &OpDesc::SetType)
      .def("input", &OpDesc::Input)
      .def("input_names", &OpDesc::InputNames)
      .def("set_input", &OpDesc::SetInput)
      .def("output", &OpDesc::Output)
      .def("output_names", &OpDesc::OutputNames)
      .def("set_output", &OpDesc::SetOutput)
      .def("has_attr", &OpDesc::HasAttr)
      .def("attr_type", &OpDesc::GetAttrType)
      .def("attr_names", &OpDesc::AttrNames)
      .def("set_attr", &OpDesc::SetAttr)
      .def("attr", &OpDesc::GetAttr)
      .def("set_block_attr", &OpDesc::SetBlockAttr)
      .def("set_serialized_attr",
           [](OpDesc &self, const std::string &name,
              const py::bytes &seriralized) {
             std::string ser(seriralized);
             self.SetAttr(name, ser);
           })
      .def("block_attr", &OpDesc::GetBlockAttr)
      .def("check_attrs", &OpDesc::CheckAttrs)
      .def("infer_shape", &OpDesc::InferShape)
      .def("infer_var_type", &OpDesc::InferVarType)
      .def("serialize_to_string", SerializeMessage<OpDesc>);
}

}  // namespace pybind
}  // namespace paddle
