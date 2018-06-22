/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/pybind/protobuf.h"

#include <deque>
#include <iostream>
#include <string>
#include <tuple>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/var_desc.h"

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

namespace pd = paddle::framework;

template <typename T>
static pybind11::bytes SerializeMessage(
    T &self) {  // NOLINT due to pybind11 convention.
  // Check IsInitialized in Python
  std::string retv;
  PADDLE_ENFORCE(self.Proto()->SerializePartialToString(&retv),
                 "Cannot serialize message");
  return retv;
}

// Bind Methods
void BindProgramDesc(pybind11::module *m) {
  pybind11::class_<pd::ProgramDesc>(*m, "ProgramDesc", "")
      .def(pybind11::init<>())
      .def("__init__",
           [](pd::ProgramDesc &self, const pd::ProgramDesc &other) {
             new (&self) pd::ProgramDesc(other);
           })
      .def("__init__",
           [](pd::ProgramDesc &self, const pybind11::bytes &binary_str) {
             std::string str(binary_str);
             new (&self) pd::ProgramDesc(str);
           })
      .def("append_block", &pd::ProgramDesc::AppendBlock,
           pybind11::return_value_policy::reference)
      .def("block", &pd::ProgramDesc::MutableBlock,
           pybind11::return_value_policy::reference)
      .def("num_blocks", &pd::ProgramDesc::Size)
      .def("flush", &pd::ProgramDesc::Flush)
      .def("get_feed_target_names", &pd::ProgramDesc::GetFeedTargetNames)
      .def("get_fetch_target_names", &pd::ProgramDesc::GetFetchTargetNames)
      .def("serialize_to_string", SerializeMessage<pd::ProgramDesc>)
      .def("parse_from_string",
           [](pd::ProgramDesc &program_desc, const std::string &data) {
             pd::proto::ProgramDesc *desc = program_desc.Proto();
             PADDLE_ENFORCE(desc->ParseFromString(data),
                            "Fail to parse ProgramDesc from string. This could "
                            "be a bug of Paddle.");
           });
}

void BindBlockDesc(pybind11::module *m) {
  pybind11::class_<pd::BlockDesc>(*m, "BlockDesc", "")
      .def_property_readonly("id", &pd::BlockDesc::ID)
      .def_property_readonly("parent", &pd::BlockDesc::Parent)
      .def("get_forward_block_idx", &pd::BlockDesc::ForwardBlockID)
      .def("set_forward_block_idx", &pd::BlockDesc::SetForwardBlockID)
      .def("append_op", &pd::BlockDesc::AppendOp,
           pybind11::return_value_policy::reference)
      .def("prepend_op", &pd::BlockDesc::PrependOp,
           pybind11::return_value_policy::reference)
      .def("insert_op", &pd::BlockDesc::InsertOp,
           pybind11::return_value_policy::reference)
      .def("remove_op", &pd::BlockDesc::RemoveOp)
      .def("var",
           [](pd::BlockDesc &self, pybind11::bytes byte_name) {
             std::string name = byte_name;
             return self.Var(name);
           },
           pybind11::return_value_policy::reference)
      .def("has_var",
           [](pd::BlockDesc &self, pybind11::bytes byte_name) {
             std::string name = byte_name;
             return self.HasVar(name);
           },
           pybind11::return_value_policy::reference)
      .def("rename_var",
           [](pd::BlockDesc &self, const pybind11::bytes &byte_name,
              const pybind11::bytes &byte_name_new) {
             std::string name = byte_name;
             std::string new_name = byte_name_new;
             self.RenameVar(name, new_name);
           })
      .def("has_var_recursive",
           [](pd::BlockDesc &self, pybind11::bytes byte_name) {
             std::string name = byte_name;
             return self.HasVarRecursive(name);
           })
      .def("find_var",
           [](pd::BlockDesc &self, pybind11::bytes byte_name) {
             std::string name = byte_name;
             return self.FindVar(name);
           },
           pybind11::return_value_policy::reference)
      .def("find_var_recursive",
           [](pd::BlockDesc &self, pybind11::bytes byte_name) {
             std::string name = byte_name;
             return self.FindVarRecursive(name);
           },
           pybind11::return_value_policy::reference)
      .def("remove_var",
           [](pd::BlockDesc &self, pybind11::bytes byte_name) {
             std::string name = byte_name;
             return self.RemoveVar(name);
           },
           pybind11::return_value_policy::reference)
      .def("all_vars", &pd::BlockDesc::AllVars,
           pybind11::return_value_policy::reference)
      .def("op_size", &pd::BlockDesc::OpSize)
      .def("op", &pd::BlockDesc::Op, pybind11::return_value_policy::reference)
      .def("serialize_to_string", SerializeMessage<pd::BlockDesc>);
}

void BindVarDsec(pybind11::module *m) {
  pybind11::class_<pd::VarDesc> var_desc(*m, "VarDesc", "");
  var_desc
      .def("name",
           [](pd::VarDesc &self) {
             pybind11::bytes name = self.Name();
             return name;
           },
           pybind11::return_value_policy::reference)
      .def("set_name", &pd::VarDesc::SetName)
      .def("set_shape", &pd::VarDesc::SetShape)
      .def("set_shapes", &pd::VarDesc::SetShapes)
      .def("set_dtype", &pd::VarDesc::SetDataType)
      .def("set_dtypes", &pd::VarDesc::SetDataTypes)
      .def("set_capacity", &pd::VarDesc::SetCapacity)
      .def("shape", &pd::VarDesc::GetShape,
           pybind11::return_value_policy::reference)
      .def("shapes", &pd::VarDesc::GetShapes,
           pybind11::return_value_policy::reference)
      .def("dtype", &pd::VarDesc::GetDataType,
           pybind11::return_value_policy::reference)
      .def("dtypes", &pd::VarDesc::GetDataTypes,
           pybind11::return_value_policy::reference)
      .def("lod_level", &pd::VarDesc::GetLoDLevel)
      .def("lod_levels", &pd::VarDesc::GetLoDLevels,
           pybind11::return_value_policy::reference)
      .def("set_lod_level", &pd::VarDesc::SetLoDLevel)
      .def("set_lod_levels", &pd::VarDesc::SetLoDLevels)
      .def("type", &pd::VarDesc::GetType)
      .def("set_type", &pd::VarDesc::SetType)
      .def("serialize_to_string", SerializeMessage<pd::VarDesc>)
      .def("persistable", &pd::VarDesc::Persistable)
      .def("set_persistable", &pd::VarDesc::SetPersistable);

  pybind11::enum_<pd::proto::VarType::Type>(var_desc, "VarType", "")
      .value("BOOL", pd::proto::VarType::BOOL)
      .value("UINT8", pd::proto::VarType::UINT8)
      .value("INT16", pd::proto::VarType::INT16)
      .value("INT32", pd::proto::VarType::INT32)
      .value("INT64", pd::proto::VarType::INT64)
      .value("FP16", pd::proto::VarType::FP16)
      .value("FP32", pd::proto::VarType::FP32)
      .value("FP64", pd::proto::VarType::FP64)
      .value("LOD_TENSOR", pd::proto::VarType::LOD_TENSOR)
      .value("SELECTED_ROWS", pd::proto::VarType::SELECTED_ROWS)
      .value("FEED_MINIBATCH", pd::proto::VarType::FEED_MINIBATCH)
      .value("FETCH_LIST", pd::proto::VarType::FETCH_LIST)
      .value("STEP_SCOPES", pd::proto::VarType::STEP_SCOPES)
      .value("LOD_RANK_TABLE", pd::proto::VarType::LOD_RANK_TABLE)
      .value("LOD_TENSOR_ARRAY", pd::proto::VarType::LOD_TENSOR_ARRAY)
      .value("CHANNEL", pd::proto::VarType::CHANNEL)
      .value("PLACE_LIST", pd::proto::VarType::PLACE_LIST)
      .value("READER", pd::proto::VarType::READER)
      .value("RAW", pd::proto::VarType::RAW);
}

void BindOpDesc(pybind11::module *m) {
  pybind11::enum_<pd::proto::AttrType>(*m, "AttrType", "")
      .value("INT", pd::proto::AttrType::INT)
      .value("INTS", pd::proto::AttrType::INTS)
      .value("FLOAT", pd::proto::AttrType::FLOAT)
      .value("FLOATS", pd::proto::AttrType::FLOATS)
      .value("STRING", pd::proto::AttrType::STRING)
      .value("STRINGS", pd::proto::AttrType::STRINGS)
      .value("BOOL", pd::proto::AttrType::BOOLEAN)
      .value("BOOLS", pd::proto::AttrType::BOOLEANS)
      .value("BLOCK", pd::proto::AttrType::BLOCK)
      .value("BLOCKS", pd::proto::AttrType::BLOCKS);

  pybind11::class_<pd::OpDesc> op_desc(*m, "OpDesc", "");
  op_desc
      .def("__init__", [](pd::OpDesc &self) { new (&self) pd::OpDesc(); },
           pybind11::return_value_policy::reference)
      .def("copy_from", &pd::OpDesc::CopyFrom)
      .def("type", &pd::OpDesc::Type)
      .def("set_type", &pd::OpDesc::SetType)
      .def("input", &pd::OpDesc::Input)
      .def("input_names", &pd::OpDesc::InputNames)
      .def("output", &pd::OpDesc::Output)
      .def("output_names", &pd::OpDesc::OutputNames)
      .def("set_input", &pd::OpDesc::SetInput)
      .def("set_output", &pd::OpDesc::SetOutput)
      .def("input_arg_names", &pd::OpDesc::InputArgumentNames)
      .def("output_arg_names", &pd::OpDesc::OutputArgumentNames)
      .def("rename_input", &pd::OpDesc::RenameInput)
      .def("rename_output", &pd::OpDesc::RenameOutput)
      .def("has_attr", &pd::OpDesc::HasAttr)
      .def("attr_type", &pd::OpDesc::GetAttrType)
      .def("attr_names", &pd::OpDesc::AttrNames)
      .def("set_attr", &pd::OpDesc::SetAttr)
      .def("attr", &pd::OpDesc::GetAttr)
      .def("set_block_attr", &pd::OpDesc::SetBlockAttr)
      .def("set_blocks_attr", &pd::OpDesc::SetBlocksAttr)
      .def("set_serialized_attr",
           [](pd::OpDesc &self, const std::string &name,
              const pybind11::bytes &seriralized) {
             std::string ser(seriralized);
             self.SetAttr(name, ser);
           })
      .def("block_attr", &pd::OpDesc::GetBlockAttr)
      .def("check_attrs", &pd::OpDesc::CheckAttrs)
      .def("infer_shape", &pd::OpDesc::InferShape)
      .def("infer_var_type", &pd::OpDesc::InferVarType)
      .def("set_is_target", &pd::OpDesc::SetIsTarget)
      .def("serialize_to_string", SerializeMessage<pd::OpDesc>)
      .def("block", &pd::OpDesc::Block,
           pybind11::return_value_policy::reference);
}

}  // namespace pybind
}  // namespace paddle
