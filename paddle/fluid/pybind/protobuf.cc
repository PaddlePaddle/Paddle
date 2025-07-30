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

#include <complex>
#include <deque>
#include <iostream>
#include <string>
#include <tuple>

#include "paddle/fluid/distributed/auto_parallel/dist_attr.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_version_proto.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/program_converter.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/framework/version.h"
#include "paddle/fluid/jit/property.h"
#include "paddle/fluid/pybind/pybind_variant_caster.h"
#include "paddle/phi/common/scalar.h"

namespace py = pybind11;

namespace paddle::pybind {

PyTypeObject *g_vartype_pytype = nullptr;
PyTypeObject *g_blockdesc_pytype = nullptr;

namespace pd = paddle::framework;
namespace jit = paddle::jit;

template <typename T>
static pybind11::bytes SerializeMessage(
    T &self) {  // NOLINT due to pybind11 convention.
  // Check IsInitialized in Python
  std::string retv;
  PADDLE_ENFORCE_EQ(self.Proto()->SerializePartialToString(&retv),
                    true,
                    common::errors::InvalidArgument(
                        "Failed to serialize input Desc to string."));
  return retv;
}

pybind11::bytes SerializeProgramDesc(
    pd::ProgramDesc &self,  // NOLINT due to pybind11 convention.
    bool legacy_format = false) {
  // Check IsInitialized in Python
  std::string retv;
  pd::ProgramDesc copy = self;
  framework::compatible::pb::OpVersionMap op_version_map(copy.OpVersionMap());

  if (legacy_format) {
    pd::no_scalar::ConvertProgram(&copy);
    copy.SetVersion(pd::kLegacyProgramVersion);
    paddle::framework::compatible::SaveOpVersions(
        paddle::framework::compatible::pb::GetLegacyOpVersions(),
        &op_version_map);
  } else {
    copy.SetVersion(pd::kCurProgramVersion);
    paddle::framework::compatible::SaveOpVersions(
        framework::compatible::get_op_version_map(), &op_version_map);
  }

  PADDLE_ENFORCE_EQ(copy.Proto()->SerializePartialToString(&retv),
                    true,
                    common::errors::InvalidArgument(
                        "Failed to serialize input Desc to string."));
  return retv;
}

template <typename T>
static void DeserializeMessage(T *self, const std::string &str) {
  PADDLE_ENFORCE_EQ(
      self->Proto()->ParsePartialFromString(str),
      true,
      common::errors::InvalidArgument("Failed to parse pb from string"));
  return;
}

// Bind Methods
void BindProgramDesc(pybind11::module *m) {
  pybind11::class_<pd::ProgramDesc, std::shared_ptr<pd::ProgramDesc>>(
      *m, "ProgramDesc", "")
      .def(pybind11::init<>())
      .def(py::init([](const pd::ProgramDesc &other) {
        return std::make_unique<pd::ProgramDesc>(other);
      }))
      .def(py::init([](const pybind11::bytes &binary_str) {
        std::string str(binary_str);
        return std::make_unique<pd::ProgramDesc>(str);
      }))
      .def("append_block",
           &pd::ProgramDesc::AppendBlock,
           pybind11::return_value_policy::reference)
      .def("block",
           &pd::ProgramDesc::MutableBlock,
           pybind11::return_value_policy::reference)
      .def("num_blocks", &pd::ProgramDesc::Size)
      .def("flush", &pd::ProgramDesc::Flush)
      .def("get_feed_target_names", &pd::ProgramDesc::GetFeedTargetNames)
      .def("get_fetch_target_names", &pd::ProgramDesc::GetFetchTargetNames)
      .def("serialize_to_string",
           &SerializeProgramDesc,
           pybind11::arg("legacy_format") = false)
      .def("need_update", &pd::ProgramDesc::NeedUpdate)
      .def("parse_from_string",
           [](pd::ProgramDesc &program_desc, const std::string &data) {
             pd::proto::ProgramDesc *desc = program_desc.Proto();
             PADDLE_ENFORCE_EQ(
                 desc->ParseFromString(data),
                 true,
                 common::errors::InvalidArgument(
                     "Failed to parse ProgramDesc from binary string."));
           })
      .def(
          "_set_version",
          [](pd::ProgramDesc &self, int64_t version) {
            return self.SetVersion(version);
          },
          pybind11::arg("version") = pd::kCurProgramVersion)
      .def("_version",
           [](pd::ProgramDesc &self) -> int64_t { return self.Version(); })
      .def("get_op_deps",
           [](const framework::ProgramDesc &program) {
             return framework::ir::GetOpDependencies(program);
           })
      .def("need_update", &pd::ProgramDesc::NeedUpdate)
      .def("cached_hash_str", [](pd::ProgramDesc &self) {
        return self.CachedHashString();
        // return pybind11::bytes(self.CachedHashString());
      });
}

void BindBlockDesc(pybind11::module *m) {
  pybind11::class_<pd::BlockDesc> blockdesc(*m, "BlockDesc", "");
  g_blockdesc_pytype = (PyTypeObject *)blockdesc.ptr();  // NOLINT
  blockdesc.def_property_readonly("id", &pd::BlockDesc::ID)
      .def_property_readonly("parent", &pd::BlockDesc::Parent)
      .def("get_forward_block_idx", &pd::BlockDesc::ForwardBlockID)
      .def("_set_forward_block_idx", &pd::BlockDesc::SetForwardBlockID)
      .def("append_op",
           &pd::BlockDesc::AppendOp,
           pybind11::return_value_policy::reference)
      .def("_prepend_op",
           &pd::BlockDesc::PrependOp,
           pybind11::return_value_policy::reference)
      .def("_insert_op",
           &pd::BlockDesc::InsertOp,
           pybind11::return_value_policy::reference)
      .def("_remove_op", &pd::BlockDesc::RemoveOp)
      .def(
          "var",
          [](pd::BlockDesc &self, pybind11::bytes byte_name) {
            std::string name = byte_name;
            return self.Var(name);
          },
          pybind11::return_value_policy::reference)
      .def(
          "has_var",
          [](pd::BlockDesc &self, pybind11::bytes byte_name) {
            std::string name = byte_name;
            return self.HasVar(name);
          },
          pybind11::return_value_policy::reference)
      .def("_rename_var",
           [](pd::BlockDesc &self,
              const pybind11::bytes &byte_name,
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
      .def("set_parent_idx", &pd::BlockDesc::SetParent)
      .def(
          "find_var",
          [](pd::BlockDesc &self, pybind11::bytes byte_name) {
            std::string name = byte_name;
            return self.FindVar(name);
          },
          pybind11::return_value_policy::reference)
      .def(
          "find_var_recursive",
          [](pd::BlockDesc &self, pybind11::bytes byte_name) {
            std::string name = byte_name;
            return self.FindVarRecursive(name);
          },
          pybind11::return_value_policy::reference)
      .def(
          "_remove_var",
          [](pd::BlockDesc &self, pybind11::bytes byte_name) {
            std::string name = byte_name;
            return self.RemoveVar(name);
          },
          pybind11::return_value_policy::reference)
      .def("all_vars",
           &pd::BlockDesc::AllVars,
           pybind11::return_value_policy::reference)
      .def("op_size", &pd::BlockDesc::OpSize)
      .def("op", &pd::BlockDesc::Op, pybind11::return_value_policy::reference)
      .def("serialize_to_string", SerializeMessage<pd::BlockDesc>)
      .def("_move_from", &pd::BlockDesc::MoveFrom);
}

void BindVarDesc(pybind11::module *m) {
  pybind11::class_<pd::VarDesc> var_desc(*m, "VarDesc", "");
  var_desc.def(pybind11::init<const std::string &>())
      .def("name", &pd::VarDesc::Name, pybind11::return_value_policy::reference)
      .def("set_name", &pd::VarDesc::SetName)
      .def("set_shape", &pd::VarDesc::SetShape)
      .def("set_shapes", &pd::VarDesc::SetShapes)
      .def("get_shape", &pd::VarDesc::GetShape)
      .def("set_dtype", &pd::VarDesc::SetDataType)
      .def("set_dtypes", &pd::VarDesc::SetDataTypes)
      .def("shape",
           &pd::VarDesc::GetShape,
           pybind11::return_value_policy::reference)
      .def("shapes",
           &pd::VarDesc::GetShapes,
           pybind11::return_value_policy::reference)
      .def("dtype",
           &pd::VarDesc::GetDataType,
           pybind11::return_value_policy::reference)
      .def("element_size",
           &pd::VarDesc::ElementSize,
           pybind11::return_value_policy::reference)
      .def("dtypes",
           &pd::VarDesc::GetDataTypes,
           pybind11::return_value_policy::reference)
      .def("lod_level", &pd::VarDesc::GetLoDLevel)
      .def("lod_levels",
           &pd::VarDesc::GetLoDLevels,
           pybind11::return_value_policy::reference)
      .def("set_lod_level", &pd::VarDesc::SetLoDLevel)
      .def("set_lod_levels", &pd::VarDesc::SetLoDLevels)
      .def("legacy_lod_level", &pd::VarDesc::GetLegacyLoDLevel)
      .def("legacy_lod_levels",
           &pd::VarDesc::GetLegacyLoDLevels,
           pybind11::return_value_policy::reference)
      .def("set_legacy_lod_level", &pd::VarDesc::SetLegacyLoDLevel)
      .def("set_legacy_lod_levels", &pd::VarDesc::SetLegacyLoDLevels)
      .def("type", &pd::VarDesc::GetType)
      .def("set_type", &pd::VarDesc::SetType)
      .def("serialize_to_string", SerializeMessage<pd::VarDesc>)
      .def("persistable", &pd::VarDesc::Persistable)
      .def("set_persistable", &pd::VarDesc::SetPersistable)
      .def("is_parameter", &pd::VarDesc::IsParameter)
      .def("set_is_parameter", &pd::VarDesc::SetIsParameter)
      .def("clear_is_parameter", &pd::VarDesc::ClearIsParameter)
      .def("has_is_parameter", &pd::VarDesc::HasIsParameter)
      .def("stop_gradient", &pd::VarDesc::StopGradient)
      .def("set_stop_gradient", &pd::VarDesc::SetStopGradient)
      .def("clear_stop_gradient", &pd::VarDesc::ClearStopGradient)
      .def("has_stop_gradient", &pd::VarDesc::HasStopGradient)
      .def("need_check_feed", &pd::VarDesc::NeedCheckFeed)
      .def("set_need_check_feed", &pd::VarDesc::SetNeedCheckFeed)
      .def("has_attr", &pd::VarDesc::HasAttr)
      .def("attr_names", &pd::VarDesc::AttrNames)
      .def("_set_attr", &pd::VarDesc::SetAttr)
      .def("remove_attr", &pd::VarDesc::RemoveAttr)
      .def("id", &pd::VarDesc::Id)
      .def("original_id", &pd::VarDesc::OriginalId)
      .def("set_original_id", &pd::VarDesc::SetOriginalId)
      .def_property("dist_attr",
                    &pd::VarDesc::MutableDistAttr,
                    &pd::VarDesc::SetDistAttr,
                    pybind11::return_value_policy::reference)
      .def("attr", &pd::VarDesc::GetAttr);

  pybind11::enum_<pd::proto::VarType::Type> vartype(var_desc, "VarType", "");
  g_vartype_pytype = (PyTypeObject *)vartype.ptr();  // NOLINT
  vartype.value("BOOL", pd::proto::VarType::BOOL)
      .value("UINT8", pd::proto::VarType::UINT8)
      .value("INT8", pd::proto::VarType::INT8)
      .value("INT16", pd::proto::VarType::INT16)
      .value("INT32", pd::proto::VarType::INT32)
      .value("INT64", pd::proto::VarType::INT64)
      .value("FP16", pd::proto::VarType::FP16)
      .value("FP32", pd::proto::VarType::FP32)
      .value("FP64", pd::proto::VarType::FP64)
      .value("BF16", pd::proto::VarType::BF16)
      .value("COMPLEX64", pd::proto::VarType::COMPLEX64)
      .value("COMPLEX128", pd::proto::VarType::COMPLEX128)
      .value("FP8_E4M3FN", pd::proto::VarType::FP8_E4M3FN)
      .value("FP8_E5M2", pd::proto::VarType::FP8_E5M2)
      .value("DENSE_TENSOR", pd::proto::VarType::DENSE_TENSOR)
      .value("SELECTED_ROWS", pd::proto::VarType::SELECTED_ROWS)
      .value("FEED_MINIBATCH", pd::proto::VarType::FEED_MINIBATCH)
      .value("FETCH_LIST", pd::proto::VarType::FETCH_LIST)
      .value("STEP_SCOPES", pd::proto::VarType::STEP_SCOPES)
      .value("DENSE_TENSOR_ARRAY", pd::proto::VarType::DENSE_TENSOR_ARRAY)
      .value("PLACE_LIST", pd::proto::VarType::PLACE_LIST)
      .value("READER", pd::proto::VarType::READER)
      .value("RAW", pd::proto::VarType::RAW)
      .value("STRING", pd::proto::VarType::STRING)
      .value("STRINGS", pd::proto::VarType::STRINGS)
      .value("VOCAB", pd::proto::VarType::VOCAB)
      .value("SPARSE_COO", pd::proto::VarType::SPARSE_COO);
}

void BindOpDesc(pybind11::module *m) {
  pybind11::enum_<pd::proto::AttrType>(*m, "AttrType", "")
      .value("INT", pd::proto::AttrType::INT)
      .value("INTS", pd::proto::AttrType::INTS)
      .value("LONG", pd::proto::AttrType::LONG)
      .value("LONGS", pd::proto::AttrType::LONGS)
      .value("FLOAT", pd::proto::AttrType::FLOAT)
      .value("FLOATS", pd::proto::AttrType::FLOATS)
      .value("FLOAT64", pd::proto::AttrType::FLOAT64)
      .value("FLOAT64S", pd::proto::AttrType::FLOAT64S)
      .value("STRING", pd::proto::AttrType::STRING)
      .value("STRINGS", pd::proto::AttrType::STRINGS)
      .value("BOOL", pd::proto::AttrType::BOOLEAN)
      .value("BOOLS", pd::proto::AttrType::BOOLEANS)
      .value("BLOCK", pd::proto::AttrType::BLOCK)
      .value("BLOCKS", pd::proto::AttrType::BLOCKS)
      .value("VAR", pd::proto::AttrType::VAR)
      .value("VARS", pd::proto::AttrType::VARS)
      .value("SCALAR", pd::proto::AttrType::SCALAR)
      .value("SCALARS", pd::proto::AttrType::SCALARS);

  pybind11::class_<pd::OpDesc> op_desc(*m, "OpDesc", "");
  op_desc
      .def(py::init([]() { return std::make_unique<pd::OpDesc>(); }),
           pybind11::return_value_policy::reference)
      .def("copy_from", &pd::OpDesc::CopyFrom)
      .def("type", &pd::OpDesc::Type)
      .def("set_type", &pd::OpDesc::SetType)
      .def("input",
           [](pd::OpDesc &self, const std::string &name) {
             return self.Input(name);
           })
      .def(
          "input_names",
          [](pd::OpDesc &self, bool with_attr_var) {
            return self.InputNames(with_attr_var);
          },
          py::arg("with_attr_var") = false)
      .def("output", &pd::OpDesc::Output)
      .def("output_names", &pd::OpDesc::OutputNames)
      .def("set_input",
           [](pd::OpDesc &self,
              const std::string &name,
              const std::vector<std::string> &vec_var_name) {
             self.SetInput(name, vec_var_name);
           })
      .def("set_output",
           [](pd::OpDesc &self,
              const std::string &name,
              const std::vector<std::string> &vec_var_name) {
             self.SetOutput(name, vec_var_name);
           })
      .def("remove_output", &pd::OpDesc::RemoveOutput)
      .def("remove_input", &pd::OpDesc::RemoveInput)
      .def(
          "input_arg_names",
          [](pd::OpDesc &self, bool with_attr_var) {
            return self.InputArgumentNames(with_attr_var);
          },
          py::arg("with_attr_var") = false)
      .def("output_arg_names", &pd::OpDesc::OutputArgumentNames)
      .def("_rename_input", &pd::OpDesc::RenameInput)
      .def("_rename_output", &pd::OpDesc::RenameOutput)
      .def(
          "has_attr",
          [](pd::OpDesc &self, const std::string &name, bool with_attr_var) {
            return self.HasAttr(name, with_attr_var);
          },
          py::arg("name"),
          py::arg("with_attr_var") = false)
      .def(
          "attr_type",
          [](pd::OpDesc &self, const std::string &name, bool with_attr_var) {
            return self.GetAttrType(name, with_attr_var);
          },
          py::arg("name"),
          py::arg("with_attr_var") = false)
      .def(
          "attr_names",
          [](pd::OpDesc &self, bool with_attr_var) {
            return self.AttrNames(with_attr_var);
          },
          py::arg("with_attr_var") = false)
      .def("_set_attr", &pd::OpDesc::SetAttr)
      .def("remove_attr", &pd::OpDesc::RemoveAttr)
      .def("_set_bool_attr", &pd::OpDesc::SetPlainAttr<bool>)
      .def("_set_int32_attr", &pd::OpDesc::SetPlainAttr<int>)
      .def("_set_int64_attr", &pd::OpDesc::SetPlainAttr<int64_t>)
      .def("_set_float32_attr", &pd::OpDesc::SetPlainAttr<float>)
      .def("_set_float64_attr", &pd::OpDesc::SetPlainAttr<double>)
      .def("_set_str_attr", &pd::OpDesc::SetPlainAttr<std::string>)

      .def("_set_bools_attr", &pd::OpDesc::SetPlainAttr<std::vector<bool>>)
      .def("_set_int32s_attr", &pd::OpDesc::SetPlainAttr<std::vector<int>>)
      .def("_set_int64s_attr", &pd::OpDesc::SetPlainAttr<std::vector<int64_t>>)
      .def("_set_float32s_attr", &pd::OpDesc::SetPlainAttr<std::vector<float>>)
      .def("_set_float64s_attr", &pd::OpDesc::SetPlainAttr<std::vector<double>>)
      .def("_set_strs_attr",
           &pd::OpDesc::SetPlainAttr<std::vector<std::string>>)

      .def("_set_scalar_attr",
           &pd::OpDesc::SetPlainAttr<paddle::experimental::Scalar>)
      .def("_set_scalars_attr",
           &pd::OpDesc::SetPlainAttr<std::vector<paddle::experimental::Scalar>>)

      .def(
          "attr",
          [](pd::OpDesc &self, const std::string &name, bool with_attr_var) {
            return self.GetAttr(name, with_attr_var);
          },
          py::arg("name"),
          py::arg("with_attr_var") = false)
      .def("set_var_attr", &pd::OpDesc::SetVarAttr)
      .def("set_vars_attr", &pd::OpDesc::SetVarsAttr)
      .def("set_block_attr", &pd::OpDesc::SetBlockAttr)
      .def("set_blocks_attr", &pd::OpDesc::SetBlocksAttr)
      .def("set_serialized_attr",
           [](pd::OpDesc &self,
              const std::string &name,
              const pybind11::bytes &serialized) {
             std::string ser(serialized);
             self.SetAttr(name, ser);
           })
      .def("_block_attr_id", &pd::OpDesc::GetBlockAttrId)
      .def("_blocks_attr_ids", &pd::OpDesc::GetBlocksAttrIds)
      .def("check_attrs", &pd::OpDesc::CheckAttrs)
      .def("infer_shape", &pd::OpDesc::InferShape)
      .def("infer_var_type", &pd::OpDesc::InferVarType)
      .def("set_is_target", &pd::OpDesc::SetIsTarget)
      .def("serialize_to_string", SerializeMessage<pd::OpDesc>)
      .def(
          "block",
          [](pd::OpDesc &self) { return self.Block(); },
          pybind11::return_value_policy::reference)
      .def("id", &pd::OpDesc::Id)
      .def("original_id", &pd::OpDesc::OriginalId)
      .def("set_original_id", &pd::OpDesc::SetOriginalId)
      .def_property("dist_attr",
                    &pd::OpDesc::MutableDistAttr,
                    &pd::OpDesc::SetDistAttr,
                    pybind11::return_value_policy::reference)
      .def("inputs", [](pd::OpDesc &self) { return self.Inputs(); })
      .def("outputs", &pd::OpDesc::Outputs)
      .def("get_attr_map", &pd::OpDesc::GetAttrMap);

  pybind11::class_<paddle::experimental::Scalar> scalar(*m, "Scalar", "");
  scalar.def(py::init<bool>())
      .def(py::init<double>())
      .def(py::init<int64_t>())
      .def(py::init<std::complex<double>>())
      .def(py::init<paddle::experimental::Scalar>())
      .def("__str__", &paddle::experimental::Scalar::ToString)
      .def("__repr__", &paddle::experimental::Scalar::ToString)
      .def("__eq__", &paddle::experimental::Scalar::operator==<paddle::Tensor>)
      .def("value",
           [](const paddle::experimental::Scalar &self)
               -> paddle::variant<bool, int64_t, double, std::complex<double>> {
             auto dtype = self.dtype();
             switch (dtype) {
               case phi::DataType::FLOAT64:
               case phi::DataType::FLOAT32:
                 return self.to<double>();
               case phi::DataType::INT32:
               case phi::DataType::INT64:
                 return self.to<int64_t>();
               case phi::DataType::BOOL:
                 return self.to<bool>();
               case phi::DataType::COMPLEX64:
               case phi::DataType::COMPLEX128:
                 // to paddle's complex to avoid ambiguous
                 // when converting bfloat16 or float16 to std::complex<double>
                 return static_cast<std::complex<double>>(
                     self.to<phi::dtype::complex<double>>());
               default:
                 PD_THROW("Invalid tensor data type `", dtype, "`.");
             }
           });
  pybind11::implicitly_convertible<bool, paddle::experimental::Scalar>();
  pybind11::implicitly_convertible<double, paddle::experimental::Scalar>();
  pybind11::implicitly_convertible<int64_t, paddle::experimental::Scalar>();
  pybind11::implicitly_convertible<std::complex<double>,
                                   paddle::experimental::Scalar>();
}

// Serialize Class Property
void BindJitProperty(pybind11::module *m) {
  pybind11::class_<jit::Property> property(*m, "Property");
  property
      .def(py::init([]() { return std::make_unique<jit::Property>(); }),
           pybind11::return_value_policy::reference)
      .def("size", &jit::Property::Size)
      .def("set_float",
           py::overload_cast<const std::string &, const float &>(
               &jit::Property::SetFloat),
           "set float",
           py::arg("name"),
           py::arg("var"))
      .def("get_float",
           py::overload_cast<const int &>(&jit::Property::GetFloat, py::const_))
      .def("get_float",
           py::overload_cast<const std::string &>(&jit::Property::GetFloat,
                                                  py::const_))
      .def("set_floats",
           py::overload_cast<const std::string &, const std::vector<float> &>(
               &jit::Property::SetFloats),
           "set list of float",
           py::arg("name"),
           py::arg("val"))
      .def("set_int",
           py::overload_cast<const std::string &, const int64_t &>(
               &jit::Property::SetInt64),
           "set int",
           py::arg("name"),
           py::arg("val"))
      .def("set_ints",
           py::overload_cast<const std::string &, const std::vector<int64_t> &>(
               &jit::Property::SetInt64s),
           "set list of int",
           py::arg("name"),
           py::arg("val"))
      .def("set_string",
           py::overload_cast<const std::string &, const std::string &>(
               &jit::Property::SetString),
           "set string",
           py::arg("name"),
           py::arg("val"))
      .def("set_strings",
           py::overload_cast<const std::string &,
                             const std::vector<std::string> &>(
               &jit::Property::SetStrings),
           "set list of string",
           py::arg("name"),
           py::arg("val"))
      .def("serialize_to_string", SerializeMessage<jit::Property>)
      .def("parse_from_string", DeserializeMessage<jit::Property>);
}

}  // namespace paddle::pybind
