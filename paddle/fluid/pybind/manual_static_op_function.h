// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/framework/custom_operator_utils.h"
#include "paddle/fluid/framework/new_executor/instruction/custom_kernel_instruction.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/fluid/pybind/op_callstack_utils.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/builtin_op.h"

namespace paddle {

namespace pybind {
static PyObject *static_api_parameter(PyObject *self,
                                      PyObject *args,
                                      PyObject *kwargs) {
  try {
    VLOG(6) << "Add parameter op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Parse Attributes
    PyObject *name_obj = PyTuple_GET_ITEM(args, 0);
    std::string name = CastPyArg2String(name_obj, "name", 0);
    // Call ir static api
    CallStackRecorder callstack_recoder("parameter");
    callstack_recoder.record();
    auto static_api_out = paddle::dialect::parameter(name);
    callstack_recoder.attach_to_ops();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *static_api_set_parameter(PyObject *self,
                                          PyObject *args,
                                          PyObject *kwargs) {
  try {
    VLOG(6) << "Add set_parameter op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *parameter_obj = PyTuple_GET_ITEM(args, 0);
    auto parameter = CastPyArg2Value(parameter_obj, "parameter", 0);

    // Parse Attributes
    PyObject *name_obj = PyTuple_GET_ITEM(args, 1);
    std::string name = CastPyArg2String(name_obj, "name", 1);
    // Call ir static api
    CallStackRecorder callstack_recoder("set_parameter");
    callstack_recoder.record();
    paddle::dialect::set_parameter(parameter, name);
    callstack_recoder.attach_to_ops();
    Py_RETURN_NONE;
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *static_api_set_persistable_value(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add shadow_output op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get OpResult from args
    PyObject *persist_value_obj = PyTuple_GET_ITEM(args, 0);
    auto persist_value = CastPyArg2Value(persist_value_obj, "persist_value", 0);

    // Parse Attributes
    PyObject *name_obj = PyTuple_GET_ITEM(args, 1);
    std::string name = CastPyArg2String(name_obj, "name", 1);
    // Call ir static api
    CallStackRecorder callstack_recoder("shadow_output");
    callstack_recoder.record();
    paddle::dialect::shadow_output(persist_value, name);
    callstack_recoder.attach_to_ops();
    Py_RETURN_NONE;
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_full(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add full op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Parse Attributes
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 0);
    PyObject *value_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 3);

    phi::DataType dtype = CastPyArg2DataTypeDirectly(dtype_obj, "full", 2);
    Place place = CastPyArg2Place(place_obj, "full", 3);

    if (!PyObject_CheckIRValue(shape_obj) &&
        !PyObject_CheckIRVectorOfValue(shape_obj) &&
        !PyObject_CheckIRValue(value_obj)) {
      std::vector<int64_t> shape = CastPyArg2Longs(shape_obj, "full", 0);
      float value = CastPyArg2Float(value_obj, "full", 1);
      auto static_api_out = paddle::dialect::full(shape, value, dtype, place);
      return ToPyObject(static_api_out);
    } else {
      pir::Value shape, value;

      if (PyObject_CheckIRValue(shape_obj)) {
        shape = CastPyArg2Value(shape_obj, "full", 0);
      } else if (PyObject_CheckIRVectorOfValue(shape_obj)) {
        std::vector<pir::Value> shape_tmp =
            CastPyArg2VectorOfValue(shape_obj, "full", 0);
        shape = paddle::dialect::stack(shape_tmp, 0);
      } else {
        std::vector<int64_t> shape_tmp = CastPyArg2Longs(shape_obj, "full", 0);
        shape = paddle::dialect::full_int_array(
            shape_tmp, phi::DataType::INT64, phi::CPUPlace());
      }

      if (PyObject_CheckIRValue(value_obj)) {
        value = CastPyArg2Value(value_obj, "full", 1);
      } else {
        float value_tmp = CastPyArg2Float(value_obj, "full", 1);
        value = paddle::dialect::full(std::vector<int64_t>{1},
                                      value_tmp,
                                      phi::DataType::FLOAT32,
                                      phi::CPUPlace());
      }

      CallStackRecorder callstack_recoder("full_with_tensor");
      callstack_recoder.record();
      auto static_api_out =
          paddle::dialect::full_with_tensor(shape, value, dtype);
      callstack_recoder.attach_to_ops();

      return ToPyObject(static_api_out);
    }
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *static_api_create_array(PyObject *self,
                                         PyObject *args,
                                         PyObject *kwargs) {
  try {
    VLOG(6) << "Add create_array op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get dtype from args
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 0);
    phi::DataType dtype =
        CastPyArg2DataTypeDirectly(dtype_obj, "create_array", 0);

    // Call ir static api
    CallStackRecorder callstack_recoder("create_array");
    callstack_recoder.record();
    auto static_api_out = paddle::dialect::create_array(dtype);
    callstack_recoder.attach_to_ops();

    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *static_api_create_array_like(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add create_array_like op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "create_array_like", 0);

    // Parse Attributes
    PyObject *value_obj = PyTuple_GET_ITEM(args, 1);
    float value = CastPyArg2Float(value_obj, "create_array_like", 1);

    // Call ir static api
    CallStackRecorder callstack_recoder("create_array_like");
    callstack_recoder.record();
    auto static_api_out = paddle::dialect::create_array_like(input, value);
    callstack_recoder.attach_to_ops();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *static_api_array_length(PyObject *self,
                                         PyObject *args,
                                         PyObject *kwargs) {
  try {
    VLOG(6) << "Add array_length op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "array_length", 0);

    // Call ir static api
    CallStackRecorder callstack_recoder("array_length");
    callstack_recoder.record();
    auto static_api_out = paddle::dialect::array_length(x);
    callstack_recoder.attach_to_ops();

    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *static_api_array_read(PyObject *self,
                                       PyObject *args,
                                       PyObject *kwargs) {
  try {
    VLOG(6) << "Add array_read op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *array_obj = PyTuple_GET_ITEM(args, 0);
    auto array = CastPyArg2Value(array_obj, "array_read", 0);

    PyObject *i_obj = PyTuple_GET_ITEM(args, 1);
    pir::Value i;
    if (PyObject_CheckIRValue(i_obj)) {
      i = CastPyArg2Value(i_obj, "array_read", 1);
    } else {
      int64_t i_tmp = CastPyArg2Int(i_obj, "array_read", 1);
      i = paddle::dialect::full(std::vector<int64_t>{1},
                                i_tmp,
                                phi::DataType::INT64,
                                phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recoder("array_read");
    callstack_recoder.record();
    auto static_api_out = paddle::dialect::array_read(array, i);
    callstack_recoder.attach_to_ops();

    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *static_api_array_write_(PyObject *self,
                                         PyObject *args,
                                         PyObject *kwargs) {
  try {
    VLOG(6) << "Add array_write_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *array_obj = PyTuple_GET_ITEM(args, 0);
    auto array = CastPyArg2Value(array_obj, "array_write_", 0);
    PyObject *x_obj = PyTuple_GET_ITEM(args, 1);
    auto x = CastPyArg2Value(x_obj, "array_write_", 1);
    PyObject *i_obj = PyTuple_GET_ITEM(args, 2);
    pir::Value i;
    if (PyObject_CheckIRValue(i_obj)) {
      i = CastPyArg2Value(i_obj, "array_write_", 2);
    } else {
      int64_t i_tmp = CastPyArg2Int(i_obj, "array_write_", 2);
      i = paddle::dialect::full(std::vector<int64_t>{1},
                                i_tmp,
                                phi::DataType::INT64,
                                phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recoder("array_write_");
    callstack_recoder.record();
    auto static_api_out = paddle::dialect::array_write_(array, x, i);
    callstack_recoder.attach_to_ops();

    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *static_api_array_to_tensor(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  try {
    VLOG(6) << "Add array_to_tensor op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    pir::Value x;
    if (PyObject_CheckIRValue(x_obj)) {
      x = CastPyArg2Value(x_obj, "array_to_tensor", 0);
    } else if (PyObject_CheckIRVectorOfValue(x_obj)) {
      std::vector<pir::Value> x_tmp =
          CastPyArg2VectorOfValue(x_obj, "array_to_tensor", 0);
      if (x_tmp.size() != 1) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Input x expects only one input, but %d are given.",
            x_tmp.size()));  // NOLINT
      }
      x = x_tmp[0];
    }

    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    auto axis = CastPyArg2Int(axis_obj, "array_to_tensor", 1);

    PyObject *use_stack_obj = PyTuple_GET_ITEM(args, 2);
    auto use_stack = CastPyArg2Boolean(use_stack_obj, "array_to_tensor", 2);

    // Call ir static api
    CallStackRecorder callstack_recoder("array_to_tensor");
    callstack_recoder.record();
    auto static_api_out = paddle::dialect::array_to_tensor(x, axis, use_stack);
    callstack_recoder.attach_to_ops();

    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_add_n_array(PyObject *self,
                                 PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add add_n_array op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *inputs_obj = PyTuple_GET_ITEM(args, 0);
    auto inputs = CastPyArg2VectorOfValue(inputs_obj, "add_n", 0);

    CallStackRecorder callstack_recoder("add_n_array");
    callstack_recoder.record();
    auto static_api_out = paddle::dialect::add_n_array(inputs);
    callstack_recoder.attach_to_ops();

    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *static_api_slice_array(PyObject *self,
                                        PyObject *args,
                                        PyObject *kwargs) {
  try {
    VLOG(6) << "Add slice_array op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "slice_array", 0);

    PyObject *starts_obj = PyTuple_GET_ITEM(args, 1);
    pir::Value starts;
    if (PyObject_CheckIRValue(starts_obj)) {
      starts = CastPyArg2Value(starts_obj, "slice_array", 1);
    } else if (PyObject_CheckIRVectorOfValue(starts_obj)) {
      std::vector<pir::Value> starts_tmp =
          CastPyArg2VectorOfValue(starts_obj, "slice_array", 1);
      starts = paddle::dialect::stack(starts_tmp, /*axis*/ 0);
    } else {
      std::vector<int64_t> starts_tmp =
          CastPyArg2Longs(starts_obj, "slice_array", 1);
      starts = paddle::dialect::full_int_array(
          starts_tmp, phi::DataType::INT64, phi::CPUPlace());
    }

    PyObject *ends_obj = PyTuple_GET_ITEM(args, 2);
    pir::Value ends;
    if (PyObject_CheckIRValue(ends_obj)) {
      ends = CastPyArg2Value(ends_obj, "slice_array", 2);
    } else if (PyObject_CheckIRVectorOfValue(ends_obj)) {
      std::vector<pir::Value> ends_tmp =
          CastPyArg2VectorOfValue(ends_obj, "slice_array", 2);
      ends = paddle::dialect::stack(ends_tmp, /*axis*/ 0);
    } else {
      std::vector<int64_t> ends_tmp =
          CastPyArg2Longs(ends_obj, "slice_array", 2);
      ends = paddle::dialect::full_int_array(
          ends_tmp, phi::DataType::INT64, phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recoder("slice_array");
    callstack_recoder.record();
    auto static_api_out = paddle::dialect::slice_array(input, starts, ends);
    callstack_recoder.attach_to_ops();

    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *static_api_slice_array_dense(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add slice_array_dense op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "slice_array_dense", 0);

    PyObject *starts_obj = PyTuple_GET_ITEM(args, 1);
    pir::Value starts;
    if (PyObject_CheckIRValue(starts_obj)) {
      starts = CastPyArg2Value(starts_obj, "slice_array_dense", 1);
    } else if (PyObject_CheckIRVectorOfValue(starts_obj)) {
      std::vector<pir::Value> starts_tmp =
          CastPyArg2VectorOfValue(starts_obj, "slice_array_dense", 1);
      starts = paddle::dialect::stack(starts_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> starts_tmp =
          CastPyArg2Longs(starts_obj, "slice_array_dense", 1);
      starts = paddle::dialect::full_int_array(
          starts_tmp, phi::DataType::INT64, phi::CPUPlace());
    }
    // Call ir static api
    CallStackRecorder callstack_recoder("slice_array_dense");
    callstack_recoder.record();
    auto static_api_out = paddle::dialect::slice_array_dense(input, starts);
    callstack_recoder.attach_to_ops();

    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

extern PyObject *eager_api_run_custom_op(PyObject *self,
                                         PyObject *args,
                                         PyObject *kwargs);

static PyObject *static_api_run_custom_op(PyObject *self,
                                          PyObject *args,
                                          PyObject *kwargs) {
  std::string op_type = CastPyArg2AttrString(PyTuple_GET_ITEM(args, 0), 0);
  VLOG(7) << "Get things from python for Custom Op: " << op_type;
  const auto &meta_info_map = OpMetaInfoMap::Instance().GetMap();
  PADDLE_ENFORCE_NE(meta_info_map.find(op_type),
                    meta_info_map.end(),
                    paddle::platform::errors::NotFound(
                        "Can't find %s in Eager OpMetaInfoMap which should be "
                        "created by LoadOpMetaInfoAndRegisterOp, please make "
                        "sure you registered your op first and try again. ",
                        op_type));

  const auto &vec_map = meta_info_map.at(op_type);
  const auto &inputs = paddle::OpMetaInfoHelper::GetInputs(vec_map[0]);
  const auto &attrs = paddle::OpMetaInfoHelper::GetAttrs(vec_map[0]);
  const auto &outputs = paddle::OpMetaInfoHelper::GetOutputs(vec_map[0]);
  const auto &inplace_map = paddle::OpMetaInfoHelper::GetInplaceMap(vec_map[0]);
  const auto &inplace_reverse_map =
      paddle::OpMetaInfoHelper::GetInplaceReverseMap(vec_map[0]);
  auto infershape_func = OpMetaInfoHelper::GetInferShapeFn(vec_map[0]);
  auto inferdtype_func = OpMetaInfoHelper::GetInferDtypeFn(vec_map[0]);

  std::string pir_op_name = paddle::framework::kCustomDialectPrefix + op_type;
  if (!inplace_map.empty()) {
    pir_op_name += "_";
  }
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::OpInfo pir_info = ctx->GetRegisteredOpInfo(pir_op_name);
  pir::OperationArgument argument(pir_info);
  std::vector<pir::Value> argument_inputs;
  std::vector<pir::Type> argument_outputs;

  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<DataType> input_dtypes;
  std::unordered_map<std::string, int> input_name2id_map;
  std::vector<std::vector<std::vector<int64_t>>> vec_input_shapes;
  std::vector<std::vector<DataType>> vec_input_dtypes;
  std::unordered_map<std::string, int> vec_input_name2id_map;
  std::vector<paddle::any> custom_attrs;
  int input_index = 0;
  int vec_input_index = 0;

  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &input = inputs.at(i);
    // Parse op_type first, so that use i + 1
    PyObject *obj = PyTuple_GET_ITEM(args, i + 1);
    // Emplace Py_None from python, this means optional inputs passed to C++,
    // use one un-initialized tensor to indicate both Tensor and
    // vector<Tensor> inputs.
    if (obj == Py_None) {
      VLOG(7) << "Add un-initialized tensor "
                 "because the optional input is None";
      if (paddle::framework::detail::IsDuplicableVar(input)) {
        vec_input_shapes.emplace_back();
        vec_input_dtypes.emplace_back();
        vec_input_name2id_map[inputs[i]] = vec_input_index;
        vec_input_index++;
      } else {
        input_shapes.emplace_back();
        input_dtypes.emplace_back();
        input_name2id_map[inputs[i]] = input_index;
        input_index++;
      }
      argument_inputs.emplace_back();
      continue;
    }
    if (paddle::framework::detail::IsDuplicableVar(input)) {
      std::vector<std::vector<int64_t>> tmp_input_shapes;
      std::vector<phi::DataType> tmp_input_dtypes;
      vec_input_name2id_map[inputs[i]] = vec_input_index;
      vec_input_index++;
      std::vector<pir::Value> input_values =
          std::move(CastPyArg2VectorOfValue(obj, op_type, i + 1));  // NOLINT
      for (auto &input_value : input_values) {
        paddle::dialect::DenseTensorType input_tensor =
            input_value.type().dyn_cast<paddle::dialect::DenseTensorType>();
        tmp_input_shapes.push_back(phi::vectorize(input_tensor.dims()));
        tmp_input_dtypes.push_back(
            paddle::dialect::TransToPhiDataType(input_tensor.dtype()));
      }
      vec_input_shapes.push_back(tmp_input_shapes);
      vec_input_dtypes.push_back(tmp_input_dtypes);
      auto input_value = paddle::dialect::stack(input_values, /*axis*/ 0);
      argument_inputs.push_back(input_value);
    } else {
      input_name2id_map[inputs[i]] = input_index;
      input_index++;
      pir::Value input_value = CastPyArg2Value(obj, op_type, i + 1);  // NOLINT
      paddle::dialect::DenseTensorType input_tensor =
          input_value.type().dyn_cast<paddle::dialect::DenseTensorType>();
      input_shapes.push_back(phi::vectorize(input_tensor.dims()));
      input_dtypes.push_back(
          paddle::dialect::TransToPhiDataType(input_tensor.dtype()));
      argument_inputs.push_back(input_value);
    }
  }
  argument.AddInputs(argument_inputs);

  // Parse op_type and inputs first, so that use 1 + inputs.size() + i
  int attr_start_idx = static_cast<int>(1 + inputs.size());
  for (size_t i = 0; i < attrs.size(); ++i) {
    const auto &attr = attrs.at(i);
    std::vector<std::string> attr_name_and_type = paddle::ParseAttrStr(attr);
    auto attr_type_str = attr_name_and_type[1];
    VLOG(7) << "Custom operator add attrs " << attr_name_and_type[0]
            << " to CustomOpKernelContext. Attribute type = " << attr_type_str;
    PyObject *obj = PyTuple_GET_ITEM(args, attr_start_idx + i);
    if (attr_type_str == "bool") {
      bool bool_attr = CastPyArg2AttrBoolean(obj, attr_start_idx + i);
      custom_attrs.push_back(bool_attr);  // NOLINT
      argument.AddAttribute(
          attr_name_and_type[0],
          pir::BoolAttribute::get(pir::IrContext::Instance(), bool_attr));
    } else if (attr_type_str == "int") {
      int int_attr = CastPyArg2AttrInt(obj, attr_start_idx + i);
      custom_attrs.push_back(int_attr);  // NOLINT
      argument.AddAttribute(
          attr_name_and_type[0],
          pir::Int32Attribute::get(pir::IrContext::Instance(), int_attr));
    } else if (attr_type_str == "float") {
      float float_attr = CastPyArg2AttrFloat(obj, attr_start_idx + i);
      custom_attrs.push_back(float_attr);  // NOLINT
      argument.AddAttribute(
          attr_name_and_type[0],
          pir::FloatAttribute::get(pir::IrContext::Instance(), float_attr));
    } else if (attr_type_str == "int64_t") {
      int64_t long_attr = CastPyArg2AttrLong(obj, attr_start_idx + i);
      custom_attrs.push_back(long_attr);  // NOLINT
      argument.AddAttribute(
          attr_name_and_type[0],
          pir::Int64Attribute::get(pir::IrContext::Instance(), long_attr));
    } else if (attr_type_str == "std::string") {
      std::string str_attr = CastPyArg2AttrString(obj, attr_start_idx + i);
      custom_attrs.push_back(str_attr);  // NOLINT
      argument.AddAttribute(
          attr_name_and_type[0],
          pir::StrAttribute::get(pir::IrContext::Instance(), str_attr));
    } else if (attr_type_str == "std::vector<int>") {
      std::vector<int> vec_int_attr =
          CastPyArg2VectorOfInt(obj, attr_start_idx + i);
      custom_attrs.push_back(vec_int_attr);
      std::vector<pir::Attribute> array_attr;
      for (size_t i = 0; i < static_cast<size_t>(vec_int_attr.size()); i++) {
        pir::Attribute attr = pir::Int32Attribute::get(
            pir::IrContext::Instance(), vec_int_attr[i]);
        array_attr.push_back(attr);
      }
      argument.AddAttribute(
          attr_name_and_type[0],
          pir::ArrayAttribute::get(pir::IrContext::Instance(), array_attr));
    } else if (attr_type_str == "std::vector<float>") {
      std::vector<float> vec_float_attr =
          CastPyArg2VectorOfFloat(obj, attr_start_idx + i);
      custom_attrs.push_back(vec_float_attr);
      std::vector<pir::Attribute> array_attr;
      for (size_t i = 0; i < static_cast<size_t>(vec_float_attr.size()); i++) {
        pir::Attribute attr = pir::FloatAttribute::get(
            pir::IrContext::Instance(), vec_float_attr[i]);
        array_attr.push_back(attr);
      }
      argument.AddAttribute(
          attr_name_and_type[0],
          pir::ArrayAttribute::get(pir::IrContext::Instance(), array_attr));
    } else if (attr_type_str == "std::vector<int64_t>") {
      std::vector<int64_t> vec_long_attr =
          CastPyArg2VectorOfInt64(obj, attr_start_idx + i);
      custom_attrs.push_back(vec_long_attr);  // NOLINT
      std::vector<pir::Attribute> array_attr;
      for (size_t i = 0; i < static_cast<size_t>(vec_long_attr.size()); i++) {
        pir::Attribute attr = pir::Int64Attribute::get(
            pir::IrContext::Instance(), vec_long_attr[i]);
        array_attr.push_back(attr);
      }
      argument.AddAttribute(
          attr_name_and_type[0],
          pir::ArrayAttribute::get(pir::IrContext::Instance(), array_attr));
    } else if (attr_type_str == "std::vector<std::string>") {
      std::vector<std::string> vec_str_attr =
          CastPyArg2VectorOfString(obj, attr_start_idx + i);
      custom_attrs.push_back(vec_str_attr);  // NOLINT
      std::vector<pir::Attribute> array_attr;
      for (size_t i = 0; i < static_cast<size_t>(vec_str_attr.size()); i++) {
        pir::Attribute attr =
            pir::StrAttribute::get(pir::IrContext::Instance(), vec_str_attr[i]);
        array_attr.push_back(attr);
      }
      argument.AddAttribute(
          attr_name_and_type[0],
          pir::ArrayAttribute::get(pir::IrContext::Instance(), array_attr));
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported `%s` type value as custom attribute now. "
          "Supported data types include `bool`, `int`, `float`, "
          "`int64_t`, `std::string`, `std::vector<int>`, "
          "`std::vector<float>`, `std::vector<int64_t>`, "
          "`std::vector<std::string>`, Please check whether "
          "the attribute data type and data type string are matched.",
          attr_type_str));
    }
  }

  paddle::framework::CheckDefaultInferShapeDtype(
      infershape_func, inferdtype_func, vec_map[0]);
  std::vector<std::vector<int64_t>> output_shapes =
      paddle::framework::RunInferShape(infershape_func,
                                       vec_map[0],
                                       input_shapes,
                                       input_name2id_map,
                                       vec_input_shapes,
                                       vec_input_name2id_map,
                                       custom_attrs);
  std::vector<phi::DataType> output_dtypes =
      paddle::framework::RunInferDtype(inferdtype_func,
                                       vec_map[0],
                                       input_dtypes,
                                       input_name2id_map,
                                       vec_input_dtypes,
                                       vec_input_name2id_map,
                                       custom_attrs);

  size_t all_values_num = 0;
  // output name -> value num (that output should hold)
  std::unordered_map<std::string, size_t> output_name2value_num;
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto &output = outputs.at(i);
    if (paddle::framework::detail::IsDuplicableVar(output)) {
      PADDLE_ENFORCE_NE(
          inplace_reverse_map.find(output),
          inplace_reverse_map.end(),
          phi::errors::InvalidArgument(
              "Only support vector output that is set for inplace, Please use "
              "`SetInplaceMap` in your output when registry custom operator."));
      const auto &input = inplace_reverse_map.at(output);
      auto index = vec_input_name2id_map[input];
      auto &input_shapes = vec_input_shapes[index];
      output_name2value_num[output] = input_shapes.size();
      all_values_num += input_shapes.size();
    } else {
      output_name2value_num[output] = 1;
      all_values_num++;
    }
  }

  PADDLE_ENFORCE_EQ(
      output_shapes.size(),
      all_values_num,
      phi::errors::InvalidArgument(
          "The number of output shapes after running custom operator's "
          "InferShapeFunc is wrong, "
          "expected contains %d Tensors' shape, but actually contains %d "
          "Tensors' shape",
          all_values_num,
          output_shapes.size()));

  PADDLE_ENFORCE_EQ(
      output_dtypes.size(),
      all_values_num,
      phi::errors::InvalidArgument(
          "The number of output dtypes after running custom operator's "
          "InferDtypeFunc is wrong, "
          "expected contains %d Tensors' dtype, but actually contains %d "
          "Tensors' dtype",
          all_values_num,
          output_dtypes.size()));

  size_t value_index = 0;
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto &output = outputs.at(i);
    if (paddle::framework::detail::IsDuplicableVar(output)) {
      auto value_num = output_name2value_num[output];
      std::vector<pir::Type> out_types;
      for (size_t j = 0; j < value_num; ++j) {
        auto ddims = phi::make_ddim(output_shapes[value_index]);
        auto dtype = output_dtypes[value_index];
        phi::DataLayout layout{DataLayout::NCHW};
        phi::LoD lod;
        out_types.push_back(paddle::dialect::DenseTensorType::get(
            pir::IrContext::Instance(),
            paddle::dialect::TransToIrDataType(dtype),
            ddims,
            layout,
            lod,
            0));
        value_index++;
      }
      pir::Type out_vector_type =
          pir::VectorType::get(pir::IrContext::Instance(), out_types);
      argument_outputs.push_back(out_vector_type);
    } else {
      auto ddims = phi::make_ddim(output_shapes[value_index]);
      auto dtype = output_dtypes[value_index];
      phi::DataLayout layout{DataLayout::NCHW};
      phi::LoD lod;
      auto out_type = paddle::dialect::DenseTensorType::get(
          pir::IrContext::Instance(),
          paddle::dialect::TransToIrDataType(dtype),
          ddims,
          layout,
          lod,
          0);
      argument_outputs.push_back(out_type);
      value_index++;
    }
  }

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
  CallStackRecorder callstack_recoder("run_custom_op");
  callstack_recoder.record();
  std::vector<pir::Value> op_results;
  pir::Operation *op =
      paddle::dialect::ApiBuilder::Instance().GetBuilder()->Build(
          std::move(argument));
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto &output = outputs.at(i);
    if (paddle::framework::detail::IsDuplicableVar(output)) {
      auto split_op = paddle::dialect::ApiBuilder::Instance()
                          .GetBuilder()
                          ->Build<pir::SplitOp>(op->result(i));
      auto split_outputs = split_op.outputs();
      op_results.insert(
          op_results.end(), split_outputs.begin(), split_outputs.end());
    } else {
      op_results.push_back(op->result(i));
    }
  }
  callstack_recoder.attach_to_ops();
  return ToPyObject(op_results);
}

static PyObject *run_custom_op(PyObject *self,
                               PyObject *args,
                               PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_abs";
    return static_api_run_custom_op(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_abs";
    return eager_api_run_custom_op(self, args, kwargs);
  }
}

static PyObject *static_api_fused_gemm_epilogue(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  try {
    VLOG(6) << "Running Static API: fused_gemm_epilogue";

    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);
    // Get OpResult from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fused_gemm_epilogue", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "fused_gemm_epilogue", 1);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 2);
    auto bias = CastPyArg2Value(bias_obj, "fused_gemm_epilogue", 2);

    // Parse Attributes if needed
    PyObject *trans_x_obj = PyTuple_GET_ITEM(args, 3);
    bool trans_x = CastPyArg2Boolean(trans_x_obj, "fused_gemm_epilogue", 3);
    PyObject *trans_y_obj = PyTuple_GET_ITEM(args, 4);
    bool trans_y = CastPyArg2Boolean(trans_y_obj, "fused_gemm_epilogue", 4);
    PyObject *activation_obj = PyTuple_GET_ITEM(args, 5);
    std::string activation =
        CastPyArg2String(activation_obj, "fused_gemm_epilogue", 5);
    // Call ir static api
    CallStackRecorder callstack_recoder("fused_gemm_epilogue");
    callstack_recoder.record();
    auto out = paddle::dialect::fused_gemm_epilogue(
        x, y, bias, trans_x, trans_y, activation);
    callstack_recoder.attach_to_ops();

    return ToPyObject(out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}
static PyObject *static_api_array_pop(PyObject *self,
                                      PyObject *args,
                                      PyObject *kwargs) {
  try {
    VLOG(6) << "Add array_pop op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "array_pop", 0);

    PyObject *index_obj = PyTuple_GET_ITEM(args, 1);
    auto index = CastPyArg2Int(index_obj, "array_pop", 1);

    // Call ir static api
    CallStackRecorder callstack_recoder("array_pop");
    callstack_recoder.record();
    auto static_api_out = paddle::dialect::array_pop(input, index);
    callstack_recoder.attach_to_ops();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

extern PyObject *eager_api_fused_gemm_epilogue(PyObject *self,
                                               PyObject *args,
                                               PyObject *kwargs);

static PyObject *fused_gemm_epilogue(PyObject *self,
                                     PyObject *args,
                                     PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fused_gemm_epilogue";
    return static_api_fused_gemm_epilogue(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fused_gemm_epilogue";
    return eager_api_fused_gemm_epilogue(self, args, kwargs);
  }
}

static PyMethodDef ManualOpsAPI[] = {
    {"set_parameter",
     (PyCFunction)(void (*)(void))static_api_set_parameter,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for set_parameter."},
    {"set_persistable_value",
     (PyCFunction)(void (*)(void))static_api_set_persistable_value,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for set_persistable_value."},
    {"parameter",
     (PyCFunction)(void (*)(void))static_api_parameter,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for parameter."},
    {"create_array",
     (PyCFunction)(void (*)(void))static_api_create_array,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for create_array."},
    {"create_array_like",
     (PyCFunction)(void (*)(void))static_api_create_array_like,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for create_array_like."},
    {"array_length",
     (PyCFunction)(void (*)(void))static_api_array_length,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for array_length."},
    {"array_read",
     (PyCFunction)(void (*)(void))static_api_array_read,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for array_read."},
    {"array_write_",
     (PyCFunction)(void (*)(void))static_api_array_write_,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for array_write_."},
    {"array_to_tensor",
     (PyCFunction)(void (*)(void))static_api_array_to_tensor,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for array_to_tensor."},
    {"add_n_array",
     (PyCFunction)(void (*)(void))static_api_add_n_array,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for add_n_array."},
    {"slice_array",
     (PyCFunction)(void (*)(void))static_api_slice_array,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for slice_array."},
    {"slice_array_dense",
     (PyCFunction)(void (*)(void))static_api_slice_array_dense,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for slice_array_dense."},
    {"fused_gemm_epilogue",
     (PyCFunction)(void (*)(void))fused_gemm_epilogue,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fused_gemm_epilogue."},
    {"_run_custom_op",
     (PyCFunction)(void (*)(void))run_custom_op,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for run_custom_op."},
    {"array_pop",
     (PyCFunction)(void (*)(void))static_api_array_pop,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for array_pop."},
    {nullptr, nullptr, 0, nullptr}};

}  // namespace pybind

}  // namespace paddle
