/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#include <Python.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace paddle {
namespace pybind {

typedef struct {
  PyObject_HEAD egr::EagerTensor eager_tensor;
} EagerTensorObject;

int TensorDtype2NumpyDtype(pten::DataType dtype);

bool PyObject_CheckLongOrConvertToLong(PyObject** obj);
bool PyObject_CheckFloatOrConvertToFloat(PyObject** obj);
bool PyObject_CheckStr(PyObject* obj);
bool CastPyArg2AttrBoolean(PyObject* obj, ssize_t arg_pos);
int CastPyArg2AttrInt(PyObject* obj, ssize_t arg_pos);
int64_t CastPyArg2AttrLong(PyObject* obj, ssize_t arg_pos);
float CastPyArg2AttrFloat(PyObject* obj, ssize_t arg_pos);
std::string CastPyArg2AttrString(PyObject* obj, ssize_t arg_pos);
egr::EagerTensor CastPyArg2EagerTensor(PyObject* obj, ssize_t arg_pos);
std::vector<egr::EagerTensor> CastPyArg2VectorOfEagerTensor(PyObject* obj,
                                                            ssize_t arg_pos);
platform::Place CastPyArg2Place(PyObject* obj, ssize_t arg_pos);
framework::proto::VarType::Type CastPyArg2ProtoType(PyObject* obj,
                                                    ssize_t arg_pos);
PyObject* ToPyObject(int value);
PyObject* ToPyObject(bool value);
PyObject* ToPyObject(int64_t value);
PyObject* ToPyObject(float value);
PyObject* ToPyObject(double value);
PyObject* ToPyObject(const char* value);
PyObject* ToPyObject(const std::string& value);
PyObject* ToPyObject(const egr::EagerTensor& value);
PyObject* ToPyObject(const std::vector<bool>& value);
PyObject* ToPyObject(const std::vector<int>& value);
PyObject* ToPyObject(const std::vector<int64_t>& value);
PyObject* ToPyObject(const std::vector<float>& value);
PyObject* ToPyObject(const std::vector<double>& value);
PyObject* ToPyObject(const std::vector<egr::EagerTensor>& value);
PyObject* ToPyObject(const platform::Place& value);
PyObject* ToPyObject(const paddle::framework::proto::VarType::Type& dtype);
PyObject* ToPyObject(const void* value);

template <typename Tuple, size_t N>
struct TupleEagerTensorResult {
  static void Run(const Tuple& out, PyObject* result) {
    TupleEagerTensorResult<Tuple, N - 1>::Run(out, result);
    PyTuple_SET_ITEM(result, N - 1, ToPyObject(std::get<N - 1>(out)));
  }
};

template <typename Tuple>
struct TupleEagerTensorResult<Tuple, 1> {
  static void Run(const Tuple& out, PyObject* result) {
    PyTuple_SET_ITEM(result, 0, ToPyObject(std::get<0>(out)));
  }
};

template <typename... Args>
PyObject* ToPyObject(const std::tuple<Args...>& out) {
  auto len = sizeof...(Args);
  PyObject* result = PyTuple_New(len);

  TupleEagerTensorResult<decltype(out), sizeof...(Args)>::Run(out, result);

  return result;
}

egr::EagerTensor GetEagerTensorFromArgs(const std::string& op_type,
                                        const std::string& arg_name,
                                        PyObject* args, ssize_t arg_idx,
                                        bool dispensable = false);
std::vector<egr::EagerTensor> GetEagerTensorListFromArgs(
    const std::string& op_type, const std::string& arg_name, PyObject* args,
    ssize_t arg_idx, bool dispensable = false);

}  // namespace pybind
}  // namespace paddle
