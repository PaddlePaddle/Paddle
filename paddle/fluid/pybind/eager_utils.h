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

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include <Python.h>

#include "paddle/fluid/eager/hooks.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/jit/function.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/selected_rows.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
namespace paddle {
class CustomOpKernelContext;
namespace framework {
class Scope;
}
namespace pybind {

namespace py = ::pybind11;
#define RETURN_PY_NONE \
  Py_INCREF(Py_None);  \
  return Py_None;

int TensorDtype2NumpyDtype(phi::DataType dtype);

bool IsEagerTensor(PyObject* obj);

bool PyObject_CheckLongOrConvertToLong(PyObject** obj);
bool PyObject_CheckFloatOrConvertToFloat(PyObject** obj);
bool PyObject_CheckStr(PyObject* obj);
bool CastPyArg2AttrBoolean(PyObject* obj, ssize_t arg_pos);
int CastPyArg2AttrInt(PyObject* obj, ssize_t arg_pos);
int64_t CastPyArg2AttrLong(PyObject* obj, ssize_t arg_pos);
size_t CastPyArg2AttrSize_t(PyObject* obj, ssize_t arg_pos);
float CastPyArg2AttrFloat(PyObject* obj, ssize_t arg_pos);
std::string CastPyArg2AttrString(PyObject* obj, ssize_t arg_pos);
paddle::CustomOpKernelContext CastPyArg2CustomOpKernelContext(PyObject* obj,
                                                              ssize_t arg_pos);
paddle::experimental::Tensor CastPyArg2Tensor(PyObject* obj, ssize_t arg_pos);
std::shared_ptr<imperative::VarBase> CastPyArg2VarBase(PyObject* obj,
                                                       ssize_t arg_pos);
std::vector<paddle::experimental::Tensor> CastPyArg2VectorOfTensor(
    PyObject* obj, ssize_t arg_pos);
platform::Place CastPyArg2Place(PyObject* obj, ssize_t arg_pos);
framework::Tensor CastPyArg2FrameworkTensor(PyObject* obj, ssize_t arg_pos);
std::vector<framework::LoDTensor> CastPyArg2VectorOfTensorBase(PyObject* obj,
                                                               ssize_t arg_pos);
std::vector<int> CastPyArg2VectorOfInt(PyObject* obj, size_t arg_pos);
std::vector<size_t> CastPyArg2VectorOfSize_t(PyObject* obj, size_t arg_pos);
std::vector<std::vector<size_t>> CastPyArg2VectorOfVectorOfSize_t(
    PyObject* obj, size_t arg_pos);
framework::proto::VarType::Type CastPyArg2ProtoType(PyObject* obj,
                                                    ssize_t arg_pos);
std::unordered_map<std::wstring, int> CastPyArg2Vocab(PyObject* obj,
                                                      ssize_t arg_pos);
std::vector<std::string> CastPyArg2Strings(PyObject* obj, ssize_t arg_pos);
std::shared_ptr<jit::Function> CastPyArg2JitFunction(PyObject* obj,
                                                     ssize_t arg_pos);

PyObject* ToPyObject(int value);
PyObject* ToPyObject(uint32_t value);
PyObject* ToPyObject(bool value);
PyObject* ToPyObject(int64_t value);
PyObject* ToPyObject(size_t value);
PyObject* ToPyObject(float value);
PyObject* ToPyObject(double value);
PyObject* ToPyObject(const char* value);
PyObject* ToPyObject(const std::string& value);
PyObject* ToPyObject(const paddle::experimental::Tensor& value,
                     bool return_py_none_if_not_initialize = false);
PyObject* ToPyObject(const paddle::experimental::Tensor& value,
                     PyObject* args,
                     const std::map<ssize_t, ssize_t>& inplace_var_idx_map);
PyObject* ToPyObject(PyObject* args, ssize_t arg_idx);
PyObject* ToPyObject(const std::vector<bool>& value);
PyObject* ToPyObject(const std::vector<int>& value);
PyObject* ToPyObject(const std::vector<int64_t>& value);
PyObject* ToPyObject(const std::vector<size_t>& value);
PyObject* ToPyObject(const std::vector<float>& value);
PyObject* ToPyObject(const std::vector<double>& value);
PyObject* ToPyObject(const std::vector<std::vector<size_t>>& value);
PyObject* ToPyObject(const std::vector<paddle::experimental::Tensor>& value,
                     bool return_py_none_if_not_initialize = false);
PyObject* ToPyObject(const platform::Place& value);
PyObject* ToPyObject(const framework::LoDTensor* value);
PyObject* ToPyObject(const phi::SelectedRows* value);
PyObject* ToPyObject(const paddle::framework::proto::VarType::Type& dtype);
PyObject* ToPyObject(const paddle::framework::proto::VarType& type);
PyObject* ToPyObject(const void* value);
PyObject* ToPyObject(
    const std::unordered_map<std::string, std::vector<std::string>>& value);
PyObject* ToPyObject(const std::unordered_map<std::wstring, int>& value);

class PyTensorHook : public egr::TensorHook {
 public:
  explicit PyTensorHook(PyObject* func) : py_func_(func) {
    Py_INCREF(py_func_);
  }

  ~PyTensorHook() {
    py::gil_scoped_acquire gil;
    Py_DECREF(py_func_);
  }

  paddle::experimental::Tensor operator()(
      const paddle::experimental::Tensor& var) override;

 private:
  PyObject* py_func_;
};

class PyVoidHook : public egr::VoidHook {
 public:
  explicit PyVoidHook(PyObject* func) : py_func_(func) { Py_INCREF(py_func_); }

  ~PyVoidHook() {
    py::gil_scoped_acquire gil;
    Py_DECREF(py_func_);
  }

  void operator()() override;

 private:
  PyObject* py_func_;
};

template <typename Tuple, size_t N>
struct TupleTensorResult {
  static void Run(const Tuple& out, PyObject* result) {
    TupleTensorResult<Tuple, N - 1>::Run(out, result);
    PyTuple_SET_ITEM(result, N - 1, ToPyObject(std::get<N - 1>(out)));
  }

  static void Run(const Tuple& out,
                  PyObject* result,
                  PyObject* args,
                  const std::map<ssize_t, ssize_t>& inplace_var_idx_map) {
    TupleTensorResult<Tuple, N - 1>::Run(
        out, result, args, inplace_var_idx_map);
    if (!inplace_var_idx_map.empty() && inplace_var_idx_map.count(N - 1)) {
      PyTuple_SET_ITEM(
          result, N - 1, ToPyObject(args, inplace_var_idx_map.at(N - 1)));
    } else {
      PyTuple_SET_ITEM(result, N - 1, ToPyObject(std::get<N - 1>(out)));
    }
  }
};

template <typename Tuple>
struct TupleTensorResult<Tuple, 1> {
  static void Run(const Tuple& out, PyObject* result) {
    PyTuple_SET_ITEM(result, 0, ToPyObject(std::get<0>(out)));
  }

  static void Run(const Tuple& out,
                  PyObject* result,
                  PyObject* args,
                  const std::map<ssize_t, ssize_t>& inplace_var_idx_map) {
    if (!inplace_var_idx_map.empty() && inplace_var_idx_map.count(0)) {
      PyTuple_SET_ITEM(result, 0, ToPyObject(args, inplace_var_idx_map.at(0)));
    } else {
      PyTuple_SET_ITEM(result, 0, ToPyObject(std::get<0>(out)));
    }
  }
};

template <typename... Args>
PyObject* ToPyObject(const std::tuple<Args...>& out) {
  auto len = sizeof...(Args);
  PyObject* result = PyTuple_New(len);

  TupleTensorResult<decltype(out), sizeof...(Args)>::Run(out, result);

  return result;
}

template <typename... Args>
PyObject* ToPyObject(const std::tuple<Args...>& out,
                     PyObject* args,
                     const std::map<ssize_t, ssize_t>& inplace_var_idx_map) {
  // For inplace op, directly return the input PyObject of the inplace tensor.
  // [Parameter]
  // out: Outputs tuple after executing op.
  // args: Input PyObject.
  // inplace_var_idx_map: Index of Tensors in inplace_map, e.g. {{value_idx,
  // arg_idx}}.
  // - value_idx: Index of inplace tensor in outputs tuple. Used to find the
  // output inplace tensor.
  // - arg_idx: Index of inplace PyObject in input args. Used to find the input
  // inplace PyObject.
  auto len = sizeof...(Args);
  PyObject* result = PyTuple_New(len);

  TupleTensorResult<decltype(out), sizeof...(Args)>::Run(
      out, result, args, inplace_var_idx_map);

  return result;
}

paddle::experimental::Scalar CastPyArg2Scalar(PyObject* obj,
                                              const std::string& op_type,
                                              ssize_t arg_pos);

paddle::experimental::Scalar CastNumpy2Scalar(PyObject* obj,
                                              const std::string& op_type,
                                              ssize_t arg_pos);

std::vector<phi::Scalar> CastPyArg2ScalarArray(PyObject* obj,
                                               const std::string& op_type,
                                               ssize_t arg_pos);

paddle::experimental::IntArray CastPyArg2IntArray(PyObject* obj,
                                                  const std::string& op_type,
                                                  ssize_t arg_pos);

paddle::Place CastPyArg2Place(PyObject* obj,
                              const std::string& op_type,
                              ssize_t arg_pos);

paddle::DataType CastPyArg2DataType(PyObject* obj,
                                    const std::string& op_type,
                                    ssize_t arg_pos);

paddle::optional<paddle::experimental::Tensor> GetOptionalTensorFromArgs(
    const std::string& op_type,
    const std::string& arg_name,
    PyObject* args,
    ssize_t arg_idx,
    bool dispensable = false);

paddle::experimental::Tensor& GetTensorFromArgs(const std::string& op_type,
                                                const std::string& arg_name,
                                                PyObject* args,
                                                ssize_t arg_idx,
                                                bool dispensable = false);

paddle::optional<std::vector<paddle::experimental::Tensor>>
GetOptionalTensorListFromArgs(const std::string& op_type,
                              const std::string& arg_name,
                              PyObject* args,
                              ssize_t arg_idx,
                              bool dispensable = false);

std::vector<paddle::experimental::Tensor> GetTensorListFromArgs(
    const std::string& op_type,
    const std::string& arg_name,
    PyObject* args,
    ssize_t arg_idx,
    bool dispensable = false);

paddle::experimental::Tensor* GetTensorPtrFromArgs(const std::string& op_type,
                                                   const std::string& arg_name,
                                                   PyObject* args,
                                                   ssize_t arg_idx,
                                                   bool dispensable = false);

std::vector<paddle::experimental::Tensor*> GetTensorPtrListFromArgs(
    const std::string& op_type,
    const std::string& arg_name,
    PyObject* args,
    ssize_t arg_idx,
    bool dispensable = false);

std::vector<paddle::experimental::Tensor*> GetTensorPtrListFromPyObject(
    PyObject* obj);

std::vector<paddle::experimental::Tensor> GetTensorListFromPyObject(
    PyObject* obj);

paddle::experimental::Tensor& GetTensorFromPyObject(PyObject* obj);

// end of Slice related methods

std::vector<paddle::framework::Scope*> GetScopePtrListFromArgs(
    const std::string& op_type,
    const std::string& arg_name,
    PyObject* args,
    ssize_t arg_idx,
    bool dispensable);

class eager_gil_scoped_release {
 public:
  eager_gil_scoped_release() { tstate = PyEval_SaveThread(); }
  ~eager_gil_scoped_release() {
    if (!tstate) return;
    PyEval_RestoreThread(tstate);
  }

 private:
  PyThreadState* tstate{nullptr};
};

}  // namespace pybind
}  // namespace paddle
