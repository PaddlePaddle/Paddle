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

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

#include <Python.h>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/eager/hooks.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/string_array.h"
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

#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/pybind/imperative.h"

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
phi::DenseTensor CastPyArg2FrameworkTensor(PyObject* obj, ssize_t arg_pos);
std::vector<phi::DenseTensor> CastPyArg2VectorOfTensorBase(PyObject* obj,
                                                           ssize_t arg_pos);
std::vector<int> CastPyArg2VectorOfInt(PyObject* obj, size_t arg_pos);
std::vector<int64_t> CastPyArg2VectorOfInt64(PyObject* obj, size_t arg_pos);
std::vector<size_t> CastPyArg2VectorOfSize_t(PyObject* obj, size_t arg_pos);
std::vector<std::vector<size_t>> CastPyArg2VectorOfVectorOfSize_t(
    PyObject* obj, size_t arg_pos);
framework::proto::VarType::Type CastPyArg2ProtoType(PyObject* obj,
                                                    ssize_t arg_pos);
paddle::framework::Vocab CastPyArg2Vocab(PyObject* obj, ssize_t arg_pos);
std::vector<std::string> CastPyArg2VectorOfString(PyObject* obj,
                                                  ssize_t arg_pos);
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
PyObject* ToPyObject(
    const std::vector<std::vector<paddle::experimental::Tensor>>& value,
    bool return_py_none_if_not_initialize = false);
PyObject* ToPyObject(const platform::Place& value);
PyObject* ToPyObject(const phi::DenseTensor* value);
PyObject* ToPyObject(const phi::SelectedRows* value);
PyObject* ToPyObject(const paddle::framework::proto::VarType::Type& dtype);
PyObject* ToPyObject(const paddle::framework::proto::VarType& type);
PyObject* ToPyObject(const void* value);
PyObject* ToPyObject(
    const std::unordered_map<std::string, std::vector<std::string>>& value);
PyObject* ToPyObject(const paddle::framework::Vocab& value);

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

class PyObjectHolder : public egr::PyObjectHolderBase {
 public:
  PyObjectHolder() { ptr_ = nullptr; }
  explicit PyObjectHolder(PyObject* ptr);
  ~PyObjectHolder() override;
  void* get() override;
  void reset(void* ptr) override;
  void inc_ref() override;
  void dec_ref() override;

 private:
  PyObject* ptr_{nullptr};
};

class PackHook : public egr::PackHookBase {
 public:
  explicit PackHook(PyObject* hook);

  ~PackHook();

  std::shared_ptr<egr::PyObjectHolderBase> operator()(
      const paddle::experimental::Tensor& tensor) override;

  void* operator()(void* py_tensor) override;

 private:
  PyObject* hook_;
};

class UnPackHook : public egr::UnPackHookBase {
 public:
  explicit UnPackHook(PyObject* hook);

  ~UnPackHook();

  paddle::experimental::Tensor operator()(
      std::shared_ptr<egr::PyObjectHolderBase> packed_value) override;

  void* operator()(void* packed_value, void* other) override;

 private:
  PyObject* hook_;
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

bool PyObject_CheckBool(PyObject** obj);

bool PyObject_CheckLongOrToLong(PyObject** obj);

bool PyObject_CheckFloatOrToFloat(PyObject** obj);

bool PyObject_CheckString(PyObject* obj);

bool CastPyArg2Boolean(PyObject* obj,
                       const std::string& op_type,
                       ssize_t arg_pos);
int CastPyArg2Int(PyObject* obj, const std::string& op_type, ssize_t arg_pos);
int64_t CastPyArg2Long(PyObject* obj,
                       const std::string& op_type,
                       ssize_t arg_pos);
float CastPyArg2Float(PyObject* obj,
                      const std::string& op_type,
                      ssize_t arg_pos);
double CastPyArg2Double(PyObject* obj,
                        const std::string& op_type,
                        ssize_t arg_pos);
phi::dtype::complex<float> CastPyArg2Complex(PyObject* obj,
                                             const std::string& op_type,
                                             ssize_t arg_pos);
std::string CastPyArg2String(PyObject* obj,
                             const std::string& op_type,
                             ssize_t arg_pos);
std::vector<bool> CastPyArg2Booleans(PyObject* obj,
                                     const std::string& op_type,
                                     ssize_t arg_pos);
std::vector<int> CastPyArg2Ints(PyObject* obj,
                                const std::string& op_type,
                                ssize_t arg_pos);
std::vector<int64_t> CastPyArg2Longs(PyObject* obj,
                                     const std::string& op_type,
                                     ssize_t arg_pos);
std::vector<float> CastPyArg2Floats(PyObject* obj,
                                    const std::string& op_type,
                                    ssize_t arg_pos);
std::vector<double> CastPyArg2Float64s(PyObject* obj,
                                       const std::string& op_type,
                                       ssize_t arg_pos);
std::vector<std::string> CastPyArg2Strings(PyObject* obj,
                                           const std::string& op_type,
                                           ssize_t arg_pos);

void CastPyArg2AttrBoolean(PyObject* obj,
                           paddle::framework::AttributeMap& attrs,  // NOLINT
                           const std::string& key,
                           const std::string& op_type,
                           ssize_t arg_pos);

void CastPyArg2AttrInt(PyObject* obj,
                       paddle::framework::AttributeMap& attrs,  // NOLINT
                       const std::string& key,
                       const std::string& op_type,
                       ssize_t arg_pos);

void CastPyArg2AttrLong(PyObject* obj,
                        paddle::framework::AttributeMap& attrs,  // NOLINT
                        const std::string& key,
                        const std::string& op_type,
                        ssize_t arg_pos);

void CastPyArg2AttrFloat(PyObject* obj,
                         paddle::framework::AttributeMap& attrs,  // NOLINT
                         const std::string& key,
                         const std::string& op_type,
                         ssize_t arg_pos);

void CastPyArg2AttrDouble(PyObject* obj,
                          paddle::framework::AttributeMap& attrs,  // NOLINT
                          const std::string& key,
                          const std::string& op_type,
                          ssize_t arg_pos);

void CastPyArg2AttrString(PyObject* obj,
                          paddle::framework::AttributeMap& attrs,  // NOLINT
                          const std::string& key,
                          const std::string& op_type,
                          ssize_t arg_pos);

void CastPyArg2AttrBooleans(PyObject* obj,
                            paddle::framework::AttributeMap& attrs,  // NOLINT
                            const std::string& key,
                            const std::string& op_type,
                            ssize_t arg_pos);

void CastPyArg2AttrInts(PyObject* obj,
                        paddle::framework::AttributeMap& attrs,  // NOLINT
                        const std::string& key,
                        const std::string& op_type,
                        ssize_t arg_pos);

void CastPyArg2AttrLongs(PyObject* obj,
                         paddle::framework::AttributeMap& attrs,  // NOLINT
                         const std::string& key,
                         const std::string& op_type,
                         ssize_t arg_pos);

void CastPyArg2AttrFloats(PyObject* obj,
                          paddle::framework::AttributeMap& attrs,  // NOLINT
                          const std::string& key,
                          const std::string& op_type,
                          ssize_t arg_pos);

void CastPyArg2AttrFloat64s(PyObject* obj,
                            paddle::framework::AttributeMap& attrs,  // NOLINT
                            const std::string& key,
                            const std::string& op_type,
                            ssize_t arg_pos);

void CastPyArg2AttrStrings(PyObject* obj,
                           paddle::framework::AttributeMap& attrs,  // NOLINT
                           const std::string& key,
                           const std::string& op_type,
                           ssize_t arg_pos);

void CastPyArg2AttrBlock(PyObject* obj,
                         paddle::framework::AttributeMap& attrs,  // NOLINT
                         const std::string& key,
                         const std::string& op_type,
                         ssize_t arg_pos);

void ConstructAttrMapFromPyArgs(
    const std::string& op_type,
    PyObject* args,
    ssize_t attr_start,
    ssize_t attr_end,
    paddle::framework::AttributeMap& attrs);  // NOLINT

std::shared_ptr<imperative::VarBase> GetVarBaseFromArgs(
    const std::string& op_type,
    const std::string& arg_name,
    PyObject* args,
    ssize_t arg_idx,
    bool dispensable = false);

std::vector<std::shared_ptr<imperative::VarBase>> GetVarBaseListFromArgs(
    const std::string& op_type,
    const std::string& arg_name,
    PyObject* args,
    ssize_t arg_idx,
    bool dispensable = false);

unsigned long GetUnsignedLongFromArgs(  // NOLINT
    const std::string& op_type,
    const std::string& arg_name,
    PyObject* args,
    ssize_t arg_idx,
    bool dispensable = false);

void InitOpsAttrTypeMap();

ssize_t GetIdxFromCoreOpsInfoMap(
    const std::unordered_map<std::string, std::vector<std::string>>&
        core_ops_info_map,
    const std::string& op_type,
    const std::string& name);

}  // namespace pybind
}  // namespace paddle
