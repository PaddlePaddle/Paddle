/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/pybind/py_hook.h"

#include <memory>

#include "paddle/fluid/frameowrk/var_type_traits.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace pybind {

namespace py = ::pybind11;

static void CheckHookResult(PyObject *origin, PyObject *result,
                            PyObject *hook) {
  PADDLE_ENFORCE_EQ(
      origin, Py_None,
      platform::errors::InvalidArgument(
          "Cannot replace a None gradient with a non-None value."));
  PADDLE_ENFORCE_EQ(
      py::isinstance<imperative::VarBase>(result), true,
      platform::errors::Unavailable(
          "Hook returned result need to be VarBase, but received `%s`.",
          py::type::of(*result)));

  auto *origin_var = (imperative::VarBase *)origin;
  auto *result_var = (imperative::VarBase *)result;

  // place check
  PADDLE_ENFORCE_EQ(origin_var->Place(), result_var->Place(),
                    platform::errors::PermissionDenied(
                        "Hook changed the device of value, original device is "
                        "%s, result device is %s.",
                        origin_var->Place(), result_var->Place()));

  // type check
  PADDLE_ENFORCE_EQ(
      origin_var->Type(), result_var->Type(),
      platform::errors::PermissionDenied(
          "Hook changed the type of value, original type is %s, result type is "
          "%s.",
          platform::demangle(framework::ToTypeName(origin_var->Type())),
          platform::demangle(framework::ToTypeName(result_var->Type()))));

  // dtype check
  PADDLE_ENFORCE_EQ(origin_var->DataType(), result_var->DataType(),
                    platform::errors::PermissionDenied(
                        "Hook changed the dtype of value, original dtype is "
                        "%s, result dtype is %s.",
                        framework::DataTypeToString(origin_var->DataType()),
                        framework::DataTypeToString(result_var->DataType())));
}

PyHook::PyHook(PyObject *hook) : hook_(hook) { Py_INCREF(hook_); }

PyHook::~PyHook() {
  py::gil_scoped_acquire gil;
  Py_DECREF(hook_);
}
s

    std::shared_ptr<VariableWrapper>
    PyHook::operator()(const std::shared_ptr<VariableWrapper> &var) {
  py::gil_scoped_acquire gil;

  // unpack to varbase
  auto varbase = imperative::VarBase(var);
  // call hook
  PyObject *origin = py::cast(varbase).ptr();
  PyObject *result = PyObject_CallFunctionObjArgs(hook, origin, nullptr);

  // check result
  PADDLE_ENFORCE_NOT_NULL(
      result, platform::errors::NotFound("The output of hook is nullptr."));
  if (result == Py_None) {
    VLOG(3) << "hook result is None.";
    return var;
  }
  CheckHookResult(origin, result, hook);

  // replace var
  auto *result_var = (imperative::VarBase *)result;
  return result_var->SharedVar();
}

}  // namespace pybind
}  // namespace paddle
