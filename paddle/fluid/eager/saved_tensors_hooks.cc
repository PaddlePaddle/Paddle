// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/saved_tensors_hooks.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"

#if !(defined(PADDLE_NO_PYTHON) && defined(PADDLE_ON_INFERENCE))
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#endif

namespace egr {
#if !(defined(PADDLE_NO_PYTHON) && defined(PADDLE_ON_INFERENCE))
PackHook::PackHook(PyObject* hook) : hook_(hook) { Py_INCREF(hook_); }

PackHook::~PackHook() {
  ::pybind11::gil_scoped_acquire gil;
  Py_DECREF(hook_);
}

PyObject* PackHook::operator()(const paddle::experimental::Tensor& tensor) {
  auto args = PyTuple_New(1);
  auto obj = paddle::pybind::ToPyObject(tensor);
  Py_INCREF(obj);
  PyTuple_SET_ITEM(args, 0, obj);
  bool grad_tmp = egr::Controller::Instance().HasGrad();
  egr::Controller::Instance().SetHasGrad(false);
  PyObject* ret = nullptr;
  {
    ::pybind11::gil_scoped_acquire gil;
    ret = PyObject_Call(hook_, args, nullptr);
    Py_XDECREF(args);
  }
  egr::Controller::Instance().SetHasGrad(grad_tmp);
  return ret;
}

PyObject* PackHook::operator()(PyObject* py_tensor) {
  auto args = PyTuple_New(1);
  Py_INCREF(py_tensor);
  PyTuple_SET_ITEM(args, 0, py_tensor);
  bool grad_tmp = egr::Controller::Instance().HasGrad();
  egr::Controller::Instance().SetHasGrad(false);
  PyObject* ret = nullptr;
  {
    ::pybind11::gil_scoped_acquire gil;
    ret = PyObject_Call(hook_, args, nullptr);
    Py_XDECREF(args);
  }

  egr::Controller::Instance().SetHasGrad(grad_tmp);
  return ret;
}

UnPackHook::UnPackHook(PyObject* hook) : hook_(hook) { Py_INCREF(hook_); }

UnPackHook::~UnPackHook() {
  ::pybind11::gil_scoped_acquire gil;
  Py_DECREF(hook_);
}

paddle::experimental::Tensor UnPackHook::operator()(PyObject* packed_value) {
  auto args = PyTuple_New(1);
  Py_INCREF(packed_value);
  PyTuple_SET_ITEM(args, 0, packed_value);
  bool grad_tmp = egr::Controller::Instance().HasGrad();
  egr::Controller::Instance().SetHasGrad(false);
  PyObject* ret = nullptr;
  ::pybind11::gil_scoped_acquire gil;
  ret = PyObject_Call(hook_, args, nullptr);
  Py_XDECREF(args);

  egr::Controller::Instance().SetHasGrad(grad_tmp);

  PADDLE_ENFORCE_EQ(paddle::pybind::IsEagerTensor(ret),
                    true,
                    paddle::platform::errors::InvalidArgument(
                        "paddle.autograd.SavedTensorsHooks only one pair "
                        "of hooks is allowed at a time."));

  auto tensor = reinterpret_cast<paddle::pybind::TensorObject*>(ret)->tensor;
  Py_XDECREF(ret);
  return tensor;
}

PyObject* UnPackHook::operator()(PyObject* packed_value, PyObject* other) {
  auto args = PyTuple_New(1);
  Py_INCREF(packed_value);
  PyTuple_SET_ITEM(args, 0, packed_value);
  bool grad_tmp = egr::Controller::Instance().HasGrad();
  egr::Controller::Instance().SetHasGrad(false);
  PyObject* ret = nullptr;
  {
    ::pybind11::gil_scoped_acquire gil;
    ret = PyObject_Call(hook_, args, nullptr);
    Py_XDECREF(args);
  }

  egr::Controller::Instance().SetHasGrad(grad_tmp);

  PADDLE_ENFORCE_EQ(paddle::pybind::IsEagerTensor(ret),
                    true,
                    paddle::platform::errors::InvalidArgument(
                        "paddle.autograd.SavedTensorsHooks only one pair "
                        "of hooks is allowed at a time."));

  return ret;
}
#endif

}  // namespace egr
