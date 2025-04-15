/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/pybind/sot/guards.h"
#include <optional>
#include "paddle/phi/api/include/tensor.h"

#if SOT_IS_SUPPORTED

#include <Python.h>
#include <frameobject.h>
#include <object.h>
#include "pybind11/numpy.h"

#if !defined(PyObject_CallOneArg) && !PY_3_9_PLUS
static inline PyObject* PyObject_CallOneArg(PyObject* func, PyObject* arg) {
  return PyObject_CallFunctionObjArgs(func, arg, NULL);
}
#endif

#if !PY_3_10_PLUS
#define Py_IsNone(x) ((x) == Py_None)
#endif

static inline bool PyObject_Equal(PyObject* a, PyObject* b) {
  if (a == b) {
    return true;
  }
  if (Py_TYPE(a) != Py_TYPE(b)) {
    return false;
  }
  int result = PyObject_RichCompareBool(a, b, Py_EQ);
  // Check for exception
  if (result == -1) {
    PyErr_Clear();
    return false;
  }
  return result;
}

std::optional<paddle::Tensor> GetTensorFromPyObject(PyObject* obj) {
  if (!paddle::pybind::PyCheckTensor(obj)) {
    // TODO(zrr1999): PyCheckTensor only check if the object is a p_tensor_type.
    return std::nullopt;
  }
  return reinterpret_cast<paddle::pybind::TensorObject*>(obj)->tensor;
}

bool LambdaGuard::check(PyObject* value) {
  PyObject* x = PyObject_CallOneArg(guard_check_fn_, value);
  if (x == nullptr) {
    PyErr_Clear();
    return false;
  }
  bool ret = PyObject_IsTrue(x);
  Py_DECREF(x);
  return ret;
}

bool GuardGroup::check(PyObject* value) {
  for (auto& guard : guards_) {
    if (!guard->check(value)) {
      return false;
    }
  }
  return true;
}

bool TypeMatchGuard::check(PyObject* value) {
  return Py_TYPE(value) == expected_;
}

bool IdMatchGuard::check(PyObject* value) { return value == expected_; }

bool ValueMatchGuard::check(PyObject* value) {
  return PyObject_Equal(value, expected_value_);
}

bool LengthMatchGuard::check(PyObject* value) {
  if (PySequence_Check(value)) {
    return PySequence_Size(value) == expected_;
  }
  if (PyMapping_Check(value)) {
    return PyMapping_Size(value) == expected_;
  }
  return false;
}

bool DtypeMatchGuard::check(PyObject* value) {
  auto tensor = GetTensorFromPyObject(value);
  if (!tensor) {
    return false;
  }
  auto dtype = tensor->type();
  return phi::TransToProtoVarType(dtype) == expected_;
}

bool ShapeMatchGuard::check(PyObject* value) {
  auto tensor = GetTensorFromPyObject(value);
  if (!tensor) {
    return false;
  }
  auto shape = tensor->shape();
  if (shape.size() != expected_.size()) {
    return false;
  }
  for (size_t i = 0; i < shape.size(); ++i) {
    if (expected_[i] && shape[i] != *expected_[i]) {
      return false;
    }
  }
  return true;
}

bool AttributeMatchGuard::check(PyObject* value) {
  PyObject* attr = PyObject_GetAttrString(value, attr_name_.c_str());
  return PyObject_Equal(attr, attr_ptr_);
}

bool LayerMatchGuard::check(PyObject* value) {
  if (value != layer_ptr_) {
    return false;
  }
  PyObject* training = PyObject_GetAttrString(value, "training");
  return (training == Py_True) == training_;
}

bool InstanceCheckGuard::check(PyObject* value) {
  return PyObject_IsInstance(value, expected_);
}

bool NumPyDtypeMatchGuard::check(PyObject* value) {
  if (value == nullptr) {
    return false;
  }

  // TODO(dev): encountered a compilation error: "declared with greater
  // visibility than the type of its field", so had to put the conversion here
  py::dtype expected_dtype = py::cast<py::dtype>(expected_);

  if (py::isinstance<py::array>(value)) {
    return py::cast<py::array>(value).dtype().is(expected_dtype);
  }

  return expected_dtype.equal(py::handle(value).get_type());
}

bool NumPyArrayValueMatchGuard::check(PyObject* value) {
  if (value == nullptr) {
    return false;
  }

  py::object py_value = py::cast<py::object>(value);
  return py::cast<py::object>(expected_)
      .attr("__eq__")(py_value)
      .attr("all")()
      .cast<bool>();
}

bool WeakRefMatchGuard::check(PyObject* value) {
  if (value == nullptr || expected_ == nullptr || Py_IsNone(expected_)) {
    return false;
  }

#if PY_3_13_PLUS
  PyObject* ref = NULL;
  int get_ref_result = PyWeakref_GetRef(expected_, &ref);
  if (get_ref_result == -1) {
    // error
    PyErr_Print();
    return false;
  }
  if (get_ref_result == 0) {
    // is dead
    return false;
  }
  bool res = PyObject_Equal(value, ref);
  Py_DECREF(ref);
  return res;
#else
  return PyObject_Equal(value, PyWeakref_GetObject(expected_));
#endif
}

PyObject* ConstantExprNode::eval(FrameProxy* frame) { return value_ptr_; }
std::string ConstantExprNode::stringify() { return py::str(value_ptr_); }

PyObject* ExternVarExprNode::eval(FrameProxy* frame) { return value_ptr_; }
std::string ExternVarExprNode::stringify() { return var_name_; }

PyObject* LocalVarExprNode::eval(FrameProxy* frame) {
#if PY_3_13_PLUS
  return PyDict_GetItemString(frame->locals, var_name_.c_str());
#elif PY_3_11_PLUS
  return PyDict_GetItemString(frame->frame->f_locals, var_name_.c_str());
#else
  return PyDict_GetItemString(frame->f_locals, var_name_.c_str());
#endif
}
std::string LocalVarExprNode::stringify() {
  return "locals[" + var_name_ + "]";
}

PyObject* GlobalVarExprNode::eval(FrameProxy* frame) {
#if PY_3_11_PLUS
  return PyDict_GetItemString(frame->frame->f_globals, var_name_.c_str());
#else
  return PyDict_GetItemString(frame->f_globals, var_name_.c_str());
#endif
}
std::string GlobalVarExprNode::stringify() {
  return "globals[" + var_name_ + "]";
}

PyObject* AttributeExprNode::eval(FrameProxy* frame) {
  PyObject* var = var_expr_->eval(frame);
  return PyObject_GetAttrString(var, attr_name_.c_str());
}
std::string AttributeExprNode::stringify() {
  std::stringstream ss;
  ss << var_expr_->stringify() << "." << attr_name_;
  return ss.str();
}

PyObject* ItemExprNode::eval(FrameProxy* frame) {
  PyObject* var = var_expr_->eval(frame);
  PyObject* key = key_expr_->eval(frame);
  return PyObject_GetItem(var, key);
}
std::string ItemExprNode::stringify() {
  std::stringstream ss;
  ss << var_expr_->stringify() << "[" << key_expr_->stringify() << "]";
  return ss.str();
}

std::optional<int> GuardNode::lookup(FrameProxy* frame) {
  // TODO(zrr1999): support multiple exprs
  auto expr = exprs.back();
  auto value = expr->eval(frame);
  if (guard->check(value)) {
    if (return_cache_index.has_value()) {
      return return_cache_index.value();
    }
    for (auto& next_guard_node : next_guard_nodes) {
      auto ret = next_guard_node->lookup(frame);
      if (ret.has_value()) {
        return ret.value();
      }
    }
  }
  return std::nullopt;
}
std::string GuardNode::stringify() {
  std::stringstream ss;
  ss << guard->get_guard_name();
  ss << "(" << exprs.back()->stringify() << ")";
  return ss.str();
}

void GuardTree::add_guard_chain(
    const std::vector<std::shared_ptr<GuardNode>>& guard_chain) {
  if (guard_chain.empty()) {
    // TODO(zrr1999): empty guard nodes means that some
    // tracker.make_faster_guard is not implemented.
    return;
  }
  for (size_t i = 1; i < guard_chain.size(); ++i) {
    guard_chain[i - 1]->next_guard_nodes.push_back(guard_chain[i]);
  }
  guard_chain.back()->return_cache_index = guard_nodes_.size();
  guard_nodes_.push_back(guard_chain.front());
}

std::optional<int> GuardTree::lookup(FrameProxy* frame) {
  for (auto& guard_node : guard_nodes_) {
    auto ret = guard_node->lookup(frame);
    if (ret.has_value()) {
      return ret.value();
    }
  }
  return std::nullopt;
}
std::string GuardTree::stringify() {
  std::stringstream ss;
  for (size_t i = 0; i < guard_nodes_.size(); ++i) {
    if (i > 0) {
      ss << " and ";
    }
    ss << guard_nodes_[i]->stringify();
  }
  return ss.str();
}

#endif
