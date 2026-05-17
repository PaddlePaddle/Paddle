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
#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/phi/api/include/tensor.h"

#if SOT_IS_SUPPORTED

#include <Python.h>
#include <frameobject.h>
#include <object.h>
#include "pybind11/numpy.h"
#if PY_3_14_PLUS
#include <internal/pycore_interpframe.h>
#endif

#if !PY_3_10_PLUS
#define Py_IsNone(x) ((x) == Py_None)
#endif

// check if the tensor is null, tensor is std::optional<paddle::Tensor>
#define HANDLE_NULL_TENSOR(tensor) \
  {                                \
    if (!tensor) {                 \
      return false;                \
    }                              \
  }

// check if the value is null and decref it
#define HANDLE_NULL_VALUE_DECREF(value) \
  {                                     \
    if ((value) == NULL) {              \
      Py_DECREF(value);                 \
      PyErr_Clear();                    \
      return false;                     \
    }                                   \
  }

// check if the value is null
#define HANDLE_NULL_VALUE(value) \
  {                              \
    if ((value) == NULL) {       \
      PyErr_Clear();             \
      return false;              \
    }                            \
  }

constexpr uint8_t kLayerNodeAttrLengthCheck = 1 << 0;
constexpr uint8_t kLayerNodeSelfLengthCheck = 1 << 1;
constexpr uint8_t kLayerNodeAttrValueCheck = 1 << 2;
constexpr uint8_t kLayerNodeMethodCodeCheck = 1 << 3;
constexpr uint8_t kLayerNodeClassWeakRefCheck = 1 << 4;
constexpr uint8_t kLayerNodeNeedsDict = kLayerNodeAttrLengthCheck |
                                        kLayerNodeAttrValueCheck |
                                        kLayerNodeMethodCodeCheck;

template <typename T>
static inline bool check_shape(
    const std::vector<std::optional<int64_t>>& expected,
    int ndim,
    const T& actual_shape,
    int64_t min_non_specialized_number) {
  if (expected.size() != static_cast<size_t>(ndim)) {
    return false;
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (!expected[i]) {
      // For dynamic dim check
      // Check the inherent constraint for dynamic dim
      // i.e. Ge(min_non_specialized_number)
      if (actual_shape[i] < min_non_specialized_number) return false;
    } else {
      // For static dim check, need exactly match
      if (actual_shape[i] != expected[i].value()) return false;
    }
  }
  return true;
}

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

static inline bool PyObject_RichEqual(PyObject* a, PyObject* b) {
  int result = PyObject_RichCompareBool(a, b, Py_EQ);
  if (result == -1) {
    PyErr_Clear();
    return false;
  }
  return result == 1;
}

static inline void AppendInt64VectorKey(std::stringstream& ss,
                                        const std::vector<int64_t>& values) {
  ss << "[";
  for (const auto& value : values) {
    ss << value << ",";
  }
  ss << "]";
}

std::optional<paddle::Tensor> GetTensorFromPyObject(PyObject* obj) {
  if (!paddle::pybind::PyCheckTensor(obj)) {
    // TODO(zrr1999): PyCheckTensor only check if the object is a p_tensor_type.
    return std::nullopt;
  }
  return reinterpret_cast<paddle::pybind::TensorObject*>(obj)->tensor;
}

static inline PyObject* GetAttrFast(
    PyObject* obj,
    const std::string& name,
    PyObject* name_object = nullptr,
    CompiledGuardAttrKind attr_kind = CompiledGuardAttrKind::GENERIC);
static inline PyObject* GetItemFast(PyObject* obj,
                                    PyObject* key,
                                    Py_hash_t key_hash = -1);
static inline Py_ssize_t GetDictSizeFast(PyObject* obj);
static inline uint64_t GetDictVersion(PyObject* obj);
static inline bool IsDictVersionUnchanged(PyObject** dict_ptr,
                                          uint64_t expected_version);
static inline bool IsDictItemAbsent(PyObject** dict_ptr,
                                    PyObject* key,
                                    Py_hash_t key_hash);
static inline bool CheckTrainingAttr(PyObject* obj, bool expected);
static inline PyObject* GetInstanceDictItem(PyObject* obj,
                                            PyObject* key,
                                            Py_hash_t key_hash = -1);
static inline PyObject* GetFunctionFieldFromObjectAttrFast(
    PyObject* obj,
    const std::string& method_name,
    PyObject* method_name_object,
    Py_hash_t method_name_hash,
    const std::string& field_name);

static inline PyObject* GetFrameLocals(FrameProxy* frame) {
#if PY_3_13_PLUS
  return frame->locals;
#elif PY_3_11_PLUS
  return frame->frame->f_locals;
#else
  return frame->f_locals;
#endif
}

static inline PyObject* GetFrameGlobals(FrameProxy* frame) {
#if PY_3_11_PLUS
  return frame->frame->f_globals;
#else
  return frame->f_globals;
#endif
}

static inline PyObject* GetFrameBuiltins(FrameProxy* frame) {
#if PY_3_11_PLUS
  return frame->frame->f_builtins;
#else
  return frame->f_builtins;
#endif
}

static inline Py_hash_t GetObjectHashNoError(PyObject* key) {
  Py_hash_t hash = PyObject_Hash(key);
  if (hash == -1) {
    PyErr_Clear();
  }
  return hash;
}

static inline PyObject* GetDictItemWithHash(PyObject* dict,
                                            PyObject* key,
                                            Py_hash_t hash) {
  if (hash != -1) {
    return _PyDict_GetItem_KnownHash(dict, key, hash);
  }
  return PyDict_GetItemWithError(dict, key);
}

static inline PyObject* GetMappingItem(PyObject* obj,
                                       PyObject* key,
                                       const std::string& name,
                                       Py_hash_t key_hash) {
  if (obj == nullptr) {
    return nullptr;
  }
  PyObject* value = nullptr;
  if (PyDict_Check(obj) && key != nullptr) {
    value = GetDictItemWithHash(obj, key, key_hash);
    if (value == nullptr) {
      PyErr_Clear();
      return nullptr;
    }
    Py_XINCREF(value);
    return value;
  }
  if (key != nullptr) {
    value = PyObject_GetAttr(obj, key);
  } else {
    value = PyObject_GetAttrString(obj, name.c_str());
  }
  if (value == nullptr) {
    PyErr_Clear();
  }
  return value;
}

struct AccessLookup {
  PyObject* value{nullptr};
  bool owned{false};
};

static inline AccessLookup GetMappingItemCached(PyObject* obj,
                                                PyObject* key,
                                                const std::string& name,
                                                Py_hash_t key_hash) {
  if (obj == nullptr) {
    return {};
  }
  PyObject* value = nullptr;
  if (PyDict_Check(obj) && key != nullptr) {
    value = GetDictItemWithHash(obj, key, key_hash);
    if (value == nullptr) {
      PyErr_Clear();
    }
    return {value, false};
  }
  if (key != nullptr) {
    value = PyObject_GetAttr(obj, key);
  } else {
    value = PyObject_GetAttrString(obj, name.c_str());
  }
  if (value == nullptr) {
    PyErr_Clear();
    return {};
  }
  return {value, true};
}

static inline PyObject* GetInstanceDictItemBorrowed(PyObject* obj,
                                                    PyObject* key,
                                                    Py_hash_t key_hash = -1) {
  if (key == nullptr) {
    return nullptr;
  }
  PyObject** dict_ptr = _PyObject_GetDictPtr(obj);
  if (dict_ptr == nullptr || *dict_ptr == nullptr) {
    PyErr_Clear();
    return nullptr;
  }
  PyObject* dict = *dict_ptr;
  if (!PyDict_Check(dict)) {
    return nullptr;
  }
  PyObject* value = GetDictItemWithHash(dict, key, key_hash);
  if (value == nullptr) {
    PyErr_Clear();
  }
  return value;
}

static inline PyObject* GetInstanceDictItem(PyObject* obj,
                                            PyObject* key,
                                            Py_hash_t key_hash) {
  PyObject* value = GetInstanceDictItemBorrowed(obj, key, key_hash);
  Py_XINCREF(value);
  return value;
}

static inline PyObject* GetTensorStopGradientBorrowed(PyObject* obj) {
  if (!paddle::pybind::PyCheckTensor(obj)) {
    return nullptr;
  }
  auto& tensor = reinterpret_cast<paddle::pybind::TensorObject*>(obj)->tensor;
  auto* meta = egr::EagerUtils::autograd_meta(&tensor);
  return meta->StopGradient() ? Py_True : Py_False;
}

static inline AccessLookup GetAttrCached(PyObject* obj,
                                         const std::string& name,
                                         PyObject* name_object,
                                         Py_hash_t name_hash,
                                         CompiledGuardAttrKind attr_kind) {
  switch (attr_kind) {
    case CompiledGuardAttrKind::TRAINING:
    case CompiledGuardAttrKind::SUB_LAYERS:
    case CompiledGuardAttrKind::FORWARD_PRE_HOOKS:
    case CompiledGuardAttrKind::FORWARD_POST_HOOKS: {
      PyObject* value =
          GetInstanceDictItemBorrowed(obj, name_object, name_hash);
      if (value != nullptr) {
        return {value, false};
      }
      break;
    }
    case CompiledGuardAttrKind::FUNC:
      if (PyMethod_Check(obj)) {
        return {PyMethod_GET_FUNCTION(obj), false};
      }
      break;
    case CompiledGuardAttrKind::CODE: {
      PyObject* func = obj;
      if (PyMethod_Check(obj)) {
        func = PyMethod_GET_FUNCTION(obj);
      }
      if (PyFunction_Check(func)) {
        return {PyFunction_GET_CODE(func), false};
      }
      break;
    }
    case CompiledGuardAttrKind::GLOBALS: {
      PyObject* func = obj;
      if (PyMethod_Check(obj)) {
        func = PyMethod_GET_FUNCTION(obj);
      }
      if (PyFunction_Check(func)) {
        return {PyFunction_GET_GLOBALS(func), false};
      }
      break;
    }
    case CompiledGuardAttrKind::STOP_GRADIENT: {
      PyObject* value = GetTensorStopGradientBorrowed(obj);
      if (value != nullptr) {
        return {value, false};
      }
      break;
    }
    case CompiledGuardAttrKind::GENERIC:
    case CompiledGuardAttrKind::CALL:
    case CompiledGuardAttrKind::FORWARD:
      break;
  }
  if (PyModule_Check(obj) && name_object != nullptr) {
    PyObject* dict = PyModule_GetDict(obj);
    if (dict != nullptr && PyDict_Check(dict)) {
      PyObject* value = GetDictItemWithHash(dict, name_object, name_hash);
      if (value != nullptr) {
        return {value, false};
      }
      PyErr_Clear();
    }
  }
  PyObject* value = nullptr;
  if (name_object != nullptr) {
    value = PyObject_GetAttr(obj, name_object);
  } else {
    value = PyObject_GetAttrString(obj, name.c_str());
  }
  if (value == nullptr) {
    PyErr_Clear();
    return {};
  }
  return {value, true};
}

static inline AccessLookup GetItemCached(PyObject* obj,
                                         PyObject* key,
                                         Py_hash_t key_hash) {
  if (PyDict_Check(obj)) {
    PyObject* value = GetDictItemWithHash(obj, key, key_hash);
    if (value == nullptr) {
      PyErr_Clear();
    }
    return {value, false};
  }
  PyObject* value = PyObject_GetItem(obj, key);
  if (value == nullptr) {
    PyErr_Clear();
    return {};
  }
  return {value, true};
}

static inline PyObject* TrainingNameObject() {
  static PyObject* training = PyUnicode_InternFromString("training");
  return training;
}

static inline Py_hash_t TrainingNameHash() {
  static Py_hash_t training_hash = GetObjectHashNoError(TrainingNameObject());
  return training_hash;
}

static inline PyObject* SubLayersNameObject() {
  static PyObject* sub_layers = PyUnicode_InternFromString("_sub_layers");
  return sub_layers;
}

static inline Py_hash_t SubLayersNameHash() {
  static Py_hash_t sub_layers_hash =
      GetObjectHashNoError(SubLayersNameObject());
  return sub_layers_hash;
}

static inline PyObject* GetAttrFast(PyObject* obj,
                                    const std::string& name,
                                    PyObject* name_object,
                                    CompiledGuardAttrKind attr_kind) {
  switch (attr_kind) {
    case CompiledGuardAttrKind::TRAINING:
    case CompiledGuardAttrKind::SUB_LAYERS:
    case CompiledGuardAttrKind::FORWARD_PRE_HOOKS:
    case CompiledGuardAttrKind::FORWARD_POST_HOOKS: {
      PyObject* value = GetInstanceDictItem(obj, name_object);
      if (value != nullptr) {
        return value;
      }
      break;
    }
    case CompiledGuardAttrKind::FUNC:
      if (PyMethod_Check(obj)) {
        PyObject* func = PyMethod_GET_FUNCTION(obj);
        Py_INCREF(func);
        return func;
      }
      break;
    case CompiledGuardAttrKind::CODE: {
      PyObject* func = obj;
      if (PyMethod_Check(obj)) {
        func = PyMethod_GET_FUNCTION(obj);
      }
      if (PyFunction_Check(func)) {
        PyObject* code = PyFunction_GET_CODE(func);
        Py_INCREF(code);
        return code;
      }
      break;
    }
    case CompiledGuardAttrKind::GLOBALS: {
      PyObject* func = obj;
      if (PyMethod_Check(obj)) {
        func = PyMethod_GET_FUNCTION(obj);
      }
      if (PyFunction_Check(func)) {
        PyObject* globals = PyFunction_GET_GLOBALS(func);
        Py_INCREF(globals);
        return globals;
      }
      break;
    }
    case CompiledGuardAttrKind::STOP_GRADIENT: {
      PyObject* value = GetTensorStopGradientBorrowed(obj);
      if (value != nullptr) {
        Py_INCREF(value);
        return value;
      }
      break;
    }
    case CompiledGuardAttrKind::GENERIC:
    case CompiledGuardAttrKind::CALL:
    case CompiledGuardAttrKind::FORWARD:
      break;
  }
  if (PyModule_Check(obj) && name_object != nullptr) {
    PyObject* dict = PyModule_GetDict(obj);
    if (dict != nullptr && PyDict_Check(dict)) {
      PyObject* value = GetDictItemWithHash(dict, name_object, -1);
      if (value != nullptr) {
        Py_INCREF(value);
        return value;
      }
      PyErr_Clear();
    }
  }
  PyObject* value = nullptr;
  if (name_object != nullptr) {
    value = PyObject_GetAttr(obj, name_object);
  } else {
    value = PyObject_GetAttrString(obj, name.c_str());
  }
  if (value == nullptr) {
    PyErr_Clear();
  }
  return value;
}

static inline PyObject* GetFunctionFieldFromObjectAttrFast(
    PyObject* obj,
    const std::string& method_name,
    PyObject* method_name_object,
    Py_hash_t method_name_hash,
    const std::string& field_name) {
  if (GetInstanceDictItemBorrowed(obj, method_name_object, method_name_hash) !=
      nullptr) {
    return nullptr;
  }

  PyObject* descriptor = _PyType_Lookup(
      reinterpret_cast<PyTypeObject*>(Py_TYPE(obj)), method_name_object);
  if (descriptor == nullptr || !PyFunction_Check(descriptor)) {
    return nullptr;
  }

  PyObject* result = nullptr;
  if (field_name == "__code__") {
    result = PyFunction_GET_CODE(descriptor);
  } else if (field_name == "__globals__") {
    result = PyFunction_GET_GLOBALS(descriptor);
  }
  return result;
}

static inline bool CheckTrainingAttr(PyObject* obj, bool expected) {
  PyObject* training_name = TrainingNameObject();
  PyObject* value =
      GetInstanceDictItemBorrowed(obj, training_name, TrainingNameHash());
  if (value != nullptr) {
    return (value == Py_True) == expected;
  }
  value = GetAttrFast(
      obj, "training", training_name, CompiledGuardAttrKind::TRAINING);
  if (value == nullptr) {
    PyErr_Clear();
    return false;
  }
  bool result = (value == Py_True) == expected;
  Py_DECREF(value);
  return result;
}

static inline bool CheckLayerMatchValue(PyObject* value,
                                        PyObject* expected,
                                        PyObject** expected_layer_dict_ptr,
                                        uint64_t expected_layer_dict_version,
                                        bool expected_bool) {
  if (value != expected) {
    return false;
  }
  if (IsDictVersionUnchanged(expected_layer_dict_ptr,
                             expected_layer_dict_version)) {
    return true;
  }
  if (expected_layer_dict_ptr != nullptr &&
      *expected_layer_dict_ptr != nullptr &&
      PyDict_Check(*expected_layer_dict_ptr)) {
    PyObject* training = GetDictItemWithHash(
        *expected_layer_dict_ptr, TrainingNameObject(), TrainingNameHash());
    if (training != nullptr) {
      return (training == Py_True) == expected_bool;
    }
    PyErr_Clear();
  }
  return CheckTrainingAttr(value, expected_bool);
}

static inline bool CheckLengthValue(PyObject* value,
                                    Py_ssize_t expected_length,
                                    bool require_dict_length) {
  if (require_dict_length) {
    return PyDict_Check(value) && GetDictSizeFast(value) == expected_length;
  }
  if (PyDict_Check(value)) {
    return GetDictSizeFast(value) == expected_length;
  }
  if (PySequence_Check(value)) {
    return PySequence_Size(value) == expected_length;
  }
  if (PyMapping_Check(value)) {
    return PyMapping_Size(value) == expected_length;
  }
  return false;
}

static inline PyObject* GetItemFast(PyObject* obj,
                                    PyObject* key,
                                    Py_hash_t key_hash) {
  if (PyDict_Check(obj)) {
    PyObject* value = GetDictItemWithHash(obj, key, key_hash);
    if (value != nullptr) {
      Py_INCREF(value);
      return value;
    }
    PyErr_Clear();
    return nullptr;
  }
  PyObject* value = PyObject_GetItem(obj, key);
  if (value == nullptr) {
    PyErr_Clear();
  }
  return value;
}

static inline Py_ssize_t GetDictSizeFast(PyObject* obj) {
  return PyDict_GET_SIZE(obj);
}

static inline uint64_t GetDictVersion(PyObject* obj) {
  if (obj == nullptr || !PyDict_Check(obj)) {
    return 0;
  }
#if PY_3_14_PLUS
  return 0;
#else
#if PY_3_12_PLUS && defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
  auto version = reinterpret_cast<PyDictObject*>(obj)->ma_version_tag;
#if PY_3_12_PLUS && defined(__clang__)
#pragma clang diagnostic pop
#endif
  return version;
#endif
}

static inline bool IsDictVersionUnchanged(PyObject** dict_ptr,
                                          uint64_t expected_version) {
  return expected_version != 0 && dict_ptr != nullptr && *dict_ptr != nullptr &&
         PyDict_Check(*dict_ptr) &&
         GetDictVersion(*dict_ptr) == expected_version;
}

static inline bool IsDictItemAbsent(PyObject** dict_ptr,
                                    PyObject* key,
                                    Py_hash_t key_hash) {
  if (dict_ptr == nullptr || *dict_ptr == nullptr || !PyDict_Check(*dict_ptr)) {
    return false;
  }
  PyObject* value = GetDictItemWithHash(*dict_ptr, key, key_hash);
  if (value != nullptr) {
    return false;
  }
  PyErr_Clear();
  return true;
}

static inline int ParseDtype(py::handle dtype) {
  try {
    return phi::TransToProtoVarType(dtype.cast<phi::DataType>());
  } catch (const py::cast_error&) {
    PyErr_Clear();
  }
  try {
    return dtype.cast<paddle::framework::proto::VarType>().type();
  } catch (const py::cast_error&) {
    PyErr_Clear();
  }
  throw py::type_error("CompiledGuard received unsupported dtype object");
}

static inline bool CheckIsNotDenseTensorHoldAllocation(PyObject* value) {
  auto tensor = GetTensorFromPyObject(value);
  HANDLE_NULL_TENSOR(tensor);

  if (!tensor->defined() ||
      (!tensor->is_dense_tensor() && !tensor->is_dist_tensor())) {
    return true;
  }

  PyObject* method =
      PyObject_GetAttrString(value, "_is_dense_tensor_hold_allocation");
  if (!method) {
    PyErr_Clear();
    return false;
  }

  if (!PyCallable_Check(method)) {
    Py_DECREF(method);
    PyErr_Clear();
    return false;
  }

  PyObject* result = PyObject_CallOneArg(method, value);
  Py_DECREF(method);
  if (result == nullptr) {
    PyErr_Clear();
    return false;
  }
  int truthy = PyObject_IsTrue(result);
  Py_DECREF(result);
  if (truthy == -1) {
    PyErr_Clear();
    return false;
  }
  return !static_cast<bool>(truthy);
}

static inline bool CheckWeakRefMatch(PyObject* value, PyObject* weak_ref) {
  if (value == nullptr || weak_ref == nullptr || Py_IsNone(weak_ref)) {
    return false;
  }

#if PY_3_13_PLUS
  PyObject* ref = nullptr;
  int get_ref_result = PyWeakref_GetRef(weak_ref, &ref);
  if (get_ref_result == -1) {
    PyErr_Clear();
    return false;
  }
  if (get_ref_result == 0) {
    return false;
  }
  bool result = value == ref || PyObject_RichEqual(value, ref);
  Py_DECREF(ref);
  return result;
#else
  PyObject* ref = PyWeakref_GetObject(weak_ref);
  if (ref == nullptr || Py_IsNone(ref)) {
    PyErr_Clear();
    return false;
  }
  return value == ref || PyObject_RichEqual(value, ref);
#endif
}

CompiledGuard::AccessCache::AccessCache(CompiledGuard* guard) : guard(guard) {
  if (guard->access_cache_values_.size() < guard->access_nodes_.size()) {
    guard->access_cache_values_.resize(guard->access_nodes_.size(), nullptr);
    guard->access_cache_evaluated_.resize(guard->access_nodes_.size(), 0);
    guard->access_cache_owned_.resize(guard->access_nodes_.size(), 0);
  }
  ++guard->access_cache_generation_;
  if (guard->access_cache_generation_ == 0) {
    std::fill(guard->access_cache_evaluated_.begin(),
              guard->access_cache_evaluated_.end(),
              0);
    guard->access_cache_generation_ = 1;
  }
  guard->access_cache_touched_.clear();
}

CompiledGuard::AccessCache::~AccessCache() {
  for (auto access_id : guard->access_cache_touched_) {
    Py_XDECREF(guard->access_cache_values_[access_id]);
    guard->access_cache_values_[access_id] = nullptr;
    guard->access_cache_owned_[access_id] = 0;
  }
  guard->access_cache_touched_.clear();
}

std::vector<CompiledGuard::AccessStep> CompiledGuard::ParseAccessPath(
    py::handle access) {
  std::vector<AccessStep> result;
  for (auto step_handle : py::reinterpret_borrow<py::iterable>(access)) {
    py::tuple step = py::reinterpret_borrow<py::tuple>(step_handle);
    if (py::len(step) != 2) {
      throw py::value_error("compiled guard access step must be a pair");
    }
    std::string kind = step[0].cast<std::string>();
    AccessStep access_step;
    access_step.name_object = py::none();
    access_step.value = py::none();
    if (kind == "local") {
      access_step.kind = AccessKind::LOCAL;
      access_step.name = step[1].cast<std::string>();
    } else if (kind == "global") {
      access_step.kind = AccessKind::GLOBAL;
      access_step.name = step[1].cast<std::string>();
    } else if (kind == "builtin") {
      access_step.kind = AccessKind::BUILTIN;
      access_step.name = step[1].cast<std::string>();
    } else if (kind == "const") {
      access_step.kind = AccessKind::CONSTANT;
      access_step.value = py::reinterpret_borrow<py::object>(step[1]);
    } else if (kind == "attr") {
      access_step.kind = AccessKind::ATTR;
      access_step.name = step[1].cast<std::string>();
      if (access_step.name == "training") {
        access_step.attr_kind = CompiledGuardAttrKind::TRAINING;
      } else if (access_step.name == "_sub_layers") {
        access_step.attr_kind = CompiledGuardAttrKind::SUB_LAYERS;
      } else if (access_step.name == "_forward_pre_hooks") {
        access_step.attr_kind = CompiledGuardAttrKind::FORWARD_PRE_HOOKS;
      } else if (access_step.name == "_forward_post_hooks") {
        access_step.attr_kind = CompiledGuardAttrKind::FORWARD_POST_HOOKS;
      } else if (access_step.name == "__func__") {
        access_step.attr_kind = CompiledGuardAttrKind::FUNC;
      } else if (access_step.name == "__code__") {
        access_step.attr_kind = CompiledGuardAttrKind::CODE;
      } else if (access_step.name == "__globals__") {
        access_step.attr_kind = CompiledGuardAttrKind::GLOBALS;
      } else if (access_step.name == "__call__") {
        access_step.attr_kind = CompiledGuardAttrKind::CALL;
      } else if (access_step.name == "forward") {
        access_step.attr_kind = CompiledGuardAttrKind::FORWARD;
      } else if (access_step.name == "stop_gradient") {
        access_step.attr_kind = CompiledGuardAttrKind::STOP_GRADIENT;
      }
    } else if (kind == "item") {
      access_step.kind = AccessKind::ITEM;
      access_step.value = py::reinterpret_borrow<py::object>(step[1]);
      access_step.key_hash = GetObjectHashNoError(access_step.value.ptr());
    } else {
      throw py::value_error("unknown compiled guard access step: " + kind);
    }
    if (access_step.kind == AccessKind::LOCAL ||
        access_step.kind == AccessKind::GLOBAL ||
        access_step.kind == AccessKind::BUILTIN ||
        access_step.kind == AccessKind::ATTR) {
      PyObject* name_object =
          PyUnicode_InternFromString(access_step.name.c_str());
      if (name_object == nullptr) {
        throw py::error_already_set();
      }
      access_step.name_object = py::reinterpret_steal<py::object>(name_object);
      access_step.key_hash =
          GetObjectHashNoError(access_step.name_object.ptr());
    }
    result.push_back(std::move(access_step));
  }
  if (result.empty()) {
    throw py::value_error("compiled guard access path cannot be empty");
  }
  return result;
}

std::vector<std::optional<int64_t>> CompiledGuard::ParseShape(
    py::handle shape) {
  std::vector<std::optional<int64_t>> result;
  for (auto dim : py::reinterpret_borrow<py::iterable>(shape)) {
    if (dim.is_none()) {
      result.emplace_back(std::nullopt);
      continue;
    }
    if (py::isinstance<py::int_>(dim)) {
      auto value = dim.cast<int64_t>();
      if (value >= 0) {
        result.emplace_back(value);
      } else {
        result.emplace_back(std::nullopt);
      }
      continue;
    }
    result.emplace_back(std::nullopt);
  }
  return result;
}

std::vector<int64_t> CompiledGuard::ParseInt64Vector(py::handle values,
                                                     const std::string& name) {
  std::vector<int64_t> result;
  try {
    for (auto value : py::reinterpret_borrow<py::iterable>(values)) {
      result.push_back(value.cast<int64_t>());
    }
  } catch (const py::cast_error&) {
    PyErr_Clear();
    throw py::type_error(name + " must be an integer sequence");
  } catch (const py::type_error&) {
    PyErr_Clear();
    throw py::type_error(name + " must be an integer sequence");
  }
  return result;
}

std::shared_ptr<CompiledGuard::GuardExpr> CompiledGuard::ParseExpr(
    py::handle expr) {
  py::tuple expr_tuple = py::reinterpret_borrow<py::tuple>(expr);
  if (py::len(expr_tuple) < 1) {
    throw py::value_error("compiled guard expression cannot be empty");
  }
  std::string kind = expr_tuple[0].cast<std::string>();
  auto result = std::make_shared<GuardExpr>();
  result->value = py::none();

  if (kind == "const") {
    if (py::len(expr_tuple) != 2) {
      throw py::value_error("const expression expects 2 fields");
    }
    result->kind = ExprKind::CONSTANT;
    result->value = py::reinterpret_borrow<py::object>(expr_tuple[1]);
  } else if (kind == "access") {
    if (py::len(expr_tuple) != 2) {
      throw py::value_error("access expression expects 2 fields");
    }
    result->kind = ExprKind::ACCESS;
    result->access_path = ParseAccessPath(expr_tuple[1]);
  } else if (kind == "unary") {
    if (py::len(expr_tuple) != 3) {
      throw py::value_error("unary expression expects 3 fields");
    }
    result->kind = ExprKind::UNARY;
    result->op = expr_tuple[1].cast<std::string>();
    result->lhs = ParseExpr(expr_tuple[2]);
  } else if (kind == "binary") {
    if (py::len(expr_tuple) != 4) {
      throw py::value_error("binary expression expects 4 fields");
    }
    result->kind = ExprKind::BINARY;
    result->op = expr_tuple[1].cast<std::string>();
    result->lhs = ParseExpr(expr_tuple[2]);
    result->rhs = ParseExpr(expr_tuple[3]);
  } else {
    throw py::value_error("unknown compiled guard expression: " + kind);
  }

  return result;
}

std::string CompiledGuard::AccessStepKey(const AccessStep& step) {
  std::stringstream ss;
  ss << static_cast<int>(step.kind) << ":";
  switch (step.kind) {
    case AccessKind::LOCAL:
    case AccessKind::GLOBAL:
    case AccessKind::BUILTIN:
      ss << step.name;
      break;
    case AccessKind::ATTR:
      ss << step.name << ":" << static_cast<int>(step.attr_kind);
      break;
    case AccessKind::CONSTANT:
    case AccessKind::ITEM: {
      PyObject* value = step.value.ptr();
      if (PyUnicode_Check(value)) {
        Py_ssize_t size = 0;
        const char* data = PyUnicode_AsUTF8AndSize(value, &size);
        if (data != nullptr) {
          ss << "str:" << std::string(data, static_cast<size_t>(size));
          break;
        }
        PyErr_Clear();
      }
      if (PyLong_Check(value)) {
        auto integer = PyLong_AsLongLong(value);
        if (!PyErr_Occurred()) {
          ss << "int:" << integer;
          break;
        }
        PyErr_Clear();
      }
      ss << "ptr:" << reinterpret_cast<uintptr_t>(value);
      break;
    }
  }
  return ss.str();
}

std::string CompiledGuard::GuardOpKey(const GuardOp& op) {
  std::stringstream ss;
  ss << static_cast<int>(op.kind) << ":" << op.access_id << ":";
  switch (op.kind) {
    case OpKind::GRAD_ENABLED:
      ss << op.expected_bool;
      break;
    case OpKind::TYPE_MATCH:
      ss << reinterpret_cast<uintptr_t>(op.expected_type);
      break;
    case OpKind::INSTANCE_CHECK:
    case OpKind::ID_MATCH:
    case OpKind::VALUE_MATCH:
    case OpKind::NUMPY_DTYPE:
    case OpKind::WEAKREF_MATCH:
      ss << reinterpret_cast<uintptr_t>(op.expected.ptr());
      if (op.kind == OpKind::VALUE_MATCH) {
        ss << ":" << op.value_match_by_identity;
      }
      break;
    case OpKind::LENGTH_MATCH:
      ss << op.expected_length << ":" << op.require_dict_length;
      break;
    case OpKind::LAYER_MATCH:
      ss << reinterpret_cast<uintptr_t>(op.expected.ptr()) << ":"
         << op.expected_bool;
      break;
    case OpKind::LAYER_MATCH_GROUP: {
      for (const auto& item : op.layer_match_items) {
        ss << item.access_id << ":"
           << reinterpret_cast<uintptr_t>(item.expected.ptr()) << ":"
           << item.expected_bool << ",";
      }
      break;
    }
    case OpKind::TENSOR_SHAPE:
    case OpKind::NUMPY_SHAPE:
      ss << op.min_non_specialized_number << ":";
      for (const auto& dim : op.expected_shape) {
        if (dim.has_value()) {
          ss << dim.value();
        } else {
          ss << "*";
        }
        ss << ",";
      }
      break;
    case OpKind::TENSOR_DTYPE:
      ss << op.expected_dtype;
      break;
    case OpKind::TENSOR_IS_DIST:
      ss << op.expected_bool;
      break;
    case OpKind::TENSOR_META:
      ss << op.expected_dtype << ":" << op.expected_bool << ":"
         << op.min_non_specialized_number << ":";
      for (const auto& dim : op.expected_shape) {
        if (dim.has_value()) {
          ss << dim.value();
        } else {
          ss << "*";
        }
        ss << ",";
      }
      break;
    case OpKind::TENSOR_DIST_META:
      ss << reinterpret_cast<uintptr_t>(op.expected_dist_info_from_tensor.ptr())
         << ":";
      AppendInt64VectorKey(ss, op.expected_dist_mesh_shape_values);
      ss << ":";
      AppendInt64VectorKey(ss, op.expected_dist_process_ids_values);
      ss << ":";
      AppendInt64VectorKey(ss, op.expected_dist_dims_mapping_values);
      ss << ":";
      AppendInt64VectorKey(ss, op.expected_dist_local_shape_values);
      break;
    case OpKind::TENSOR_NOT_HOLD_ALLOCATION:
      break;
    case OpKind::EXPR_MATCH:
      ss << reinterpret_cast<uintptr_t>(op.expr.get());
      break;
  }
  return ss.str();
}

std::string CompiledGuard::AccessPathKey(size_t access_id) const {
  std::vector<size_t> path_ids;
  std::optional<size_t> current = access_id;
  while (current.has_value()) {
    path_ids.push_back(current.value());
    current = access_nodes_[current.value()].parent;
  }
  std::stringstream ss;
  for (auto iter = path_ids.rbegin(); iter != path_ids.rend(); ++iter) {
    if (iter != path_ids.rbegin()) {
      ss << "/";
    }
    ss << AccessStepKey(access_nodes_[*iter].step);
  }
  return ss.str();
}

std::string CompiledGuard::LookupGuardOpKey(const GuardOp& op) const {
  std::stringstream ss;
  ss << static_cast<int>(op.kind) << ":";
  if (op.kind != OpKind::GRAD_ENABLED && op.kind != OpKind::EXPR_MATCH) {
    ss << AccessPathKey(op.access_id) << ":";
  }
  switch (op.kind) {
    case OpKind::GRAD_ENABLED:
      ss << op.expected_bool;
      break;
    case OpKind::TYPE_MATCH:
      ss << reinterpret_cast<uintptr_t>(op.expected_type);
      break;
    case OpKind::INSTANCE_CHECK:
      ss << reinterpret_cast<uintptr_t>(op.expected.ptr()) << ":"
         << op.expected_is_dict;
      break;
    case OpKind::ID_MATCH:
    case OpKind::NUMPY_DTYPE:
    case OpKind::WEAKREF_MATCH:
      ss << reinterpret_cast<uintptr_t>(op.expected.ptr());
      break;
    case OpKind::VALUE_MATCH:
      ss << reinterpret_cast<uintptr_t>(op.expected.ptr()) << ":"
         << op.value_match_by_identity;
      break;
    case OpKind::LENGTH_MATCH:
      ss << op.expected_length << ":" << op.require_dict_length;
      break;
    case OpKind::LAYER_MATCH:
      ss << reinterpret_cast<uintptr_t>(op.expected.ptr()) << ":"
         << op.expected_bool;
      break;
    case OpKind::LAYER_MATCH_GROUP: {
      const auto append_attr_length_check =
          [&](const LayerMatchItem::AttrLengthCheck& check) {
            ss << "attr_len(" << AccessPathKey(check.access_id) << ":"
               << static_cast<int>(check.attr_kind) << ":"
               << reinterpret_cast<uintptr_t>(check.name_object.ptr()) << ":"
               << check.expected_length << ":" << check.require_dict_length
               << ")";
          };
      const auto append_method_code_check =
          [&](const LayerMatchItem::MethodCodeCheck& check) {
            ss << "method_code(" << AccessPathKey(check.access_id) << ":"
               << check.method_name << ":"
               << reinterpret_cast<uintptr_t>(check.expected_code.ptr()) << ":"
               << check.expected_instance_override_absent << ")";
          };
      const auto append_self_length_check =
          [&](const LayerMatchItem::SelfLengthCheck& check) {
            ss << "self_len(" << AccessPathKey(check.access_id) << ":"
               << check.expected_length << ":" << check.require_dict_length
               << ")";
          };
      const auto append_attr_value_check =
          [&](const LayerMatchItem::AttrValueCheck& check) {
            ss << "attr_value(" << AccessPathKey(check.access_id) << ":"
               << static_cast<int>(check.attr_kind) << ":"
               << reinterpret_cast<uintptr_t>(check.name_object.ptr()) << ":"
               << reinterpret_cast<uintptr_t>(check.expected.ptr()) << ":"
               << check.value_match_by_identity << ")";
          };
      const auto append_class_weakref_check =
          [&](const LayerClassWeakRefCheck& check) {
            ss << "class_weakref(" << AccessPathKey(check.access_id) << ":"
               << reinterpret_cast<uintptr_t>(check.name_object.ptr()) << ":"
               << reinterpret_cast<uintptr_t>(check.expected.ptr()) << ")";
          };
      const auto append_tree_node = [&](const auto& self,
                                        size_t node_index) -> void {
        const auto& node = op.layer_sub_layer_tree_nodes[node_index];
        ss << "node(" << node_index << ":";
        if (node.item_index.has_value()) {
          ss << "item=" << node.item_index.value();
        } else {
          ss << "item=*";
        }
        ss << ":flags=" << static_cast<int>(node.check_flags) << ":";
        for (const auto& check : node.self_length_checks) {
          append_self_length_check(check);
        }
        for (const auto& check : node.attr_length_checks) {
          append_attr_length_check(check);
        }
        for (const auto& check : node.attr_value_checks) {
          append_attr_value_check(check);
        }
        for (const auto& check : node.method_code_checks) {
          append_method_code_check(check);
        }
        for (const auto& check : node.class_weakref_checks) {
          append_class_weakref_check(check);
        }
        for (const auto& child : node.children) {
          ss << "child(" << reinterpret_cast<uintptr_t>(child.key.ptr()) << ":"
             << child.key_hash << "->";
          self(self, child.node_index);
          ss << ")";
        }
        ss << ")";
      };

      for (const auto& item : op.layer_match_items) {
        ss << "item(" << AccessPathKey(item.access_id) << ":"
           << reinterpret_cast<uintptr_t>(item.expected.ptr()) << ":"
           << item.expected_bool << ":" << item.use_sub_layer_tree << ":"
           << item.cache_value << ":";
        for (const auto& check : item.attr_length_checks) {
          append_attr_length_check(check);
        }
        for (const auto& check : item.method_code_checks) {
          append_method_code_check(check);
        }
        ss << "),";
      }
      ss << "|method:";
      for (const auto& group : op.layer_method_code_groups) {
        ss << group.method_name << ":"
           << reinterpret_cast<uintptr_t>(group.expected_code.ptr()) << ":"
           << reinterpret_cast<uintptr_t>(group.expected_type) << ",";
      }
      ss << "|tree:";
      for (const auto& root : op.layer_sub_layer_tree_roots) {
        ss << "root(" << AccessPathKey(root.base_access_id) << ":";
        append_tree_node(append_tree_node, root.node_index);
        ss << "),";
      }
      break;
    }
    case OpKind::TENSOR_SHAPE:
    case OpKind::NUMPY_SHAPE:
      ss << op.min_non_specialized_number << ":";
      for (const auto& dim : op.expected_shape) {
        ss << (dim.has_value() ? std::to_string(dim.value()) : "*") << ",";
      }
      break;
    case OpKind::TENSOR_DTYPE:
      ss << op.expected_dtype;
      break;
    case OpKind::TENSOR_IS_DIST:
      ss << op.expected_bool;
      break;
    case OpKind::TENSOR_META:
      ss << op.expected_dtype << ":" << op.expected_bool << ":"
         << op.min_non_specialized_number << ":";
      for (const auto& dim : op.expected_shape) {
        ss << (dim.has_value() ? std::to_string(dim.value()) : "*") << ",";
      }
      break;
    case OpKind::TENSOR_DIST_META:
      ss << reinterpret_cast<uintptr_t>(op.expected_dist_info_from_tensor.ptr())
         << ":";
      AppendInt64VectorKey(ss, op.expected_dist_mesh_shape_values);
      ss << ":";
      AppendInt64VectorKey(ss, op.expected_dist_process_ids_values);
      ss << ":";
      AppendInt64VectorKey(ss, op.expected_dist_dims_mapping_values);
      ss << ":";
      AppendInt64VectorKey(ss, op.expected_dist_local_shape_values);
      break;
    case OpKind::TENSOR_NOT_HOLD_ALLOCATION:
      break;
    case OpKind::EXPR_MATCH:
      ss << reinterpret_cast<uintptr_t>(op.expr.get());
      break;
  }
  return ss.str();
}

size_t CompiledGuard::InternAccessPath(
    const std::vector<AccessStep>& access_path) {
  std::optional<size_t> parent;
  std::string key;
  size_t current = 0;
  for (const auto& step : access_path) {
    if (parent.has_value()) {
      key = std::to_string(parent.value()) + "/" + AccessStepKey(step);
    } else {
      key = AccessStepKey(step);
    }
    auto it = access_node_ids_.find(key);
    if (it == access_node_ids_.end()) {
      current = access_nodes_.size();
      access_nodes_.push_back(AccessNode{parent, step});
      access_node_ids_.emplace(key, current);
    } else {
      current = it->second;
    }
    parent = current;
  }
  return current;
}

void CompiledGuard::DeduplicateOps() {
  std::unordered_set<std::string> seen;
  std::vector<GuardOp> deduplicated;
  deduplicated.reserve(ops_.size());
  for (auto& op : ops_) {
    if (seen.insert(GuardOpKey(op)).second) {
      deduplicated.push_back(std::move(op));
    }
  }
  std::unordered_map<size_t, PyTypeObject*> value_match_types;
  for (const auto& op : deduplicated) {
    if (op.kind == OpKind::VALUE_MATCH) {
      value_match_types[op.access_id] =
          reinterpret_cast<PyTypeObject*>(Py_TYPE(op.expected.ptr()));
    }
  }

  std::vector<GuardOp> compacted;
  compacted.reserve(deduplicated.size());
  for (auto& op : deduplicated) {
    if (op.kind == OpKind::TYPE_MATCH) {
      auto iter = value_match_types.find(op.access_id);
      if (iter != value_match_types.end() && iter->second == op.expected_type) {
        continue;
      }
    }
    compacted.push_back(std::move(op));
  }
  ops_ = std::move(compacted);
}

void CompiledGuard::FuseTensorMetaOps() {
  struct TensorMetaIndices {
    std::optional<size_t> shape;
    std::optional<size_t> dtype;
    std::optional<size_t> is_dist;
  };
  std::unordered_map<size_t, TensorMetaIndices> indices;
  for (size_t i = 0; i < ops_.size(); ++i) {
    auto& entry = indices[ops_[i].access_id];
    switch (ops_[i].kind) {
      case OpKind::TENSOR_SHAPE:
        entry.shape = i;
        break;
      case OpKind::TENSOR_DTYPE:
        entry.dtype = i;
        break;
      case OpKind::TENSOR_IS_DIST:
        entry.is_dist = i;
        break;
      default:
        break;
    }
  }

  std::vector<unsigned char> removed(ops_.size(), 0);
  for (const auto& item : indices) {
    const auto& meta = item.second;
    if (!meta.shape.has_value() || !meta.dtype.has_value() ||
        !meta.is_dist.has_value()) {
      continue;
    }
    auto& fused = ops_[meta.shape.value()];
    fused.kind = OpKind::TENSOR_META;
    fused.expected_dtype = ops_[meta.dtype.value()].expected_dtype;
    fused.expected_bool = ops_[meta.is_dist.value()].expected_bool;
    removed[meta.dtype.value()] = 1;
    removed[meta.is_dist.value()] = 1;
  }

  if (std::any_of(removed.begin(), removed.end(), [](unsigned char value) {
        return value != 0;
      })) {
    std::vector<GuardOp> fused_ops;
    fused_ops.reserve(ops_.size());
    for (size_t i = 0; i < ops_.size(); ++i) {
      if (!removed[i]) {
        fused_ops.push_back(std::move(ops_[i]));
      }
    }
    ops_ = std::move(fused_ops);
  }
}

void CompiledGuard::FuseLayerMatchOps() {
  std::vector<LayerMatchItem> items;
  std::vector<std::vector<AccessStep>> item_access_paths;
  std::unordered_map<size_t, size_t> access_id_to_item;
  std::vector<LayerMethodCodeGroup> method_code_groups;
  std::unordered_map<std::string, size_t> method_code_group_ids;
  items.reserve(ops_.size());
  item_access_paths.reserve(ops_.size());
  std::optional<size_t> first_layer_match;
  std::vector<unsigned char> removed(ops_.size(), 0);

  for (size_t i = 0; i < ops_.size(); ++i) {
    auto& op = ops_[i];
    if (op.kind != OpKind::LAYER_MATCH) {
      continue;
    }
    if (!first_layer_match.has_value()) {
      first_layer_match = i;
    } else {
      removed[i] = 1;
    }

    LayerMatchItem item;
    item.access_id = op.access_id;
    item.expected = op.expected;
    item.expected_layer_dict_ptr = op.expected_layer_dict_ptr;
    item.expected_layer_dict_version = op.expected_layer_dict_version;
    item.expected_bool = op.expected_bool;
    access_id_to_item.emplace(op.access_id, items.size());
    item_access_paths.push_back(op.access_path);
    items.push_back(std::move(item));
  }

  for (size_t i = 0; i < ops_.size(); ++i) {
    auto& op = ops_[i];
    if (op.kind != OpKind::VALUE_MATCH || op.access_path.size() < 3 ||
        !PyCode_Check(op.expected.ptr())) {
      continue;
    }
    const auto& code_step = op.access_path.back();
    if (code_step.kind != AccessKind::ATTR ||
        code_step.attr_kind != CompiledGuardAttrKind::CODE) {
      continue;
    }

    std::optional<size_t> parent_access_id;
    const AccessStep* method_step = nullptr;
    if (op.access_path.size() >= 3) {
      const auto& maybe_call = op.access_path[op.access_path.size() - 2];
      if (maybe_call.kind == AccessKind::ATTR &&
          maybe_call.attr_kind == CompiledGuardAttrKind::CALL) {
        std::vector<AccessStep> parent_path(op.access_path.begin(),
                                            op.access_path.end() - 2);
        parent_access_id = InternAccessPath(parent_path);
        method_step = &maybe_call;
      }
    }
    if (!parent_access_id.has_value() && op.access_path.size() >= 4) {
      const auto& maybe_func = op.access_path[op.access_path.size() - 2];
      const auto& maybe_forward = op.access_path[op.access_path.size() - 3];
      if (maybe_func.kind == AccessKind::ATTR &&
          maybe_func.attr_kind == CompiledGuardAttrKind::FUNC &&
          maybe_forward.kind == AccessKind::ATTR &&
          maybe_forward.attr_kind == CompiledGuardAttrKind::FORWARD) {
        std::vector<AccessStep> parent_path(op.access_path.begin(),
                                            op.access_path.end() - 3);
        parent_access_id = InternAccessPath(parent_path);
        method_step = &maybe_forward;
      }
    }
    if (!parent_access_id.has_value() || method_step == nullptr) {
      continue;
    }
    auto item_iter = access_id_to_item.find(parent_access_id.value());
    if (item_iter == access_id_to_item.end()) {
      continue;
    }
    auto& item = items[item_iter->second];
    std::stringstream group_key;
    group_key << reinterpret_cast<uintptr_t>(Py_TYPE(item.expected.ptr()))
              << ":" << method_step->name << ":"
              << reinterpret_cast<uintptr_t>(op.expected.ptr());
    auto group_iter = method_code_group_ids.find(group_key.str());
    size_t group_index = 0;
    if (group_iter == method_code_group_ids.end()) {
      group_index = method_code_groups.size();
      method_code_group_ids.emplace(group_key.str(), group_index);
      method_code_groups.push_back(LayerMethodCodeGroup{
          method_step->name,
          method_step->name_object,
          op.expected,
          reinterpret_cast<PyTypeObject*>(Py_TYPE(item.expected.ptr())),
          method_step->key_hash});
    } else {
      group_index = group_iter->second;
    }
    bool expected_instance_override_absent =
        IsDictItemAbsent(item.expected_layer_dict_ptr,
                         method_step->name_object.ptr(),
                         method_step->key_hash);
    items[item_iter->second].method_code_checks.push_back(
        LayerMatchItem::MethodCodeCheck{op.access_id,
                                        group_index,
                                        method_step->name,
                                        method_step->name_object,
                                        op.expected,
                                        method_step->key_hash,
                                        expected_instance_override_absent});
    removed[i] = 1;
  }

  for (size_t i = 0; i < ops_.size(); ++i) {
    auto& op = ops_[i];
    if (op.kind != OpKind::LENGTH_MATCH || op.access_path.size() < 2) {
      continue;
    }
    const auto& attr_step = op.access_path.back();
    if (attr_step.kind != AccessKind::ATTR) {
      continue;
    }
    if (attr_step.attr_kind != CompiledGuardAttrKind::SUB_LAYERS &&
        attr_step.attr_kind != CompiledGuardAttrKind::FORWARD_PRE_HOOKS &&
        attr_step.attr_kind != CompiledGuardAttrKind::FORWARD_POST_HOOKS) {
      continue;
    }
    std::vector<AccessStep> parent_path(op.access_path.begin(),
                                        op.access_path.end() - 1);
    auto parent_access_id = InternAccessPath(parent_path);
    auto item_iter = access_id_to_item.find(parent_access_id);
    if (item_iter == access_id_to_item.end()) {
      continue;
    }
    items[item_iter->second].attr_length_checks.push_back(
        LayerMatchItem::AttrLengthCheck{op.access_id,
                                        attr_step.attr_kind,
                                        attr_step.name_object,
                                        attr_step.key_hash,
                                        op.expected_length,
                                        op.require_dict_length});
    removed[i] = 1;
  }

  struct TreeBuildNode {
    std::optional<size_t> item_index;
    std::vector<LayerMatchItem::SelfLengthCheck> self_length_checks;
    std::vector<LayerMatchItem::AttrLengthCheck> attr_length_checks;
    std::vector<LayerMatchItem::AttrValueCheck> attr_value_checks;
    std::vector<LayerMatchItem::MethodCodeCheck> method_code_checks;
    std::vector<LayerClassWeakRefCheck> class_weakref_checks;
    std::vector<LayerSubLayerTreeChild> children;
    std::unordered_map<std::string, size_t> child_ids;
  };

  using SubLayerSegment = std::pair<AccessStep, AccessStep>;
  const auto parse_sub_layer_path =
      [&](const std::vector<AccessStep>& path,
          std::vector<AccessStep>* base_path,
          std::vector<SubLayerSegment>* segments) -> bool {
    std::optional<size_t> first_sub_layers;
    for (size_t i = 0; i + 1 < path.size(); ++i) {
      if (path[i].kind == AccessKind::ATTR &&
          path[i].attr_kind == CompiledGuardAttrKind::SUB_LAYERS &&
          path[i + 1].kind == AccessKind::ITEM) {
        first_sub_layers = i;
        break;
      }
    }
    if (!first_sub_layers.has_value() || first_sub_layers.value() == 0) {
      return false;
    }
    base_path->assign(path.begin(), path.begin() + first_sub_layers.value());
    for (size_t i = first_sub_layers.value(); i < path.size();) {
      if (i + 1 >= path.size() || path[i].kind != AccessKind::ATTR ||
          path[i].attr_kind != CompiledGuardAttrKind::SUB_LAYERS ||
          path[i + 1].kind != AccessKind::ITEM) {
        return false;
      }
      segments->emplace_back(path[i], path[i + 1]);
      i += 2;
    }
    return !segments->empty();
  };

  std::vector<TreeBuildNode> tree_build_nodes;
  std::vector<LayerSubLayerTreeRoot> tree_roots;
  std::unordered_map<size_t, size_t> root_ids;
  std::unordered_map<size_t, size_t> tree_node_by_access_id;

  for (size_t item_index = 0; item_index < item_access_paths.size();
       ++item_index) {
    std::vector<AccessStep> base_path;
    std::vector<SubLayerSegment> segments;
    if (!parse_sub_layer_path(
            item_access_paths[item_index], &base_path, &segments)) {
      continue;
    }

    size_t base_access_id = InternAccessPath(base_path);
    auto root_iter = root_ids.find(base_access_id);
    size_t current_node = 0;
    if (root_iter == root_ids.end()) {
      current_node = tree_build_nodes.size();
      tree_build_nodes.emplace_back();
      root_ids.emplace(base_access_id, current_node);
      tree_roots.push_back(LayerSubLayerTreeRoot{base_access_id, current_node});
      tree_node_by_access_id.emplace(base_access_id, current_node);
    } else {
      current_node = root_iter->second;
    }

    std::vector<AccessStep> current_path = base_path;
    for (const auto& segment : segments) {
      const auto& key = segment.second;
      const auto key_string = AccessStepKey(key);
      auto child_iter =
          tree_build_nodes[current_node].child_ids.find(key_string);
      if (child_iter == tree_build_nodes[current_node].child_ids.end()) {
        size_t child_node = tree_build_nodes.size();
        tree_build_nodes.emplace_back();
        tree_build_nodes[current_node].child_ids.emplace(key_string,
                                                         child_node);
        tree_build_nodes[current_node].children.push_back(
            LayerSubLayerTreeChild{key.value, key.key_hash, child_node});
        current_node = child_node;
      } else {
        current_node = child_iter->second;
      }
      current_path.push_back(segment.first);
      current_path.push_back(segment.second);
      tree_node_by_access_id.emplace(InternAccessPath(current_path),
                                     current_node);
    }

    tree_build_nodes[current_node].item_index = item_index;
    items[item_index].use_sub_layer_tree = true;
  }

  for (size_t i = 0; i < ops_.size(); ++i) {
    if (removed[i]) {
      continue;
    }
    auto& op = ops_[i];
    if (op.kind != OpKind::VALUE_MATCH || op.access_path.size() < 3 ||
        !PyCode_Check(op.expected.ptr())) {
      continue;
    }
    const auto& code_step = op.access_path.back();
    if (code_step.kind != AccessKind::ATTR ||
        code_step.attr_kind != CompiledGuardAttrKind::CODE) {
      continue;
    }

    std::optional<size_t> parent_access_id;
    const AccessStep* method_step = nullptr;
    if (op.access_path.size() >= 3) {
      const auto& maybe_call = op.access_path[op.access_path.size() - 2];
      if (maybe_call.kind == AccessKind::ATTR &&
          maybe_call.attr_kind == CompiledGuardAttrKind::CALL) {
        std::vector<AccessStep> parent_path(op.access_path.begin(),
                                            op.access_path.end() - 2);
        parent_access_id = InternAccessPath(parent_path);
        method_step = &maybe_call;
      }
    }
    if (!parent_access_id.has_value() && op.access_path.size() >= 4) {
      const auto& maybe_func = op.access_path[op.access_path.size() - 2];
      const auto& maybe_forward = op.access_path[op.access_path.size() - 3];
      if (maybe_func.kind == AccessKind::ATTR &&
          maybe_func.attr_kind == CompiledGuardAttrKind::FUNC &&
          maybe_forward.kind == AccessKind::ATTR &&
          maybe_forward.attr_kind == CompiledGuardAttrKind::FORWARD) {
        std::vector<AccessStep> parent_path(op.access_path.begin(),
                                            op.access_path.end() - 3);
        parent_access_id = InternAccessPath(parent_path);
        method_step = &maybe_forward;
      }
    }
    if (!parent_access_id.has_value() || method_step == nullptr) {
      continue;
    }
    auto tree_iter = tree_node_by_access_id.find(parent_access_id.value());
    if (tree_iter == tree_node_by_access_id.end()) {
      continue;
    }

    PyObject* expected_parent = nullptr;
    auto item_iter = access_id_to_item.find(parent_access_id.value());
    if (item_iter != access_id_to_item.end()) {
      expected_parent = items[item_iter->second].expected.ptr();
    } else {
      auto& tree_node = tree_build_nodes[tree_iter->second];
      if (tree_node.item_index.has_value()) {
        expected_parent = items[tree_node.item_index.value()].expected.ptr();
      }
    }
    std::stringstream group_key;
    PyTypeObject* expected_type = nullptr;
    if (expected_parent != nullptr) {
      expected_type = reinterpret_cast<PyTypeObject*>(Py_TYPE(expected_parent));
      group_key << reinterpret_cast<uintptr_t>(expected_type);
    } else {
      group_key << "dynamic";
    }
    group_key << ":" << method_step->name << ":"
              << reinterpret_cast<uintptr_t>(op.expected.ptr());
    auto group_iter = method_code_group_ids.find(group_key.str());
    size_t group_index = 0;
    if (group_iter == method_code_group_ids.end()) {
      group_index = method_code_groups.size();
      method_code_group_ids.emplace(group_key.str(), group_index);
      method_code_groups.push_back(
          LayerMethodCodeGroup{method_step->name,
                               method_step->name_object,
                               op.expected,
                               expected_type,
                               method_step->key_hash});
    } else {
      group_index = group_iter->second;
    }
    PyObject** expected_parent_dict_ptr =
        expected_parent != nullptr ? _PyObject_GetDictPtr(expected_parent)
                                   : nullptr;
    bool expected_instance_override_absent =
        IsDictItemAbsent(expected_parent_dict_ptr,
                         method_step->name_object.ptr(),
                         method_step->key_hash);
    tree_build_nodes[tree_iter->second].method_code_checks.push_back(
        LayerMatchItem::MethodCodeCheck{op.access_id,
                                        group_index,
                                        method_step->name,
                                        method_step->name_object,
                                        op.expected,
                                        method_step->key_hash,
                                        expected_instance_override_absent});
    removed[i] = 1;
  }

  for (size_t i = 0; i < ops_.size(); ++i) {
    if (removed[i]) {
      continue;
    }
    auto& op = ops_[i];
    if (op.kind != OpKind::LENGTH_MATCH) {
      continue;
    }
    auto tree_iter = tree_node_by_access_id.find(op.access_id);
    if (tree_iter == tree_node_by_access_id.end()) {
      continue;
    }
    tree_build_nodes[tree_iter->second].self_length_checks.push_back(
        LayerMatchItem::SelfLengthCheck{
            op.access_id, op.expected_length, op.require_dict_length});
    removed[i] = 1;
  }

  for (size_t i = 0; i < ops_.size(); ++i) {
    if (removed[i]) {
      continue;
    }
    auto& op = ops_[i];
    if (op.kind != OpKind::WEAKREF_MATCH || op.access_path.size() < 3) {
      continue;
    }
    const auto& class_step = op.access_path[op.access_path.size() - 2];
    const auto& attr_step = op.access_path.back();
    if (class_step.kind != AccessKind::ATTR || class_step.name != "__class__" ||
        attr_step.kind != AccessKind::ATTR) {
      continue;
    }
    std::vector<AccessStep> parent_path(op.access_path.begin(),
                                        op.access_path.end() - 2);
    auto parent_access_id = InternAccessPath(parent_path);
    auto tree_iter = tree_node_by_access_id.find(parent_access_id);
    if (tree_iter == tree_node_by_access_id.end()) {
      continue;
    }
    tree_build_nodes[tree_iter->second].class_weakref_checks.push_back(
        LayerClassWeakRefCheck{op.access_id,
                               attr_step.name_object,
                               attr_step.key_hash,
                               op.expected});
    removed[i] = 1;
  }

  for (size_t i = 0; i < ops_.size(); ++i) {
    if (removed[i]) {
      continue;
    }
    auto& op = ops_[i];
    if (op.kind != OpKind::LENGTH_MATCH || op.access_path.size() < 2) {
      continue;
    }
    const auto& attr_step = op.access_path.back();
    if (attr_step.kind != AccessKind::ATTR) {
      continue;
    }
    if (attr_step.attr_kind != CompiledGuardAttrKind::SUB_LAYERS &&
        attr_step.attr_kind != CompiledGuardAttrKind::FORWARD_PRE_HOOKS &&
        attr_step.attr_kind != CompiledGuardAttrKind::FORWARD_POST_HOOKS) {
      continue;
    }
    std::vector<AccessStep> parent_path(op.access_path.begin(),
                                        op.access_path.end() - 1);
    auto parent_access_id = InternAccessPath(parent_path);
    auto tree_iter = tree_node_by_access_id.find(parent_access_id);
    if (tree_iter == tree_node_by_access_id.end()) {
      continue;
    }
    tree_build_nodes[tree_iter->second].attr_length_checks.push_back(
        LayerMatchItem::AttrLengthCheck{op.access_id,
                                        attr_step.attr_kind,
                                        attr_step.name_object,
                                        attr_step.key_hash,
                                        op.expected_length,
                                        op.require_dict_length});
    removed[i] = 1;
  }

  for (size_t i = 0; i < ops_.size(); ++i) {
    if (removed[i]) {
      continue;
    }
    auto& op = ops_[i];
    if (op.kind != OpKind::VALUE_MATCH || op.access_path.size() < 2) {
      continue;
    }
    const auto& attr_step = op.access_path.back();
    if (attr_step.kind != AccessKind::ATTR ||
        attr_step.attr_kind != CompiledGuardAttrKind::GENERIC) {
      continue;
    }
    std::vector<AccessStep> parent_path(op.access_path.begin(),
                                        op.access_path.end() - 1);
    auto parent_access_id = InternAccessPath(parent_path);
    auto tree_iter = tree_node_by_access_id.find(parent_access_id);
    if (tree_iter == tree_node_by_access_id.end()) {
      continue;
    }
    tree_build_nodes[tree_iter->second].attr_value_checks.push_back(
        LayerMatchItem::AttrValueCheck{op.access_id,
                                       attr_step.attr_kind,
                                       attr_step.name_object,
                                       attr_step.key_hash,
                                       op.expected,
                                       op.value_match_by_identity});
    removed[i] = 1;
  }

  std::unordered_set<size_t> access_ids_needed_by_remaining_ops;
  for (size_t i = 0; i < ops_.size(); ++i) {
    if (removed[i] ||
        (first_layer_match.has_value() && i == first_layer_match.value())) {
      continue;
    }
    if (ops_[i].kind == OpKind::GRAD_ENABLED ||
        ops_[i].kind == OpKind::EXPR_MATCH) {
      continue;
    }
    std::optional<size_t> access_id = ops_[i].access_id;
    while (access_id.has_value()) {
      access_ids_needed_by_remaining_ops.insert(access_id.value());
      access_id = access_nodes_[access_id.value()].parent;
    }
  }
  for (auto& item : items) {
    item.cache_value =
        access_ids_needed_by_remaining_ops.find(item.access_id) !=
        access_ids_needed_by_remaining_ops.end();
  }

  if (items.size() <= 1 || !first_layer_match.has_value()) {
    return;
  }

  auto& fused = ops_[first_layer_match.value()];
  fused.kind = OpKind::LAYER_MATCH_GROUP;
  fused.layer_match_items = std::move(items);
  fused.layer_method_code_groups = std::move(method_code_groups);
  fused.layer_sub_layer_tree_roots = std::move(tree_roots);
  fused.layer_sub_layer_tree_nodes.reserve(tree_build_nodes.size());
  for (auto& node : tree_build_nodes) {
    uint8_t check_flags = 0;
    if (!node.attr_length_checks.empty()) {
      check_flags |= kLayerNodeAttrLengthCheck;
    }
    if (!node.self_length_checks.empty()) {
      check_flags |= kLayerNodeSelfLengthCheck;
    }
    if (!node.attr_value_checks.empty()) {
      check_flags |= kLayerNodeAttrValueCheck;
    }
    if (!node.method_code_checks.empty()) {
      check_flags |= kLayerNodeMethodCodeCheck;
    }
    if (!node.class_weakref_checks.empty()) {
      check_flags |= kLayerNodeClassWeakRefCheck;
    }
    fused.layer_sub_layer_tree_nodes.push_back(
        LayerSubLayerTreeNode{node.item_index,
                              check_flags,
                              std::move(node.self_length_checks),
                              std::move(node.attr_length_checks),
                              std::move(node.attr_value_checks),
                              std::move(node.method_code_checks),
                              std::move(node.class_weakref_checks),
                              std::move(node.children)});
  }

  std::vector<GuardOp> fused_ops;
  fused_ops.reserve(ops_.size() - fused.layer_match_items.size() + 1);
  for (size_t i = 0; i < ops_.size(); ++i) {
    if (!removed[i]) {
      fused_ops.push_back(std::move(ops_[i]));
    }
  }
  ops_ = std::move(fused_ops);
}

CompiledGuard::CompiledGuard(const py::list& specs) {
  ops_.reserve(py::len(specs));
  for (auto spec_handle : specs) {
    py::tuple spec = py::reinterpret_borrow<py::tuple>(spec_handle);
    if (py::len(spec) < 1) {
      throw py::value_error("compiled guard spec cannot be empty");
    }
    std::string kind = spec[0].cast<std::string>();
    GuardOp op;
    op.expected = py::none();

    if (kind == "grad_enabled") {
      if (py::len(spec) != 2) {
        throw py::value_error("grad_enabled guard expects 2 fields");
      }
      op.kind = OpKind::GRAD_ENABLED;
      op.expected_bool = spec[1].cast<bool>();
    } else if (kind == "expr_match") {
      if (py::len(spec) != 2) {
        throw py::value_error("expr_match guard expects 2 fields");
      }
      op.kind = OpKind::EXPR_MATCH;
      op.expr = ParseExpr(spec[1]);
    } else {
      if (py::len(spec) < 3) {
        throw py::value_error(kind + " guard expects an access path");
      }
      op.access_path = ParseAccessPath(spec[1]);
      op.access_id = InternAccessPath(op.access_path);
      op.expected = py::reinterpret_borrow<py::object>(spec[2]);

      if (kind == "type_match") {
        op.kind = OpKind::TYPE_MATCH;
        op.expected_type = reinterpret_cast<PyTypeObject*>(op.expected.ptr());
      } else if (kind == "instance_check") {
        op.kind = OpKind::INSTANCE_CHECK;
        op.expected_is_dict =
            op.expected.ptr() == reinterpret_cast<PyObject*>(&PyDict_Type);
      } else if (kind == "id_match") {
        op.kind = OpKind::ID_MATCH;
      } else if (kind == "value_match") {
        op.kind = OpKind::VALUE_MATCH;
        PyObject* expected = op.expected.ptr();
        op.value_match_by_identity = PyCode_Check(expected) ||
                                     PyBool_Check(expected) ||
                                     Py_IsNone(expected);
      } else if (kind == "length_match") {
        op.kind = OpKind::LENGTH_MATCH;
        op.expected_length = spec[2].cast<Py_ssize_t>();
        if (!ops_.empty()) {
          const auto& previous = ops_.back();
          if (previous.kind == OpKind::INSTANCE_CHECK &&
              previous.access_id == op.access_id && previous.expected_is_dict) {
            op.require_dict_length = true;
            ops_.pop_back();
          }
        }
      } else if (kind == "layer_match") {
        op.kind = OpKind::LAYER_MATCH;
        op.expected_bool = op.expected.attr("training").cast<bool>();
        op.expected_layer_dict_ptr = _PyObject_GetDictPtr(op.expected.ptr());
        if (op.expected_layer_dict_ptr != nullptr &&
            *op.expected_layer_dict_ptr != nullptr) {
          op.expected_layer_dict_version =
              GetDictVersion(*op.expected_layer_dict_ptr);
        }
      } else if (kind == "tensor_shape") {
        if (py::len(spec) != 4) {
          throw py::value_error("tensor_shape guard expects 4 fields");
        }
        op.kind = OpKind::TENSOR_SHAPE;
        op.expected_shape = ParseShape(spec[2]);
        op.min_non_specialized_number = spec[3].cast<int64_t>();
      } else if (kind == "tensor_dtype") {
        op.kind = OpKind::TENSOR_DTYPE;
        op.expected_dtype = ParseDtype(spec[2]);
      } else if (kind == "tensor_is_dist") {
        op.kind = OpKind::TENSOR_IS_DIST;
        op.expected_bool = spec[2].cast<bool>();
      } else if (kind == "tensor_dist_meta") {
        if (py::len(spec) != 7) {
          throw py::value_error("tensor_dist_meta guard expects 7 fields");
        }
        op.kind = OpKind::TENSOR_DIST_META;
        op.expected_dist_mesh_shape =
            py::reinterpret_borrow<py::object>(spec[2]);
        op.expected_dist_process_ids =
            py::reinterpret_borrow<py::object>(spec[3]);
        op.expected_dist_dims_mapping =
            py::reinterpret_borrow<py::object>(spec[4]);
        op.expected_dist_local_shape =
            py::reinterpret_borrow<py::object>(spec[5]);
        op.expected_dist_info_from_tensor =
            py::reinterpret_borrow<py::object>(spec[6]);
        if (!PyCallable_Check(op.expected_dist_info_from_tensor.ptr())) {
          throw py::type_error(
              "tensor_dist_meta requires a callable extractor");
        }
        op.expected_dist_mesh_shape_values =
            ParseInt64Vector(spec[2], "tensor_dist_meta mesh shape");
        op.expected_dist_process_ids_values =
            ParseInt64Vector(spec[3], "tensor_dist_meta process ids");
        op.expected_dist_dims_mapping_values =
            ParseInt64Vector(spec[4], "tensor_dist_meta dims mapping");
        op.expected_dist_local_shape_values =
            ParseInt64Vector(spec[5], "tensor_dist_meta local shape");
      } else if (kind == "tensor_not_hold_allocation") {
        op.kind = OpKind::TENSOR_NOT_HOLD_ALLOCATION;
      } else if (kind == "numpy_dtype") {
        op.kind = OpKind::NUMPY_DTYPE;
      } else if (kind == "numpy_shape") {
        if (py::len(spec) != 4) {
          throw py::value_error("numpy_shape guard expects 4 fields");
        }
        op.kind = OpKind::NUMPY_SHAPE;
        op.expected_shape = ParseShape(spec[2]);
        op.min_non_specialized_number = spec[3].cast<int64_t>();
      } else if (kind == "weakref_match") {
        op.kind = OpKind::WEAKREF_MATCH;
      } else {
        throw py::value_error("unknown compiled guard op: " + kind);
      }
    }
    if (op.kind == OpKind::VALUE_MATCH && !ops_.empty()) {
      const auto& previous = ops_.back();
      if (previous.kind == OpKind::TYPE_MATCH &&
          previous.access_id == op.access_id &&
          previous.expected_type == Py_TYPE(op.expected.ptr())) {
        ops_.pop_back();
      }
    }
    ops_.push_back(std::move(op));
  }
  DeduplicateOps();
  FuseTensorMetaOps();
  FuseLayerMatchOps();
}

PyObject* CompiledGuard::EvalAccess(
    FrameProxy* frame, const std::vector<AccessStep>& access_path) const {
  PyObject* current = nullptr;
  for (size_t i = 0; i < access_path.size(); ++i) {
    const auto& step = access_path[i];
    PyObject* next = nullptr;
    switch (step.kind) {
      case AccessKind::LOCAL:
        next = GetMappingItem(GetFrameLocals(frame),
                              step.name_object.ptr(),
                              step.name,
                              step.key_hash);
        break;
      case AccessKind::GLOBAL:
        next = GetMappingItem(GetFrameGlobals(frame),
                              step.name_object.ptr(),
                              step.name,
                              step.key_hash);
        break;
      case AccessKind::BUILTIN:
        next = GetMappingItem(GetFrameBuiltins(frame),
                              step.name_object.ptr(),
                              step.name,
                              step.key_hash);
        break;
      case AccessKind::CONSTANT:
        next = step.value.ptr();
        Py_INCREF(next);
        break;
      case AccessKind::ATTR:
        if (current == nullptr) {
          return nullptr;
        }
        next = GetAttrFast(
            current, step.name, step.name_object.ptr(), step.attr_kind);
        break;
      case AccessKind::ITEM:
        if (current == nullptr) {
          return nullptr;
        }
        next = GetItemFast(current, step.value.ptr(), step.key_hash);
        break;
    }
    Py_XDECREF(current);
    current = next;
    if (current == nullptr) {
      return nullptr;
    }
  }
  return current;
}

PyObject* CompiledGuard::EvalAccessNode(FrameProxy* frame,
                                        size_t access_id,
                                        AccessCache* cache) const {
  auto& values = cache->guard->access_cache_values_;
  auto& evaluated = cache->guard->access_cache_evaluated_;
  auto& owned = cache->guard->access_cache_owned_;
  const auto generation = cache->guard->access_cache_generation_;
  if (evaluated[access_id] == generation) {
    return values[access_id];
  }

  const auto& node = access_nodes_[access_id];
  const auto cache_result = [&](PyObject* result, bool result_owned) {
    values[access_id] = result;
    evaluated[access_id] = generation;
    owned[access_id] = result_owned;
    if (result_owned) {
      cache->guard->access_cache_touched_.push_back(access_id);
    }
    return result;
  };

  if (node.step.kind == AccessKind::ATTR &&
      (node.step.attr_kind == CompiledGuardAttrKind::CODE ||
       node.step.attr_kind == CompiledGuardAttrKind::GLOBALS) &&
      node.parent.has_value()) {
    const auto& parent_node = access_nodes_[node.parent.value()];
    if (parent_node.step.kind == AccessKind::ATTR &&
        parent_node.parent.has_value()) {
      if (parent_node.step.attr_kind == CompiledGuardAttrKind::CALL) {
        PyObject* base =
            EvalAccessNode(frame, parent_node.parent.value(), cache);
        if (base != nullptr) {
          PyObject* result = GetFunctionFieldFromObjectAttrFast(
              base,
              "__call__",
              parent_node.step.name_object.ptr(),
              parent_node.step.key_hash,
              node.step.name);
          if (result != nullptr) {
            return cache_result(result, false);
          }
        }
      } else if (parent_node.step.attr_kind == CompiledGuardAttrKind::FUNC) {
        const auto& method_node = access_nodes_[parent_node.parent.value()];
        if (method_node.step.kind == AccessKind::ATTR &&
            method_node.parent.has_value()) {
          PyObject* base =
              EvalAccessNode(frame, method_node.parent.value(), cache);
          if (base != nullptr) {
            PyObject* result = GetFunctionFieldFromObjectAttrFast(
                base,
                method_node.step.name,
                method_node.step.name_object.ptr(),
                method_node.step.key_hash,
                node.step.name);
            if (result != nullptr) {
              return cache_result(result, false);
            }
          }
        }
      }
    }
  }

  PyObject* current = nullptr;
  if (node.parent.has_value()) {
    current = EvalAccessNode(frame, node.parent.value(), cache);
    if (current == nullptr) {
      return cache_result(nullptr, false);
    }
  }

  AccessLookup next;
  const auto& step = node.step;
  switch (step.kind) {
    case AccessKind::LOCAL:
      next = GetMappingItemCached(GetFrameLocals(frame),
                                  step.name_object.ptr(),
                                  step.name,
                                  step.key_hash);
      break;
    case AccessKind::GLOBAL:
      next = GetMappingItemCached(GetFrameGlobals(frame),
                                  step.name_object.ptr(),
                                  step.name,
                                  step.key_hash);
      break;
    case AccessKind::BUILTIN:
      next = GetMappingItemCached(GetFrameBuiltins(frame),
                                  step.name_object.ptr(),
                                  step.name,
                                  step.key_hash);
      break;
    case AccessKind::CONSTANT:
      next = {step.value.ptr(), false};
      break;
    case AccessKind::ATTR:
      if (current == nullptr) {
        break;
      }
      next = GetAttrCached(current,
                           step.name,
                           step.name_object.ptr(),
                           step.key_hash,
                           step.attr_kind);
      break;
    case AccessKind::ITEM:
      if (current == nullptr) {
        break;
      }
      next = GetItemCached(current, step.value.ptr(), step.key_hash);
      break;
  }

  return cache_result(next.value, next.owned);
}

PyObject* CompiledGuard::EvalExpr(FrameProxy* frame,
                                  const GuardExpr& expr) const {
  switch (expr.kind) {
    case ExprKind::CONSTANT: {
      PyObject* value = expr.value.ptr();
      Py_INCREF(value);
      return value;
    }
    case ExprKind::ACCESS:
      return EvalAccess(frame, expr.access_path);
    case ExprKind::UNARY: {
      PyObject* value = EvalExpr(frame, *expr.lhs);
      if (value == nullptr) {
        return nullptr;
      }
      PyObject* result = nullptr;
      if (expr.op == "+") {
        result = PyNumber_Positive(value);
      } else if (expr.op == "-") {
        result = PyNumber_Negative(value);
      } else if (expr.op == "~") {
        result = PyNumber_Invert(value);
      } else if (expr.op == "not" || expr.op == "!") {
        int truth = PyObject_IsTrue(value);
        if (truth != -1) {
          result = PyBool_FromLong(!truth);
        }
      } else if (expr.op == "bool") {
        int truth = PyObject_IsTrue(value);
        if (truth != -1) {
          result = PyBool_FromLong(truth);
        }
      } else {
        PyErr_SetString(
            PyExc_TypeError,
            ("unsupported compiled guard unary op: " + expr.op).c_str());
      }
      Py_DECREF(value);
      if (result == nullptr) {
        PyErr_Clear();
      }
      return result;
    }
    case ExprKind::BINARY: {
      PyObject* lhs = EvalExpr(frame, *expr.lhs);
      if (lhs == nullptr) {
        return nullptr;
      }
      PyObject* rhs = EvalExpr(frame, *expr.rhs);
      if (rhs == nullptr) {
        Py_DECREF(lhs);
        return nullptr;
      }

      PyObject* result = nullptr;
      if (expr.op == "==") {
        result = PyObject_RichCompare(lhs, rhs, Py_EQ);
      } else if (expr.op == "!=") {
        result = PyObject_RichCompare(lhs, rhs, Py_NE);
      } else if (expr.op == "<") {
        result = PyObject_RichCompare(lhs, rhs, Py_LT);
      } else if (expr.op == "<=") {
        result = PyObject_RichCompare(lhs, rhs, Py_LE);
      } else if (expr.op == ">") {
        result = PyObject_RichCompare(lhs, rhs, Py_GT);
      } else if (expr.op == ">=") {
        result = PyObject_RichCompare(lhs, rhs, Py_GE);
      } else if (expr.op == "+") {
        result = PyNumber_Add(lhs, rhs);
      } else if (expr.op == "-") {
        result = PyNumber_Subtract(lhs, rhs);
      } else if (expr.op == "*") {
        result = PyNumber_Multiply(lhs, rhs);
      } else if (expr.op == "/") {
        result = PyNumber_TrueDivide(lhs, rhs);
      } else if (expr.op == "//") {
        result = PyNumber_FloorDivide(lhs, rhs);
      } else if (expr.op == "%") {
        result = PyNumber_Remainder(lhs, rhs);
      } else if (expr.op == "**") {
        result = PyNumber_Power(lhs, rhs, Py_None);
      } else if (expr.op == "<<") {
        result = PyNumber_Lshift(lhs, rhs);
      } else if (expr.op == ">>") {
        result = PyNumber_Rshift(lhs, rhs);
      } else if (expr.op == "&") {
        result = PyNumber_And(lhs, rhs);
      } else if (expr.op == "|") {
        result = PyNumber_Or(lhs, rhs);
      } else if (expr.op == "^") {
        result = PyNumber_Xor(lhs, rhs);
      } else {
        PyErr_SetString(
            PyExc_TypeError,
            ("unsupported compiled guard binary op: " + expr.op).c_str());
      }

      Py_DECREF(lhs);
      Py_DECREF(rhs);
      if (result == nullptr) {
        PyErr_Clear();
      }
      return result;
    }
  }
  return nullptr;
}

bool CompiledGuard::CheckTensorDistMeta(PyObject* value, const GuardOp& op) {
  auto tensor = GetTensorFromPyObject(value);
  if (!tensor || !tensor->is_dist_tensor()) {
    return false;
  }

  PyObject* dist_info =
      PyObject_CallOneArg(op.expected_dist_info_from_tensor.ptr(), value);
  if (dist_info == nullptr) {
    PyErr_Clear();
    return false;
  }

  PyObject* mesh = PyObject_GetAttrString(dist_info, "mesh");
  if (mesh == nullptr) {
    Py_DECREF(dist_info);
    PyErr_Clear();
    return false;
  }

  PyObject* mesh_shape = PyObject_GetAttrString(mesh, "shape");
  PyObject* process_ids = PyObject_GetAttrString(mesh, "process_ids");
  PyObject* dims_mapping = PyObject_GetAttrString(dist_info, "dims_mapping");
  PyObject* local_shape = PyObject_GetAttrString(dist_info, "local_shape");

  bool result =
      mesh_shape != nullptr && process_ids != nullptr &&
      dims_mapping != nullptr && local_shape != nullptr &&
      PyObject_RichEqual(mesh_shape, op.expected_dist_mesh_shape.ptr()) &&
      PyObject_RichEqual(process_ids, op.expected_dist_process_ids.ptr()) &&
      PyObject_RichEqual(dims_mapping, op.expected_dist_dims_mapping.ptr()) &&
      PyObject_RichEqual(local_shape, op.expected_dist_local_shape.ptr());

  Py_XDECREF(mesh_shape);
  Py_XDECREF(process_ids);
  Py_XDECREF(dims_mapping);
  Py_XDECREF(local_shape);
  Py_DECREF(mesh);
  Py_DECREF(dist_info);
  PyErr_Clear();
  return result;
}

bool CompiledGuard::CheckOp(FrameProxy* frame,
                            const GuardOp& op,
                            AccessCache* cache) const {
  if (op.kind == OpKind::GRAD_ENABLED) {
    return egr::Controller::Instance().HasGrad() == op.expected_bool;
  }

  if (op.kind == OpKind::EXPR_MATCH) {
    PyObject* value = EvalExpr(frame, *op.expr);
    if (value == nullptr) {
      PyErr_Clear();
      return false;
    }
    int truth = PyObject_IsTrue(value);
    Py_DECREF(value);
    if (truth == -1) {
      PyErr_Clear();
      return false;
    }
    return truth == 1;
  }

  if (op.kind == OpKind::LAYER_MATCH_GROUP) {
    const auto cache_borrowed_value = [&](size_t access_id, PyObject* value) {
      auto& guard = *cache->guard;
      const auto generation = guard.access_cache_generation_;
      if (guard.access_cache_evaluated_[access_id] == generation) {
        return;
      }
      guard.access_cache_values_[access_id] = value;
      guard.access_cache_evaluated_[access_id] = generation;
      guard.access_cache_owned_[access_id] = 0;
    };

    auto& method_code_required = cache->guard->layer_method_code_required_;
    if (method_code_required.size() < op.layer_method_code_groups.size()) {
      method_code_required.resize(op.layer_method_code_groups.size(), 0);
    }
    std::fill(method_code_required.begin(),
              method_code_required.begin() + op.layer_method_code_groups.size(),
              0);
    auto& dynamic_method_code_required =
        cache->guard->layer_method_code_dynamic_required_;
    if (dynamic_method_code_required.size() <
        op.layer_method_code_groups.size()) {
      dynamic_method_code_required.resize(op.layer_method_code_groups.size());
    }
    for (size_t i = 0; i < op.layer_method_code_groups.size(); ++i) {
      dynamic_method_code_required[i].clear();
    }

    constexpr size_t kDynamicMethodCodeGroup =
        std::numeric_limits<size_t>::max();
    const auto check_attr_length_checks =
        [&](PyObject* value,
            PyObject* layer_dict,
            const std::vector<LayerMatchItem::AttrLengthCheck>&
                attr_length_checks) -> bool {
      for (const auto& length_check : attr_length_checks) {
        PyObject* attr_value = nullptr;
        bool use_generic_access = true;
        if (layer_dict != nullptr) {
          attr_value = GetDictItemWithHash(layer_dict,
                                           length_check.name_object.ptr(),
                                           length_check.name_hash);
          if (attr_value != nullptr) {
            use_generic_access = false;
            cache_borrowed_value(length_check.access_id, attr_value);
          } else {
            PyErr_Clear();
          }
        }
        if (use_generic_access) {
          attr_value = EvalAccessNode(frame, length_check.access_id, cache);
        }
        if (attr_value == nullptr) {
          PyErr_Clear();
          return false;
        }
        if (!CheckLengthValue(attr_value,
                              length_check.expected_length,
                              length_check.require_dict_length)) {
          PyErr_Clear();
          return false;
        }
      }
      return true;
    };

    const auto check_self_length_checks =
        [&](PyObject* value,
            const std::vector<LayerMatchItem::SelfLengthCheck>&
                self_length_checks) -> bool {
      for (const auto& length_check : self_length_checks) {
        cache_borrowed_value(length_check.access_id, value);
        PyObject* sub_layers = GetInstanceDictItemBorrowed(
            value, SubLayersNameObject(), SubLayersNameHash());
        if (sub_layers != nullptr && PyDict_Check(sub_layers)) {
          if (GetDictSizeFast(sub_layers) != length_check.expected_length) {
            return false;
          }
          continue;
        }
        PyErr_Clear();
        if (!CheckLengthValue(value,
                              length_check.expected_length,
                              length_check.require_dict_length)) {
          PyErr_Clear();
          return false;
        }
      }
      return true;
    };

    const auto check_attr_value_checks =
        [&](PyObject* value,
            PyObject* layer_dict,
            const std::vector<LayerMatchItem::AttrValueCheck>&
                attr_value_checks) -> bool {
      for (const auto& value_check : attr_value_checks) {
        PyObject* attr_value = nullptr;
        bool use_generic_access = true;
        if (layer_dict != nullptr) {
          attr_value = GetDictItemWithHash(
              layer_dict, value_check.name_object.ptr(), value_check.name_hash);
          if (attr_value != nullptr) {
            use_generic_access = false;
            cache_borrowed_value(value_check.access_id, attr_value);
          } else {
            PyErr_Clear();
          }
        }
        if (use_generic_access) {
          attr_value = EvalAccessNode(frame, value_check.access_id, cache);
        }
        if (attr_value == nullptr) {
          PyErr_Clear();
          return false;
        }
        if (value_check.value_match_by_identity) {
          if (attr_value != value_check.expected.ptr()) {
            return false;
          }
        } else if (!PyObject_Equal(attr_value, value_check.expected.ptr())) {
          return false;
        }
      }
      return true;
    };

    const auto check_method_code_checks =
        [&](PyObject* value,
            PyObject* layer_dict,
            const std::vector<LayerMatchItem::MethodCodeCheck>&
                method_code_checks,
            bool layer_dict_unchanged) -> bool {
      for (const auto& code_check : method_code_checks) {
        PyObject* override = nullptr;
        if (!(layer_dict_unchanged &&
              code_check.expected_instance_override_absent)) {
          if (layer_dict != nullptr) {
            override = GetDictItemWithHash(layer_dict,
                                           code_check.method_name_object.ptr(),
                                           code_check.method_name_hash);
          } else {
            override =
                GetInstanceDictItemBorrowed(value,
                                            code_check.method_name_object.ptr(),
                                            code_check.method_name_hash);
          }
        }
        if (override != nullptr) {
          PyObject* code = EvalAccessNode(frame, code_check.access_id, cache);
          if (code == nullptr) {
            PyErr_Clear();
            return false;
          }
          if (code != code_check.expected_code.ptr()) {
            return false;
          }
          continue;
        }

        if (code_check.group_index == kDynamicMethodCodeGroup) {
          PyObject* descriptor =
              _PyType_Lookup(reinterpret_cast<PyTypeObject*>(Py_TYPE(value)),
                             code_check.method_name_object.ptr());
          if (descriptor == nullptr || !PyFunction_Check(descriptor)) {
            return false;
          }
          PyObject* code = PyFunction_GET_CODE(descriptor);
          if (code != code_check.expected_code.ptr()) {
            return false;
          }
        } else {
          const auto& group =
              op.layer_method_code_groups[code_check.group_index];
          if (group.expected_type == nullptr) {
            auto* type = reinterpret_cast<PyTypeObject*>(Py_TYPE(value));
            auto& required_types =
                dynamic_method_code_required[code_check.group_index];
            if (std::find(required_types.begin(), required_types.end(), type) ==
                required_types.end()) {
              required_types.push_back(type);
            }
          } else {
            method_code_required[code_check.group_index] = 1;
          }
        }
      }
      return true;
    };

    const auto check_class_weakref_checks =
        [&](PyObject* value,
            const std::vector<LayerClassWeakRefCheck>& class_weakref_checks)
        -> bool {
      auto* type = reinterpret_cast<PyTypeObject*>(Py_TYPE(value));
      for (const auto& weakref_check : class_weakref_checks) {
        PyObject* descriptor =
            _PyType_Lookup(type, weakref_check.name_object.ptr());
        if (descriptor == nullptr) {
          return false;
        }
        cache_borrowed_value(weakref_check.access_id, descriptor);
        if (!CheckWeakRefMatch(descriptor, weakref_check.expected.ptr())) {
          return false;
        }
      }
      return true;
    };

    const auto check_layer_item = [&](PyObject* value,
                                      const LayerMatchItem& item) -> bool {
      if (item.cache_value) {
        cache_borrowed_value(item.access_id, value);
      }
      if (!CheckLayerMatchValue(value,
                                item.expected.ptr(),
                                item.expected_layer_dict_ptr,
                                item.expected_layer_dict_version,
                                item.expected_bool)) {
        return false;
      }
      if (item.attr_length_checks.empty() && item.method_code_checks.empty()) {
        return true;
      }
      bool layer_dict_unchanged = IsDictVersionUnchanged(
          item.expected_layer_dict_ptr, item.expected_layer_dict_version);
      PyObject* layer_dict = nullptr;
      if (item.expected_layer_dict_ptr != nullptr &&
          *item.expected_layer_dict_ptr != nullptr &&
          PyDict_Check(*item.expected_layer_dict_ptr)) {
        layer_dict = *item.expected_layer_dict_ptr;
      }
      return check_attr_length_checks(
                 value, layer_dict, item.attr_length_checks) &&
             check_method_code_checks(value,
                                      layer_dict,
                                      item.method_code_checks,
                                      layer_dict_unchanged);
    };

    enum class LayerTreeStatus : uint8_t {
      MATCH,
      MISS,
      UNSUPPORTED,
    };

    const auto check_sub_layer_tree =
        [&](const auto& self,
            PyObject* value,
            size_t node_index) -> LayerTreeStatus {
      const auto& node = op.layer_sub_layer_tree_nodes[node_index];
      if (node.item_index.has_value() &&
          !check_layer_item(value,
                            op.layer_match_items[node.item_index.value()])) {
        return LayerTreeStatus::MISS;
      }
      PyObject* layer_dict = nullptr;
      if ((node.check_flags & kLayerNodeNeedsDict) != 0) {
        PyObject** layer_dict_ptr = _PyObject_GetDictPtr(value);
        if (layer_dict_ptr != nullptr && *layer_dict_ptr != nullptr &&
            PyDict_Check(*layer_dict_ptr)) {
          layer_dict = *layer_dict_ptr;
        }
      }
      if (((node.check_flags & kLayerNodeAttrLengthCheck) != 0 &&
           !check_attr_length_checks(
               value, layer_dict, node.attr_length_checks)) ||
          ((node.check_flags & kLayerNodeSelfLengthCheck) != 0 &&
           !check_self_length_checks(value, node.self_length_checks)) ||
          ((node.check_flags & kLayerNodeAttrValueCheck) != 0 &&
           !check_attr_value_checks(
               value, layer_dict, node.attr_value_checks)) ||
          ((node.check_flags & kLayerNodeMethodCodeCheck) != 0 &&
           !check_method_code_checks(
               value,
               layer_dict,
               node.method_code_checks,
               node.item_index.has_value() &&
                   IsDictVersionUnchanged(
                       op.layer_match_items[node.item_index.value()]
                           .expected_layer_dict_ptr,
                       op.layer_match_items[node.item_index.value()]
                           .expected_layer_dict_version))) ||
          ((node.check_flags & kLayerNodeClassWeakRefCheck) != 0 &&
           !check_class_weakref_checks(value, node.class_weakref_checks))) {
        return LayerTreeStatus::MISS;
      }
      if (node.children.empty()) {
        return LayerTreeStatus::MATCH;
      }

      PyObject* sub_layers = GetInstanceDictItemBorrowed(
          value, SubLayersNameObject(), SubLayersNameHash());
      if (sub_layers == nullptr) {
        PyErr_Clear();
        return LayerTreeStatus::UNSUPPORTED;
      }
      if (!PyDict_Check(sub_layers)) {
        return LayerTreeStatus::UNSUPPORTED;
      }
      for (const auto& child : node.children) {
        PyObject* child_value =
            GetDictItemWithHash(sub_layers, child.key.ptr(), child.key_hash);
        if (child_value == nullptr) {
          PyErr_Clear();
          return LayerTreeStatus::MISS;
        }
        auto child_status = self(self, child_value, child.node_index);
        if (child_status != LayerTreeStatus::MATCH) {
          return child_status;
        }
      }
      return LayerTreeStatus::MATCH;
    };

    bool check_tree_items_with_generic_access = false;
    for (const auto& root : op.layer_sub_layer_tree_roots) {
      PyObject* value = EvalAccessNode(frame, root.base_access_id, cache);
      if (value == nullptr) {
        PyErr_Clear();
        return false;
      }
      auto status =
          check_sub_layer_tree(check_sub_layer_tree, value, root.node_index);
      if (status == LayerTreeStatus::MISS) {
        return false;
      }
      if (status == LayerTreeStatus::UNSUPPORTED) {
        check_tree_items_with_generic_access = true;
      }
    }

    for (const auto& item : op.layer_match_items) {
      if (item.use_sub_layer_tree && !check_tree_items_with_generic_access) {
        continue;
      }
      PyObject* value = EvalAccessNode(frame, item.access_id, cache);
      if (value == nullptr) {
        PyErr_Clear();
        return false;
      }
      if (!check_layer_item(value, item)) {
        return false;
      }
    }
    for (size_t i = 0; i < op.layer_method_code_groups.size(); ++i) {
      if (!method_code_required[i]) {
        const auto& dynamic_required_types = dynamic_method_code_required[i];
        if (dynamic_required_types.empty()) {
          continue;
        }
        const auto& group = op.layer_method_code_groups[i];
        for (auto* type : dynamic_required_types) {
          PyObject* descriptor =
              _PyType_Lookup(type, group.method_name_object.ptr());
          if (descriptor == nullptr || !PyFunction_Check(descriptor)) {
            return false;
          }
          PyObject* code = PyFunction_GET_CODE(descriptor);
          if (code != group.expected_code.ptr()) {
            return false;
          }
        }
        continue;
      }
      const auto& group = op.layer_method_code_groups[i];
      PyObject* descriptor =
          _PyType_Lookup(group.expected_type, group.method_name_object.ptr());
      if (descriptor == nullptr || !PyFunction_Check(descriptor)) {
        return false;
      }
      PyObject* code = PyFunction_GET_CODE(descriptor);
      if (code != group.expected_code.ptr()) {
        return false;
      }
    }
    return true;
  }

  PyObject* value = EvalAccessNode(frame, op.access_id, cache);
  if (value == nullptr) {
    PyErr_Clear();
    return false;
  }

  bool result = false;
  switch (op.kind) {
    case OpKind::TYPE_MATCH:
      result = Py_TYPE(value) == op.expected_type;
      break;
    case OpKind::INSTANCE_CHECK: {
      if (op.expected_is_dict) {
        result = PyDict_Check(value);
      } else {
        int instance_result = PyObject_IsInstance(value, op.expected.ptr());
        if (instance_result == -1) {
          PyErr_Clear();
          result = false;
        } else {
          result = instance_result == 1;
        }
      }
      break;
    }
    case OpKind::ID_MATCH:
      result = value == op.expected.ptr();
      break;
    case OpKind::VALUE_MATCH:
      if (op.value_match_by_identity) {
        result = value == op.expected.ptr();
      } else {
        result = PyObject_Equal(value, op.expected.ptr());
      }
      break;
    case OpKind::LENGTH_MATCH:
      result =
          CheckLengthValue(value, op.expected_length, op.require_dict_length);
      if (!result) {
        PyErr_Clear();
      }
      break;
    case OpKind::LAYER_MATCH: {
      result = CheckLayerMatchValue(value,
                                    op.expected.ptr(),
                                    op.expected_layer_dict_ptr,
                                    op.expected_layer_dict_version,
                                    op.expected_bool);
      break;
    }
    case OpKind::LAYER_MATCH_GROUP:
      result = false;
      break;
    case OpKind::TENSOR_SHAPE: {
      auto tensor = GetTensorFromPyObject(value);
      if (!tensor) {
        result = false;
        break;
      }
      auto shape = tensor->shape();
      result = check_shape<std::vector<int64_t>>(op.expected_shape,
                                                 shape.size(),
                                                 shape,
                                                 op.min_non_specialized_number);
      break;
    }
    case OpKind::TENSOR_DTYPE: {
      auto tensor = GetTensorFromPyObject(value);
      if (!tensor) {
        result = false;
        break;
      }
      result = phi::TransToProtoVarType(tensor->type()) == op.expected_dtype;
      break;
    }
    case OpKind::TENSOR_IS_DIST: {
      auto tensor = GetTensorFromPyObject(value);
      if (!tensor) {
        result = false;
        break;
      }
      result = tensor->is_dist_tensor() == op.expected_bool;
      break;
    }
    case OpKind::TENSOR_META: {
      auto tensor = GetTensorFromPyObject(value);
      if (!tensor) {
        result = false;
        break;
      }
      auto shape = tensor->shape();
      result =
          check_shape<std::vector<int64_t>>(op.expected_shape,
                                            shape.size(),
                                            shape,
                                            op.min_non_specialized_number) &&
          phi::TransToProtoVarType(tensor->type()) == op.expected_dtype &&
          tensor->is_dist_tensor() == op.expected_bool;
      break;
    }
    case OpKind::TENSOR_DIST_META:
      result = CheckTensorDistMeta(value, op);
      break;
    case OpKind::TENSOR_NOT_HOLD_ALLOCATION:
      result = CheckIsNotDenseTensorHoldAllocation(value);
      break;
    case OpKind::NUMPY_DTYPE: {
      py::dtype expected_dtype = py::cast<py::dtype>(op.expected);
      if (py::isinstance<py::array>(value)) {
        result = py::reinterpret_borrow<py::array>(py::handle(value))
                     .dtype()
                     .is(expected_dtype);
      } else {
        result = expected_dtype.equal(py::handle(value).get_type());
      }
      break;
    }
    case OpKind::NUMPY_SHAPE: {
      if (!py::isinstance<py::array>(value)) {
        result = false;
        break;
      }
      py::array array = py::reinterpret_borrow<py::array>(py::handle(value));
      result = check_shape<const Py_ssize_t*>(op.expected_shape,
                                              array.ndim(),
                                              array.shape(),
                                              op.min_non_specialized_number);
      break;
    }
    case OpKind::WEAKREF_MATCH:
      result = CheckWeakRefMatch(value, op.expected.ptr());
      break;
    case OpKind::EXPR_MATCH:
      result = false;
      break;
    case OpKind::GRAD_ENABLED:
      result = false;
      break;
  }

  return result;
}

bool CompiledGuard::check(FrameProxy* frame) {
  AccessCache cache(this);
  for (const auto& op : ops_) {
    if (!CheckOp(frame, op, &cache)) {
      return false;
    }
  }
  return true;
}

std::string CompiledGuard::stringify() const {
  std::stringstream ss;
  ss << "CompiledGuard(num_ops=" << ops_.size() << ")";
  return ss.str();
}

CompiledGuardLookup::CompiledGuardLookup() { nodes_.emplace_back(); }

void CompiledGuardLookup::add_guard(const std::shared_ptr<CompiledGuard>& guard,
                                    int cache_index) {
  if (!guard) {
    return;
  }
  size_t node_index = 0;
  for (size_t op_index = 0; op_index < guard->ops_.size(); ++op_index) {
    const auto key = guard->LookupGuardOpKey(guard->ops_[op_index]);
    auto& edges = nodes_[node_index].edges;
    auto iter =
        std::find_if(edges.begin(), edges.end(), [&](const TrieEdge& edge) {
          return edge.key == key;
        });
    if (iter == edges.end()) {
      size_t child = nodes_.size();
      edges.push_back(TrieEdge{key, guard, op_index, child});
      nodes_.emplace_back();
      node_index = child;
    } else {
      node_index = iter->child;
    }
  }
  if (!nodes_[node_index].return_cache_index.has_value()) {
    nodes_[node_index].return_cache_index = cache_index;
  }
}

CompiledGuard::AccessCache* CompiledGuardLookup::LookupContext::GetCache(
    CompiledGuard* guard) {
  auto iter = caches.find(guard);
  if (iter != caches.end()) {
    return iter->second.get();
  }
  auto cache = std::make_unique<CompiledGuard::AccessCache>(guard);
  auto* cache_ptr = cache.get();
  caches.emplace(guard, std::move(cache));
  return cache_ptr;
}

std::optional<int> CompiledGuardLookup::LookupNode(
    FrameProxy* frame, size_t node_index, LookupContext* context) const {
  const auto& node = nodes_[node_index];
  if (node.return_cache_index.has_value()) {
    return node.return_cache_index.value();
  }
  for (const auto& edge : node.edges) {
    auto* guard = edge.guard.get();
    auto* cache = context->GetCache(guard);
    if (!guard->CheckOp(frame, guard->ops_[edge.op_index], cache)) {
      continue;
    }
    auto result = LookupNode(frame, edge.child, context);
    if (result.has_value()) {
      return result;
    }
  }
  return std::nullopt;
}

std::optional<int> CompiledGuardLookup::lookup(FrameProxy* frame) {
  LookupContext context;
  return LookupNode(frame, 0, &context);
}

std::string CompiledGuardLookup::stringify() const {
  std::stringstream ss;
  ss << "CompiledGuardLookup(num_nodes=" << nodes_.size() << ")";
  return ss.str();
}

#endif
