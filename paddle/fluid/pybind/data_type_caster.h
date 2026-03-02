/* Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// This header provides a singleton cache for phi::DataType Python objects and
// a custom pybind11 type_caster specialization that ensures ALL C++ → Python
// conversions of DataType return the same cached Python object (singleton).
//
// This guarantees `paddle.float32 is value.dtype` for all code paths,
// including PIR ops (which go through pybind11 auto-cast) and eager ops
// (which go through ToPyObject).
//
// Include this header BEFORE any pybind11 binding code that involves DataType.
// It must be included in every translation unit that might cast DataType to
// Python (pybind.cc, pir.cc, native_meta_tensor.cc, etc.).

#pragma once

#include <Python.h>

#include "paddle/phi/common/data_type.h"
#include "pybind11/pybind11.h"

namespace paddle {
namespace pybind {

/// Global singleton cache for DataType Python objects.
/// Initialized once after py::enum_<DataType> registration, stores a
/// reference to each enum value's class attribute (e.g., DataType.FLOAT32).
class DataTypeSingletonCache {
 public:
  static constexpr size_t kCacheSize =
      static_cast<size_t>(phi::DataType::NUM_DATA_TYPES);

  static DataTypeSingletonCache& Instance() {
    static DataTypeSingletonCache instance;
    return instance;
  }

  /// Initialize the cache by fetching enum attributes from the registered
  /// py::enum_<DataType> type object. Must be called once after enum
  /// registration in PYBIND11_MODULE.
  void Init(PyTypeObject* data_type_pytype) {
    // The order MUST match the enum definition in
    // paddle/phi/common/data_type.h (excluding NUM_DATA_TYPES / ALL_DTYPE).
    static const char* kNames[] = {
        "UNDEFINED", "BOOL",     "UINT8",         "INT8",        "UINT16",
        "INT16",     "UINT32",   "INT32",         "UINT64",      "INT64",
        "FLOAT32",   "FLOAT64",  "COMPLEX64",     "COMPLEX128",  "PSTRING",
        "FLOAT16",   "BFLOAT16", "FLOAT8_E4M3FN", "FLOAT8_E5M2",
    };
    static_assert(sizeof(kNames) / sizeof(kNames[0]) == kCacheSize,
                  "kNames must match DataType enum size");

    auto* type_obj = reinterpret_cast<PyObject*>(data_type_pytype);
    for (size_t i = 0; i < kCacheSize; ++i) {
      // PyObject_GetAttrString returns a new reference; we hold it forever.
      cache_[i] = PyObject_GetAttrString(type_obj, kNames[i]);
    }
    initialized_ = true;
  }

  bool IsInitialized() const { return initialized_; }

  /// Get the cached singleton PyObject* for the given DataType.
  /// Returns nullptr if cache is not initialized or index is out of range.
  /// The returned reference is borrowed (caller must Py_INCREF if needed).
  PyObject* Get(phi::DataType dtype) const {
    if (!initialized_) return nullptr;
    auto idx = static_cast<size_t>(dtype);
    if (idx >= kCacheSize) return nullptr;
    return cache_[idx];
  }

 private:
  DataTypeSingletonCache() = default;

  bool initialized_ = false;
  PyObject* cache_[kCacheSize] = {};
};

}  // namespace pybind
}  // namespace paddle

// Custom pybind11 type_caster specialization for phi::DataType.
// This MUST be defined before any pybind11 code that casts DataType,
// and it must be inside the pybind11::detail namespace.
//
// The key insight: pybind11's default type_caster_base creates a new Python
// object on every C++ → Python cast via make_new_instance(). By specializing
// the type_caster, we intercept the cast and return the cached singleton
// instead, ensuring identity (`is`) comparison works.
//
// Note: pybind11's enum __init__ (e.g., `core.DataType(10)`) bypasses this
// caster, so those objects are NOT singletons. The value-based `tp_hash`
// (DataTypeEnumHash) ensures hash consistency for such edge cases.
//
// During enum registration (before cache init), we fall back to the default
// pybind11 behavior so that the enum values are created normally.
namespace pybind11 {
namespace detail {

template <>
struct type_caster<phi::DataType> : public type_caster_base<phi::DataType> {
  using base = type_caster_base<phi::DataType>;

 public:
  // Python → C++ direction: reuse default behavior
  using base::load;

  // C++ → Python direction: return cached singleton if available
  static handle cast(const phi::DataType& src,
                     return_value_policy policy,
                     handle parent) {
    auto& cache = paddle::pybind::DataTypeSingletonCache::Instance();
    if (cache.IsInitialized()) {
      PyObject* cached = cache.Get(src);
      if (cached) {
        Py_INCREF(cached);
        return handle(cached);
      }
    }
    // Fallback: during enum registration or for unknown values
    return base::cast(src, policy, parent);
  }

  static handle cast(phi::DataType&& src,
                     return_value_policy policy,
                     handle parent) {
    return cast(src, policy, parent);
  }
};

}  // namespace detail
}  // namespace pybind11
