/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
// clang-format off
#include "paddle/utils/Util.h"

#ifndef PADDLE_NO_PYTHON
// must include the following two blocks, otherwise,
// gcc compiler may produce warning
#ifdef __APPLE__
#define _POSIX_SOURCE
#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700
#endif

#ifdef _POSIX_C_SOURCE
#define __TEMP_POSIX_C_SOURCE _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif
#ifdef _XOPEN_SOURCE
#define __TEMP_XOPEN_SOURCE _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif
#include <Python.h>
#include <frameobject.h>
#endif

#include <stdarg.h>
#include <map>
#include <mutex>
// clang-format on

namespace paddle {

std::string callPythonFunc(const std::string& moduleName,
                           const std::string& funcName,
                           const std::vector<std::string>& args);

#ifndef PADDLE_NO_PYTHON

/**
 * Global lock guard of python C-api invokes.
 * NOTE: the lock of this guard is reentrant or recursive.
 */
class PyGuard {
 public:
  PyGuard();
  PyGuard(const PyGuard& other) = delete;
  PyGuard& operator=(const PyGuard& other) = delete;

 private:
  std::lock_guard<std::recursive_mutex> guard_;
};

struct PyObjectDeleter {
  void operator()(PyObject* obj) {
    if (obj) {
      Py_DECREF(obj);
    }
  }
};

typedef std::unique_ptr<PyObject, PyObjectDeleter> PyObjectPtr;

PyObjectPtr callPythonFuncRetPyObj(const std::string& moduleName,
                                   const std::string& funcName,
                                   const std::vector<std::string>& args);

PyObjectPtr createPythonClass(const std::string& moduleName,
                              const std::string& className,
                              const std::vector<std::string>& args,
                              const std::map<std::string, std::string>& kwargs);

#define CHECK_PY(x) CHECK((x) != nullptr) << ::paddle::py::getPyCallStack()

namespace py {
PyObjectPtr import(const std::string& moduleName);

/**
 * Cast a PyLong or PyInt to int type T.
 * @tparam T return type.
 * @param [in] obj PyLong or PyInt object.
 * @param [out] ok status for casting. False if error occured. nullptr if user
 *                 don't care is ok or not.
 * @return The value of python object, or 0 if not ok.
 */
template <typename T>
T castInt(PyObject* obj, bool* ok = nullptr) {
  if (PyLong_Check(obj)) {
    if (ok) *ok = true;
    return (T)PyLong_AsUnsignedLong(obj);
  } else if (PyInt_Check(obj)) {
    if (ok) *ok = true;
    return (T)PyInt_AsLong(obj);
  } else {
    if (ok) *ok = false;
    return (T)0;
  }
}

/**
 * Invoke repr of python object.
 *
 * Just like toString method in java.
 */
char* repr(PyObject* obj);

/**
 * Invoke repr of python object.
 */
inline char* repr(const PyObjectPtr& obj) { return repr(obj.get()); }

/**
 * Get Python Error Stack String.
 */
std::string getPyCallStack();

/**
 * Object Helper for PyObjectPtr.
 *
 * Implements getAttr method for object.
 */
class ObjectHelper {
 public:
  explicit ObjectHelper(const PyObjectPtr& obj) : obj_(obj) {}

  /**
   * get attribute
   */
  inline PyObject* getAttr(const std::string& field) const {
    auto obj = PyObject_GetAttrString(obj_.get(), field.c_str());
    CHECK_PY(obj) << "Cannot get attribute on python object " << obj_.get();
    return obj;
  }

  /**
   * Get Int attribute
   * @param [in] field  attribute name.
   * @param [out] ok true if this attribute is int.
   * @tparam T int type.
   * @return int value.
   */
  template <typename T>
  T getIntAttr(const std::string& field, bool* ok = nullptr) const {
    PyObjectPtr tmp(getAttr(field));
    return castInt<T>(tmp.get(), ok);
  }

  /**
   * Get int attribute. Log(Fatal) when not ok
   * @param field attribute name.
   * @return int value.
   */
  template <typename T>
  T getIntAttrWithError(const std::string& field) const {
    bool ok;
    T tmp = getIntAttr<T>(field, &ok);
    CHECK(ok) << "Cannot get integer attribute on object " << obj_.get();
    return tmp;
  }

  /**
   * Get bool attribute.
   * @param field
   * @param [out] isBoolType return true if attribute is bool type. If the
   *                         attribute is not bool type, then an implicit
   *                         conversion will happens, and will return the
   *                         conversion result.
   *
   *                         Such as, if the attribute is 1, then the return
   *                         value of function will be true, but the isBoolType
   *                         will return false.
   * @return
   */
  bool getBoolAttr(const std::string& field, bool* isBoolType = nullptr) const {
    PyObjectPtr tmp(getAttr(field));
    if (isBoolType) {
      *isBoolType = PyBool_Check(tmp.get());
    }
    return PyObject_IsTrue(tmp.get());
  }

 private:
  const PyObjectPtr& obj_;
};

/**
 * Python Sequence Helper
 *
 * The python sequence means list or tuple.
 */
class SequenceHelper {
 public:
  explicit SequenceHelper(const PyObjectPtr& seq) : seq_(seq.get()) {
    CHECK(PySequence_Check(seq_));
  }

  explicit SequenceHelper(PyObject* seq) : seq_(seq) {
    CHECK(PySequence_Check(seq_));
  }

  inline size_t size() const { return (size_t)PySequence_Size(seq_); }

  inline PyObject* operator[](size_t i) const {
    return PySequence_Fast_GET_ITEM(seq_, i);
  }

  inline double getDouble(size_t i) const {
    auto* ptr = (*this)[i];
    return PyFloat_AsDouble(ptr);
  }

  /**
   * Set a sequence item o[i] = obj;
   * @param i index
   * @param obj setted item.
   * @param steal if steal = true, sequence will move object in iteself,
   *              just like std::move. Otherwise, it will increase reference
   *              count. Default is false.
   */
  inline void set(size_t i, const PyObjectPtr& obj, bool steal = false) {
    this->set(i, obj.get(), steal);
  }

  /**
   * Set a sequence item o[i] = obj;
   */
  inline void set(size_t i, PyObject* obj, bool steal = false) {
    if (!steal) {
      Py_XINCREF(obj);
    }
    if (PyTuple_Check(seq_)) {
      CHECK_NE(PyTuple_SetItem(seq_, i, obj), -1) << getPyCallStack();
    } else {
      CHECK_NE(PySequence_SetItem(seq_, i, obj), -1) << getPyCallStack();
    }
  }

 private:
  PyObject* seq_;
};

class DictHelper {
 public:
  explicit DictHelper(PyObject* d) : dict_(d) {}

  explicit DictHelper(const PyObjectPtr& d) : dict_(d.get()) {}

  void set(const std::string& key, PyObject* item) {
    PyDict_SetItemString(dict_, key.c_str(), item);
  }

  void setBool(const std::string& key, bool b) {
    this->set(key, PyBool_FromLong(b));
  }

  void setStringList(const std::string& key,
                     const std::vector<std::string>& items) {
    auto* list = PyList_New(items.size());
    for (size_t i = 0; i < items.size(); ++i) {
      PyList_SetItem(list, i, PyString_FromString(items[i].c_str()));
    }
    this->set(key, list);
  }

 private:
  inline void checkDict() { CHECK(PyDict_Check(this->dict_)); }

  PyObject* dict_;
};

inline static bool isCallable(const PyObjectPtr& obj) {
  return PyCallable_Check(obj.get());
}

/**
 * Wrap a callable object.
 */
class CallableHelper {
 public:
  explicit CallableHelper(const PyObjectPtr& obj) : obj_(obj) {
    CHECK(py::isCallable(obj_));
  }

  ~CallableHelper() {}

  /**
   * reset args, and create new tuple.
   * @param sz args size.
   */
  void setArgsSize(size_t sz) { args.reset(PyTuple_New(sz)); }

  /**
   * Get args sequence. User can set/get by SequenceHelper.
   */
  SequenceHelper getArgs() { return SequenceHelper(args); }

  /**
   * Call python method, return an object.
   */
  PyObject* operator()() {
    PyGuard guard;
    return PyObject_Call(obj_.get(), args.get(), kwargs.get());
  }

 private:
  const PyObjectPtr& obj_;
  PyObjectPtr args;
  PyObjectPtr kwargs;
};

inline static PyObject* iterNext(const PyObjectPtr& context, bool* atEnd) {
  PyGuard g;
  PyObject* data = PyIter_Next(context.get());
  if (data == nullptr) {
    if (PyErr_ExceptionMatches(PyExc_StopIteration)) {
      PyErr_Clear();
      *atEnd = true;
      return nullptr;
    } else if (PyErr_Occurred()) {
      CHECK_PY(data) << "Calling iterator next error";
      return nullptr;
    } else {
      *atEnd = false;
      return data;  // just return none in iterator.
    }
  } else {
    *atEnd = false;
    return data;
  }
}
}  // namespace py

#endif

/**
 * Initialize python.
 */
void initPython(int argc, char** argv);

}  // namespace paddle
