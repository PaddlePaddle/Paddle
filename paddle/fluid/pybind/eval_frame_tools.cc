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

#include "paddle/fluid/pybind/eval_frame_tools.h"

#include <Python.h>
#include <frameobject.h>
#include <set>

#include "glog/logging.h"

class SkipCodeInfo {
 public:
  static SkipCodeInfo& Instance();
  void add_custom_skip_code(PyObject* code);
  void clear_code_info();

  std::set<PyObject*> no_skip_code;
  std::set<PyObject*> customed_skip_code;

 private:
  SkipCodeInfo() {
    VLOG(1) << "[fei] construction";
    no_skip_code = std::set<PyObject*>();
    customed_skip_code = std::set<PyObject*>();
  }
};

SkipCodeInfo& SkipCodeInfo::Instance() {
  static SkipCodeInfo _instance;
  return _instance;
}

void SkipCodeInfo::add_custom_skip_code(PyObject* code) {
  VLOG(1) << "[fei] before add size: " << customed_skip_code.size();
  customed_skip_code.insert(code);
  VLOG(1) << "[fei] after add size: " << customed_skip_code.size();
}

void SkipCodeInfo::clear_code_info() {
  VLOG(1) << "[fei] clear";
  no_skip_code.clear();
  customed_skip_code.clear();
}

int need_skip(PyObject* frame) {
  auto& skip_info = SkipCodeInfo::Instance();
  PyObject* code = (PyObject*)((PyFrameObject*)frame)->f_code;  // NOLINT

  VLOG(1) << "[fei] set size: " << skip_info.customed_skip_code.size();
  auto search = skip_info.customed_skip_code.find(code);
  if (search != skip_info.customed_skip_code.end()) {
    VLOG(1) << "[fei] check True";
    return 1;
  } else {
    VLOG(1) << "[fei] check False";
    return 0;
  }
}

PyObject* zskip(PyObject* files) {
  auto& skip_info = SkipCodeInfo::Instance();
  Py_ssize_t size = PyTuple_GET_SIZE(files);
  for (Py_ssize_t i = 0; i < size; i++) {
    PyObject* obj = PyTuple_GetItem(files, i);
    skip_info.add_custom_skip_code(obj);
  }
  return Py_None;
}
