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

#include <unordered_set>

#include "paddle/common/errors.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/phi/core/enforce.h"

/*============================ Dict Tree ================================*/

class TreeNode {
 public:
  TreeNode() = default;
  ~TreeNode() { clear(); }
  void clear();
  int add_prefix(const char* filename);
  int check_filename(const char* filename);

 private:
  int is_prefix;
  TreeNode* children[256];
};

void TreeNode::clear() {
  for (int i = 0; i < 256; i++) {
    if (children[i] != nullptr) delete children[i];
  }
}

int TreeNode::add_prefix(const char* filepath) {
  if (is_prefix) return 0;
  if (filepath[0] == '\0') return 1;

  int ch = (int)filepath[0];  // NOLINT
  if (children[ch] == nullptr) {
    TreeNode* node = new TreeNode();
    children[ch] = node;
  }

  if (children[ch]->add_prefix(filepath + 1)) is_prefix = 1;

  return 0;
}

int TreeNode::check_filename(const char* filename) {
  int cur_idx = 0;
  TreeNode* cur_node = this;

  while (filename[cur_idx] != '\0') {
    cur_node = cur_node->children[(int)filename[cur_idx]];  // NOLINT
    if (cur_node == nullptr) return 0;
    if (cur_node->is_prefix) return 1;
    cur_idx += 1;
  }

  return 0;
}

/*========================== utils  ==========================*/

const char* pystr_to_cstr(PyObject* pystr) {
  if (PyUnicode_Check(pystr))
    return PyUnicode_AsUTF8(pystr);
  else
    PADDLE_THROW(phi::errors::InvalidArgument("Input PyObject is not string!"));
}

/*========================== SkipCodeInfo ===============================*/

class SkipCodeInfo {
 public:
  static SkipCodeInfo& Instance();
  void clear_code_info();

  void add_no_skip_code(PyCodeObject* code);
  void add_skip_file_prefix(PyObject* filepath);

  int is_no_skip_code(PyCodeObject* code);
  int in_skip_path(PyObject* filename);

 private:
  SkipCodeInfo() {
    no_skip_codes = std::unordered_set<PyCodeObject*>();
    skip_codes = std::unordered_set<PyCodeObject*>();
    root = new TreeNode();
  }
  ~SkipCodeInfo() { clear_code_info(); }
  std::unordered_set<PyCodeObject*> no_skip_codes;
  std::unordered_set<PyCodeObject*> skip_codes;
  TreeNode* root;
};

SkipCodeInfo& SkipCodeInfo::Instance() {
  static SkipCodeInfo _instance;
  return _instance;
}

void SkipCodeInfo::clear_code_info() {
  no_skip_codes.clear();
  skip_codes.clear();
  root->clear();
}

void SkipCodeInfo::add_no_skip_code(PyCodeObject* code) {
  no_skip_codes.insert(code);
}

void SkipCodeInfo::add_skip_file_prefix(PyObject* filepath) {
  const char* path = pystr_to_cstr(filepath);
  root->add_prefix(path);
}

int SkipCodeInfo::is_no_skip_code(PyCodeObject* code) {
  return no_skip_codes.find(code) != no_skip_codes.end();
}

int SkipCodeInfo::in_skip_path(PyObject* filename) {
  const char* name = pystr_to_cstr(filename);
  return root->check_filename(name);
}

/*========================== code status ==============================*/
enum CodeState { UNKNOW, WITH_GRAPH, WITHOUT_GRAPH };

class CodeInfo {
 public:
  CodeState state;
  int counter;
};

class CodeStatus {
 public:
  static CodeStatus& Instance();
  int is_code_without_graph(PyCodeObject* code);
  void set_with_graph(PyCodeObject* code);
  void add_with_graph_code(PyCodeObject* code);
  void clear();

 private:
  CodeStatus() { code_map = std::unordered_map<PyCodeObject*, CodeInfo*>(); }
  ~CodeStatus() { clear(); }
  std::unordered_map<PyCodeObject*, CodeInfo*> code_map;
};

CodeStatus& CodeStatus::Instance() {
  static CodeStatus _instance;
  return _instance;
}

int CodeStatus::is_code_without_graph(PyCodeObject* code) {
  CodeInfo* code_info;
  if (code_map.find(code) != code_map.end()) {
    code_info = code_map[code];
  } else {
    code_info = new CodeInfo();
    code_map.emplace(code, code_info);
  }
  if (code_info->state == WITHOUT_GRAPH) return 1;
  if (code_info->state == UNKNOW) {
    code_info->counter += 1;
    if (code_info->counter >= 10) code_info->state = WITHOUT_GRAPH;
  }
  return 0;
}

void CodeStatus::set_with_graph(PyCodeObject* code) {
  CodeInfo* code_info;
  if (code_map.find(code) != code_map.end()) {
    code_info = code_map[code];
    code_info->state = WITH_GRAPH;
  }
}

void CodeStatus::add_with_graph_code(PyCodeObject* code) {
  CodeInfo* code_info;
  if (code_map.find(code) != code_map.end()) {
    code_info = code_map[code];
    code_info->state = WITH_GRAPH;
  } else {
    code_info = new CodeInfo();
    code_info->state = WITH_GRAPH;
    code_map.emplace(code, code_info);
  }
}

void CodeStatus::clear() {
  for (auto iter = code_map.begin(); iter != code_map.end(); iter++) {
    delete iter->second;
  }
  code_map.clear();
}

/*========================== interfaces ===============================*/

int need_skip(FrameObject* frame) {
  auto& skip_info = SkipCodeInfo::Instance();
  PyCodeObject* code = frame->f_code;  // NOLINT
  PyObject* co_filename = code->co_filename;

  if (skip_info.is_no_skip_code(code)) {
    return 0;
  }

#if PY_VERSION_HEX >= 0x030b0000
  const char* filename = pystr_to_cstr(co_filename);
  PyObject* _filename = NULL;
  if (memcmp(filename, "<frozen", 7) == 0) {
    PyObject* f_globals = frame->f_globals;
    _filename = PyDict_GetItemString(f_globals, "__file__");
    if (_filename != NULL) {
      Py_INCREF(_filename);
      co_filename = _filename;
    }
  }
#endif

  int result = skip_info.in_skip_path(co_filename);

#if PY_VERSION_HEX >= 0x030b0000
  if (_filename != NULL) Py_DECREF(_filename);
#endif
  return result;
}

int is_code_without_graph(PyCodeObject* code) {
  auto& code_status = CodeStatus::Instance();
  return code_status.is_code_without_graph(code);
}

/*========================== pybind ===============================*/
PyObject* set_with_graph(PyObject* code) {
  auto& code_status = CodeStatus::Instance();
  code_status.set_with_graph((PyCodeObject*)code);  // NOLINT
  return Py_None;
}

PyObject* setup_codes_with_graph(PyObject* code_tuple) {
  auto& code_status = CodeStatus::Instance();
  Py_ssize_t size = PyTuple_GET_SIZE(code_tuple);
  for (Py_ssize_t i = 0; i < size; i++) {
    PyCodeObject* code =
        (PyCodeObject*)PyTuple_GetItem(code_tuple, i);  // NOLINT
    code_status.add_with_graph_code(code);
  }
  return Py_None;
}

PyObject* no_skip_codes(PyObject* code_tuple) {
  auto& skip_info = SkipCodeInfo::Instance();
  Py_ssize_t size = PyTuple_GET_SIZE(code_tuple);
  for (Py_ssize_t i = 0; i < size; i++) {
    PyCodeObject* code =
        (PyCodeObject*)PyTuple_GetItem(code_tuple, i);  // NOLINT
    skip_info.add_no_skip_code(code);
  }
  return Py_None;
}

PyObject* skip_file_prefix(PyObject* filepath_tuple) {
  auto& skip_info = SkipCodeInfo::Instance();
  Py_ssize_t size = PyTuple_GET_SIZE(filepath_tuple);
  for (Py_ssize_t i = 0; i < size; i++) {
    PyObject* code = PyTuple_GetItem(filepath_tuple, i);
    skip_info.add_skip_file_prefix(code);
  }
  return Py_None;
}
