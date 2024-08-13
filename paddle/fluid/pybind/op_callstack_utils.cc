// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <Python.h>
#include <frameobject.h>

#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/op_callstack_utils.h"

pir::Attribute CallStackRecorder::GetOpCallstackInfo() {
  PyObject* traceback_str = PyUnicode_FromString("traceback");
  PyObject* traceback_module = PyImport_Import(traceback_str);

  if (nullptr == traceback_module) {
    Py_DECREF(traceback_str);
    Py_DECREF(traceback_module);
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "Failed to import traceback module while getting callstack information "
        "for %s.",
        api_name_));
  }
  PyObject* tb = PyObject_GetAttrString(traceback_module, "extract_stack");
  PyObject* stack = PyObject_CallObject(tb, nullptr);
  if (nullptr == stack) {
    Py_DECREF(tb);
    Py_DECREF(traceback_str);
    Py_DECREF(traceback_module);
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "Failed to get callstack object while getting callstack information "
        "for "
        "%s.",
        api_name_));
  }
  Py_ssize_t stack_size = PyList_Size(stack);
  std::vector<pir::Attribute> op_callstack_infos;
  for (Py_ssize_t i = 0; i < stack_size; ++i) {
    PyObject* frame_summary = PyList_GetItem(stack, i);
    PyObject* filename = PyObject_GetAttrString(frame_summary, "filename");
    PyObject* lineno = PyObject_GetAttrString(frame_summary, "lineno");
    PyObject* name = PyObject_GetAttrString(frame_summary, "name");
    PyObject* line = PyObject_GetAttrString(frame_summary, "line");
    PyObject* callstack_info = PyUnicode_FromFormat(
        "  File \"%S\", line %S, in %S", filename, lineno, name);
    PyObject* callstack_source_line = PyUnicode_FromFormat("    %S", line);
    op_callstack_infos.push_back(
        pir::StrAttribute::get(pir::IrContext::Instance(),
                               std::string(PyUnicode_AsUTF8(callstack_info))));
    op_callstack_infos.push_back(pir::StrAttribute::get(
        pir::IrContext::Instance(),
        std::string(PyUnicode_AsUTF8(callstack_source_line))));
    Py_DECREF(callstack_info);
    Py_DECREF(callstack_source_line);
    Py_DECREF(filename);
    Py_DECREF(lineno);
    Py_DECREF(name);
    Py_DECREF(line);
  }
  Py_DECREF(tb);
  Py_DECREF(traceback_str);
  Py_DECREF(traceback_module);
  return pir::ArrayAttribute::get(pir::IrContext::Instance(),
                                  op_callstack_infos);
}

void CallStackRecorder::Record() {
  auto before_insertion_point =
      paddle::dialect::ApiBuilder::Instance().GetCurrentInsertionPoint();
  before_insertion_iterator_ = (--before_insertion_point.second);
  before_insertion_block_ = before_insertion_point.first;
}

void CallStackRecorder::AttachToOps() {
  before_insertion_iterator_++;
  pir::Attribute callstack_info_attr = GetOpCallstackInfo();
  pir::InsertionPoint after_insertion_point =
      paddle::dialect::ApiBuilder::Instance().GetCurrentInsertionPoint();
  PADDLE_ENFORCE_EQ(before_insertion_block_,
                    after_insertion_point.first,
                    common::errors::PreconditionNotMet(
                        "The block obtained before and after calling the "
                        "static API %s is inconsistent.",
                        api_name_));
  auto after_insertion_iterator = after_insertion_point.second;
  for (auto block_iterator = before_insertion_iterator_;
       block_iterator != after_insertion_iterator;
       block_iterator++) {
    block_iterator->set_attribute(paddle::framework::OpProtoAndCheckerMaker::
                                      OpCreationCallstackAttrName(),
                                  callstack_info_attr);
  }
}
