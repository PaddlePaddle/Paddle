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

#include "paddle/fluid/pybind/op_callstack_utils.h"

pir::Attribute get_op_callstack_info() {
  PyThreadState* tstate = PyThreadState_GET();
  std::vector<pir::Attribute> op_callstack_infos;
  if (NULL != tstate && NULL != tstate->frame) {
    PyFrameObject* frame = tstate->frame;

    while (NULL != frame) {
      int line = PyCode_Addr2Line(frame->f_code, frame->f_lasti);
      const char* filename = PyUnicode_AsUTF8(frame->f_code->co_filename);
      const char* funcname = PyUnicode_AsUTF8(frame->f_code->co_name);
      std::string callstack_info =
          (filename + std::string("(") + std::to_string(line) +
           std::string(")") + funcname + "\n");
      op_callstack_infos.push_back(
          pir::StrAttribute::get(pir::IrContext::Instance(), callstack_info));
      frame = frame->f_back;
    }
  }
  return pir::ArrayAttribute::get(pir::IrContext::Instance(),
                                  op_callstack_infos);
}
