/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_call_stack.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {

void InsertCallStackInfo(const std::string &type, const AttributeMap &attrs,
                         platform::EnforceNotMet *exception) {
  if (attrs.count("sub_block") != 0) {
    return;
  }

  const std::vector<std::string> *callstack = nullptr;
  auto iter = attrs.find(OpProtoAndCheckerMaker::OpCreationCallstackAttrName());
  if (iter != attrs.end()) {
    callstack = &boost::get<std::vector<std::string>>(iter->second);
    if (callstack->empty()) callstack = nullptr;
  }

  std::ostringstream sout;
  std::ostringstream sout_py_trace;
  // Step 1. Construct python call stack string
  if (callstack) {
    sout_py_trace << "\n------------------------------------------\n";
    sout_py_trace << "Python Call Stacks (More useful to users):";
    sout_py_trace << "\n------------------------------------------\n";
    for (auto &line : *callstack) {
      sout_py_trace << line;
    }
  }
  // Step 2. Insert python traceback into err_str_
  std::size_t found = exception->err_str_.rfind(
      "\n----------------------\nError Message "
      "Summary:\n----------------------\n");
  if (found != std::string::npos) {
    exception->err_str_.insert(found, sout_py_trace.str());
  } else {
    exception->err_str_.append(sout_py_trace.str());
  }
  // Step 3. Construct final call stack & append error op name
  sout << exception->err_str_;
  if (callstack) {
    sout << "  [operator < " << type << " > error]";
  }
  exception->err_str_ = sout.str();
}

}  // namespace framework
}  // namespace paddle
