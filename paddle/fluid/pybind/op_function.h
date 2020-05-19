// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/pybind/imperative.h"

namespace py = pybind11;
namespace paddle {
namespace pybind {
static inline void ConstructAttrMapFromPyArgs(framework::AttributeMap* attrs,
                                              const py::args& args) {
  PADDLE_ENFORCE_EQ(
      args.size() % 2, 0,
      platform::errors::InvalidArgument(
          "The number of arguments for arributes should be even."));
  for (size_t i = 0; i < args.size(); i += 2) {
    auto name = args[i].cast<std::string>();
    auto value = args[i + 1].cast<framework::Attribute>();
    (*attrs)[name] = value;
  }
}

static inline std::vector<std::shared_ptr<imperative::VarBase>>
ConstructDuplicableOutput(const size_t num) {
  auto tracer = imperative::GetCurrentTracer();
  std::vector<std::shared_ptr<imperative::VarBase>> res;
  res.reserve(num);
  for (size_t i = 0; i < num; i++) {
    auto var_base_name = tracer->GenerateUniqueName();
    res.emplace_back(new imperative::VarBase(var_base_name));
  }
  return res;
}
}  // namespace pybind
}  // namespace paddle

// This include must be the last line
#include "paddle/fluid/pybind/op_function_impl.h"
