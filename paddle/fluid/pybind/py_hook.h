/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include <Python.h>
#include <memory>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "paddle/fluid/imperative/hooks.h"
#include "paddle/fluid/imperative/type_defs.h"

namespace paddle {
namespace pybind {

using VariableWrapper = imperative::VariableWrapper;

class PyHook : Hook {
 public:
  explicit PyHook(PyObject* hook);
  ~PyHook() override;

  std::shared_ptr<VariableWrapper> operator()(
      const std::shared_ptr<VariableWrapper>& var) override;

 private:
  // users may add multiple hooks for one op
  PyObject* hook_;
};

}  // namespace pybind
}  // namespace paddle
