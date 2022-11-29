// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <Python.h>
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/hooks.h"
#include "paddle/fluid/pybind/pyobject_holder.h"
#include "paddle/fluid/pybind/saved_tensors_hooks.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

namespace paddle {
namespace pybind {

#if !(defined(PADDLE_NO_PYTHON) && defined(PADDLE_ON_INFERENCE))
class PackHookImpl : public PackHook {
 public:
  explicit PackHookImpl(PyObject* hook);

  ~PackHookImpl();

  std::shared_ptr<PyObjectHolder> operator()(
      const paddle::experimental::Tensor& tensor) override;

  void* operator()(void* py_tensor) override;

 private:
  PyObject* hook_;
};

class UnPackHookImpl : public UnPackHook {
 public:
  explicit UnPackHookImpl(PyObject* hook);

  ~UnPackHookImpl();

  paddle::experimental::Tensor operator()(
      std::shared_ptr<PyObjectHolder> packed_value) override;

  void* operator()(void* packed_value, void* other) override;

 private:
  PyObject* hook_;
};
#endif

}  // namespace pybind
}  // namespace paddle
