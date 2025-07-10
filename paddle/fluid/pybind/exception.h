/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/platform/enforce.h"
#include "pybind11/pybind11.h"

#define EAGER_TRY try {
#define EAGER_CATCH_AND_THROW_RETURN_NULL             \
  }                                                   \
  catch (...) {                                       \
    ThrowExceptionToPython(std::current_exception()); \
    return nullptr;                                   \
  }

#define EAGER_CATCH_AND_THROW_RETURN_NEG              \
  }                                                   \
  catch (...) {                                       \
    ThrowExceptionToPython(std::current_exception()); \
    return -1;                                        \
  }

namespace paddle {
namespace pybind {

void BindException(pybind11::module* m);
void ThrowExceptionToPython(std::exception_ptr p);

}  // namespace pybind
}  // namespace paddle
