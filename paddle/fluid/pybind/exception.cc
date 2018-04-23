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

#include "paddle/fluid/pybind/exception.h"

namespace paddle {
namespace pybind {

void BindException(pybind11::module* m) {
  static pybind11::exception<platform::EnforceNotMet> exc(*m, "EnforceNotMet");
  pybind11::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const platform::EnforceNotMet& e) {
      exc(e.what());
    }
  });

  m->def("__unittest_throw_exception__",
         [] { PADDLE_THROW("test exception"); });
}

}  // namespace pybind
}  // namespace paddle
