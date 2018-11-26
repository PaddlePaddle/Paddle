/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/pybind/imperative.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/imperative/tracer.h"

namespace paddle {
namespace pybind {

// Bind Methods
void BindTracer(pybind11::module *m) {
  pybind11::class_<imperative::Tracer>(*m, "Tracer", "")
      .def(pybind11::init<>())
      .def("trace", &imperative::Tracer::Trace)
      .def_property("scope",
                    [](const imperative::Tracer &self) { return self.Scope(); },
                    [](imperative::Tracer &self, framework::Scope *scope) {
                      self.SetScope(scope);
                    },
                    R"DOC()DOC")
      .def_property("block",
                    [](const imperative::Tracer &self) { return self.Block(); },
                    [](imperative::Tracer &self, framework::BlockDesc *block) {
                      self.SetBlock(block);
                    },
                    R"DOC()DOC");
}

}  // namespace pybind
}  // namespace paddle
