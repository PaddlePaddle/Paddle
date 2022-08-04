/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/pybind/jit.h"

#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/platform/place.h"

#include "paddle/fluid/jit/engine/executor_engine.h"
#include "paddle/fluid/jit/engine/pe_engine.h"
#include "paddle/fluid/jit/function_schema.h"
#include "paddle/fluid/jit/layer.h"
#include "paddle/fluid/jit/serializer.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {

PyTypeObject *g_executor_engine_pytype = nullptr;
PyTypeObject *g_pe_engine_pytype = nullptr;
using Variable = paddle::framework::Variable;

void BindJit(pybind11::module *m) {
  py::class_<jit::Layer>(*m, "Layer", R"DOC(Layer Class.)DOC")
      .def("function_dict",
           &jit::Layer::FunctionMap,
           py::return_value_policy::reference);

  py::class_<jit::ExecutorEngine, std::shared_ptr<jit::ExecutorEngine>>
      executor_engine(*m, "ExectorFunction", R"DOC(ExectorFunction Class.)DOC");
  g_executor_engine_pytype =
      reinterpret_cast<PyTypeObject *>(executor_engine.ptr());
  executor_engine.def("info", &jit::ExecutorEngine::Info);

  py::class_<jit::PEEngine, std::shared_ptr<jit::PEEngine>> pe_engine(
      *m, "PEEngine", R"DOC(PEEngine Class.)DOC");
  g_pe_engine_pytype = reinterpret_cast<PyTypeObject *>(pe_engine.ptr());
  pe_engine.def("info", &jit::PEEngine::Info);

  py::class_<jit::FunctionInfo, std::shared_ptr<jit::FunctionInfo>>(
      *m, "FunctionInfo", R"DOC(FunctionInfo Class.)DOC")
      .def("name", &jit::FunctionInfo::FunctionName)
      .def("input_names", &jit::FunctionInfo::InputArgNames)
      .def("output_names", &jit::FunctionInfo::OutputArgNames);

  m->def("Load",
         [](const std::string &path, const platform::CPUPlace &cpu_place) {
           return paddle::jit::Load(path, cpu_place);
         });

  m->def("Load",
         [](const std::string &path, const platform::CUDAPlace &cuda_place) {
           return paddle::jit::Load(path, cuda_place);
         });
}

}  // namespace pybind
}  // namespace paddle
