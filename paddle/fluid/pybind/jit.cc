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

<<<<<<< HEAD
#include "paddle/fluid/jit/function.h"
#include "paddle/fluid/jit/function_schema.h"
#include "paddle/fluid/jit/layer.h"
=======
#include "paddle/fluid/jit/executor_function.h"
#include "paddle/fluid/jit/function_schema.h"
#include "paddle/fluid/jit/layer.h"
#include "paddle/fluid/jit/pe_function.h"
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
#include "paddle/fluid/jit/serializer.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {

<<<<<<< HEAD
PyTypeObject *g_jit_function_pytype = nullptr;
=======
PyTypeObject *g_executor_function_pytype = nullptr;
PyTypeObject *g_pe_function_pytype = nullptr;
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
using Variable = paddle::framework::Variable;

void BindJit(pybind11::module *m) {
  py::class_<jit::Layer>(*m, "Layer", R"DOC(Layer Class.)DOC")
<<<<<<< HEAD
      .def("function_names", &jit::Layer::FunctionNames)
      .def("function", &jit::Layer::Function)
      .def("function_info", &jit::Layer::FunctionInfo);

  py::class_<jit::Function, std::shared_ptr<jit::Function>> function(
      *m, "Function", R"DOC(Function Class.)DOC");
  g_jit_function_pytype = reinterpret_cast<PyTypeObject *>(function.ptr());
=======
      .def("function_dict",
           &jit::Layer::FunctionMap,
           py::return_value_policy::reference);

  py::class_<jit::ExecutorFunction, std::shared_ptr<jit::ExecutorFunction>>
      executor_function(
          *m, "ExectorFunction", R"DOC(ExectorFunction Class.)DOC");
  g_executor_function_pytype =
      reinterpret_cast<PyTypeObject *>(executor_function.ptr());
  executor_function.def("info", &jit::ExecutorFunction::Info);

  py::class_<jit::PEFunction, std::shared_ptr<jit::PEFunction>> pe_function(
      *m, "PEFunction", R"DOC(PEFunction Class.)DOC");
  g_pe_function_pytype = reinterpret_cast<PyTypeObject *>(pe_function.ptr());
  pe_function.def("info", &jit::PEFunction::Info);
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

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
