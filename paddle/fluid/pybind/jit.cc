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
#include "glog/logging.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/jit/function.h"
#include "paddle/fluid/jit/function_schema.h"
#include "paddle/fluid/jit/layer.h"
#include "paddle/fluid/jit/serializer.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/pybind/eval_frame.h"
#include "paddle/fluid/pybind/eval_frame_tools.h"
#include "paddle/utils/pybind.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {

PyTypeObject *g_jit_function_pytype = nullptr;
using Variable = paddle::framework::Variable;

void BindJit(pybind11::module *m) {
  py::class_<jit::Layer>(*m, "Layer", R"DOC(Layer Class.)DOC")
      .def("function_names", &jit::Layer::FunctionNames)
      .def("function", &jit::Layer::Function)
      .def("function_info", &jit::Layer::FunctionInfo);

  py::class_<jit::Function, std::shared_ptr<jit::Function>> function(
      *m, "Function", R"DOC(Function Class.)DOC");
  g_jit_function_pytype = reinterpret_cast<PyTypeObject *>(function.ptr());

  py::class_<jit::FunctionInfo, std::shared_ptr<jit::FunctionInfo>>(
      *m, "FunctionInfo", R"DOC(FunctionInfo Class.)DOC")
      .def("name", &jit::FunctionInfo::FunctionName)
      .def("input_names", &jit::FunctionInfo::InputArgNames)
      .def("output_names", &jit::FunctionInfo::OutputArgNames);

  m->def("Load", [](const std::string &path, const phi::CPUPlace &cpu_place) {
    return paddle::jit::Load(path, cpu_place);
  });

  m->def("Load", [](const std::string &path, const phi::GPUPlace &cuda_place) {
    return paddle::jit::Load(path, cuda_place);
  });
}

void BindEvalFrame(pybind11::module *m) {
  PyInit__eval_frame();
  m->def(
      "set_eval_frame",
      [](const py::object &py_func) {
        VLOG(5) << "start call set_eval_frame_py.";
        auto ret = set_eval_frame_py(py_func.ptr());
        auto obj = py::reinterpret_borrow<py::object>(ret);
        return obj;
      },
      py::arg("callback"));

  m->def(
      "sot_setup_codes_with_graph",
      [](const py::object &py_codes) {
        auto ret = setup_codes_with_graph(py_codes.ptr());
        auto obj = py::reinterpret_borrow<py::object>(ret);
        return obj;
      },
      py::arg("py_codes"));

  m->def(
      "sot_set_with_graph",
      [](const py::object &py_codes) {
        auto ret = set_with_graph(py_codes.ptr());
        auto obj = py::reinterpret_borrow<py::object>(ret);
        return obj;
      },
      py::arg("py_codes"));

  m->def(
      "eval_frame_no_skip_codes",
      [](const py::object &py_codes) {
        auto ret = no_skip_codes(py_codes.ptr());
        auto obj = py::reinterpret_borrow<py::object>(ret);
        return obj;
      },
      py::arg("py_codes"));

  m->def(
      "eval_frame_skip_file_prefix",
      [](const py::object &py_codes) {
        auto ret = skip_file_prefix(py_codes.ptr());
        auto obj = py::reinterpret_borrow<py::object>(ret);
        return obj;
      },
      py::arg("py_codes"));
}

}  // namespace pybind
}  // namespace paddle
